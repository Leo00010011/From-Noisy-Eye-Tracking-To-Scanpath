"""Lightweight image-reliance statistics gathered during ``validate()``.

Unlike the offline diagnostic suite (``image_reliance.py``), this needs no ``InferenceRecorder``:
the decoder layers call ``probe_begin`` / ``probe_cross`` / ``probe_gate`` / ``probe_end`` when a probe is attached, and the
probe accumulates running sums on-device (no per-step host sync). Stats per decoder layer:

- ``*_res_ratio`` = mean ||x|| / mean ||ca_out|| — the residual stream *before* the cross-attention
  is added (and before any norm) over the cross-attention output. Large ⇒ the modality is swamped.
- ``gate_logit`` = mean of ``fc_gate([img, dec])`` (pre-sigmoid), ``gate`` = mean sigmoid, i.e. the
  per-channel weight on the image; ``gate_sat_img`` / ``gate_sat_dec`` = fraction of channels with
  gate > 0.9 / < 0.1; ``gate_img_share`` = mean ||g*img|| / (mean ||g*img|| + mean ||(1-g)*dec||),
  the magnitude-aware share of the fused state that comes from the image (the gate alone ignores
  that the two inputs have very different norms).

Only valid query tokens are counted (``tgt_mask`` for the fixation decoder, ``src_mask`` for the eye
decoder), and each query position once per batch: the autoregressive sampler re-runs the whole
prefix when the KV cache is off, so positions already seen in the batch are skipped.
"""
import torch


def _probe_of(module):
    probe = getattr(module, "_reliance_probe", None)
    if probe is None or not probe.active:
        return None
    return probe


class RelianceProbe:
    def __init__(self, model):
        self.active = False
        self.layers = []  # (key, module)
        self._attach(model)
        self.reset()

    def _attach(self, model):
        from src.model.blocks import DeformableDecoder, DeformableDoubleInputDecoder
        root = getattr(model, "_orig_mod", model)  # torch.compile wrapper
        for name, module in root.named_modules():
            if isinstance(module, DeformableDoubleInputDecoder) and name.startswith("decoder."):
                key = f"dec{name.split('.')[1]}"
            elif isinstance(module, DeformableDecoder) and name.startswith("eye_decoder."):
                key = f"eye{name.split('.')[1]}"
            else:
                continue
            module._reliance_probe = self
            module._reliance_key = key
            self.layers.append((key, module))

    def detach(self):
        for _, module in self.layers:
            module._reliance_probe = None
        self.layers = []

    def __len__(self):
        return len(self.layers)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def reset(self):
        self.sums = {}
        self._seen = {}
        self._masks = {}

    def start_batch(self, src_mask=None, tgt_mask=None):
        self._seen = {}
        self._masks = {"eye": src_mask, "dec": tgt_mask}

    # ── accumulation ──────────────────────────────────────────────────────
    def _valid(self, module, kind, x):
        """(B, Nq) bool of query tokens to count this call (valid and not yet seen in this batch)."""
        key = module._reliance_key
        B, Nq = x.shape[:2]
        start = getattr(module, "_reliance_start", 0)
        pos = torch.arange(start, start + Nq, device=x.device)
        seen = self._seen.get(key, 0)
        valid = (pos >= seen).unsqueeze(0).expand(B, Nq)
        mask = self._masks.get(kind)
        if mask is not None:
            idx = pos.clamp(max=mask.size(1) - 1)
            valid = valid & mask.to(x.device).bool()[:, idx] & (pos < mask.size(1)).unsqueeze(0)
        return valid

    def _add(self, name, value):
        self.sums[name] = self.sums.get(name, 0) + value

    def cross_update(self, module, kind, stream, x, ca_out, valid):
        key = module._reliance_key
        w = valid.to(x.dtype)
        self._add(f"{key}/{stream}_res", (x.norm(dim=-1) * w).sum())
        self._add(f"{key}/{stream}_ca", (ca_out.norm(dim=-1) * w).sum())
        self._add(f"{key}/{stream}_n", w.sum())

    def gate_update(self, module, logit, gate, img, dec, valid):
        key = module._reliance_key
        w = valid.to(logit.dtype)
        n = w.sum()
        D = logit.size(-1)
        self._add(f"{key}/gate_logit", (logit.mean(-1) * w).sum())
        self._add(f"{key}/gate", (gate.mean(-1) * w).sum())
        self._add(f"{key}/gate_sat_img", ((gate > 0.9).to(w.dtype).sum(-1) / D * w).sum())
        self._add(f"{key}/gate_sat_dec", ((gate < 0.1).to(w.dtype).sum(-1) / D * w).sum())
        self._add(f"{key}/gate_img_norm", ((gate * img).norm(dim=-1) * w).sum())
        self._add(f"{key}/gate_dec_norm", (((1 - gate) * dec).norm(dim=-1) * w).sum())
        self._add(f"{key}/gate_n", n)

    def end_call(self, module, valid_pos_max):
        key = module._reliance_key
        self._seen[key] = max(self._seen.get(key, 0), valid_pos_max)

    # ── summary ───────────────────────────────────────────────────────────
    def summary(self):
        """Flat ``{metric_name: float}``; per layer plus a mean over layers per metric family."""
        s = {k: float(v) for k, v in self.sums.items()}
        out = {}
        for key, _ in self.layers:
            for stream in ("img", "gaze"):
                n = s.get(f"{key}/{stream}_n", 0.0)
                ca = s.get(f"{key}/{stream}_ca", 0.0)
                if n > 0 and ca > 0:
                    out[f"{stream}_res_ratio_{key}"] = s[f"{key}/{stream}_res"] / ca
            n = s.get(f"{key}/gate_n", 0.0)
            if n > 0:
                for stat in ("gate_logit", "gate", "gate_sat_img", "gate_sat_dec"):
                    out[f"{stat}_{key}"] = s[f"{key}/{stat}"] / n
                total = s[f"{key}/gate_img_norm"] + s[f"{key}/gate_dec_norm"]
                if total > 0:
                    out[f"gate_img_share_{key}"] = s[f"{key}/gate_img_norm"] / total
        # layer-mean per family (e.g. img_res_ratio_dec, gate_eye ...)
        families = {}
        for name, value in out.items():
            base, key = name.rsplit("_", 1)
            prefix = "dec" if key.startswith("dec") else "eye"
            families.setdefault(f"{base}_{prefix}", []).append(value)
        for fam, values in families.items():
            out[f"{fam}_mean"] = sum(values) / len(values)
        return out


def probe_cross(module, kind, stream, x, ca_out):
    """Record residual-vs-cross-attention norms; ``kind`` ∈ {"eye", "dec"}, ``stream`` ∈ {"img", "gaze"}."""
    probe = _probe_of(module)
    if probe is None:
        return
    with torch.no_grad():
        valid = probe._valid(module, kind, x)
        probe.cross_update(module, kind, stream, x.float(), ca_out.float(), valid)


def probe_gate(module, gate_module, img, dec):
    """Recompute the GatedFusion projection (cheap) and record its statistics."""
    probe = _probe_of(module)
    if probe is None:
        return
    with torch.no_grad():
        valid = probe._valid(module, "dec", dec)
        logit = gate_module.fc_gate(torch.cat((img, dec), dim=-1)).float()
        probe.gate_update(module, logit, torch.sigmoid(logit), img.float(), dec.float(), valid)


def probe_begin(module):
    """Snapshot the KV-cache length (= position of the first query token) before self-attention."""
    if _probe_of(module) is None:
        return
    module._reliance_start = (module.get_cached_input_count()
                              if hasattr(module, "get_cached_input_count") else 0)


def probe_end(module, x):
    """Mark this call's query positions as seen (call once, after all probe_* calls of a layer)."""
    probe = _probe_of(module)
    if probe is None:
        return
    probe.end_call(module, getattr(module, "_reliance_start", 0) + x.size(1))

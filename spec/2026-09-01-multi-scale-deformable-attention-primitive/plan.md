# Plan — F1: Multi-scale Deformable Attention Primitive

## Context and Design Decisions

**Why F1 is first.** The migration (TechStack §"Deformable param-layout compatibility") hinges on
one dual constraint: the extended `DeformableAttention` must (a) reproduce Mask2Former's
`MSDeformAttn` parameter layout so F2 can load pretrained pixel-decoder weights by name/shape, and
(b) collapse to today's exact single-scale shapes and numerics at `n_levels=1` so existing
checkpoints load with zero key mismatch and identical activations. Nothing downstream can be built
until this contract is proven, so F1 ships before F2/F4.

**Extend in place, keep the class name.** Locked decision (Roadmap F1, TechStack locked-decision 2):
generalize `DeformableAttention` rather than fork a new class. The class name is preserved so old
`state_dict`s map by name. The documented escape hatch — splitting into a sibling
`MultiScaleDeformableAttention` — is invoked only if multi-level branching measurably degrades the
single-scale path; the plan below keeps both paths in one class with a cheap `n_levels==1` fast
lane, so the escape hatch is not expected to trigger.

**Param layout mirrors Mask2Former exactly.** From
`mask2former/.../ops/modules/ms_deform_attn.py`: `sampling_offsets` is
`Linear(d, n_heads·n_levels·n_points·2)`, `attention_weights` is
`Linear(d, n_heads·n_levels·n_points)`, softmax over the flattened `n_levels·n_points` axis, per-level
`offset_normalizer = [Wₗ, Hₗ]`. We reproduce this ordering (`head → level → point → xy`) so a
reshape `view(n_heads, n_levels, n_points, ...)` lines up with the checkpoint. At `n_levels=1` this
is exactly today's `(n_heads, n_points, ...)` with a size-1 level axis inserted — byte-identical
after `view(-1)`.

**Signature choice — overload `spatial_shape`, zero decoder edits** (clarified with the user). We
keep the existing keyword `spatial_shape` and let it be either a `(H,W)` tuple (legacy, 1 level) or
a `(n_levels,2)` tensor, and accept an optional `level_start_index`. This means F1 touches **only**
`DeformableAttention`; `DeformableDecoder`/`DeformableDoubleInputDecoder` keep their current calls
verbatim and stay byte-identical. F4 later opts into the tensor form.

**Pure PyTorch, no CUDA op** (locked decision 5). Sampling is `F.grid_sample` per level, looped over
`n_levels` (typically ≤ 3), keeping the module `torch.compile`-free-friendly and
`InferenceRecorder`-compatible. We deliberately do **not** vendor Mask2Former's
`ms_deform_attn_core_pytorch`; F1's core is self-contained and its multi-scale numerics are
cross-checked against the pretrained kernel only in F2, per the user's validation choice.

**KV-cache stays.** The projected value depends only on `value`, so it is cached once when
`enable_memory_kv_cache()` is set — unchanged contract, generalized to the flattened multi-level
memory. Precondition (documented): memory geometry (`spatial_shape`) is fixed while the cache is
warm, which holds for autoregressive inference where the image memory is constant across decode
steps.

**Init parity vs. byte-identity.** Mask2Former additionally zeros `value_proj`/`output_proj`
*biases*; today's class does not. Since a loaded checkpoint overwrites those biases, the init
difference is irrelevant to weight loading. To preserve fresh-init byte-identity with existing runs
(FR2), F1 **keeps today's init** (does not add the bias-zeroing). This is a conscious divergence
from Mask2Former's `_reset_parameters` and is safe.

---

## Step 1 — Generalize the constructor (`src/model/blocks.py`, `DeformableAttention.__init__`)

Add `n_levels: int = 1` to the signature (place it after `num_points`, before `attn_dropout`, so
existing keyword calls are unaffected; all current call sites use keywords). Store `self.n_levels`.
Resize the two projections:

```python
self.n_levels = n_levels
self.sampling_offsets  = nn.Linear(embed_dim, num_heads * n_levels * num_points * 2, **factory_kwargs)
self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * num_points,     **factory_kwargs)
# value_proj / output_proj / dropout / cache flags: unchanged
```

Everything else in `__init__` (head_dim, geometric_sigma, attn_dropout, normalize_grid_init, cache
flags) is unchanged. Call `self._reset_parameters()` at the end as today.

## Step 2 — Generalize the star-pattern init (`_reset_parameters`)

Build the bias with an explicit level axis so it collapses to today's vector at `n_levels=1`:

```python
nn.init.constant_(self.sampling_offsets.weight.data, 0.)
thetas = torch.arange(self.num_heads, dtype=self.dtype, device=self.device) * (2.0 * math.pi / self.num_heads)
grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)                  # (H, 2)
if self.normalize_grid_init:
    grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
# (H, 2) -> (H, L, P, 2)
grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.n_levels, self.num_points, 1)
for i in range(self.num_points):
    grid_init[:, :, i, :] *= (i + 1)
with torch.no_grad():
    self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
nn.init.constant_(self.attention_weights.weight.data, 0.)
nn.init.constant_(self.attention_weights.bias.data, 0.)
nn.init.xavier_uniform_(self.value_proj.weight.data)
nn.init.xavier_uniform_(self.output_proj.weight.data)   # biases left at PyTorch default (see design note)
```

At `n_levels=1`, `view(H,1,1,2).repeat(1,1,P,1)` flattens in the same `head→point→xy` order as
today's `unsqueeze(1).repeat(1,P,1)` — proven in validation Group 1.

## Step 3 — Input normalization helper inside `forward`

At the top of `forward`, normalize the polymorphic inputs to canonical multi-level tensors:

```python
# spatial_shape -> shapes tensor (L,2) + per-level (H,W) list
if torch.is_tensor(spatial_shape):
    shapes = spatial_shape.to(torch.long)              # (L, 2)
else:                                                  # legacy (H, W) tuple
    shapes = query.new_tensor([spatial_shape], dtype=torch.long)  # (1, 2)
L = shapes.shape[0]
if L != self.n_levels:
    raise ValueError(f"spatial_shape has {L} levels but module has n_levels={self.n_levels}")
level_sizes = (shapes[:, 0] * shapes[:, 1]).tolist()   # [H_l*W_l]
if sum(level_sizes) != value.shape[1]:
    raise ValueError("Σ Hₗ·Wₗ does not match value length")
if level_start_index is None:
    level_start_index = torch.tensor([0, *itertools.accumulate(level_sizes)][:-1], device=value.device)

# reference_points -> (B, Nq, L, 2)
if reference_points.shape[-1] != 2:
    raise ValueError("reference_points last dim must be 2")
if reference_points.dim() == 3:                        # (B, Nq, 2) -> broadcast across levels
    reference_points = reference_points[:, :, None, :].expand(-1, -1, L, -1)
elif reference_points.shape[2] != L:
    raise ValueError("reference_points level axis must equal n_levels")
```

(`itertools` is imported at module top; `level_start_index` is accepted for API parity and future
padding-mask support but the split below is driven by `level_sizes`.)

## Step 4 — Project + (optionally cache) the value, split per level

```python
bs = query.shape[0]
if self.value_cache is not None:
    value_levels = self.value_cache                    # list of (bs*H, head_dim, Hₗ, Wₗ)
else:
    v = self.value_proj(value)                         # (bs, ΣHₗWₗ, embed_dim)
    v = v.view(bs, value.shape[1], self.num_heads, self.head_dim)
    v_split = v.split(level_sizes, dim=1)              # L × (bs, HₗWₗ, heads, head_dim)
    value_levels = []
    for (H, W), v_l in zip(shapes.tolist(), v_split):
        # (bs, HₗWₗ, heads, head_dim) -> (bs*heads, head_dim, Hₗ, Wₗ)
        v_l = v_l.permute(0, 2, 3, 1).reshape(bs * self.num_heads, self.head_dim, H, W)
        value_levels.append(v_l)
    if self.cache_memory_kv:
        self.value_cache = value_levels
```

This preserves the current cache semantics (build once, reuse) while generalizing the artifact from
a single reshaped map to a per-level list. `clear_kv_cache` / `disable_*` already null
`self.value_cache` — no change needed.

## Step 5 — Offsets, weights, joint softmax

```python
num_queries = query.shape[1]
sampling_offsets = self.sampling_offsets(query).view(
    bs, num_queries, self.num_heads, self.n_levels, self.num_points, 2)
attention_weights = self.attention_weights(query).view(
    bs, num_queries, self.num_heads, self.n_levels * self.num_points)
attention_weights = F.softmax(attention_weights, -1).view(
    bs, num_queries, self.num_heads, self.n_levels, self.num_points)
attention_weights = self.dropout(attention_weights)

if self.training and self.geometric_sigma > 0:
    sampling_offsets = sampling_offsets + torch.randn_like(sampling_offsets) * self.geometric_sigma

# per-level offset normalizer [Wₗ, Hₗ]
offset_normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)      # (L, 2)
sampling_locations = reference_points[:, :, None, :, None, :] \
    + sampling_offsets / offset_normalizer[None, None, None, :, None, :]   # (B,Nq,H,L,P,2)
```

Note the softmax is over `n_levels·n_points` jointly (FR4). At `n_levels=1` it is softmax over
`num_points`, matching today. Record the four recorder tensors here (FR7) before sampling.

## Step 6 — Per-level grid_sample and reduce

```python
sampling_grids = 2 * sampling_locations - 1                                # [0,1]->[-1,1]
sampled_per_level = []
for l, v_l in enumerate(value_levels):
    # grid for this level: (B,Nq,H,P,2) -> (B*H, Nq, P, 2)
    grid_l = sampling_grids[:, :, :, l].permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampled = F.grid_sample(v_l, grid_l, mode='bilinear',
                            padding_mode='zeros', align_corners=False)     # (B*H, head_dim, Nq, P)
    sampled_per_level.append(sampled)

# stack levels: (B*H, head_dim, Nq, L, P) -> flatten L,P
sampled = torch.stack(sampled_per_level, dim=-2).flatten(-2)               # (B*H, head_dim, Nq, L*P)
weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
    bs * self.num_heads, 1, num_queries, self.n_levels * self.num_points)
output = (sampled * weights).sum(-1)                                       # (B*H, head_dim, Nq)
output = output.view(bs, self.num_heads * self.head_dim, num_queries).transpose(1, 2)
return self.output_proj(output)
```

At `n_levels=1` the loop runs once and `stack(...).flatten(-2)` is a no-op reshape, so the tensor
algebra reduces to today's single-`grid_sample` path (validated in Group 1).

## Step 7 — Confirm decoders are untouched

Verify (no code change) that `DeformableDecoder.__cross_attention` (`blocks.py:926`) and
`DeformableDoubleInputDecoder.__cross_attention2` (`blocks.py:1029`) still call with
`spatial_shape=self.spatial_shape` (a tuple) and `(B,Nq,2)` reference points, and that the new
`forward` handles that path. This is asserted by the Group 4 tests, not by edits.

## Step 8 — Tests (`tests/test_ms_deformable_attention.py`)

New pytest module covering Groups 1–4 in `validation.md`. Mirror the style of existing
`tests/test_*.py` (plain `pytest`, CPU, small dummy tensors, fixed seeds).

---

## Implementation Order

1. **Step 1** — add `n_levels` to `__init__`, resize projections.
2. **Step 2** — level-aware star-pattern `_reset_parameters`.
3. **Step 3** — polymorphic input normalization in `forward`.
4. **Step 4** — value projection, per-level split, KV-cache list.
5. **Step 5** — offsets, weights, joint softmax, per-level locations, recorder hooks.
6. **Step 6** — per-level `grid_sample`, level+point reduction, output projection.
7. **Step 7** — confirm (via tests) `DeformableDecoder` / `DeformableDoubleInputDecoder` unchanged.
8. **Step 8** — write `tests/test_ms_deformable_attention.py` (Groups 1–4).

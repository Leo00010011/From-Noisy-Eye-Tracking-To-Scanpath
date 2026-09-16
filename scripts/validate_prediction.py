# validate_prediction.py  --  mirrors tools/rescore/contract.py (FR4) of the eval repo
# Copied verbatim from few-shot-scanpath spec/handoff/prediction_contract.md §5 (stdlib only).
# Usage: python scripts/validate_prediction.py fixations.json pred_seed0.json [...]
import hashlib, json, math, sys
from collections import Counter

GT_SHA = "46c6926f6075f4c7038138ea5116feaeec52efb201104133d6c96a7032c5ac9b"
W, H = 512, 384

def main(gt_path, *pred_paths):
    assert hashlib.sha256(open(gt_path, "rb").read()).hexdigest() == GT_SHA, "wrong fixations.json"
    gt = json.load(open(gt_path, encoding="utf-8"))
    keys = {(r["name"], r["subject"]) for r in gt if r["split"] == "test"}
    assert len(keys) == 1062
    for p in pred_paths:
        pred = json.load(open(p, encoding="utf-8"))
        assert isinstance(pred, list), "top level must be a list"
        for i, r in enumerate(pred):
            assert all(k in r for k in ("name", "subject", "X", "Y", "T")), (i, "missing keys")
            assert isinstance(r["name"], str), (i, "name")
            assert type(r["subject"]) is int, (i, "subject must be a JSON int")
            assert len(r["X"]) == len(r["Y"]) == len(r["T"]), (i, "ragged")
            for v in r["X"] + r["Y"]:
                assert type(v) in (int, float) and math.isfinite(v) and v >= 0, (i, v)
            for t in r["T"]:
                assert type(t) is int and t >= 0, (i, "T must be int ms", t)
        c = Counter((r["name"], r["subject"]) for r in pred)
        assert all(n == 1 for n in c.values()), "duplicate keys"
        assert set(c) == keys, f"key set differs: missing {len(keys - set(c))}, extra {len(set(c) - keys)}"
        xs = [v for r in pred for v in r["X"]]; ys = [v for r in pred for v in r["Y"]]
        assert max(xs) <= W and max(ys) <= H, f"out of 512x384: max X {max(xs)}, max Y {max(ys)}"
        L = [len(r["X"]) for r in pred]
        print(f"OK {p}: {len(pred)} records | X {min(xs):.1f}-{max(xs):.1f} | "
              f"Y {min(ys):.1f}-{max(ys):.1f} | T {min(t for r in pred for t in r['T'])}-"
              f"{max(t for r in pred for t in r['T'])} ms | length mean {sum(L)/len(L):.2f} "
              f"min {min(L)} max {max(L)} | empty {L.count(0)} | short(1-2) {sum(1 <= l < 3 for l in L)}")

if __name__ == "__main__":
    main(*sys.argv[1:])

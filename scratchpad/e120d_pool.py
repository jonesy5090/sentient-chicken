"""E120 -- pool the H0 ladder across every lineage block.

Eight lineages left the primary at t=1.47 with both blocks agreeing in sign. The per-
lineage standard deviation of the final-assay difference is 1.54 against an effect of
0.90, so ~24 lineages are needed for 80% power -- the design was underpowered rather
than the result ambiguous, and adding lineages is the honest response to that rather
than reading harder into eight.

Run:  PYTHONPATH=. python scratchpad/e120d_pool.py
"""

import glob
import json
import os

import numpy as np


def load():
    blocks = []
    for path in sorted(glob.glob("scratchpad/e120c_cache*.json")):
        c = json.load(open(path))
        if "intact" in c and "yoked" in c:
            blocks.append((os.path.basename(path), c))
    return blocks


def paired(blocks, fn):
    """intact - yoked, one value per lineage, concatenated across blocks."""
    out = []
    for _, c in blocks:
        a = np.array([fn(r) for r in c["intact"]])
        b = np.array([fn(r) for r in c["yoked"]])
        out.append(a - b)
    return np.concatenate(out)


def bar_for(n):
    """Two-sided 0.05 t threshold, df = n-1."""
    table = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
             10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
             16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
             22: 2.074, 23: 2.069}
    return table.get(n - 1, 2.064)


def report(label, d, want_negative=False):
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / se
    bar = bar_for(len(d))
    ok = abs(t) > bar
    direction = "H0" if ((d.mean() < 0) == want_negative) else "against H0"
    print(f"  {label:<42} {d.mean():+.4f} +- {se:.4f}  t={t:+5.2f}  "
          f"(n={len(d)}, bar {bar:.3f})  {'CLEARS' if ok else 'null'}, {direction}")
    return t, ok


def main():
    blocks = load()
    print(f"E120 pooled ladder -- {len(blocks)} blocks, "
          f"{sum(len(c['intact']) for _, c in blocks)} lineages per arm\n")

    print("PRIMARY -- reduction in catches per dive, intact minus yoked")
    print("(positive = intact reduced predation more; H0 predicts positive)")
    d = paired(blocks, lambda r: r["improve_caught"])
    report("reduction in catches/dive", d, want_negative=False)
    print(f"    per-lineage: {np.round(d, 4)}")
    print(f"    lineages favouring intact: {int((d > 0).sum())}/{len(d)}")

    print("\nSECONDARY (pre-declared) -- final flocks re-assayed on an INTACT world,")
    print("so this measures the evolved brain rather than the channel it was reared under")
    print("(negative = intact-reared flocks are caught less; H0 predicts negative)")
    for k, lbl in (("caught", "final catches/hen"),
                   ("per_dive", "final catches/dive")):
        dd = paired(blocks, lambda r, k=k: r["final_intact"][k])
        report(lbl, dd, want_negative=True)
        print(f"    lineages favouring intact: {int((dd < 0).sum())}/{len(dd)}")

    print("\nCONFOUND CHECKS -- the channel has no mechanistic route to hunger.")
    print("If these move as much as predation does, the effect is not about information.")
    report("hunger improvement", paired(blocks, lambda r: r["improve_hunger"]))
    report("final hunger", paired(blocks, lambda r: r["final_intact"]["hunger"]))

    print("\nPER BLOCK, primary and the final-assay secondary:")
    for name, c in blocks:
        a = np.array([r["improve_caught"] for r in c["intact"]])
        b = np.array([r["improve_caught"] for r in c["yoked"]])
        p = a - b
        fa = np.array([r["final_intact"]["caught"] for r in c["intact"]])
        fb = np.array([r["final_intact"]["caught"] for r in c["yoked"]])
        f = fa - fb
        print(f"  {name:<28} primary {p.mean():+.4f}   final caught/hen {f.mean():+.4f}")

    print("\nReference: a hand-planted comprehension (auditory_scaffold) is worth")
    print("-1.195 catches/hen at this density (E120 6c). That is the ceiling this")
    print("selected effect should be read against, not zero.")


if __name__ == "__main__":
    main()

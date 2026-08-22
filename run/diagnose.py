"""Can the pallium reach a muscle?

E001 returned a null: learning changed the connectome but not behaviour. Before
spending hours on longer runs, check the cheapest explanation -- that the cortical
pathway never gains enough influence over the motor output for anything it learns to
matter.

Motor output is `sigmoid(reflex + cortical + bias)`. The innate arc drives it with
weights of 5-8; the cortical readout starts at 0.05 scale. If the cortical term stays
small against the reflex term, the pallium is shouting into a sigmoid that has
already saturated, and run length is irrelevant.

    usage:  python -m run.diagnose --minutes 20
"""

import argparse

import jax
import jax.numpy as jnp

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--chunk", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta-out", type=float, default=None,
                    help="override the readout learning rate")
    ap.add_argument("--readout-scale", type=float, default=None,
                    help="override the initial cortical readout scale")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP
    pc = plasticity.PlasticConfig(enabled=True)
    if args.eta_out is not None:
        pc = pc._replace(eta_out=args.eta_out)

    key = jax.random.key(args.seed)
    w = world.reset(key, cfg)
    p = connectome.build(
        jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS, n_hens=cfg.n_hens,
        readout_scale=(0.05 if args.readout_scale is None else args.readout_scale))
    x = brain.initial_state(p, cfg.n_hens)

    _w, _x, p_end, _ps, _k, s = simulate.simulate(
        w, x, p, jax.random.fold_in(key, 2), cfg, args.minutes * 60.0,
        args.chunk, pc)

    print(f"cortical influence over {args.minutes:.0f} min of chicken time "
          f"(eta_out={pc.eta_out:g}, readout_scale="
          f"{0.05 if args.readout_scale is None else args.readout_scale:g})\n")
    hdr = (f"{'t (min)':>8} {'|reflex|':>10} {'|cortical|':>11} {'ratio':>8} "
           f"{'|W_out|':>9} {'|W|':>9} {'hunger':>7}")
    print(hdr)
    print("-" * len(hdr))

    n = len(s.t_s)
    for i in range(0, n, max(1, n // 10)):
        refl = float(s.reflex_drive[i])
        cort = float(s.cortical_drive[i])
        print(f"{float(s.t_s[i]) / 60:>8.1f} {refl:>10.3f} {cort:>11.4f} "
              f"{cort / (refl + 1e-9):>8.3f} {float(s.w_out_norm[i]):>9.4f} "
              f"{float(s.w_norm[i]):>9.5f} {float(s.hunger[i]):>7.3f}")

    r0, r1 = float(s.cortical_drive[0]), float(s.cortical_drive[-1])
    wo0, wo1 = float(s.w_out_norm[0]), float(s.w_out_norm[-1])
    ratio = r1 / (float(s.reflex_drive[-1]) + 1e-9)

    print(f"\ncortical drive : {r0:.4f} -> {r1:.4f}  ({r1 / (r0 + 1e-9):.2f}x)")
    wn0, wn1 = float(s.w_norm[0]), float(s.w_norm[-1])
    print(f"|W_out| readout: {wo0:.4f} -> {wo1:.4f}  ({wo1 / (wo0 + 1e-9):.2f}x)")
    print(f"|W| recurrent  : {wn0:.5f} -> {wn1:.5f}  ({wn1 / (wn0 + 1e-9):.2f}x)")
    # Which of the two answers "is the rule active?" depends on the readout mode.
    # Under `hebbian_readout` the neuromodulator is replaced by a constant for `W_out`
    # only, so |W_out| moves whether or not any reward arrived and cannot report on a
    # reward term. |W| stays reward-gated. E065/E066/E068 read |W_out|, got an
    # identical reassuring number in every condition, and concluded the rule was
    # active while `sickness_penalty` was supplying ~0.007% of reinforcement (E068).
    print("  (|W| is the reward-gated one -- read it, not |W_out|, when asking "
          "whether a reward\n   term reached the weights. See E068.)")
    print(f"final ratio    : {ratio:.3f} cortical/reflex")

    print()
    if ratio < 0.05:
        print("VERDICT: the pallium cannot reach a muscle. Whatever it learns is")
        print("  swamped by the innate arc, so run length is not the problem.")
        print("  Fix the readout (eta_out, readout_scale) before running E002.")
    elif ratio < 0.25:
        print("VERDICT: cortical influence is present but weak. Learning could")
        print("  matter at the margin; a longer run is worth doing, but raising")
        print("  the readout scale is likely the larger effect.")
    else:
        print("VERDICT: the cortical pathway has real influence over behaviour.")
        print("  The E001 null is not explained by the readout -- look at the")
        print("  learning signal or the run length instead.")


if __name__ == "__main__":
    main()

"""E122 -- make vigilance cost something, then re-run the ladder.

E121 measured a communication-free hill in the fitness landscape: taking the crouch rate
from 0.399 to 0.790 is worth +0.233 fitness with no information involved, so "always
crouch" beats "crouch when told" and an alarm call has nothing to sell. The cause is that
foraging here is SEARCH-limited -- patches cover 0.28% of the floor and holding hunger at
0.46 needs feeding only 4% of the time -- so interrupting feeding costs almost nothing.

Part A sweeps patch size and intake rate looking for a coop where an information-free
increase in crouching stops paying. The crouch increase is applied as a constant added to
`b_motor[M_CROUCH]`: the purest information-free vigilance there is, with no alarm
direction anywhere in it.

**The calibration target never mentions the channel.** Tuning a world until communication
helps would be worthless; tuning it until unconditional vigilance stops being free is a
statement about the world alone, and the ladder is run once, afterwards, at a frozen
setting.

Run:  PYTHONPATH=. python scratchpad/e122_vigilance_cost.py calibrate
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, plasticity
from run import evolve, simulate
from scratchpad.e120_ladder_checks import founders

N_SEEDS = 4
LIFETIME_S = 600.0
HAWK = 20.0
CAUGHT_W = 0.1049                      # E120's repeatable-variance calibration
RADII = (0.3, 1.0, 2.0)
RATES = (3.0e-2, 1.0e-2, 5.0e-3)
BIASES = (0.0, 0.5, 1.0, 1.5)
CACHE = "scratchpad/e122_cache.json"


def make_cfg(radius, rate):
    return spec.DEFAULT_COOP._replace(
        hawk_period_s=HAWK, channel_mode="intact",
        call_log_steps=spec.YOKE_LOG_STEPS,
        peck_radius=radius, peck_food_rate=rate)


def run_cell(c, bias, seed):
    """One lifetime with `bias` added to the crouch channel's resting drive."""
    pc = evolve.NO_LEARNING
    key = jax.random.key(8800 + seed)
    p = founders(jax.random.fold_in(key, 0), c)
    if bias:
        bm = np.asarray(p.b_motor).copy()
        bm[spec.M_CROUCH] += bias
        p = p._replace(b_motor=jnp.asarray(bm))
    w0 = world.reset(jax.random.fold_in(key, 1), c)
    x0 = brain.initial_state(p, c.n_hens)
    ps = plasticity.initial_state(p, c.n_hens, pc)
    w_end, *_ = simulate.rollout_quiet(
        w0, x0, p, jax.random.fold_in(key, 2), c, int(LIFETIME_S / c.dt), ps, pc)
    _, _, _, _, _, tr = simulate.rollout(
        world.reset(jax.random.fold_in(key, 1), c),
        brain.initial_state(p, c.n_hens), p, jax.random.fold_in(key, 2), c,
        int(60.0 / c.dt), plasticity.initial_state(p, c.n_hens, pc), pc)
    return dict(
        crouch=float(jnp.mean(tr.motor[:, :, spec.M_CROUCH])),
        peck=float(jnp.mean(tr.motor[:, :, spec.M_PECK])),
        caught=float(jnp.mean(w_end.n_caught_any)),
        hunger=float(jnp.mean(w_end.hunger)),
        drives=float(jnp.mean(w_end.hunger + w_end.cold + w_end.thirst)))


def calibrate():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for radius in RADII:
        for rate in RATES:
            key = f"r{radius}_p{rate}"
            if key in cache:
                print(f"[{key}] cached")
                continue
            c = make_cfg(radius, rate)
            cell = {}
            for bias in BIASES:
                runs = [run_cell(c, bias, s) for s in range(N_SEEDS)]
                cell[str(bias)] = runs
            cache[key] = cell
            json.dump(cache, open(CACHE, "w"), indent=1)
            b0 = cell["0.0"]
            print(f"[{key}] crouch {np.mean([r['crouch'] for r in b0]):.3f} "
                  f"hunger {np.mean([r['hunger'] for r in b0]):.4f} "
                  f"caught {np.mean([r['caught'] for r in b0]):.2f}", flush=True)
    summarise(cache)


def summarise(cache):
    print("\n" + "=" * 104)
    print("A. Does an INFORMATION-FREE increase in crouching still pay?")
    print("   net = caught_weight*(catches saved) - (drives cost). Positive = "
          "unconditional vigilance wins.")
    print("=" * 104)
    print(f"{'patch r':>8} {'intake':>8} {'bias':>5} {'crouch':>7} {'caught':>7} "
          f"{'hunger':>7} {'d_pred':>8} {'d_driv':>8} {'NET':>9}")
    best = []
    for radius in RADII:
        for rate in RATES:
            key = f"r{radius}_p{rate}"
            if key not in cache:
                continue
            cell = cache[key]
            b0 = cell["0.0"]
            c0 = np.mean([r["caught"] for r in b0])
            d0 = np.mean([r["drives"] for r in b0])
            cr0 = np.mean([r["crouch"] for r in b0])
            h0 = np.mean([r["hunger"] for r in b0])
            print(f"{radius:>8} {rate:>8.1e} {0.0:>5.1f} {cr0:>7.3f} {c0:>7.2f} "
                  f"{h0:>7.4f} {'--':>8} {'--':>8} {'--':>9}")
            for bias in BIASES[1:]:
                v = cell[str(bias)]
                cr = np.mean([r["crouch"] for r in v])
                cg = np.mean([r["caught"] for r in v])
                dr = np.mean([r["drives"] for r in v])
                hu = np.mean([r["hunger"] for r in v])
                dpred = CAUGHT_W * (c0 - cg)
                ddriv = -(dr - d0)
                net = dpred + ddriv
                flag = ""
                if cr >= 1.9 * cr0:
                    flag = "  <- ~doubled crouch"
                    best.append((abs(net), key, bias, net, hu, cr, cg))
                print(f"{'':>8} {'':>8} {bias:>5.1f} {cr:>7.3f} {cg:>7.2f} "
                      f"{hu:>7.4f} {dpred:>+8.4f} {ddriv:>+8.4f} {net:>+9.4f}{flag}")
    print("=" * 104)
    print("E121's reference at the default coop: net +0.233 at roughly doubled crouch.")
    if best:
        best.sort()
        print("\nSettings nearest the target (|net| smallest at ~doubled crouch), with the")
        print("0.3-0.7 hunger band enforced:")
        for a, key, bias, net, hu, cr, cg in best[:6]:
            ok = "OK" if 0.3 <= hu <= 0.7 else "REJECTED (hunger outside 0.3-0.7)"
            print(f"  {key:>18} bias {bias:.1f}: net {net:+.4f}  hunger {hu:.4f}  "
                  f"crouch {cr:.3f}  caught {cg:.2f}   {ok}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if what == "calibrate":
        calibrate()
    elif what == "show":
        summarise(json.load(open(CACHE)))

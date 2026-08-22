"""E090 part 2: does a hunger term on M_PECK give CONDITIONAL aversion?

E089 found the gakel scaffold behaviourally inert -- 3.5% peck suppression against a
food drive of 7.0, both deep in sigmoid saturation -- and that no SCAFFOLD_WEIGHT
resolves it, because with two terms either the warning always wins or it never does.

The arithmetic in E090 section 5 says neither term alone works and both together do,
because saturation makes a hunger term invisible when nothing is wrong and decisive once
a warning has pulled the drive out of it. This measures that in the running model.

Four falsifiers, all pre-registered:
  primary        >=25% suppression at low hunger (the assay bar)
  conditionality suppression at hunger 0.8 must be <80% of suppression at 0.2
  neonatal       unwarned pecking must differ by <2% between hunger 0.2 and 0.8
  regression     no other ethogram assay changes state
"""
import sys, time
sys.path.insert(0, 'scratchpad')
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from coop.spec import CoopConfig
from hen import brain, connectome, regions
from run import probes, simulate

CFG = spec.DEFAULT_COOP
WS = (1.5, 5.0, 7.0, 9.0)
HS = (0.0, 4.0, 8.0)
MIN_MOD, MAX_COND, MAX_NEONATAL = probes.MIN_MODULATION, 0.80, 0.02


def peck_under(cfg, hunger, gakel_on, W, H):
    """Mean M_PECK for a hen on food, with or without a gakel call in earshot."""
    cfg = cfg._replace(n_hens=2, n_food=1)
    w = probes._staged(cfg, pos=[[10.0, 10.0], [10.0, 10.6]], heading=0.0,
                       food=[[10.05, 10.0]], hunger=hunger)
    ch = spec.CALL_MOTOR_IDX.index(spec.M_CALL_GAKEL if gakel_on else spec.M_CALL_CONTACT)
    w = w._replace(calls=jnp.zeros((cfg.n_hens, spec.N_CALLS)).at[1, ch].set(1.0))
    p = connectome.build(jax.random.key(probes.GENOME_SEED), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens, gakel_scaffold=True,
                         gakel_peck_weight=W, hunger_peck_weight=H)
    x = brain.initial_state(p, cfg.n_hens)
    _w, _x, _p, _ps, _k, tr = simulate.rollout(w, x, p, jax.random.key(7), cfg, 1)
    return float(jnp.mean(tr.motor[:, 0, spec.M_PECK]))


print("E090 part 2 -- hunger-gated pecking, measured in the running model")
print(f"bars: suppression >={100*MIN_MOD:.0f}% at low hunger; hunger-0.8 suppression "
      f"<{100*MAX_COND:.0f}% of hunger-0.2; unwarned spread <{100*MAX_NEONATAL:.0f}%\n")
t0 = time.perf_counter()
print(f"{'W':>5}{'H':>5}{'unwarned .2/.8':>18}{'warned .2':>11}{'warned .8':>11}"
      f"{'supp .2':>9}{'supp .8':>9}{'cond':>7}{'neo':>6}")
best = []
for W in WS:
    for H in HS:
        un_s = peck_under(CFG, 0.2, False, W, H)
        un_h = peck_under(CFG, 0.8, False, W, H)
        wa_s = peck_under(CFG, 0.2, True, W, H)
        wa_h = peck_under(CFG, 0.8, True, W, H)
        s_s = (un_s - wa_s) / max(un_s, 1e-9)
        s_h = (un_h - wa_h) / max(un_h, 1e-9)
        neo = abs(un_h - un_s) / max(un_s, 1e-9)
        cond = s_h < MAX_COND * s_s
        ok = s_s >= MIN_MOD and cond and neo < MAX_NEONATAL
        if ok:
            best.append((W, H, s_s, s_h, neo))
        print(f"{W:>5.1f}{H:>5.1f}{un_s:>9.4f}/{un_h:<8.4f}{wa_s:>11.4f}{wa_h:>11.4f}"
              f"{100*s_s:>8.1f}%{100*s_h:>8.1f}%{'yes' if cond else '':>7}"
              f"{'ok' if neo < MAX_NEONATAL else 'FAIL':>6}")

print(f"\nconfigurations clearing all three: "
      + (", ".join(f"W={w:.0f} H={h:.0f}" for w, h, *_ in best) or "NONE"))
if best:
    # prefer the strongest sated suppression among those that clear
    W, H, s_s, s_h, neo = max(best, key=lambda r: r[2])
    print(f"strongest: W={W:.0f} H={H:.0f} -- sated {100*s_s:.1f}% suppression, "
          f"starving {100*s_h:.1f}%, unwarned spread {100*neo:.2f}%")
    print(f"\nregression: full ethogram at W={W:.0f} H={H:.0f}")
    import run.probes as P
    orig = P._connectome
    P._CONNECTOME_CACHE.clear()

    def patched(n_hens, gakel_scaffold=False):
        ck = (n_hens, gakel_scaffold, "e090", W, H)
        if ck not in P._CONNECTOME_CACHE:
            P._CONNECTOME_CACHE[ck] = connectome.build(
                jax.random.key(P.GENOME_SEED), regions.DEFAULT_REGIONS, n_hens=n_hens,
                gakel_scaffold=gakel_scaffold, gakel_peck_weight=W, hunger_peck_weight=H)
        return P._CONNECTOME_CACHE[ck]

    P._connectome = patched
    res = P.run_all()
    P._connectome = orig
    for fn, r in zip(P.ALL, res):
        flag = "PASS" if r.passed else "FAIL"
        was = "was XFAIL" if fn.__name__ in P.EXPECTED_FAILURES else ""
        print(f"  {flag:<5} {r.name:<48} {was}")
    print(f"  {sum(r.passed for r in res)}/{len(res)} passed")
print(f"\nwall clock: {time.perf_counter()-t0:.0f} s")

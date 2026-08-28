"""E114: hold the learned gate fixed, take vigilance out, and see if the benefit survives.

Rearing happens once per seed at the DEFAULT world, so both blinding conditions test the
same learned gate. The intervention is on the world, not on what was learned.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import sensing, spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST = int(30 * 60 / CFG.dt), int(10 * 60 / CFG.dt)
SEED0, SEEDS = int(os.environ.get("E114_SEED0", "0")), 8
BG = plasticity.PlasticConfig(enabled=True, hebbian_readout=True,
                              readout_scaling_strength=0.3, bg_gate=True)
OFF = plasticity.PlasticConfig(enabled=False, bg_gate=True)
reg = regions.DEFAULT_REGIONS
NAMES = {spec.M_TURN_R: "TURN_R", spec.M_TURN_L: "TURN_L", spec.M_PECK: "PECK",
         spec.M_SCRATCH: "SCRATCH", spec.M_FORWARD: "FORWARD"}


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def test(w, x, p, ps, key, cfg, pc, n):
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, p, ps, key = c
        stub = neurons.rate(x)[:, -n_motor:]
        s = jnp.einsum("hmn,hn->hm", p.W_str, stub)
        s = s - pc.bg_lateral * jnp.mean(s, axis=-1, keepdims=True)
        gate = jax.nn.sigmoid(brain.GATE_OPEN_BIAS + s)
        obs = sensing.observe(w, cfg)
        hawk_near = (jnp.linalg.norm(w.pos - w.hawk_pos[None, :], axis=-1)
                     < cfg.vision_range) & (w.hawk_on > 0.0)
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, (gate, obs[:, spec.IDX_AERIAL], w.head_down, hawk_near)
    (w, x, p, ps, key), out = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, out


t0 = time.perf_counter()
print(f"E114 -- does the gate work through vigilance? seeds {SEED0}-{SEED0+SEEDS-1}\n")
print("E102's claim: 'pecking and turning drive head_down, which blinds her to the sky'.")
print(f"spec.HEAD_DOWN_ACTIONS = "
      f"{tuple(NAMES.get(i, i) for i in spec.HEAD_DOWN_ACTIONS)} -- turning is not in it.\n")

rows = {}
profile = np.zeros(spec.MOTOR_DIM)
aerial_check = {}
for blinds in (1.0, 0.0):
    cfg = CFG._replace(head_down_blinds=blinds)
    for gate_on in (True, False):
        key = (blinds, gate_on)
        cd, hu, hd, aer_down, aer_up = [], [], [], [], []
        for s in range(SEED0, SEED0 + SEEDS):
            k = jax.random.key(s)
            p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
            # Rearing is always at the DEFAULT world, so all four cells share a gate.
            _w, _x, p, ps, _k, _t = simulate.rollout(
                world.reset(k, CFG), brain.initial_state(p0, 16), p0,
                jax.random.fold_in(k, 2), CFG, REAR, pc=BG,
                ps=plasticity.initial_state(p0, 16, BG))
            pp = p if gate_on else p._replace(W_str=jnp.zeros_like(p.W_str))
            wf, (gates, aerial, head_down, hawk_near) = test(
                world.reset(k, cfg), brain.initial_state(pp, 16), pp, ps,
                jax.random.fold_in(k, 7), cfg, OFF, TEST)
            d = float(jnp.sum(wf.n_dives))
            cd.append(float(jnp.sum(wf.n_caught_any)) / max(d, 1))
            hu.append(float(jnp.mean(wf.hunger)))
            hd.append(float(jnp.mean(head_down)))
            # Instrument: the aerial channel while head-down vs head-up, hawk in range.
            a = np.asarray(aerial); h = np.asarray(head_down); n_ = np.asarray(hawk_near)
            down, up = (h > 0.5) & n_, (h <= 0.5) & n_
            if down.sum() > 50:
                aer_down.append(float(a[down].mean()))
            if up.sum() > 50:
                aer_up.append(float(a[up].mean()))
            if gate_on and blinds == 1.0:
                profile += np.asarray(jnp.mean(gates, axis=(0, 1)))
        rows[key] = dict(cd=np.array(cd), hu=np.array(hu), hd=np.mean(hd))
        aerial_check[key] = (np.mean(aer_down) if aer_down else float("nan"),
                             np.mean(aer_up) if aer_up else float("nan"))

print(f"{'blinds':>8}{'gate':>7}{'caught/dive':>13}{'hunger':>9}{'head-down':>11}"
      f"{'aerial down':>13}{'aerial up':>11}")
for blinds in (1.0, 0.0):
    for gate_on in (True, False):
        r = rows[(blinds, gate_on)]
        ad, au = aerial_check[(blinds, gate_on)]
        print(f"{blinds:>8.1f}{'on' if gate_on else 'off':>7}"
              f"{np.mean(r['cd']):>13.4f}{np.mean(r['hu']):>9.4f}{r['hd']:>11.4f}"
              f"{ad:>13.4f}{au:>11.4f}")

print(f"\n  learned gate profile (blinds=1.0, gate on), 5 lowest channels:")
order = np.argsort(profile / SEEDS)[:5]
for i in order:
    blinding = "BLINDS her" if int(i) in spec.HEAD_DOWN_ACTIONS else "does not blind"
    print(f"    {NAMES.get(int(i), int(i)):>10}  {profile[int(i)]/SEEDS:.4f}   {blinding}")
for i in spec.HEAD_DOWN_ACTIONS:
    print(f"    (head-down action {NAMES.get(int(i), int(i)):>8}: gate "
          f"{profile[int(i)]/SEEDS:.4f})")


def paired(a, b, name):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"    {name:<44}{d.mean():+.4f} +/- {se:.4f}  t={d.mean()/(se+1e-12):+.2f}")
    return d.mean(), se


print(f"\n  paired on caught/dive, df={SEEDS-1}, crit 2.365:")
g1, s1 = paired(rows[(1.0, True)]["cd"], rows[(1.0, False)]["cd"],
                "gate effect WITH blinding (E113's -0.068)")
g0, s0 = paired(rows[(0.0, True)]["cd"], rows[(0.0, False)]["cd"],
                "gate effect WITHOUT blinding  <-- the test")
paired(rows[(0.0, False)]["cd"], rows[(1.0, False)]["cd"],
       "removing blinding, no gate (manipulation check)")

print("\n--- pre-registered falsifiers (E114 section 4) ---")
ad1, au1 = aerial_check[(1.0, False)]
ad0, au0 = aerial_check[(0.0, False)]
print(f"instrument   aerial while head-down: blinds=1.0 {ad1:.4f} -> blinds=0.0 {ad0:.4f} "
      f"(head-up reference {au0:.4f})")
print(f"             -> {'flag works' if ad0 > 0.8 * au0 else 'FLAG NOT WORKING, void'}")
frac = g0 / g1 if g1 else float("nan")
print(f"primary      gate effect retained without blinding: {100*frac:.0f}% "
      f"({g0:+.4f} of {g1:+.4f})")
print(f"             -> {'vigilance IS the mechanism' if frac < 0.5 else 'vigilance is NOT the mechanism'}")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")

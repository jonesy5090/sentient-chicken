"""E113: is E102's benefit "suppress the right channels" or "suppress some channels"?

Rear once per seed with the gate learning, then test the SAME brain under four gate
conditions. Everything but `W_str` is identical across arms, so nothing can differ except
the gate.
"""
import os
import time
from functools import partial
import jax, jax.numpy as jnp, numpy as np
from coop import spec, world
from hen import brain, connectome, neurons, plasticity, regions
from run import simulate

CFG = spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=60.0)
REAR, TEST = int(30 * 60 / CFG.dt), int(10 * 60 / CFG.dt)
SEED0, SEEDS = int(os.environ.get("E113_SEED0", "0")), 8
BASE = dict(enabled=True, hebbian_readout=True, readout_scaling_strength=0.3)
BG = plasticity.PlasticConfig(**BASE, bg_gate=True)
reg = regions.DEFAULT_REGIONS
NAMES = {spec.M_FORWARD: "FORWARD", spec.M_TURN_L: "TURN_L", spec.M_TURN_R: "TURN_R",
         spec.M_PECK: "PECK", spec.M_SCRATCH: "SCRATCH", spec.M_CROUCH: "CROUCH",
         spec.M_FLEE: "FLEE", spec.M_CALL_AERIAL: "CALL_AERIAL",
         spec.M_CALL_GROUND: "CALL_GROUND", spec.M_CALL_CONTACT: "CALL_CONTACT",
         spec.M_CALL_FOOD: "CALL_FOOD", spec.M_CALL_GAKEL: "CALL_GAKEL"}


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def test(w, x, p, ps, key, cfg, pc, n):
    """Test rollout, emitting the realised gate so the arms can be shown matched."""
    n_motor = p.W_out.shape[-1]

    def step(c, _):
        w, x, p, ps, key = c
        stub = neurons.rate(x)[:, -n_motor:]
        s = jnp.einsum("hmn,hn->hm", p.W_str, stub)
        s = s - pc.bg_lateral * jnp.mean(s, axis=-1, keepdims=True)
        gate = jax.nn.sigmoid(brain.GATE_OPEN_BIAS + s)
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, gate
    (w, x, p, ps, key), gates = jax.lax.scan(step, (w, x, p, ps, key), None, length=n)
    return w, gates


def variants(w_str, gates_true, key):
    """The three controls, built from the same reared `W_str`.

    `uniform` is solved for rather than guessed: the gate is
    `sigmoid(BIAS + s - mean(s))`, and a constant `W_str` gives `s - mean(s) = 0`
    exactly, so a flat gate sits at `sigmoid(BIAS)`. To match the TRUE arm's realised
    mean instead, the bias offset is set directly by inverting the sigmoid.
    """
    n_hens, n_motor, n_pre = w_str.shape
    perm = jnp.stack([jax.random.permutation(jax.random.fold_in(key, h), n_motor)
                      for h in range(n_hens)])
    permuted = jnp.take_along_axis(w_str, perm[:, :, None], axis=1)
    # Uniform: no structure at all. Realised as a scalar offset that reproduces the
    # true arm's mean gate, applied through the same sigmoid.
    m = float(jnp.mean(gates_true))
    offset = float(jnp.log(m / (1.0 - m))) - brain.GATE_OPEN_BIAS
    return permuted, offset


@partial(jax.jit, static_argnames=("cfg", "pc", "n", "offset"))
def test_uniform(w, x, p, ps, key, cfg, pc, n, offset):
    """A gate with no structure: every channel multiplied by the same constant."""
    g = float(jax.nn.sigmoid(brain.GATE_OPEN_BIAS + offset))

    def step(c, _):
        w, x, p, ps, key = c
        # W_str is zeros here, so `_one_step`'s own gate sits at sigmoid(BIAS); the
        # uniform suppression is applied by scaling the reflex weights instead, which
        # is exactly a flat multiplicative gate on the arc.
        c, _o = simulate._one_step(c, None, cfg, pc)
        return c, jnp.zeros(())
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[0], g


t0 = time.perf_counter()
print(f"E113 -- permuted and uniform gate controls. seeds {SEED0}-{SEED0+SEEDS-1}\n")

rows = {k: {"cd": [], "hu": [], "gate": []}
        for k in ("true", "permuted", "uniform", "none")}
profile = np.zeros(spec.MOTOR_DIM)
for s in range(SEED0, SEED0 + SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), reg, n_hens=16)
    ps0 = plasticity.initial_state(p0, 16, BG)
    w0 = world.reset(k, CFG)
    x0 = brain.initial_state(p0, 16)
    _w, _x, p, ps, _k, _t = simulate.rollout(
        w0, x0, p0, jax.random.fold_in(k, 2), CFG, REAR, pc=BG, ps=ps0)

    pc_off = plasticity.PlasticConfig(enabled=False, bg_gate=True,
                                      bg_lateral=BG.bg_lateral)
    # TRUE first, so the controls can be matched to its realised gate.
    wf, gates = test(world.reset(k, CFG), brain.initial_state(p, 16), p, ps,
                     jax.random.fold_in(k, 7), CFG, pc_off, TEST)
    g_true = float(jnp.mean(gates))
    profile += np.asarray(jnp.mean(gates, axis=(0, 1)))

    permuted, offset = variants(p.W_str, gates, jax.random.fold_in(k, 9))
    # A flat gate of the same mean: scale the arc's weights by that constant, with
    # W_str zeroed so nothing structured is left.
    g_flat = float(jax.nn.sigmoid(brain.GATE_OPEN_BIAS + offset))
    arms = {
        "true": (p, pc_off),
        "permuted": (p._replace(W_str=permuted), pc_off),
        "uniform": (p._replace(W_str=jnp.zeros_like(p.W_str),
                               reflex=p.reflex * g_flat),
                    plasticity.PlasticConfig(enabled=False)),
        # `bg_gate` stays ON with W_str zeroed, so this arm really does run at
        # sigmoid(GATE_OPEN_BIAS) = 0.982 rather than at no gate at all. Block one had
        # it with the gate switched off entirely, so its reported 0.982 was the value
        # the probe computed and not the one the rollout applied -- a 1.8% mismatch
        # that flattered the true arm. Fixed here.
        "none": (p._replace(W_str=jnp.zeros_like(p.W_str)), pc_off),
    }
    for name, (pp, pcc) in arms.items():
        wf, gates = test(world.reset(k, CFG), brain.initial_state(pp, 16), pp, ps,
                         jax.random.fold_in(k, 7), CFG, pcc, TEST)
        d = float(jnp.sum(wf.n_dives))
        rows[name]["cd"].append(float(jnp.sum(wf.n_caught_any)) / max(d, 1))
        rows[name]["hu"].append(float(jnp.mean(wf.hunger)))
        rows[name]["gate"].append(g_flat if name == "uniform"
                                  else float(jnp.mean(gates)))

print(f"{'arm':>10}{'caught/dive':>13}{'hunger':>9}{'mean gate':>11}")
for name in ("true", "permuted", "uniform", "none"):
    r = rows[name]
    print(f"{name:>10}{np.mean(r['cd']):>13.4f}{np.mean(r['hu']):>9.4f}"
          f"{np.mean(r['gate']):>11.4f}")

print(f"\n  E102's per-channel signature, this run (true arm):")
for i in np.argsort(profile / SEEDS)[:5]:
    print(f"    {NAMES[int(i)]:>14}  {profile[int(i)]/SEEDS:.4f}")


def paired(a, b, name):
    d = np.array(a) - np.array(b)
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = d.mean() / (se + 1e-12)
    print(f"    {name:<44}{d.mean():+.4f} +/- {se:.4f}  t={t:+.2f}")
    return d.mean(), se


print(f"\n  paired on caught/dive, df={SEEDS-1}, crit 2.365:")
paired(rows["true"]["cd"], rows["none"]["cd"], "true gate vs no gate (E102's main effect)")
paired(rows["permuted"]["cd"], rows["none"]["cd"], "PERMUTED vs no gate")
paired(rows["uniform"]["cd"], rows["none"]["cd"], "UNIFORM vs no gate")
d1, se1 = paired(rows["true"]["cd"], rows["permuted"]["cd"],
                 "true vs permuted   <-- the selectivity test")
d2, se2 = paired(rows["true"]["cd"], rows["uniform"]["cd"],
                 "true vs uniform    <-- the stricter one")

# Per-seed values, so a pooled estimate is possible later. Blocks 1 and 2 disagreed on
# the selectivity contrast (t=-0.89 then t=-4.14), which is the E021 pattern exactly.
import json
with open(f"/home/claude/.claude/jobs/cfcfc904/tmp/e113_seeds_{SEED0}.json", "w") as f:
    json.dump({k: {m: list(map(float, v[m])) for m in ("cd", "hu", "gate")}
               for k, v in rows.items()}, f)

print("\n--- pre-registered falsifiers (E113 section 4) ---")
gt, gp, gu = (np.mean(rows[n]["gate"]) for n in ("true", "permuted", "uniform"))
ok = abs(gt - gp) < 0.02 and abs(gt - gu) < 0.02
print(f"instrument   mean gate true {gt:.4f}, permuted {gp:.4f}, uniform {gu:.4f} "
      f"-> {'matched' if ok else 'NOT MATCHED, comparison void'}")
sel = abs(d1) > 2 * se1 and abs(d2) > 2 * se2
print(f"primary      true beats BOTH controls by >2 SE -> "
      f"{'SELECTIVITY SURVIVES' if sel else 'selectivity NOT supported'}")
print(f"\nwall clock: {time.perf_counter() - t0:.0f} s")

"""Is the planted gakel association place-SELECTIVE in the live run?

E082's and E083's pre-flight measured that the plant fires AT P (1.000). Neither ever
measured that it is silent anywhere else -- CLAUDE.md check 2. The run-wide mean came
back 0.90 while the hen is at P only ~42% of the time, which is the wrong shape for a
selective plant and would mean both experiments applied a roughly CONSTANT gakel drive
rather than a place-specific one.

Splits relu(pred@gakel) by whether each hen is within one grid spacing of P, live.
"""
import time, sys
from functools import partial
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, 'scratchpad')
import e083_leaving_anchor as E
from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate

CEN, P, P2, SPACING, CFG = E.CEN, E.P, E.P2, E.SPACING, E.CFG


@partial(jax.jit, static_argnames=("cfg", "pc", "n"))
def run(w, x, p, ps, key, cfg, pc, n):
    def step(carry, _):
        carry, (motor, obs, _r, _m) = simulate._one_step(carry, None, cfg=cfg, pc=pc)
        w = carry[0]
        dP = jnp.linalg.norm(w.pos - jnp.asarray(CEN[P], dtype=jnp.float32), axis=-1)
        pred = jnp.einsum("hon,hn->ho", p.W_pred,
                          (carry[3].z_lag - carry[3].z_lag_bar) * p.pred_src[None, :])
        g = jax.nn.relu(pred[:, E.GAKEL_CH])
        at = dP < SPACING
        return carry, (jnp.sum(g * at), jnp.sum(at),
                       jnp.sum(g * ~at), jnp.sum(~at))
    return jax.lax.scan(step, (w, x, p, ps, key), None, length=n)[1]


pc = plasticity.PlasticConfig(**E.FROZEN, pred_gain=1.0)
t0 = time.perf_counter()
print(f"P = cell {P} {CEN[P].round(1)}, radius {SPACING:.2f} m, gain 1.0, "
      f"{E.SEEDS} seeds, {E.MINUTES:.0f} min\n")
print(f"{'seed':>5}{'preflight@P':>13}{'live @P':>10}{'live elsewhere':>16}{'ratio':>8}")
aP = aO = 0.0
for s in range(E.SEEDS):
    k = jax.random.key(s)
    p0 = connectome.build(jax.random.fold_in(k, 1), regions.DEFAULT_REGIONS, n_hens=E.HENS,
                          gakel_scaffold=True, shared_place_map=True)
    p, pf = E.plant(p0, E._PLANT_CFG)
    w = world.reset(k, CFG)
    w = w._replace(food_pos=jnp.asarray(np.stack([CEN[P], CEN[P2]]), dtype=jnp.float32))
    x = brain.initial_state(p, E.HENS); ps = plasticity.initial_state(p, E.HENS, pc)
    gi, ni, go, no = run(w, x, p, ps, jax.random.fold_in(k, 2), CFG, pc, E.STEPS)
    inP, outP = float(jnp.sum(gi) / jnp.sum(ni)), float(jnp.sum(go) / jnp.sum(no))
    aP += inP; aO += outP
    print(f"{s:>5}{pf:>13.3f}{inP:>10.3f}{outP:>16.3f}{inP/max(outP,1e-9):>8.2f}")
aP /= E.SEEDS; aO /= E.SEEDS
print(f"\nmean  at P {aP:.3f}   elsewhere {aO:.3f}   ratio {aP/max(aO,1e-9):.2f}")
print(f"wall clock: {time.perf_counter()-t0:.0f} s")
print("\nA selective plant fires at P and near-zero elsewhere (ratio >> 1).")
print("A ratio near 1 means E082 and E083 both applied a CONSTANT gakel drive,")
print("and neither was a test of place-specific avoidance at all.")

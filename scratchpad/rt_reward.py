"""Does the reward guard run where the defect can appear? (E027, review finding 5)

`tests/test_plasticity.py::test_reward_is_not_dominated_by_one_component` runs at
`spec.DEFAULT_COOP`, whose `hawk_period_s` is 900. The H4 world runs at 20 -- a hawk
45x more often. If `n_struck` dominates the reward at the rate H4 actually uses, the
guard written to catch exactly that class of defect is blind to it, for the same reason
the E019 audio defect survived: the test runs at the one configuration where the problem
does not occur.

Same freeze-and-measure method as the guard itself, so the numbers are comparable.
"""
import jax, jax.numpy as jnp
from coop import sensing, spec, world
from hen import brain, connectome, plasticity, regions
from hen.plasticity import PlasticConfig

FIELDS = ("hunger", "thirst", "cold", "vigour", "n_struck")


def shares(cfg, steps=3_000, label=""):
    pc = PlasticConfig()
    w = world.reset(jax.random.key(0), cfg)
    p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    contrib = {k: [] for k in FIELDS}
    struck_steps = 0
    for t in range(steps):
        obs = sensing.observe(w, cfg)
        x, motor, _ = brain.step(x, obs, p, cfg.dt)
        wn = world.step(w, motor, jax.random.fold_in(jax.random.key(4), t), cfg)
        r = plasticity.reward(w, wn, cfg, pc)
        for k in FIELDS:
            contrib[k].append(jnp.mean(r - plasticity.reward(
                w, wn._replace(**{k: getattr(w, k)}), cfg, pc)))
        struck_steps += int(jnp.sum(wn.n_struck - w.n_struck) > 0)
        w = wn
    var = {k: float(jnp.var(jnp.array(v))) for k, v in contrib.items()}
    total = max(sum(var.values()), 1e-12)
    print(f"{label}  (hawk every {cfg.hawk_period_s:.0f} s, {cfg.n_hens} hens)")
    print(f"  steps in which some hen was struck: {struck_steps} / {steps}")
    for k in FIELDS:
        bar = "#" * int(40 * var[k] / total)
        print(f"    {k:<10}{100*var[k]/total:>6.1f}%  {bar}")
    worst = max(var, key=var.get)
    print(f"  -> guard asserts every share < 80%; worst is {worst} at "
          f"{100*var[worst]/total:.1f}% "
          f"({'PASSES' if var[worst]/total < 0.8 else 'WOULD FAIL'})\n")


print("Reward variance decomposition, freeze-one-field method (same as the guard).\n")
shares(spec.DEFAULT_COOP, label="guard config -- what the test suite runs")
shares(spec.DEFAULT_COOP._replace(n_hens=16, hawk_period_s=20.0),
       label="H4 config  -- what E024/E026 actually ran")

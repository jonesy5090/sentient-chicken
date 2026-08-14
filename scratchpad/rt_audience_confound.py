"""Does the H3 audience assay differ in ONE thing (audience) or also in heard calls?"""
import jax, jax.numpy as jnp
from coop import spec, sensing
from hen import brain, connectome, regions
from run import simulate, audience

cfg = spec.DEFAULT_COOP
n = cfg.n_hens
p = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=n)

for legacy in (False, True):
    c = cfg._replace(legacy_audio=legacy)
    print(f"\n=== legacy_audio={legacy} ===")
    for aud in (False, True):
        w = audience._staged(c, n, audience=aud, hawk=True, food=False)
        x = brain.initial_state(p, n)
        wt, _x, _p, _ps, _k, tr = simulate.rollout(w, x, p, jax.random.key(11), c, 300)
        obs = tr.obs[-1, 0]        # focal hen, last step
        au = obs[spec.AUDIO_LO:spec.AUDIO_HI]
        flock_vis = obs[jnp.array([spec.vis_index(b, spec.CLS_FLOCKMATE) for b in range(spec.N_BINS)])]
        print(f" audience={aud!s:5}  audio(contact,food,aerial,ground)="
              f"[{au[0]:.3f} {au[1]:.3f} {au[2]:.3f} {au[3]:.3f}]"
              f"  max flockmate-vis={float(jnp.max(flock_vis)):.3f}"
              f"  aerial={float(obs[spec.IDX_AERIAL]):.3f}"
              f"  focal alarm-call={float(tr.motor[-1,0,spec.M_CALL_AERIAL]):.3f}")

# what the assay reports
print("\nassay() on a naive hen:", audience.assay(p, cfg, n))
print("comprehension (bare arc):", audience.comprehension(p, cfg, n))
p_s = connectome.build(jax.random.key(1), regions.DEFAULT_REGIONS, n_hens=n, auditory_scaffold=True)
print("comprehension (scaffold):", audience.comprehension(p_s, cfg, n))
print("assay() scaffold:", audience.assay(p_s, cfg, n))

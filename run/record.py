"""Record a rollout to a flat binary trajectory for the offline 3D viewer.

`coop/world.py`'s state is already 2D -- `pos` is `(H, 2)` -- so the browser viewer
renders a stylised 3D scene from a 2D floor plan rather than anything the sim itself
computes. This intentionally never touches the compiled scan: it is a second,
*outer* scan (frame / sim-step) that snapshots world state every `steps_per_frame`
inner steps and discards the steps between, exactly the pattern `simulate._chunked`
already uses to produce developmental summaries without materialising every step.

    usage:  python -m run.record --minutes 10 --fps 15
            python -m run.record --minutes 10 --plastic --name "h4 dive"
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from coop import spec, world
from hen import brain, connectome, plasticity, regions
from run import simulate
from run.simulate import NO_PLASTICITY

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Per-frame fields pulled off `World`, in write order. Static geometry (food_pos,
# water_pos, coop size) goes in meta.json instead -- it never changes and repeating
# it every frame would be pure waste against the memory-bandwidth argument this
# project already takes seriously for the sim itself.
_FIELDS = ("pos", "heading", "head_down", "calls",
           "hawk_pos", "hawk_on", "fox_pos", "fox_on",
           "food_amount", "struck")


def _snapshot(w: world.World):
    return (w.pos, w.heading, w.head_down, w.calls,
            w.hawk_pos, w.hawk_on, w.fox_pos, w.fox_on,
            w.food_amount, w.n_strike_events)


@partial(jax.jit, static_argnames=("cfg", "pc", "n_frames", "steps_per_frame"))
def _record(w, x, p, ps, key, cfg, pc, n_frames: int, steps_per_frame: int):
    def frame(carry, _):
        carry, _ = jax.lax.scan(
            partial(simulate._one_step, cfg=cfg, pc=pc), carry,
            None, length=steps_per_frame)
        w = carry[0]
        return carry, _snapshot(w)

    carry, frames = jax.lax.scan(frame, (w, x, p, ps, key), None, length=n_frames)
    return carry, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--hens", type=int, default=spec.DEFAULT_COOP.n_hens)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="viewer frames per biological second (subsamples the sim)")
    ap.add_argument("--plastic", action="store_true")
    ap.add_argument("--name", type=str, default=None,
                    help="label shown in the run picker (default: auto)")
    ap.add_argument("--desc", type=str, default="")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    pc = plasticity.PlasticConfig(enabled=args.plastic)
    seconds = args.minutes * 60.0
    steps_per_frame = max(1, round(1.0 / (cfg.dt * args.fps)))
    n_frames = max(1, round(seconds / (steps_per_frame * cfg.dt)))
    frame_dt = steps_per_frame * cfg.dt

    key = jax.random.key(args.seed)
    w = world.reset(key, cfg)
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens)
    x = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    food_pos, water_pos = w.food_pos, w.water_pos

    print(f"recording {cfg.n_hens} hens, {seconds / 60:.1f} min "
          f"({n_frames} frames @ {1 / frame_dt:.1f} fps)"
          f"{' [plastic]' if args.plastic else ''}")

    t0 = time.perf_counter()
    (w, x, p, ps, key), frames = _record(
        w, x, p, ps, jax.random.fold_in(key, 2), cfg, pc if pc.enabled else NO_PLASTICITY,
        n_frames, steps_per_frame)
    jax.block_until_ready(frames[0])
    wall = time.perf_counter() - t0
    print(f"wall clock: {wall:.1f}s ({seconds / wall:.0f}x real time)")

    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = args.name.strip().lower().replace(" ", "-") if args.name else \
        ("plastic" if args.plastic else "innate")
    run_id = f"{ts}_{slug}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {name: np.asarray(a, dtype=np.float32) for name, a in zip(_FIELDS, frames)}
    layout = {}
    buf = bytearray()
    offset = 0
    for name in _FIELDS:
        a = arrays[name]
        layout[name] = {"offset": offset, "shape": list(a.shape)}
        b = a.tobytes()
        buf += b
        offset += len(b)
    (out_dir / "trajectory.bin").write_bytes(bytes(buf))

    meta = {
        "id": run_id,
        "name": args.name or run_id,
        "description": args.desc,
        "created_at": ts,
        "seed": args.seed,
        "hens": cfg.n_hens,
        "plastic": bool(args.plastic),
        "minutes": args.minutes,
        "n_frames": n_frames,
        "frame_dt": frame_dt,
        "coop_size": cfg.size,
        "n_food": cfg.n_food,
        "n_water": cfg.n_water,
        "food_pos": np.asarray(food_pos, dtype=np.float32).tolist(),
        "water_pos": np.asarray(water_pos, dtype=np.float32).tolist(),
        "layout": layout,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

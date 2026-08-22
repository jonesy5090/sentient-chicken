"""Record a rollout to a flat binary trajectory for the offline 3D viewer.

`coop/world.py`'s state is already 2D -- `pos` is `(H, 2)` -- so the browser viewer
renders a stylised 3D scene from a 2D floor plan rather than anything the sim itself
computes. This intentionally never touches the compiled scan: it is a second,
*outer* scan (frame / sim-step) that snapshots world state every `steps_per_frame`
inner steps and discards the steps between, exactly the pattern `simulate._chunked`
already uses to produce developmental summaries without materialising every step.

    usage:  python -m run.record --minutes 10 --fps 15
            python -m run.record --minutes 10 --plastic --name "h4 dive"

Writing `--desc`: this is the only thing standing between the viewer and total
confusion. Whoever loads a run in the browser (`viz/web/index.html`'s sidebar shows
it verbatim) may have zero context on this project -- no idea what stage it's from,
what's been built, or why any of it matters. Treat `--desc` as the one paragraph
they'll read before watching, and answer three things in plain language, no jargon,
no experiment numbers:

  1. What is being tested, in a sentence a non-scientist would understand -- "a
     flock of simulated chickens" is the shared premise everyone already gets from
     looking at the scene; say what's actually being asked beyond that.
  2. What they will *see* happening in THIS recording specifically -- only the
     mechanics actually active (a predator, sickness, calls, learning switched on
     or off), not a tour of every feature the model has ever grown.
  3. What this recording demonstrates -- a specific finding, stated plainly
     ("the flock did not learn to avoid it"), or that it's just a demo of the
     mechanics with no result attached. Never imply a finding that isn't there.

Example, for a T2 recording:
    "A flock of simulated chickens forage, drink, and watch for a hawk and a fox.
    One feeder is secretly 'poisoned' and which one rotates every five minutes; a
    hen who eats from it turns sickly and flashes red for about a minute, and
    calls out (a purple dot) to warn nearby flockmates. Learning is switched off
    in this recording, so it's a demo of the mechanics only -- no claim about
    whether the flock gets better at avoiding it is being made here."

An empty `--desc` still records (never blocks the run), but prints a reminder --
see `main()` below.
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
           "food_amount", "struck",
           "sick_on", "food_contaminated")


def _snapshot(w: world.World):
    return (w.pos, w.heading, w.head_down, w.calls,
            w.hawk_pos, w.hawk_on, w.fox_pos, w.fox_on,
            w.food_amount, w.n_strike_events,
            w.sick_on.astype(jnp.float32), w.food_contaminated.astype(jnp.float32))


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


def record_and_save(key, cfg, pc, minutes: float, fps: float = 15.0, seed: int = 0,
                    name: str | None = None, desc: str = "",
                    problem: str = "") -> Path:
    """Roll out a recording and write it to `runs/`. Returns the output directory.

    Shared by `main()` below and by other entry points' `--record` flag (`run.lifetime`
    and similar) -- callers that already built a `key`/`cfg`/`pc` for their own run pass
    the same ones here so the recording matches exactly what they reported, rather than
    duplicating the capture-and-write logic per caller.
    """
    plastic = bool(pc.enabled)
    seconds = minutes * 60.0
    steps_per_frame = max(1, round(1.0 / (cfg.dt * fps)))
    n_frames = max(1, round(seconds / (steps_per_frame * cfg.dt)))
    frame_dt = steps_per_frame * cfg.dt

    w = world.reset(key, cfg)
    # The T2 scaffolds are opt-in everywhere else (E076/E077: two of them silently moved
    # every downstream baseline), so the recorder has to ask for them explicitly too.
    p = connectome.build(jax.random.fold_in(key, 1), regions.DEFAULT_REGIONS,
                         n_hens=cfg.n_hens,
                         gakel_scaffold=cfg.contamination_enabled,
                         shared_place_map=cfg.place_cells_enabled,
                         place_to_hippocampus=cfg.place_cells_enabled)
    x = brain.initial_state(p, cfg.n_hens)
    ps = plasticity.initial_state(p, cfg.n_hens, pc)
    food_pos, water_pos = w.food_pos, w.water_pos

    print(f"recording {cfg.n_hens} hens, {seconds / 60:.1f} min "
          f"({n_frames} frames @ {1 / frame_dt:.1f} fps)"
          f"{' [plastic]' if plastic else ''}")

    t0 = time.perf_counter()
    (w, x, p, ps, key), frames = _record(
        w, x, p, ps, jax.random.fold_in(key, 2), cfg, pc if plastic else NO_PLASTICITY,
        n_frames, steps_per_frame)
    jax.block_until_ready(frames[0])
    wall = time.perf_counter() - t0
    print(f"wall clock: {wall:.1f}s ({seconds / wall:.0f}x real time)")

    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = name.strip().lower().replace(" ", "-") if name else \
        ("plastic" if plastic else "innate")
    run_id = f"{ts}_{slug}"
    out_dir = RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {n: np.asarray(a, dtype=np.float32) for n, a in zip(_FIELDS, frames)}
    layout = {}
    buf = bytearray()
    offset = 0
    for n in _FIELDS:
        a = arrays[n]
        layout[n] = {"offset": offset, "shape": list(a.shape)}
        b = a.tobytes()
        buf += b
        offset += len(b)
    (out_dir / "trajectory.bin").write_bytes(bytes(buf))

    meta = {
        "id": run_id,
        "name": name or run_id,
        "description": desc,
        # What the project was stuck on when this was recorded. Separate from
        # `description`, which says what is on screen: this says why anyone was
        # looking. It is expected to go stale, which is what dates the run.
        "problem": problem,
        "created_at": ts,
        "seed": seed,
        "hens": cfg.n_hens,
        "plastic": plastic,
        "minutes": minutes,
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
    if not problem.strip():
        print("no --problem given -- the viewer will not show what this recording was "
              "made to investigate")
    if not desc.strip():
        print("no --desc given -- the viewer's sidebar will show nothing to explain "
             "this run to someone with no context. See this module's docstring for "
             "what a good one covers, then edit this run's meta.json's "
             "\"description\" field directly (no need to re-record).")
    return out_dir


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
    ap.add_argument("--desc", type=str, default="",
                    help="a paragraph a total newcomer could read and understand "
                         "what they're about to watch and why -- see this file's "
                         "module docstring for the full guidance and an example")
    ap.add_argument("--problem", type=str, default="",
                    help="the short-term goal this recording sits inside: what is "
                         "currently blocking, and what would unblock it. Distinct from "
                         "--desc, which explains what is on screen. This is a snapshot "
                         "of project state at recording time and is expected to go "
                         "stale -- that is the point, it dates the run.")
    ap.add_argument("--t2", action="store_true",
                    help="turn on the T2 scaffolds: a rotating contaminated feeder, "
                         "sickness, the gakel call, allocentric place cells and the "
                         "innate response to hearing a gakel call. Off by default "
                         "everywhere, so the recorder asks for them explicitly too.")
    ap.add_argument("--hawk-period", type=float, default=None,
                    help="seconds between hawk dives (default: DEFAULT_COOP's 900s -- "
                         "a several-minute recording will often show none at all; "
                         "try 20-60 for a demo that reliably has predation in it)")
    args = ap.parse_args()

    cfg = spec.DEFAULT_COOP._replace(n_hens=args.hens)
    if args.t2:
        cfg = cfg._replace(contamination_enabled=True, place_cells_enabled=True)
    if args.hawk_period is not None:
        cfg = cfg._replace(hawk_period_s=args.hawk_period)
    pc = plasticity.PlasticConfig(enabled=args.plastic)
    key = jax.random.key(args.seed)

    record_and_save(key, cfg, pc, args.minutes, args.fps, args.seed,
                    args.name, args.desc, args.problem)


if __name__ == "__main__":
    main()

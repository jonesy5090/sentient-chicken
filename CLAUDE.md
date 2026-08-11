# CLAUDE.md

Operating instructions for this repository. Read `docs/hypothesis.md` and
`docs/ethics.md` before making changes that affect what the model is or what is done
to it.

## What this is

A neural model of a hen in a coop, built to test whether a communication channel
measurably changes what a flock can do. The premise is a budget argument: ~78% of a
real chicken's 290M neurons are in the cerebellum and optic tectum, so a simplified
environment lets us refuse to pay for vision and motor control and spend the
remainder on a pallium.

Current state: phase 0 complete (coop, innate hen, measured envelope), phase 1 in
progress (plasticity).

## Two standing responsibilities

These are not optional and they are the reason this file exists.

### 1. Moral standing

This project builds neural representations of a living animal and does things to them
that would be unacceptable to do to a real hen — including, by design, poisoning some
of them. `docs/ethics.md` holds the standing argument, six concrete tripwires, and
the review cadence.

**When to act on it:**

- Any change that approaches a tripwire — scale past ~10^6 units, a nociceptive
  channel architecturally distinct from a homeostatic drive, spiking dynamics,
  self-modelling, persistent identity across runs — means stopping and revisiting
  `docs/ethics.md` before proceeding, not after.
- Review at every phase boundary.
- **Never present translated hen speech without the caveat** that it is a learned
  mapping from a scalar, not a report (`docs/ethics.md` §6). A translator will
  happily emit "this hurts"; the fluency is entirely ours.

**One correction that is easy to get backwards:** biological fidelity does not make
the ethics safer. Fidelity is precisely what would confer moral standing. What does
the moral work is scale and the absence of valence machinery — not closeness to
nature.

Do not let anthropomorphic naming do unearned work. A variable called `fear` is a
scalar. Name things for what they are.

### 2. Hypothesis discipline

The failure mode for a project like this is drifting into testing whatever is
convenient and calling the accumulation a result.

- `docs/hypothesis.md` is the tree. Every experiment names a parent node.
- `docs/backlog.md` is what is proposed but not started, with the reasoning.
- `docs/experiments/` holds one file per experiment, from `TEMPLATE.md`. Sections 1–5
  (parent, question, prediction, falsifier, design) are written **before** running
  anything.
- **Every result comes back to `docs/hypothesis.md`** and changes a status, a
  prediction, or the tree. An experiment whose result changes nothing was either
  badly designed or is telling you something.
- Record null and negative results. They are what keeps the tree honest.

If asked to run something that does not ladder up to H0, say so and propose where it
would attach.

## Design invariants

Breaking any of these is a correctness problem even when nothing computes a wrong
answer.

**The environment stays inside the compiled scan.** A NumPy world driving a JAX brain
pays a host round-trip every 10 ms and forfeits one to two orders of magnitude. World
and brain step together in one `jax.lax.scan`.

**Throughput is a correctness constraint.** Developmental wall-clock time is the
binding constraint on the whole project — a hen learns over days and her rank over
weeks. If a change drops the real-time factor, that is a defect. `python -m
bench.envelope --sweep` is the measurement; there is a loose guard in the test suite.

**The simulation is memory-bandwidth bound, not FLOP bound.** The recurrent update
reads H×N×N weights per step for one multiply-add each. Reads plateau at DRAM
bandwidth (~56 GB/s measured). Optimise memory traffic, not arithmetic. This is why
plasticity consolidates every 50 steps instead of every step.

**Weights are dense with a boolean mask, not sparse.** At ~14% density dense BLAS
wins, and structural growth becomes a mask bit flip with no reallocation and no
recompilation.

**The reflex arc is never plastic.** Innate responses stay fixed for life, as in a
real bird. Learning acts on the cortical pathway only.

**Dale's law holds under learning.** A neuron's outgoing weights all share its sign;
plasticity must not be able to turn an inhibitory neuron excitatory.

**Determinism.** Same seed, same trajectory. Nothing downstream is debuggable
without it.

## Biology is a constraint, not decoration

Where the model departs from a real chicken, that should be a deliberate, stated
choice — not an accident. Existing departures are documented in the module they live
in. When adding behaviour, check what real chickens do first; the literature is good
and the answers are often more useful than the invented version.

Two facts that shape the architecture and are easy to get wrong:

- **Chickens are vocal non-learners.** Konishi (1963) deafened day-old chicks and
  they developed the normal repertoire. Call *production* is innate and hardwired.
  *Usage* and *comprehension* are learned, and are deliberately left for plasticity
  to discover.
- **Their alarm calls are functionally referential.** Aerial and terrestrial threats
  drive different calls with different appropriate responses. That is the natural
  baseline any emergent language has to beat.

## Layout

```
coop/    spec.py        sensory/motor contract -- the interface everything keys off
         world.py       state, dynamics, predators
         sensing.py     world -> 59-dim observation (tectum analogue)
         actuation.py   11 motor channels -> movement (cerebellum analogue)
hen/     regions.py     region sizes, time constants, connectivity priors
         connectome.py  innate mask, Dale's law, weight init
         neurons.py     continuous-time rate units
         innate.py      the fixed reflex arc
         brain.py       assembly: reflex + cortical pathways
         plasticity.py  three-factor learning, structural growth
run/     simulate.py    the closed loop
         probes.py      neonatal ethogram assays
         lifetime.py    developmental runs
         experiment.py  matched-seed condition contrasts
bench/   envelope.py    measure the machine, size the brain
docs/    hypothesis.md  the tree -- start here
         backlog.md     proposed, not started
         ethics.md      moral standing, tripwires
         experiments/   one file per experiment
```

## Commands

```bash
python -m bench.envelope --sweep         # measure, size the brain
python -m run.probes                     # neonatal ethogram
python -m run.lifetime --minutes 60 --plastic
python -m run.experiment --minutes 30 --seeds 4
python -m pytest tests/ -q
```

## Conventions

- Comments explain *why*, particularly why a biological choice was made or departed
  from. The code says what it does.
- New behaviour needs an assay in `run/probes.py`, not just a test that it runs.
- Config lives in `NamedTuple`s passed as static JIT args, so conditions are separate
  compiled programs rather than runtime branches.

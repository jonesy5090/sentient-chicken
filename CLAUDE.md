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

Current state: phase 0 proven; phase 1 **built but not proven** — the learning rule is
currently a clean null (E020), and saying "phase 1 complete" is exactly the
built-versus-proven conflation this file warns about. `docs/hypothesis.md` is
authoritative; `/status` will read it for you.

## Why this is worth doing, and therefore what not to break

This is a genuinely interesting experiment rather than a toy, and the things that
make it interesting are specific and fragile. Each of the following looks like
incidental complexity from inside the code, and removing any of them quietly turns
the project back into a demo. If you find yourself simplifying one, stop.

**The natural baseline is real and unusually strong.** Chickens have functionally
referential alarm calls — aerial and terrestrial threats drive different calls with
different appropriate responses — and they are vocal *non-learners*, so production is
innate while usage and comprehension are learned. That gives every phase a published
result to be checked against rather than a vibe. It is why `hen/innate.py` hardwires
call production and deliberately omits the audience effect.

**The head-down gate is the whole thesis in one line.** `coop/sensing.py` zeroes the
aerial channel when the hen is pecking. Communication only pays when the receiver
lacks something the sender has, so without that asymmetry no signal is ever worth
making, at any brain size. It reads like an odd sensory quirk. It is the load-bearing
wall. Measured at 0.01 vs 0.87 aerial signal in the same bird seconds apart.

**The control design is what makes the result falsifiable.** The headline comparison
is an intact channel against a *shuffled* one — identical brain, identical bandwidth,
zero information — plus a lesion of a trained flock. Anything less rigorous cannot
distinguish "language helps" from "more neurons help". `docs/backlog.md` §1 is not
bureaucracy; it is the experiment.

**The negative results are assets.** E001 is a null and it is the most useful file in
`docs/experiments/`. E002 found a ceiling nobody predicted. E003 caught a bug in our
own statistics. E004 found that structural growth *hurts*. A version of this project
that reports only its successes would be worth much less.

**The ethics question is live, not ceremonial.** See below.

## How to explain this project

The person you are reporting to **knows neuroscience basics** — neurons, synapses,
plasticity, what a brain region is, roughly what a reflex arc does. Do not explain
those.

What does need translating is everything from the *other* two fields this project
sits in: machine learning, and experimental statistics. Those are where the jargon
piles up and where a confident-sounding sentence can hide a thing nobody has checked.

**Translate these on sight**, every time, not just the first:

| instead of | say |
|---|---|
| "the readout gradient was insufficient" | "what she learned couldn't reach her muscles" |
| "credit assignment failure" | "she had no way to tell which of her actions caused the good thing" |
| "the representation is degenerate" | "her brain state for *hawk* and for *alarm call* were nearly identical — she couldn't tell them apart" |
| "saturated dynamics" | "her neurons were all firing flat out, so nothing could stand out against the background" |
| "superadditive interaction" | "doing both at once hurt more than the two problems added together" |
| "t=3.85, p<0.05" | "the effect is bigger than the run-to-run noise, so it's real" |
| "confounded" | "two things changed at once, so we can't tell which one did it" |
| "eligibility trace" | "a fading record of which synapses were active recently, so a reward arriving a moment later knows who to credit" |

**Three habits that matter more than vocabulary:**

1. **Lead with what happened, then the number.** "The hens got worse at feeding —
   4.7% of the time against 6.2%" beats "hunger change +0.062 ± 0.016 (t=3.85)".
   The number is evidence for the sentence, not a substitute for it.

2. **Name the model part *and* its biological counterpart.** "`W_out`, the connection
   from the thinking part of the brain out to the muscles". A reader who knows
   neuroscience can follow the biology immediately and does not need to hold a
   variable name in their head.

3. **When something fails, explain the failure mechanically.** Not "the hypothesis
   was not supported" but "she could learn, but nothing she learned could change what
   she did". Mechanism is what makes a null result useful instead of disappointing.

**Say plainly when a result is bad, uncertain, or was previously reported wrong.**
This project has overturned its own findings repeatedly and the record of *how* is
its most valuable output. A summary that always sounds like progress is not tracking
anything.

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

**The `status` skill** (`.claude/skills/status/`) answers the three standing
questions — what we are trying to achieve, what has been built versus what has been
*proven*, and what is next — from these files rather than from memory. Use it when
asked where the project is, and before starting significant new work, to check the
work ladders up to the tree.

**The `red-team` skill** (`.claude/skills/red-team/`) buys an outside reader. It
commissions an adversarial review from an agent with none of the conversation's
context, then verifies every finding independently before acting on any of it. Use it
when successive experiments keep generating explanations for the same null, before an
expensive run, or whenever the work feels stuck in a loop.

It exists because accumulated context is what makes this project productive and what
makes it blind. The first time it ran it found that every communication experiment for
eighteen experiments had been measuring a channel that carried no information. It was
also wrong about two things, which is why verification is the skill's central rule
rather than a footnote.

- `docs/hypothesis.md` is the tree. Every experiment names a parent node.
- `docs/backlog.md` is what is proposed but not started, with the reasoning.
- `docs/experiments/` holds one file per experiment, from `TEMPLATE.md`. Sections 1–5
  (parent, question, prediction, falsifier, design) are written **before** running
  anything.
- **Every result comes back to `docs/hypothesis.md`** and changes a status, a
  prediction, or the tree. An experiment whose result changes nothing was either
  badly designed or is telling you something.
- **No status changes on one seed block.** A significant result must replicate on a
  fresh block of seeds before it moves the tree. E021 measured the same contrast as
  E020 on seeds 12–23 instead of 0–11 and got a 4.4× larger standard error, turning
  t=3.84 into t=0.01. The pairing and the t table are both fine; what is not safe is
  assuming a block's *variance* is representative. A homogeneous block makes a small
  difference look decisive.
- **Test a post-hoc observation on data that did not generate it.** E021's question A
  came from noticing something in E020. Re-reading it off E020's own seeds would have
  confirmed it; fresh seeds falsified it, sign and all.
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

**When a term moves, measure it in its new home.** Three separate defects (E019) were
quantities checked in the place they had just been moved *from* — a call cost verified
in the metric after being relocated into the reward, a gain documented at 0.9 while the
code ran 0.70, a diagnosis made before a re-baselining and never re-checked after. Each
survived because the obvious verification was performed and looked at the wrong place.

**Guard tests must run at the configuration where the defect appears.** The auditory
channel carried zero information at the default 16 hens for eighteen experiments,
because the suite ran at 4 — the one flock size where it still worked.

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

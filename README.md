# sentient-chicken

**Chickens already have words. This project asks what happens if we give them more.**

---

## Start here

A hen who spots a hawk makes a different sound than one who spots a fox.

That difference is not emotional intensity. Hens who hear the first call run for cover
and crouch. Hens who hear the second stand up and scan the ground. The call carries
*which kind of danger* — and the birds respond appropriately even when there is
nothing to see. Biologists call this **functional reference**, and it has been
documented in chickens since the 1980s.

There is a second fact that makes the first one stranger. Chickens do not *learn*
their calls. Konishi deafened day-old chicks in 1963 and they grew up producing the
normal repertoire perfectly — no model to copy, no feedback, nothing. The vocabulary
is built in, like a heartbeat.

So a chicken is born with words she never learned, meaning things she was never
taught. What she *does* learn is when to say them, and what to do when she hears one.

**This project builds a chicken, from neurons up, and then asks: if you gave her the
ability to invent new words, what would she use them for?**

---

## Is that a real question?

It is, and it has a real answer that we do not know yet.

The honest guess, from the literature on how communication systems emerge, is that a
flock would first invent signals for **danger** and **food** — which chickens already
have — and then, if anything, for **each other**. Chickens have a strict dominance
hierarchy; "pecking order" is not a metaphor, it is the original literal use of the
phrase. Recognising individuals is expensive. A name is a compression trick.

**Names are the most likely thing to be invented that nature did not provide.** That
is a testable prediction, and it is the kind of result that would be worth having.

---

## The trick that makes it affordable

A real chicken has about 290 million neurons. Simulating that faithfully is a
supercomputer problem — roughly a trillion connections, terabytes of live state,
hundreds of GPUs. Not a hobby project.

But look at where the neurons actually are:

| brain structure | what it does | share of the brain |
|---|---|---|
| cerebellum | coordinating a real body | **63%** |
| optic tectum | processing panoramic vision | **15%** |
| telencephalon | *everything we care about* | 23% |

**Roughly 78% of a chicken's brain is spent on having a body and eyes.** Put a hen in
a simple enough world and you can decline to pay for that. Here, 42 million neurons of
visual processing become twelve directional "what's over there" sensors, and 182
million neurons of motor control become eleven numbers like *walk*, *peck*, *crouch*.

That is not a claim about biology — you cannot free up cerebellum and spend it on
grammar; brains do not work that way. It is a **budget decision**, and it is what
brings the project down from a supercomputer to a laptop.

---

## The one idea everything depends on

Here is the part that took the longest to get right, and it is not in the brain at
all.

**A signal is only worth making if the listener does not already know.**

If every hen can always see every hawk, no alarm call is ever worth anything, no
matter how many neurons you give the bird. Communication needs an *information gap*.

Chickens supply one for free: **a hen with her beak in the dirt cannot see the sky.**
Foraging and watching for danger are physically incompatible. So there is one line in
this codebase that matters more than the rest:

```python
aerial = aerial * (1.0 - w.head_down)   # she is looking down; she cannot see it
```

And it works. Here is the same simulated hen, with the same hawk overhead, seconds
apart — the only difference is whether she happens to be pecking:

| | sees the hawk | crouches |
|---|---|---|
| head up | 0.87 | 0.85 |
| pecking | **0.01** | **0.06** |

Across an hour, the flock is head-down about **64% of the time**. Blind to the sky,
and dependent on whichever flockmate happens to be looking up.

**That gap is where language becomes worth having.** The pressure lives in the world,
not in the neurons.

---

## Where the project is now

The hen is born knowing a handful of reflexes and nothing else — she pecks at small
objects, crouches at things overhead, runs from things on the ground, calls when she
is lonely, and huddles when she is cold. All of that is measured against real
published chick behaviour. **7 out of 7 behavioural tests pass.**

Then she learns — or she did, until we found out why.

**Ten experiments in, the honest state of this project is that a result we had
replicated turned out to depend on a defect, and it did not survive fixing it.** That
arc is the most interesting thing here, so it gets told straight:

| | what happened |
|---|---|
| **E001** | Nothing. Twelve thousand synapses rewiring themselves, zero effect on behaviour. A busy brain doing nothing. |
| **E002** | Found out why. The pathway from the "thinking" part of the brain to the muscles was effectively frozen — she could learn, but she could not *act* on what she learned. Also found a ceiling nobody expected: give the learned pathway too much control and behaviour gets **worse**, because a hen who overrides her instincts with a half-trained brain is worse off than one who just follows instinct. |
| **E003** | Fixed that. The effect appeared — and we caught a bug in our own *statistics* that would have let us declare victory on a result that was still noise. |
| **E004** | Ran it properly, twelve times. **It holds.** Learning works. |
| **E005–E007** | Three attempts to get one hen to *understand* another's alarm call. All null, each for a different real reason. |
| **E008–E009** | Found the actual blocker, and it was underneath everything else: **the brain was saturated.** A hen's internal state for "heard an alarm call" and "saw a hawk" differed by less than 1%. She could not tell them apart, so nothing downstream could learn from the difference. |
| **E010** | Fixed the saturation. **The learning result collapsed** — t=3.93 to t=0.08. |

That last row is the uncomfortable one, and it is why this section no longer says
learning works.

The reason is worth understanding, because it is not simply "we broke it". A
saturated brain emits a nearly constant signal, and a constant signal is *harmless* —
it just shifts a baseline. A responsive brain emits a varying signal, and an untrained
bird cannot yet use variation, so it acts as noise. **Saturation had been accidentally
protecting the hen from her own inexperience.** Every parameter that made the original
result work had been tuned against the broken version, and inherited its assumptions.

So E004 was a real, replicated, pre-registered finding that turned out to be narrower
than it looked. Nothing has been withdrawn — both results are recorded in full — but
the question is open again, and the next step is retuning what depended on the old
regime.

An unplanned discovery along the way: the hens that **grow** new connections do
*worse* than the ones that only prune. The best learner throws away 42% of the brain
she was born with. That inverts the obvious expectation that more plasticity is
better, and it now has its own open question.

Not yet built: memory of places, social hierarchy, and the language itself.

---

## How we would know if we were fooling ourselves

This is the part that decides whether the project is science or a demo.

The tempting experiment is: give one flock language, give another flock a smaller
brain, see who wins. **That experiment is worthless.** If the small-brained flock
loses, you cannot tell whether it lost for want of language or for want of neurons.

So the real control is a flock with an **identical brain** and a **scrambled
channel** — every hen hears a random flockmate instead of the right one. Same amount
of chatter, same cost, zero information. If language-flock beats scrambled-flock,
then *information* is doing the work and nothing else can explain it.

And the decisive test is a lesion: take a flock that has already learned, **switch the
channel off**, and see how much they lose. Birds that talk a lot are not evidence.
Birds that fall apart when silenced are.

Written down in advance, in [`docs/backlog.md`](docs/backlog.md), along with the
results that would prove us wrong.

---

## Should we be doing this at all?

The plan involves deliberately poisoning some of these hens to see whether they warn
each other. That deserves an argument rather than a shrug, and it gets one in
[`docs/ethics.md`](docs/ethics.md).

The short version contains one thing worth knowing, because most people guess it
backwards: **making the model more realistic makes the ethics harder, not easier.**
Fidelity to a real chicken is exactly what would give the thing moral standing. If the
model is a crude sketch, there is nobody home to wrong.

What actually does the work is scale — each hen is 512 simple units, about 0.4% of a
fruit fly — and the absence of any machinery for suffering. "Hunger" here is a number
that goes up and down. There is no pain pathway, because none has been written.

There are six specific tripwires that would stop the work, agreed in advance. One is
specific to this project and worth flagging now: eventually we plan to *translate*
what the hens say into English. A translator will cheerfully print **"this hurts."**
It will be wrong. It will be a number we defined, dressed in a word we chose, and it
will be far more persuasive than it deserves to be.

---

## The technical part

<details>
<summary><b>How it runs, and what it costs</b></summary>

Written in JAX. The world and the brain step together inside a single compiled loop —
a Python environment driving an accelerated brain would pay a round-trip every 10 ms
and lose one to two orders of magnitude.

Measured on a 4-core CPU (`python -m bench.envelope --sweep`):

```
 neurons   hens      steps/s   real-time     W (MB)   GB/s read    hrs/day
     128     16        8,761       87.6x        1.0         9.2       0.27
     256     16        5,853       58.5x        4.2        24.5       0.41
     512     16        3,360       33.6x       16.8        56.4       0.71
    1024     16          733        7.3x       67.1        49.2       3.28
```

Watch the `GB/s read` column plateau around 56. The simulation is **memory-bandwidth
bound, not compute bound** — it reads every weight once per step and does a single
multiply with it. More memory bandwidth buys neurons roughly linearly; more CPU cores
buy almost nothing. That one measurement set the size of the brain.

At the default 512 neurons the flock runs ~30x faster than reality, so one chicken-day
takes about 48 minutes and a month of rearing runs overnight. **Wall-clock time, not
memory, is the binding constraint on the whole project** — a hen learns her
surroundings over days and her rank over weeks.

</details>

<details>
<summary><b>How the hen is built</b></summary>

Two pathways run in parallel to every muscle:

- **the reflex arc** — senses straight to action, fixed at birth, never changes
- **the cortical path** — senses → pallium → motor, plastic, starts nearly silent

A newly hatched hen is almost pure reflex. The pallium is wired and running but has
nothing useful to say yet, and it has to *earn* influence over behaviour rather than
being handed it. E002 showed both that this earning has to be possible (it wasn't) and
that it must not go too far (behaviour gets worse).

Learning is a three-factor rule: two local traces of neural activity, gated by a
global reward signal. The reward is not a task score handed down from outside — it is
**homeostatic improvement**. Being fed, watered, warm and uneaten. Any foraging
strategy has to be discovered from that alone.

Constraints enforced in the code and the tests: a neuron's outgoing connections all
share its sign (learning cannot flip an inhibitory neuron to excitatory), new synapses
can only appear where an axon could physically reach, and the reflex arc is never
touched by learning.

</details>

<details>
<summary><b>Layout and commands</b></summary>

```
coop/    spec.py       the sensory/motor contract everything keys off
         world.py      state, dynamics, predators
         sensing.py    world -> 59-dim observation (the eyes)
         actuation.py  11 motor channels -> movement (the body)
hen/     regions.py    region sizes, time constants, wiring priors
         connectome.py the innate brain, Dale's law, initialisation
         neurons.py    continuous-time rate units
         innate.py     the fixed reflex arc: what a chick is born knowing
         brain.py      assembly of the two pathways
         plasticity.py three-factor learning, structural growth
run/     simulate.py   the closed loop
         probes.py     behavioural assays against published chick behaviour
         lifetime.py   developmental runs
         experiment.py matched-seed contrasts between conditions
         diagnose.py   is the learned pathway reaching the muscles?
bench/   envelope.py   measure the machine, size the brain
docs/    hypothesis.md the tree of claims -- start here
         backlog.md    the language experiment, designed but not built
         ethics.md     moral standing, tripwires, review log
         experiments/  one file per experiment, predictions written first
```

```bash
pip install -r requirements.txt

python -m bench.envelope --sweep                  # size the brain for your machine
python -m run.probes                              # the newly-hatched behaviour tests
python -m run.lifetime --minutes 60 --plastic     # rear a flock
python -m run.experiment --minutes 20 --seeds 12  # matched-seed A/B
python -m pytest tests/ -q
```

</details>

<details>
<summary><b>How the project keeps itself honest</b></summary>

The failure mode for something like this is testing whatever is convenient and calling
the pile a result. Three rules:

1. [`docs/hypothesis.md`](docs/hypothesis.md) is a tree of claims. **No experiment
   runs without naming the claim it feeds.**
2. Every experiment gets a file in [`docs/experiments/`](docs/experiments/) with its
   prediction and its falsifier **written before the run**. E004's prediction was
   committed to git while the run was still executing.
3. Every result comes back and changes something. Nulls are recorded, not buried —
   E001 is a null and it is the most useful file in the directory.

Ask `/status` at any point and get the same three answers: what we are trying to
achieve, what has been built versus what has actually been *proven*, and what is next.

</details>

---

## Current state

**Phase 0** (a credible newly-hatched hen) and **phase 1** (she learns) are done and
measured. **Phase 2** — does she learn *when* to call, the way a real chicken does? —
is next. **Phase 4** is the language experiment, and it is designed but not built.

Full detail in [`docs/hypothesis.md`](docs/hypothesis.md).

---

## Sources

The biology is not decorative and none of it is invented.

Chicken neuron counts: [Frontiers in Neuroanatomy 2022](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2022.1048261/full) ·
[Olkowicz et al., PNAS 2016](https://www.pnas.org/doi/10.1073/pnas.1517131113).
Deafened chicks develop normal calls: [Konishi 1963](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1439-0310.1963.tb01156.x).
Aerial vs ground alarm calls and audience effects: [Evans & Marler](https://pubmed.ncbi.nlm.nih.gov/3396311/) ·
[Animal Behaviour 1983](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589).
The vigilance/foraging trade-off: [Royal Society Open Science](https://royalsocietypublishing.org/doi/10.1098/rsos.150135).
How structured languages emerge: [iterated learning](https://arxiv.org/pdf/1910.05291) ·
[symbol emergence](https://arxiv.org/pdf/2303.04544).
Ethics of simulated minds: [Metzinger's moratorium argument](https://www.worldscientific.com/doi/abs/10.1142/S270507852150003X).

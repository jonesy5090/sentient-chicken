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

## The result: it works, and here is exactly how much

This is the headline, and it is new.

Take a flock of sixteen hens. Send a hawk at them every twenty seconds. Then count
only the moments that matter — a hen who is **in danger and had her head down when the
hawk committed to its dive**. She cannot see it. The only way she finds out is if
somebody tells her.

| | caught, when blind |
|---|---|
| deaf flock | **72.5%** |
| flock that can hear each other | **58.2%** |

**A hen who can hear her flockmates is caught about 20 percentage points less often, in
exactly the moments she could not see the hawk herself.** Measured across 24 separate
worlds: **−0.198 ± 0.059**, which in plain terms means the effect is more than three
times bigger than the run-to-run noise, so it is real rather than luck.

It was also replicated properly. Twelve worlds gave −0.187. Twelve *fresh* worlds,
chosen before they were run, gave −0.208. The project has been burned by this exact
thing before — a result that looked decisive on one batch of random seeds and evaporated
on the next — so no finding moves the record until it survives a second batch.

**Three honest limits, stated here rather than in a footnote:**

- **Nothing is learned.** There is no plasticity anywhere in this experiment. The hens
  understand each other because we wired them to, the way a real chick is born already
  finding fear calls alarming. This proves a **working channel helps**. It does not
  prove language can be *acquired*, which is a different and harder claim.
- **The signal only means "danger."** It does not yet mean *which* danger, or *where*.
  That is the next rung and it is not built.
- **It is one task, at one brain size.** Sixteen hens, a pallium half again its default
  size, one kind of threat.

---

## How we would know if we were fooling ourselves

This is the part that decides whether the project is science or a demo, and it is where
most of the twenty-six experiments actually went.

The tempting experiment is: give one flock language, give another flock a smaller
brain, see who wins. **That experiment is worthless.** If the small-brained flock
loses, you cannot tell whether it lost for want of language or for want of neurons.

So the control has to be a flock with an **identical brain** and a **channel carrying
no information** — same amount of chatter, same energetic cost, nothing useful in it.
If the talking flock beats that one, then *information* is doing the work and nothing
else can explain it.

**The first version of that control did not work, and finding out is the most useful
thing in this repository.**

The obvious design is to scramble *who you hear*: every hen gets a random flockmate's
call instead of her own neighbour's. It sounds airtight. It is not, and the reason is
almost funny — in a 20-metre run, **every hen can already hear every other hen.** So
scrambling the sender changes almost nothing. What survives is "somebody is screaming
right now", and that is nearly the whole message.

We measured it rather than arguing about it. How well does the channel predict "a hawk
is coming for *me*"?

| channel | predicts the hawk |
|---|---|
| intact | **+0.56** |
| scrambled sender | **+0.55** ← supposed to be zero |
| **shifted in time** | **−0.13** |

The scrambled control **kept 98% of the information it existed to destroy.** Eighteen
experiments' worth of conclusions had been drawn against instruments in that condition.

The control that works is **yoked**: play each hen the flock's *real* calls, with all
their real timing and urgency, shifted by more than the length of a hawk dive. Every
statistical property is preserved. The only thing removed is that the calls are no
longer about *now*.

**That control comes out flat — +0.035, indistinguishable from noise, in both batches of
seeds.** So the benefit is not the noise, not the arousal, not the energy cost of
calling. It is the information.

And a second control, in the other direction: a flock with a perfectly good channel and
**no ability to interpret it** gets no benefit either. The channel needs a sender *and*
a receiver.

**Every future control gets measured this way before the experiment runs, never after.**

---

## The lesson that cost twenty-six experiments

Here is the arc, told straight, because it is the most interesting thing here and a
version of this project that reported only its successes would be worth much less.

Getting to that result took twenty-six experiments. **Every single thing that blocked it
turned out to be a broken measuring instrument, not a fact about the brain.**

| what we concluded | what was actually true |
|---|---|
| hens will never understand each other | a full-volume alarm from the next bird moved the listener's ear by **0.0000** — they were deaf |
| learning destroys the brain | one predator strike was mis-scaled into a reward signal **150× too large** |
| the pallium cannot tell two things apart | it had been built with **zero inhibitory neurons** — every neuron in it was excitatory |
| learning does not improve foraging | hens start at exactly the hunger level the metric measures *around*, so it was scoring a coin flip |
| a scrambled channel is as good as a real one | the scramble **kept 98%** of what it was meant to destroy |
| hearing an alarm does not save a hen | the world gave her **no interval** in which a warning could possibly arrive — the hawk appeared already on top of her |

Twenty-five experiments reasoning carefully about a bird whose instruments were broken.
Each time, the project generated a plausible *mechanism* to explain the null instead of
asking whether the null could have come out any other way.

One of these deserves its own sentence. The hens were given an innate startle response
to alarm calls — and it was arithmetically incapable of hiding anyone. Hearing a call
pushed her crouch to `0.269`. Actually hiding required more than `0.5`. Both numbers
were written by the same person, in the same repository, and never once multiplied
together.

**So the standing rule is now: before running an experiment, prove the instrument could
detect a positive result.** Not "does the code run" — whether success is *physically
reachable*. Does the thing you are manipulating actually move? Does your control destroy
what it claims to? Is there a path from cause to effect, in the arithmetic? A null result
means nothing at all until those are answered.

---

## Where the learning stands, which is less happy

The hen is born knowing a handful of reflexes and nothing else — she pecks at small
objects, crouches at things overhead, runs from things on the ground, calls when she is
lonely, and huddles when she is cold. All of it is measured against real published chick
behaviour. **7 out of 7 behavioural tests pass**, and that part has held up throughout.

Then she learns. Except she doesn't.

| | what happened |
|---|---|
| **E001** | Nothing. Twelve thousand synapses rewiring themselves, zero effect on behaviour. |
| **E002** | Found out why: the path from the "thinking" brain to the muscles was frozen. Also found a ceiling — too much learned control makes behaviour **worse**. |
| **E003–E004** | Fixed it, and an effect appeared. ~~Learning works.~~ Later reinterpreted, and it rests on a single batch of seeds that has never been replicated. |
| **E005–E009** | Five attempts to get one hen to *understand* another's alarm call. All null. All of them run on a channel we later discovered was silent. |
| **E010–E012** | Three wrong diagnoses before the right one: a cost added for a *different* experiment was being charged to hunger, tripling it and destroying the metric. |
| **E013** | **The first clean test: learning makes hens significantly worse.** |
| **E014–E016** | The brain damage was a units bug. Fixing it restored the connectome and **changed the behaviour not at all** — so the damage and the harm were two separate things that merely happened together. |
| **E019** | Three defects found by an outside reviewer. The hens could not hear each other, the learned pathway could only slide a single number up and down, and **98% of the reward signal was the cost of calling**. |
| **E020–E021** | Re-ran the clean test with those fixed. **The harm is gone.** So is the confidence: a result that looked decisive on one batch of seeds was worth nothing on the next. |
| **E022–E023** | Second outside review. The pallium had **no inhibitory neurons at all**. Fixed, and everything measured before it became un-comparable. |
| **E024–E026** | The control failed, the risk metric was confounded three ways, and the world had no warning interval. All three fixed → **H4 supported.** |

**The honest state: a clean null.** A hen who learns forages indistinguishably from one
who cannot — **+0.011 ± 0.012** across 24 seeds. Learning does not hurt her. It does not
help her either. The earlier headline that it *harmed* her has been withdrawn, and the
withdrawal sits in the record next to the claim it replaced.

Nothing is deleted here. E013's refutation, E015's decomposition of the harm, E016's
staging result and E020's finding that exploration had become costly have all been
superseded or struck through. **The route is the record**, and given how much of it was
wrong, the route is worth more than any single claim on it.

An unplanned finding along the way that still stands: the hens that **grow** new
connections do *worse* than the ones that only prune. That inverts the obvious
expectation and has its own open question.

**Not yet built:** memory of places, a dominance hierarchy, generational turnover, and
the thing the whole project is named for — a channel on which a hen can invent a word
that was never wired into her.

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
surroundings over days and her rank over weeks. A change that slows the simulation
down is treated as a defect, not a trade-off.

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
share its sign (learning cannot flip an inhibitory neuron to excitatory), excitatory and
inhibitory cells are mixed *within* every region rather than segregated between them,
new synapses can only appear where an axon could physically reach, and the reflex arc is
never touched by learning.

</details>

<details>
<summary><b>Layout and commands</b></summary>

```
coop/    spec.py       the sensory/motor contract everything keys off
         world.py      state, dynamics, predators
         sensing.py    world -> 59-dim observation (the eyes), and the channel modes
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
         h4.py         the channel ladder: deaf / yoked / intact / no-comprehension
         audience.py   playback and comprehension assays
         diagnose.py   is the learned pathway reaching the muscles?
bench/   envelope.py   measure the machine, size the brain
docs/    hypothesis.md the tree of claims -- start here
         backlog.md    what is proposed but not started, and why
         ethics.md     moral standing, tripwires, review log
         experiments/  one file per experiment, predictions written first
```

```bash
pip install -r requirements.txt

python -m bench.envelope --sweep                  # size the brain for your machine
python -m run.probes                              # the newly-hatched behaviour tests
python -m run.lifetime --minutes 60 --plastic     # rear a flock
python -m run.experiment --minutes 20 --seeds 12  # matched-seed A/B
python -m run.h4 --minutes 10 --seeds 12          # the channel ladder
python -m pytest tests/ -q                        # 51 tests
```

</details>

<details>
<summary><b>How the project keeps itself honest</b></summary>

The failure mode for something like this is testing whatever is convenient and calling
the pile a result. Four rules:

1. [`docs/hypothesis.md`](docs/hypothesis.md) is a tree of claims. **No experiment
   runs without naming the claim it feeds.**
2. Every experiment gets a file in [`docs/experiments/`](docs/experiments/) with its
   prediction and its falsifier **written before the run**. E004's prediction was
   committed to git while the run was still executing.
3. Every result comes back and changes something. Nulls are recorded, not buried —
   E001 is a null and it is one of the most useful files in the directory.
4. **No claim moves the record on one batch of random seeds.** E021 measured the same
   comparison on a fresh batch and watched a decisive-looking result become nothing at
   all. A batch that happens to be homogeneous makes a small difference look certain.

Two tools sit alongside those rules. `/status` answers the same three questions at any
point — what we are trying to achieve, what has been *built* versus what has been
*proven*, and what is next — by reading the files rather than the conversation.
`/red-team` buys an outside reader: it commissions a critique from someone with none of
the project's accumulated context, then **independently re-measures every finding before
acting on any of it.**

That second one is not ceremony. The first time it ran, it found that every
communication experiment for eighteen experiments had been measuring a channel that
carried no information. It was also confidently wrong about two things, which is why the
re-measurement step is the rule rather than a footnote.

</details>

---

## Current state

**Phase 0** — a credible newly-hatched hen — is done and measured, and it holds up: she
does seven documented chick behaviours nobody taught her.

**Phase 1** — she learns — is *built* but not *proven*, and the distinction is the honest
part. The machinery all works: eligibility traces, a neuromodulator, synapses that grow
and prune. What it does not do is make her measurably better at anything. The current
result is a clean null. An earlier version of this file called phase 1 complete, on the
strength of a finding that turned out to be an artefact of three defects in the
environment and the learning rule.

**The channel works, and that is the first real result past the hatchling stage.** A
flock that can hear each other is caught 20 percentage points less often when blind, and
a flock whose calls carry no information about the present gets no benefit at all. But
the hens were *born* understanding each other rather than learning to — so what is proven
is that communication pays, not that it can be acquired.

**The two halves of the thesis have now come apart cleanly.** The bird does not yet
learn. The channel does real work. Joining those up — a flock that *learns* what a call
means, and then invents one nature never gave it — is the rest of the project.

Full detail in [`docs/hypothesis.md`](docs/hypothesis.md), which is the authority when
this file and it disagree.

---

## Sources

The biology is not decorative and none of it is invented.

Chicken neuron counts: [Frontiers in Neuroanatomy 2022](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2022.1048261/full) ·
[Olkowicz et al., PNAS 2016](https://www.pnas.org/doi/10.1073/pnas.1517131113).
Deafened chicks develop normal calls: [Konishi 1963](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1439-0310.1963.tb01156.x).
Aerial vs ground alarm calls and audience effects: [Evans & Marler](https://pubmed.ncbi.nlm.nih.gov/3396311/) ·
[Animal Behaviour 1983](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589).
The vigilance/foraging trade-off: [Royal Society Open Science](https://royalsocietypublishing.org/doi/10.1098/rsos.150135).
Naive birds learning what to fear by watching others: [Curio 1978](https://onlinelibrary.wiley.com/doi/10.1111/j.1439-0310.1978.tb00254.x).
How structured languages emerge: [iterated learning](https://arxiv.org/pdf/1910.05291) ·
[symbol emergence](https://arxiv.org/pdf/2303.04544).
Ethics of simulated minds: [Metzinger's moratorium argument](https://www.worldscientific.com/doi/abs/10.1142/S270507852150003X).

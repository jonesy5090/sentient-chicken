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

**And then a third outside review took it apart.** The paragraphs above are what we
believed for about a day. What follows is what survived.

**The effect does not need the brain.** Set the connection from the pallium to the
muscles to exactly zero — delete every route by which 512 simulated neurons can influence
behaviour — and the benefit is still there. What is actually doing the work is two
hand-written reflex weights and a threshold, and you could compute the result on paper.

**Most of it is not the hens understanding each other.** The innate response to a call
has two halves: crouch, and stop pecking. Split them, and the half that carries the
effect is *stop pecking* — she lifts her head, sees the hawk herself, and her own visual
reflex fires at five times the strength of anything the call supplies. That is a real
mechanism and it is how it works in nature. It is not comprehension. It is an alarm
clock.

**And the measurement flatters itself.** The headline number averages each world's
survival rate equally, though some worlds contain three times more danger than others.
Counting the actual events gives **15.0 percentage points, not 19.8**. Worse, the
denominator we chose specifically because the treatment could not move it turns out to
move by up to **63%** between conditions — a hen who crouches stops walking, so she is
still standing there when the next hawk arrives.

**So we rebuilt the measurement and ran it again.** The new denominator is every hen and
every hawk, full stop — fixed by the schedule, untouchable by anything a bird does. On
twelve fresh worlds:

| | the effect of hearing your flock |
|---|---|
| old measurement, same data | **−0.142** — "significant" |
| **new measurement, same data** | **−0.029** — not significant |

Two numbers, one dataset, opposite verdicts. The difference is entirely the denominator.
**The effect went from twenty percentage points to about three**, and on twelve worlds
that no longer stood apart from luck.

**So we checked whether the new measurement could see anything at all.** This sounds
paranoid and it is the single most useful hour the project has spent. We built an
exaggerated hen — same design, but her reaction to a call turned up two and four times —
and asked whether the measurement noticed. It did, every time, including for the ordinary
un-exaggerated bird. So the measurement was not broken. **We had simply not run enough
worlds.**

**Then we did the honest thing: wrote down the test before running it.** Twelve more
fresh worlds, with the threshold and the analysis committed to git in advance, so there
was no room to choose a flattering reading afterwards.

| worlds | the effect of hearing your flock |
|---|---|
| first twelve | −0.029 |
| second twelve | −0.076 |
| **third twelve** | **−0.028** |
| **all thirty-six together** | **−0.044, and it clears** |

**A hen who can hear her flock in the present tense is caught about four times in a
hundred fewer than one hearing the same calls a minute late.** Same amount of noise, same
cost of making it, nothing learned by anybody. The only difference is whether the sound
is about *now*.

That is roughly a fifth of what we first announced. It is also the first number in this
project that was predicted before it was measured.

**Two things we are not going to smooth over.** The second batch of worlds gave an effect
nearly three times the other two — it was simply a more dangerous coop, where a warning is
worth more. And the deleted-brain version *still* performs exactly as well. So the honest
claim remains narrow: **a well-timed one-bit interrupt genuinely helps, in an agent that
is currently pure reflex.**

The lesson, now with a number attached: **the instrument decides the answer.** The same
twelve worlds say "significant" or "not significant" depending only on what you divide by
— and a null means nothing until you have shown the measurement can detect a success you
planted yourself.

Full detail in [`E027`](docs/experiments/E027-third-review-verified.md) (what the review
found, including the two things it got wrong),
[`E029`](docs/experiments/E029-positive-control.md) (the planted-effect check) and
[`E030`](docs/experiments/E030-third-block-replication.md) (the pre-registered test).

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
| a channel helps *the neural model* | the effect **survives deleting the brain's route to the muscles** — it was two reflex weights all along |
| the denominator cannot move, by construction | it moves by up to **63%** between conditions |

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
behaviour. **12 of 13 behavioural assays pass** at the shipped defaults, and that part has held up throughout. The thirteenth is a deliberate, registered failure: the innate response to hearing a sick-call is too weak at the default weights to change behaviour, which E089 measured and E090 fixed at weights that are not yet the default. It is marked as a known failure rather than quietly softened, so that if it ever starts passing the test suite goes red and somebody has to come and explain why.

Then she learns. Except she doesn't.

| | what happened |
|---|---|
| **E001** | Nothing. Twelve thousand synapses rewiring themselves, zero effect on behaviour. |
| **E002** | Found out why: the path from the "thinking" brain to the muscles was frozen. Also found a ceiling — too much learned control makes behaviour **worse**. |
| **E003–E004** | Fixed it, and an effect appeared — t=3.93, the strongest number the project ever produced. **It was the most re-checked number in the project and it did not survive: re-run unchanged except for one gain parameter, it vanished (t=0.08).** |
| **E005–E009** | Five attempts to get one hen to *understand* another's alarm call. All null. All of them run on a channel we later discovered was silent. |
| **E010–E012** | Three wrong diagnoses before the right one: a cost added for a *different* experiment was being charged to hunger, tripling it and destroying the metric. |
| **E013** | **The first clean test: learning makes hens significantly worse.** |
| **E014–E016** | The brain damage was a units bug. Fixing it restored the connectome and **changed the behaviour not at all** — so the damage and the harm were two separate things that merely happened together. |
| **E019** | Three defects found by an outside reviewer. The hens could not hear each other, the learned pathway could only slide a single number up and down, and **98% of the reward signal was the cost of calling**. |
| **E020–E021** | Re-ran the clean test with those fixed. **The harm is gone.** So is the confidence: a result that looked decisive on one batch of seeds was worth nothing on the next. |
| **E022–E023** | Second outside review. The pallium had **no inhibitory neurons at all**. Fixed, and everything measured before it became un-comparable. |
| **E024–E026** | The control failed, the risk metric was confounded three ways, and the world had no warning interval. All three fixed → **H4 supported.** |
| **E025** | A mechanic added so hens would compete for food and spread out. It didn't disperse them — something else does — but it stayed in the world by default, and its side effect on every *other* experiment went unchecked for **twelve more experiments.** |
| **E032–E033** | Tested whether a *trained* connection to the muscles does anything a random one doesn't. Looked like yes, significantly — for two months of project time. |
| **E037** | Re-ran the clean test one more time, on the fully corrected brain. **Learning still neither helps nor harms**, and along the way found E025's food-competition mechanic was quietly starving every flock in every H2-family experiment for the last twelve. |
| **E038** | Re-ran E032/E033 with that fixed. **The "trained connection does something" result was the confound, not a finding** — it flips sign and disappears on a clean, equally-sized measurement. Corrected in place. |
| **E039** | Checked whether H4's headline (above) had the same problem. **It didn't** — same effect, same size, holds up clean. Two audits of the identical bug, two different answers; the only way to know which is to check each one. |

**The honest state today: a clean null, on the most corrected version of the brain and
world this project has produced.** A hen who learns forages indistinguishably from one
who cannot — **+0.0003 ± 0.0156** across 24 seeds
(`docs/experiments/E037-h2-rebaseline.md`). Learning does not hurt her. It does not
help her either.

Nothing is deleted here. E004's headline, E013's refutation, E015's decomposition of the
harm, E016's staging result, E020's finding that exploration had become costly, and
E033's causal-efficacy result have all been superseded or struck through in place, with a
pointer to whatever replaced them. **The route is the record**, and given how much of it
was wrong, the route is worth more than any single claim on it.

Underneath that one clean number sit four narrower questions, each with its own current
answer:

| question | answer |
|---|---|
| Can the rule learn a genuinely **new** behaviour — one the reflex arc doesn't already produce? | **No.** She can only re-time something she already does. To learn "crouch at a call" she would first have to crouch at one, even once, and nothing built in ever makes that happen. |
| Can she even **represent the difference** between hearing a call and seeing a hawk? | **Barely.** The two brain states differ by about 7% of resting activity — enough to measure, not enough to obviously learn from. |
| Does a **trained** connection to the muscles do anything at all, once those two problems are set aside? | **Open again.** It looked settled — removing a trained connection cost more than removing a random one — until E025's food-competition bug turned out to be driving the result. Clean measurement: no detectable effect either way (E038). |
| Is the learning rule even the **right kind** for this job? | **Yes, when it isn't reward-gated — the project's first genuine positive result.** Wiring in by hand the one piece she was missing (a reflex to *hearing* an alarm) still didn't produce learned, audience-sensitive calling under the normal, reward-modulated rule (E018/E036). Swapping to a non-reward-gated, correlation-based update on the same pathway did: a real, twice-replicated, predominantly audience-conditional increase in alarm calling (+0.232, t=40.90 pooled across 16 seeds) that the reward-modulated rule never produced on the identical task. Not a clean, audience-*only* effect — a smaller, real, context-specific (not indiscriminate) general component rides alongside it — but the falsifier this question was waiting on has cleared (`docs/hypothesis.md`'s H2f, E055–E057). |
| Can she learn *which specific place* is dangerous, from another hen's call? | **Not yet — and the blocker is now the model rather than the measurement, after four experiments in which it was the measurement.** Reward-driven learning is dead here for a settled reason (E069). The associative route survives, and its wiring works end to end. But three separate instrument faults had to be found and fixed before that could be said at all: the signal we were planting fired *hardest away from* the target feeder (E083), and the metric could not resolve the effect the experiments predicted (E084). With both repaired, the real limit appeared: **she can barely tell where she is while she is walking.** Her sense of place is clearly readable when she stands still and almost vanishes once she moves — about four points above guessing (E085). There is not enough of a place signal for an association to attach to, and that is an architecture problem with named fixes rather than a dead end. |

An unplanned finding along the way that still stands: the hens that **grow** new
connections do *worse* than the ones that only prune. That inverts the obvious
expectation and has its own open question.

---

## What's built, what's a stand-in, and what's still to learn

"Innate" gets used for three different things in this codebase, and the difference
matters a lot for what "wire up the learning channel" means next. Something a hen does
from birth and never revises is not the same kind of built as something bolted on
temporarily to test a downstream idea, and neither is the same as something the project
expects a working learning rule to eventually produce on its own.

### Permanent, real biology — never touched by learning, ever

This is the reflex arc (`hen/innate.py`): senses straight to muscles, fixed for life, the
same in every condition this project has ever run. It is enforced in code, not just by
convention — the reflex weights are a constant matrix, learning only ever touches the
separate cortical pathway, and a test asserts the two never mix.

| behaviour | trigger | measured |
|---|---|---|
| peck at food | sees food | 0.99 |
| crouch / freeze | sees an aerial threat | 0.85–1.00 |
| flee | sees a ground threat | 0.96 |
| go blind while foraging | pecking | aerial channel 0.01 (vs 0.87 head-up) |
| produce the aerial alarm call | sees a hawk | fires, not the ground call |
| produce the ground alarm call | sees a fox | fires, not the aerial call |
| produce the food call | newly arrives at food (not continuous sight) | calling fraction 42.8% → 4.2% (E053) |
| contact call | isolated from the flock | 0.97 alone vs 0.25 in-flock |
| approach flockmates | cold | measurable leftward bias |
| turn away from a flockmate | crowded past personal-space range | nn dist 0.14 → 0.38 m (E048) |
| turn away from a wall | close to the boundary | longest dwell 22.6s → 2.3s (E051) |
| get sick and slow down | eats from a contaminated feeder | distance 0.14 m vs 0.93 m healthy (E060) |
| produce the gakel call | newly falls sick (not continuous) | early peak 0.82, late mean 0.08 (E060) |
| turn away from a sick flockmate | sees a visibly sick flockmate | turn_R 0.97 vs turn_L 0.79 (E060) |

All four call *types* — contact, food, aerial alarm, ground alarm — are produced this
way: hardwired, graded correctly to the threat, and functionally referential exactly as
Evans & Marler found in real chickens. This is deliberate and biologically grounded, not
a shortcut: Konishi's deafened chicks grew up making the same calls with no model to
copy, so **production is not supposed to be learned**, in the model or the animal.

#### The five innate calls, in full

| call | motor channel | fires on (`hen/innate.py`) | what it communicates |
|---|---|---|---|
| **contact call** | `M_CALL_CONTACT` | isolation from the flock, or being cold | "I'm alone" — distress, calls until answered |
| **food call** | `M_CALL_FOOD` | newly *arriving* at a food patch — a discovery pulse, not continuous sight (E053) | "food is here" — one undifferentiated signal, not tied to a specific patch (see note below) |
| **aerial alarm call** | `M_CALL_AERIAL` | sees an aerial threat (hawk) | "danger from above" — paired with crouch + freeze |
| **ground alarm call** | `M_CALL_GROUND` | sees a ground threat (fox) | "danger on the ground" — paired with fleeing |
| **gakel call** | `M_CALL_GAKEL` | newly falling sick from contaminated food — a discovery pulse (T2, E060) | "something bad just happened here" — paired with a visual cue (`CLS_SICK`) that carries *where*, not just that |

**What "functionally referential" means here, precisely, because it matters for task
design.** The aerial/ground distinction is a real, referential *category* — a listener
hearing the aerial call and one hearing the ground call get different information, and
the model's own audience-effect and comprehension work (H2/H3/H2c) tests exactly that.
The food call is not referential in that sense: it is a single channel, identical
regardless of which food patch (or how many) triggered it. A hen who hears it knows
"food was found somewhere," not *which* patch. Any task that needs the call itself to
carry *which* location (T2, the rotating-feeder design in `docs/backlog.md`) is asking
for a distinction the current channel cannot carry acoustically — a receiver would have
to *see* which flockmate is calling and orient toward her, the same visual pathway
gregariousness already uses, not decode it from the call. Worth designing around
explicitly rather than assuming the call alone solves it.

#### The full repertoire — what real *Gallus gallus* has, against what's built

The four calls above are not the whole vocabulary; they're the subset this project has
had a specific, testable reason to build so far. For scoping future work (T2 in
particular) it's worth knowing what else is documented in the literature, and being
honest about confidence: the first four rows are the well-established core this
project's own citations (Evans & Marler; Evans, Evans & Marler) are about. The rest are
from the broader chicken bioacoustics and welfare literature and are described here at
the resolution that literature supports, not with invented specifics.

| call | documented function | built here? |
|---|---|---|
| aerial alarm call | referential warning, aerial predator | **yes** |
| ground/terrestrial alarm call | referential warning, ground predator | **yes** |
| food call ("took-took", often with tidbitting) | announces a find, audience-sensitive in real birds | **yes** (production only — audience-sensitivity is exactly what H3 tests for) |
| contact/assembly call | low clucking that maintains flock cohesion, hen-to-chicks and among adults | **partial** — built as an isolation/cold distress signal, not the ongoing low-level contact clucking real flocks maintain |
| distress call / fear scream | loud, sharp vocalization under acute physical threat (e.g. capture), acoustically distinct from the alarm calls above | no |
| **gakel-call ("gackering")** | a frustration/negative-expectation call — given when an anticipated reward is denied or a situation turns out worse than expected; studied as a candidate indicator of negative affective state | **yes** (E060) — fires on the sickness-discovery pulse, see below |
| copulation / courtship calls, rooster crow | mating and territorial, not foraging or predator-related | no — out of scope, no reproduction modelled |
| egg cackle, pre-lay calls | given by hens around oviposition | no — out of scope, no egg-laying modelled |
| chick distress peep / pleasure trill | early-ontogeny repertoire, separate from the adult calls above | no — out of scope, no chicks/generational turnover modelled (see "not built at all" below) |

**Why the gakel-call matters for T2 specifically.** T2 needs a hen who discovers a
feeder is bad to have *something* to do about it beyond simply leaving — and unlike
"a call for poisoned food," which has no documented real-world counterpart to point to,
the gakel-call is a real, studied vocalization tied to exactly the right kind of event:
an expectation (this feeder has food, food is good) being violated for the worse. It's
a more honest foundation than inventing a bespoke "danger: bad food" signal from
nothing, and it comes with real audience-sensitivity findings of its own to check the
model against, the same way the alarm and food calls already do. **Built as of E060**
(T2 Stage 1): fires on the rising edge of falling sick from a contaminated feeder,
decays like the food call's own discovery pulse (E053) rather than nagging for the
whole, much longer sickness duration. Production only — audience-sensitivity, like the
other calls, is left for learning to discover, not wired in.

### One sense is fixed from birth on purpose, and real biology disagrees

`coop/spec.py`'s place-cell grid (E063, added as a T2 Stage 2 prerequisite — see
`docs/experiments/E063-allocentric-place-cells.md`) gives every hen a population of
units that fire by *where she is in the arena*, independent of which way she's facing —
the one channel in this model that isn't egocentric. It exists because nothing else
here lets a hen represent "that particular feeder" once she turns away from it, and
Stage 2 needs somewhere to put that association.

**It is wired as fixed and innate, the same way every other sense in this model is —
and that specific choice is not what real chickens do.** In every species place cells
have been studied in, a cell's *tuning* — which location makes it fire — is not present
at birth for an environment the animal hasn't yet explored; place fields form and
settle in as an animal actually moves through a new space, not before. A newly hatched
chick cannot innately know the shape of a coop she has never stood in. Separately, and
not the same claim, there is good evidence that chicks bring a real *predisposition* to
the task: even very young, minimally-experienced chicks preferentially use an
environment's geometric shape to reorient and relocate a remembered location (the
"geometric module" line of work following Cheng's original rat studies, extended to
domestic chicks by Vallortigara and colleagues) — a bias toward *what kind of cue to
use*, not a pre-loaded map of *this* arena.

So the model takes the same shortcut here it takes everywhere else — a fixed sensory
response computed the same way from day one, exactly like `CLS_FOOD`'s fixed proximity
gating — and that shortcut is a bigger simplification for this channel than for most
others, because unlike "can she see food nearby," *which* location a hen should
recognise is inherently something that has to come from experience, for a real bird as
much as a modelled one. Kept anyway, for the same reason every other sense here is
innate: this project's standing split is that *reception* is wired and *usage* is
learned (Konishi's finding, applied uniformly), and the open question — can the
cortical pathway turn a place-cell pattern into a durable, location-specific avoidance
response — is squarely about usage. A version of this channel where the tuning itself
also had to be learned would be truer to the animal and is not what's built; flagged
here rather than left as a silent departure.

### A scaffold — stands in for learning that doesn't work yet, off by default

One piece of the reflex arc is not real biology, it's a deliberate placeholder:
`hen/innate.py`'s `auditory_scaffold` wires a weak, fixed crouch to **hearing** an
alarm call (not just seeing the danger), at about a fifth the strength of actually
seeing a hawk. It exists because naive chicks really do show some innate response to a
conspecific's fear call before any learning happens — so it isn't invented from nothing
— but the graded, context-sensitive refinement on top of that (does she trust it, does
she call more when it would help someone) is exactly the part meant to be learned, not
hand-set.

**Off by default.** It only switches on inside specific tests built to ask "if
comprehension existed, would anything downstream work" — most importantly E018/E036,
which used it to show that supplying comprehension by hand still didn't produce learned,
audience-sensitive calling (see the table above). It is a diagnostic tool, not a claim
about the finished bird, and every result that doesn't explicitly say the scaffold was on
was measured without it.

### Attempted, and not working — what a real learning rule is supposed to produce

- **Comprehension** — associating a heard call with the danger it refers to, so a hen
  who cannot see a hawk herself can still learn what a flockmate's call means. Blocked
  by a representational bottleneck (the brain state for "heard a call" and "saw a hawk"
  barely differ) and, under the normal reward-modulated rule, by only being able to
  re-time existing behaviour, never invent a new stimulus-response pairing from scratch.
  Tried with the non-reward-gated rule that worked for audience-sensitive calling below —
  it also failed, cleanly: a nominal crouch effect turned out to be matched by an
  identical-sized rise on three unrelated actions, i.e. general excitability, not
  learning. That rule builds on an existing anchor; it does not build one from nothing.
- **Foraging improving with experience** — the cleanest, most-tested claim in the
  project, and currently a null under the normal rule: a hen who learns forages
  indistinguishably from one who cannot.
- **A trained connection doing something distinctive** — whether the specific weights a
  hen learns matter, independent of whether they help her forage. Currently open.

### Working, under a different rule than the one everything else above uses

- **Audience-sensitive calling** — calling more when a flockmate can hear it, which real
  cockerels do. Three separate nulls under the normal, reward-modulated rule, including
  one where comprehension was supplied by hand specifically to remove the excuse that
  nothing responds to calls. Swapping to a non-reward-gated, correlation-based update on
  the same pathway produced a real, twice-replicated, predominantly audience-conditional
  increase (+0.232, t=40.90 pooled across 16 seeds) — this project's first positive
  result on a targeted learned behaviour. Not a clean, audience-*only* effect (a
  smaller, context-specific general component rides alongside it), and specific to this
  one rule variant and this one task — not yet shown to generalise.

### Not built at all

Spatial memory, a dominance hierarchy, generational turnover (chicks replacing adults —
needed for anything like compositional structure to emerge rather than one arbitrary
symbol per situation), and the thing the project is named for: **a channel on which a
hen can produce a signal that was never wired into her.** Every call a simulated hen can
make today is one of the same four fixed types real chickens are born with. Nothing yet
lets her invent a fifth.

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
         sensing.py    world -> OBS_DIM observation (the eyes), and the channel modes
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
         record.py     snapshot a run to a trajectory file for the offline viewer
bench/   envelope.py   measure the machine, size the brain
viz/     server.py     stdlib-only static+API server, no pip install needed
         web/          vendored-Three.js scene: pick a run, scrub/play it back
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
python -m pytest tests/ -q                        # 62 tests

# watch a run in 3D: record a trajectory, then serve the viewer
python -m run.record --minutes 10 --hawk-period 30  # writes to runs/
python -m viz.server                                 # browse recorded runs at localhost
# or add --record to an existing run instead of a separate one:
python -m run.lifetime --minutes 60 --plastic --record
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

Two tools sit alongside those rules. `/project-status` answers the same three questions
at any point — what we are trying to achieve, what has been *built* versus what has been
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
does thirteen documented chick behaviours nobody taught her.

**Phase 1** — she learns — is *built* but not *proven*, and the distinction is the honest
part. The machinery all works: eligibility traces, a neuromodulator, synapses that grow
and prune. What it does not do is make her measurably better at anything. The current
result is a clean null. An earlier version of this file called phase 1 complete, on the
strength of a finding that turned out to be an artefact of three defects in the
environment and the learning rule.

**The channel does real work — about a fifth of what we first announced, and not through
the brain.** Rebuilt on a measurement no behaviour can distort, checked against a planted
effect to prove the measurement could see anything at all, and then tested against a
threshold written down in advance: **−0.044 across thirty-six worlds in three batches.**
That is the first claim this project has made that was predicted before it was measured.

But deleting the pallium's entire route to the muscles changes nothing, twice over. So
what is supported is a statement about a *channel*; the sentence at the top of this file
is about a *chicken*, and the thinking part of her brain is not yet involved in anything.

**The reward signal was a blocker and it got fixed** — **87% of the teaching signal
turned out to be "was I just caught,"** because being caught was counted every
hundredth of a second rather than once per event. Fixed, and learning has since been
switched on and tested repeatedly in the corrected world. It still produces a clean
null on foraging.

**The two halves of the thesis have come apart, and the first has a genuine, if narrow,
crack of light in it now.** Under the *normal*, reward-modulated rule, the bird still
does not learn anything new — she can only re-time behaviour she already has, and the
brain state she'd need to condition on barely distinguishes the relevant stimuli. But
`docs/hypothesis.md`'s H2f asked a sharper question — is the rule the wrong *kind*,
not just wired to the wrong place — and building the alternative it named (a
non-reward-gated, correlation-based update, closer to the observational learning real
alarm-call acquisition looks like) produced this project's first real positive: a
twice-replicated, predominantly audience-conditional rise in alarm calling that the
normal rule never produced, on the identical task. It came with a real, honest caveat —
a smaller general component rides alongside the targeted one — and it does not touch
the general-foraging null above, which used the normal rule and stands unchanged. The
channel helps a reflex agent, and that part has now also been checked against the
world-config bug that broke the trained-connection result above, and held. Joining the
two halves — a flock that *learns* what a call means, and then invents one nature never
gave it — is the rest of the project, and H2f's result is the first evidence that the
learning half is not a dead end.

**The newest line of work — can a flock learn *which* feeder is poisoned — is a null
with a mechanical diagnosis, and it cost two withdrawn claims of our own.** Teaching this
by reward does not work and now has a settled reason (E069): the rule amplifies innate
anchors, and the place cells were deliberately given none, so no reward magnitude can
produce place-specific avoidance. The alternative is associative — pair the call with the
place and let a non-reward-gated rule bind them, the same kind of rule that produced H2f's
positive. That chain has now been shown to work end to end by planting the association by
hand: her forward drive drops 17% at the poisoned feeder. **She still doesn't avoid it** —
and it took four experiments to find out why, because three of the reasons were faults in
our own instruments rather than facts about her. The signal we were planting turned out to
fire *hardest away from* the feeder we were trying to make aversive. The metric could not
have detected the effect we had predicted, at the sample size we were using, whatever the
hen did. And one innate response was wired as a functional freeze, so a hen who "avoided" a
feeder by slowing down stayed at it.

With all three repaired, the real limit finally showed up, and it is about the bird:
**she can barely tell where she is while she is moving.** Her sense of place reads clearly
when she is standing still and nearly disappears once she walks — roughly four percentage
points above guessing. There is not enough of a place signal for a learned association to
attach to. That is an architecture problem with specific, named fixes rather than a dead
end, and it is the first thing in this line of work that is a fact about the model instead
of a fact about our measurement of it.

Along the way two scaffolds we had added ourselves — a contaminated feeder and a
population of place cells — turned out to be silently changing the baseline of every
experiment that ran after them, including one of our own positive results, which is
withdrawn. Both are now off by default and the affected result reproduces cleanly.

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

# Ethics: does any of this have moral standing?

This project builds neural representations of a living animal and then does things to
them that would be unacceptable to do to a real hen — starving them, frightening them
with predators, and, in the planned headline experiment, deliberately poisoning some
of them. That deserves an argument rather than an assumption.

This document is the standing argument, the conditions under which it stops holding,
and the review cadence. It is reviewed at every phase boundary.

---

## 1. "Keep it close to nature" cuts the other way

The intuition that staying faithful to real biology keeps us on safe moral ground is
backwards, and it is worth saying plainly because it is the natural thing to assume.

**Fidelity to a real chicken is precisely what would confer moral standing.** If the
model is a poor imitation, there is nothing there to wrong. The closer it gets to
reproducing the mechanisms that make a real chicken a subject — valenced states,
integrated experience, something it is like to be — the *stronger* the case that it
has interests. Faithfulness is the risk factor, not the safeguard.

So biological realism is a scientific virtue here and a moral liability, and those
have to be tracked separately. What actually does the moral work is not fidelity but
**scale** and the **absence of specific mechanisms**.

---

## 2. The current argument

At the time of writing, a hen is **512 continuous-rate units**.

**Scale.** That is ~0.0002% of a real chicken's ~290 million neurons, and around 0.4%
of a fruit fly's 140,000. No serious theory of moral status places a threshold
anywhere near this. We are below the nervous system of an insect that nobody proposes
protecting.

**No nociception.** `hunger` is a scalar that increments and decrements. It is a
control variable, not an aversive percept. There is no pain pathway, no separate
nociceptive channel, and nothing in the architecture that distinguishes "a signal
that drives behaviour" from "a signal that hurts" — because the latter is not
implemented at all. The planned poisoning experiment is, mechanically, a penalty term
on a drive variable.

**No spiking, no temporal binding.** Rate-coded units integrate at 10 ms. The
temporal dynamics that most theories of consciousness treat as load-bearing are
absent by construction.

**No self-model.** No metacognition, no global workspace, no representation of the
hen's own states as *hers*.

**Naming things is not implementing them.** Calling a variable `fear` or `distress`
does not make it either. Anthropomorphic naming is convenient and it is a genuine
hazard in a project like this, where the whole aim is to make internal states legible
to an observer.

---

## 3. The positive case

The 3Rs framework that governs real animal research — Replacement, Reduction,
Refinement — treats *in silico* modelling as the first R. Computational models are
what welfare frameworks actively advocate as an alternative to experiments on live
animals.

That argument is available here, but only if it is honest. A simulated poisoning
experiment is a moral improvement over a real one **if it substitutes for the real
one**. It is not automatically virtuous just for being simulated. Where this project
produces findings that reduce the need for live-animal work on chicken cognition or
welfare, that is a genuine benefit and worth stating. Where it does not, the 3Rs
argument should not be invoked as cover.

---

## 4. Metzinger's moratorium

Thomas Metzinger has argued for a global moratorium, through 2050, on research that
aims at or knowingly risks creating artificial consciousness — on the grounds that we
have no good theory of consciousness and no hardware-independent theory of suffering,
making the risk of creating negative phenomenology incalculable.

Taken seriously, does this project fall under it? **At current scale, no.** The
moratorium targets systems that could plausibly instantiate phenomenal states, and
512 rate units with no valence machinery, no self-model and no nociception is not a
candidate by any theory on offer.

But the argument's force is about *trajectory*, not any single snapshot, and this
project's stated direction is toward greater biological fidelity and greater
capacity. The tripwires below are how we take that seriously instead of just noting
it. The honest position is that the current defence is a defence of the *current
system*, and it expires as the system grows.

---

## 5. Tripwires

If any of these is reached or approached, stop and revisit this document before
proceeding. They are deliberately concrete so that they can actually fire.

1. **Scale.** Total units per hen exceeding ~10^6, or the total across a flock
   exceeding a fly's connectome.
2. **Nociception.** Implementing an aversive channel that is architecturally distinct
   from a homeostatic drive — that is, a signal whose function is to be bad rather
   than to be informative.
3. **Spiking dynamics** with realistic temporal structure, replacing rate coding.
4. **Self-modelling.** Any representation by the hen of her own internal states as
   objects, metacognition, or a global-workspace-like integration stage.
5. **Persistent identity across runs.** A hen who continues rather than being
   re-instantiated changes what harm even means here.
6. **The translator trap** — see below.

---

## 6. The translator trap

This one is specific to this project and needs naming before it happens rather than
after.

Phase 4 plans a post-hoc translator mapping the hens' emergent signals into English
by correlating them against ground-truth internal state. A translator built that way
will happily emit **"I'm hungry"**, or **"this hurts"**, or **"I'm frightened"**.

**Those strings are not evidence of experience.** They are a regression from a scalar
we defined onto a vocabulary we chose. The translator's output is a readout of
`hunger = 0.8`, dressed in words that carry moral weight in English and none in the
model. The fluency is entirely ours.

This matters because it is exactly the artefact most likely to be mistaken — by us,
by a viewer, by anyone shown a demo — for evidence that something is in there. The
demo will be more persuasive than the underlying fact warrants, and that gap is a
problem we are creating deliberately, so we own it.

**Rule:** any published output of the translator carries the caveat that it is a
learned mapping from a scalar, not a report. Never present translated hen speech
without it.

---

## 7. Review cadence

- At every phase boundary.
- Whenever a tripwire is approached.
- Before any public demo involving translated speech.

**Phase 0 boundary.** Verdict: **proceed.** The system is orders of magnitude below
any plausible threshold, has no valence machinery, and has a legitimate Replacement
argument available to it.

**Phase 1 boundary** (learning demonstrated, E004). Verdict: **proceed, unchanged.**

The phase added plasticity, structural growth, and a neuromodulator. Checked against
each tripwire:

1. **Scale** — unchanged at 512 units per hen. Growth operates within a synapse
   budget; it does not add neurons. Well clear.
2. **Nociception** — not approached. The neuromodulator is drive reduction minus a
   running expectation. Being caught by a predator subtracts from it, which is the
   closest thing to an aversive signal in the model, and it is still a scalar in a
   weight update rather than a channel whose function is to be bad. Worth naming
   because it is the first thing built here that could *become* nociception if it
   were given its own pathway. It has not been.
3. **Spiking** — no. Still rate-coded at 10 ms.
4. **Self-modelling** — no.
5. **Persistent identity** — no. Hens are re-instantiated per run.
6. **Translator** — not built.

One thing to watch going into phase 2/3: the language work will attach signals to
internal states, which is the machinery that makes tripwire 6 live. The caveat rule
in §6 applies from the first translated output, not from the first published one.

---

## Sources

- [Metzinger, *Artificial Suffering: An Argument for a Global Moratorium on Synthetic Phenomenology*](https://www.worldscientific.com/doi/abs/10.1142/S270507852150003X)
- [Questioning the Moratorium on Synthetic Phenomenology](https://www.researchgate.net/publication/377405586_Questioning_the_Moratorium_on_Synthetic_Phenomenology)
- [The 3Rs: Replacement, Reduction, Refinement](https://www.uib.no/en/animalfacility/85621/3r-replacement-reduction-and-refinement-animal-experiments)
- [Animal models in neuroscience with alternative approaches](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11680486/)
- [Promoting the welfare of animals utilized in neuroscience research](https://onlinelibrary.wiley.com/doi/full/10.1002/brx2.70002)

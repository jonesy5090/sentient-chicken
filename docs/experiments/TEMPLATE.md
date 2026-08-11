# E000 — short title

> Copy this file to `E0NN-slug.md`. Fill in sections 1–5 **before** running anything.
> Sections 6–8 are written afterwards. An experiment with no pre-registered
> prediction and no falsifier is not an experiment, it is a demo.

## 1. Parent hypothesis

Which node in `docs/hypothesis.md` does this feed? If none, stop — it either belongs
in `docs/backlog.md` as a proposed new branch, or it does not belong in the project.

## 2. Question

One sentence.

## 3. Prediction

Stated numerically wherever possible, and *before* the run. "X will be lower than Y"
is acceptable; "we will observe interesting dynamics" is not.

## 4. Falsifier

What result would count against the parent hypothesis? If nothing could, this
experiment cannot inform anything and should not be run.

## 5. Design

- **Conditions**, and what is held identical across them (seeds, genome, coop,
  predator arrivals — matched unless there is a reason not to).
- **Primary metric**, chosen now, not after looking.
- **Secondary metrics**, explicitly marked as exploratory.
- **Replicates**: how many seeds, and why that many.
- **Command** to reproduce, with the commit hash.

## 6. Result

Numbers, with uncertainty. Include the negative and null results; they are the ones
that keep the hypothesis tree honest.

## 7. Interpretation

What this does and does not license concluding. Name the alternative explanations you
cannot rule out — particularly any confound between conditions.

## 8. Consequence

**Required.** What changed as a result:

- status update in `docs/hypothesis.md`
- new or removed backlog items
- an ethics review, if a tripwire was approached
- nothing, and why the experiment was still worth running

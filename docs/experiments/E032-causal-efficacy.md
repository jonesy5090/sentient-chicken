# E032 — causal efficacy: does a *trained* readout do anything?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2e** — the cortical pathway is behaviourally inert, so H2 is not testable through it.
Opened by [E031](E031-the-credit-window-is-not-the-blocker.md).

## 2. Question

E031 found an **untrained** `W_out` contributes nothing to feeding at 0×, 1× or 10× gain,
on a metric that detects a halved reflex at t=4.32. But an untrained readout is a random
projection, and random drive is roughly zero-mean, so it may simply average out. **A
learned readout is structured and could do what a random one cannot.** E031 could not
separate those, and said so.

This does. It is `docs/backlog.md` §5's *causal efficacy* test — *"take a trained flock
and mute the channel at test time; the drop in performance is the amount of work it was
doing"* — applied to the readout rather than the channel. It has sat unrun since the
backlog was written.

## 3. Design

Two rearing conditions × two test conditions, matched seeds. Every test branch forks from
the **identical** end-of-rearing world and brain state, so the lesion is the only
difference — a genuine within-subject manipulation rather than two separate runs.

| rearing (20 min) | test (5 min, no plasticity) |
|---|---|
| **plastic** | intact `W_out` |
| **plastic** | `W_out` ≡ 0 |
| **fixed** (no plasticity) | intact `W_out` |
| **fixed** | `W_out` ≡ 0 |

**Primary quantity — the interaction:**

```
(trained: intact − lesioned)  −  (fixed: intact − lesioned)
```

The fixed pair is the control for the lesion itself. Any drop it shows is what removing a
*random* projection costs, which E031 measured at zero; the trained pair's drop is what
removing a *learned* one costs. The difference is the work learning did.

**Metric:** `fed %` over the test window, the metric E022 recommended as primary for H2.

## 4. Manipulation check — and the run is void without it

**`|W_out|` must actually change during rearing.** E001 was a null for three experiments
because `eta_out` was so small that `|W_out|` grew 1.00× — the readout was frozen and
nobody checked. If training does not move `W_out` here, this experiment tests nothing and
the result must be discarded rather than reported.

Gate, fixed in advance: **mean |ΔW_out| / |W_out| > 0.05** between hatch and end of
rearing. Reported whatever it is.

## 5. Prediction and falsifier

**H2e predicts the interaction is ~0**: lesioning a trained readout costs no more than
lesioning a random one, because the pathway is inert either way.

**Falsifier for H2e — and this is the outcome worth hoping for:** the interaction is
significantly negative, i.e. trained flocks lose measurably more when their readout is
cut. That would mean the pathway *is* efficacious once structured, E031's inertness is a
property of random weights only, and **H2's null returns to being a fact about the
learning rule** rather than the architecture.

Threshold: two-tailed t at 11 df = **2.201**, 12 matched seeds.

**Secondary, reported but not decisive:** whether the trained-intact flock feeds better
than fixed-intact at all — which is H2's original question, on a fresh block.

## 6. Result

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._

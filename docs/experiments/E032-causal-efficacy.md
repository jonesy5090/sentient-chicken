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

## 6. An error in §5, recorded rather than edited

**The falsifier's sign label contradicts its own description, and it was committed that
way.** §5 reads: *"the interaction is significantly **negative**, i.e. trained flocks lose
measurably more when their readout is cut."* Those clauses disagree. Losing more when cut
means a **larger drop**, and the interaction as defined in §3 —
`(trained drop) − (fixed drop)` — is then **positive**.

The substantive claim is the verbal one; the sign word is the error. Left in place and
corrected here, following [E018](E018-innate-auditory-reflex.md), which recorded a
mis-derived prediction rather than quietly editing it. **The direction being tested is
unchanged, so this does not licence choosing a sign after the fact** — and it happens not
to matter, because the result does not reach threshold either way.

## 7. Result

12 matched seeds, 20 min rearing + 5 min test, 16 hens.

```
MANIPULATION CHECK: mean |dW_out| / |W_out| during rearing = 0.0963   PASSES (gate 0.05)
                    fixed flocks drift 0.000000                        as required

rearing         intact  lesioned      drop
trained         13.782    13.555    +0.227
fixed           13.591    13.905    -0.314

PRIMARY interaction  +0.5412 +/- 0.2537   t=2.13   threshold 2.201  NOT SIGNIFICANT
secondary, both intact: fed % +0.1913 +/- 0.4473   t=0.43
```

**The manipulation check is clean.** Training moves the readout by 9.6%, and fixed flocks
drift by exactly zero — so this is not E001's frozen `W_out`, and the two arms differ in
the one thing they are supposed to.

## 8. Interpretation

**Not significant, and the direction is the interesting part.**

- Lesioning an **untrained** readout **helps**: −0.314. A random projection into the motor
  drive is mildly harmful, which is E002's ceiling finding — an untrained pallium
  overriding good reflexes — seen from the other side.
- Lesioning a **trained** readout **hurts**: +0.227. After rearing, the hen is slightly
  better off keeping it.
- The difference, +0.541 ± 0.254, is **t=2.13 against a 2.201 threshold**. It misses.

So the honest statement is: **there is a hint that training converts a mildly harmful
readout into a mildly useful one, and it is not strong enough to claim.** On one seed
block it could not move a status anyway — the E021 rule.

**H2e is neither confirmed nor falsified.** Its prediction was an interaction of ~0; the
measurement is closer to +0.5 but cannot exclude 0 at this n. What E031 established
stands: an *untrained* pathway is inert-to-harmful. What E032 was built to settle — whether
a *trained* one is different — remains open by a margin of 0.07 in t.

**H2's own question is still null.** Trained against fixed, both intact: +0.191 ± 0.447,
t=0.43. Whatever the trained readout is doing, it does not show up as better foraging.

## 9. Consequence

- **H2e stays `UNDER TEST`.** Do not report this as support for either side. (It still
  does, after E033's second block and [E038](E038-h2e-depletion-audit.md)'s correction of
  it — this file's caution held up better than what followed it.)
- **A second block is the obvious next move and it is cheap in a machine that can run
  uninterrupted.** 12 fresh seeds at the same design; if the interaction holds near +0.5
  the pooled estimate clears comfortably, and if it collapses the way E021's did, H2e
  gains real support. **Declare the pooling in advance this time** — E029/E030 is the
  template.
- **The `--rear 20` cost is the binding constraint**: ~6 min per trained cell, because
  consolidation writes the full `(H,N,N)` tensor every 50 steps. A second block is ~80
  minutes of uninterrupted compute.
- **If a second block lands near zero**, H2 is a fact about the architecture and E007's
  unresolved additive-versus-multiplicative gating question is the next real piece of
  work. If it lands near +0.5, the readout does earn influence through learning, and H2's
  null returns to being about the *rule* — most likely its magnitude rather than its
  timing, since E031 already ruled out the credit window.

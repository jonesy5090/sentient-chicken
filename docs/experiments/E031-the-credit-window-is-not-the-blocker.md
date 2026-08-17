# E031 — the credit window is not the blocker, and H2 may not be testable

> **Diagnostic**, written after the fact and labelled as such. Three measurements, each
> with its acceptance criterion fixed before it ran. None of them tests H2; they test
> whether H2 *can* be tested.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement. Currently
`UNDER TEST`, a clean null across 24 seeds.

## 2. Question

The 0.2 s credit window has been the leading explanation for H2's null since E022 and was
promoted to the top of the queue by E028. `hen/plasticity.py:34-37` states it in the
source: *"anything that has to bridge a longer gap than that is not learnable by this rule
as written."* It has been cited four times and **never measured.**

## 3. Measurement one — is the task's payoff actually delayed? **No.**

`scratchpad/creditgap.py`, 120 s, 16 hens.

```
feeding events per hen         : 426.7   (min 244, max 765)
mean gap between feeds         : 0.3 s
steps with any reward movement : 100.0%

corr( reward(t), peck(t - lag) )
   lag 0.00 s   0.1180
   lag 0.20 s   0.1097      <- edge of the rule's window
   lag 5.00 s   0.1770      <- peak
```

**The premise is wrong.** Reward is not sparse and does not arrive minutes later: a hen
feeds every **0.3 seconds** and the reward signal moves on every step. Two thirds of the
peak peck–reward correlation is already available at lag 0, inside the window. The larger
correlation at 5 s is consistent with foraging bouts persisting — she is still in the
patch — rather than a causal bridge the rule must span.

**The credit window is not what blocks H2.** A line in a docstring, cited for four
experiments as a mechanism, does not survive its first measurement. This is the fourth
time in this project a plausible sentence next to the code turned out to be a claim
rather than a fact.

## 4. Measurement two — can the cortical pathway move H2's metric at all?

If the pathway learning is supposed to act through cannot change the number H2 measures,
then no learning rule can produce an effect and the null says nothing. E027 and E030 both
found `W_out` lesions leaving *predation* outcomes untouched; this asks the same on H2's
own metric.

`scratchpad/inertness.py`, 12 matched seeds × 5 min, 16 hens, no plasticity.

| condition | fed % | vs fixed |
|---|---|---|
| fixed | 3.671 | — |
| **lesioned (`W_out` = 0)** | 3.531 | **−0.139 ± 0.204, t=0.68** |
| **amplified (`W_out` × 10)** | 3.561 | **−0.110 ± 0.203, t=0.54** |
| peck reflex × 0.5 *(positive control)* | 3.213 | **−0.458 ± 0.106, t=4.32** |

**Deleting the cortical pathway and multiplying it tenfold produce the same result:
nothing.** The metric is demonstrably not blind — it detects a halved innate reflex at
t=4.32, an effect three times smaller in nothing but consistency.

## 5. What this does and does not establish

**Establishes:** an **untrained** cortical pathway contributes nothing measurable to
feeding, at default gain or at 10×, on a metric sensitive enough to detect a reflex
change. H2's null therefore cannot be attributed to the learning rule until the pathway
is shown able to move the metric at all.

**Does not establish** that a *trained* pathway could not. `W_out` here is a random
projection, and random drive is roughly zero-mean, so it may simply average out. A
learned `W_out` is structured and could in principle shift behaviour where a random one
does not. **This is the distinction the experiment cannot make, and it should not be
glossed.**

Two existing results lean the same way without settling it: E019 measured a learned
`ΔW_out` moving the cortical contribution by 0.7% of its own magnitude, and E027/E030
found lesioning `W_out` leaves predation outcomes unchanged.

## 6. Consequence

- **The credit window is struck as H2's explanation.** Not deferred — measured and
  refuted. Do not sweep `tau_slow`; it would be a fix to a problem that does not exist.
- **H2's null is downgraded to uninformative** pending §7. Every result from E001 onward
  that was read as "the rule does not learn" is compatible with "the pathway it learns
  through cannot reach the metric".
- **New node H2e** proposed in `docs/hypothesis.md`.
- **The decisive next experiment already exists in the backlog and has never been run.**
  `docs/backlog.md` §5 calls it *causal efficacy*: take a trained flock and mute the
  channel — or here, lesion a **trained** `W_out` — and measure the drop. If a trained
  pathway is as inert as a random one, H2 is not a fact about the rule but about the
  architecture, and the fix is E007's unresolved additive-versus-multiplicative question,
  not a better learning rule.

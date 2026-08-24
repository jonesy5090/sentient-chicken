# E109 — what the rule writes: can the readout learn anything but "more of the same"?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, directly. The first *downstream* explanation attempted for
H2's null, after three upstream ones failed.

---

## 2. Question

[E108](E108-what-the-rule-reads.md) closed the third upstream explanation in eight
experiments. At the instant `consolidate` runs, the teaching signal separates feeding
windows at **AUC 0.955** and the presynaptic factor at **0.731**. Both factors are present
and informative, simultaneously. Whatever is wrong is not that a signal fails to arrive.

So look at what the rule *writes*. The readout update is

```
dw_out = eta_out · m · dz_motor ⊗ dz_slow[-n_motor:]
```

and the change it makes to the cortical drive is

```
Δcortical = dw_out @ stub = m · (dz_slow · stub) · dz_motor
```

**The direction of the change is exactly `dz_motor`.** Everything else in that expression
is a scalar. So whatever `dz_motor` points at is the only direction in motor space the
readout can ever be pushed, on any single consolidation.

And `dz_motor` traces the **motor output** — `sigmoid(reflex + cortical + b_motor + noise)`
— which is dominated by the reflex arc. `plasticity.py`'s own docstring says the noise
placement is deliberate for exactly this reason: "an exploratory action that happens to
pay off gets credited to the synapses that produced it".

**If `dz_motor` is mostly the arc's own deviation, then the rule can only ever add more of
what the arc is already doing.** At a food patch the arc already pecks; a rule that
reinforces "peck harder when at a patch" produces no behavioural change, because that is
the behaviour already. Learning would be able to amplify the innate policy and never
redirect it — which is a mechanical account of H2's null that lives entirely downstream,
in what the rule does, and is untouched by all three failed upstream explanations.

The three-factor rule's postsynaptic factor is the **only** term in it that has never been
measured.

## 3. Prediction

1. **`dz_motor` is dominated by the reflex arc.** Cosine between `dz_motor` and the
   deviation of a reflex-only counterfactual motor output above **0.8**, under the default
   instrumental rule.
2. **The drive's variance decomposes as mostly reflex.** Reflex share above 80%, cortical
   below 20%, under the default rule at 30 minutes.
3. **`hebbian_readout` is different, and that is the point of including it.** E106
   measured `|cortical|` at 1.606 against a comparable reflex drive, so under that rule
   the cortical share should be large and the cosine correspondingly lower. If the
   alignment is high under the instrumental rule and low under hebbian, the two rules fail
   for *different* reasons and the tree should stop treating them as variants of one
   thing.
4. **Exploration noise contributes little at 30 minutes.** `explore_sigma` decays with
   `explore_tau_s = 3600 s`, so at 1800 s it is at ~2/3 of hatch value — I expect noise to
   be a real but minority share, and if it is *large* then what the rule mostly credits is
   randomness, which is a different finding again.

## 4. Falsifier

**Primary.** Cosine between `dz_motor` and the reflex deviation **below 0.5** under the
instrumental rule. The rule would then have an independent output direction, this
explanation joins the other three in the discard pile, and — importantly — H2's null would
have **no** remaining mechanistic candidate anywhere in the rule, which should be recorded
as the finding rather than patched with a fifth guess.

**Instrument falsifier — checked and reported before the headline.** The reflex-only
counterfactual must actually differ from the real motor output. If `|cortical|` is
negligible the cosine is 1.0 by construction and measures nothing. Reported as mean
`|cortical|` against mean `|reflex|`; the probe is void if the ratio is below 0.01.

**Reconstruction falsifier.** The traces are recomputed offline from per-step drives, so
that the counterfactuals can be traced the same way. My offline `z_motor` must match
`ps.z_motor` from the live scan carry to within 1e-4. If it does not, my reconstruction is
not the rule's and no counterfactual built on it means anything. This is the check that
makes the rest trustworthy and it is reported first.

**Triviality falsifier.** If the noise share of `dz_motor`'s variance exceeds 90%, the
alignment measurement is measuring noise and is uninformative regardless of its value.

## 5. Design

**Read-only probe, no new flag** — as E108. The trajectory comes from
`simulate._one_step`, so the traces are the ones the rule consolidates on.

A nested scan: outer loop over consolidation windows, inner loop of `interval = 50` steps.
The inner loop emits **every step**: `motor`, `drives.reflex`, `drives.cortical`, and
`p.b_motor`. Traces are then rebuilt offline in numpy with the rule's own constants
(`tau_motor = 0.10 s`, `baseline_tau_s = 20.0 s`), which is what lets the counterfactuals
be traced identically:

- `motor_full = sigmoid(reflex + cortical + b_motor)` — reconstruction check against
  `ps.z_motor`
- `motor_reflex = sigmoid(reflex + b_motor)` — the arc alone
- `motor_cortical = sigmoid(cortical + b_motor)` — the learned pathway alone

each traced at `tau_motor`, centred at `baseline_tau_s`, and sampled at the boundaries.

**Measured**: cosine between `dz_motor` and each counterfactual's deviation; the variance
decomposition of the motor drive into reflex, cortical and their covariance; the noise
share, from the difference between the emitted `motor` and the deterministic
`sigmoid(reflex + cortical + b_motor)`; and all of the above restricted to windows in
which she fed, since those are the windows where `m` is large and the update actually
lands.

**Arms**, 4 seeds, 16 hens: untrained; reared 30 min under the **default instrumental**
rule; reared 30 min under **`hebbian_readout`**.

### Cost

~10 minutes.

---

## 6. Result

### 6a. The instrument, checked first

**The reconstruction falsifier fired on the first run** — `max |offline − live| = 9.1e-01`
against a bar of 1e-4 — and no cosine was read until it was fixed. Two causes, both mine:

- A reared arm's live traces have been running for 30 minutes, and I started the offline
  reconstruction at zero. Counterfactuals now start from the live `PlasticState`.
- `update_traces` builds its whole return value from the **old** state, so `z_motor_bar`
  advances toward the *previous* `z_motor`, not the one computed in the same call. I had
  used the new one.

After both: **`max |offline − live| = 1.79e-07`** in every arm and both seed blocks. The
reconstruction is the rule's, so the counterfactuals built on it mean something.

`|cortical| / |reflex|` is 0.11 (instrumental) and 0.85 (hebbian), both far above the 0.01
below which the probe would be void.

### 6b. The headline — the primary falsifier does not fire

Cosine between `dz_motor` — the entire direction of the readout's update — and the
deviation of a reflex-only counterfactual. 4 seeds per block, 600 consolidation windows.

| arm | seeds 0–3 | **seeds 4–7** | feeding windows only (0–3 / 4–7) |
|---|---|---|---|
| untrained | 0.9814 | 0.9799 | 0.9916 / 0.9910 |
| **instrumental (default)** | **0.9822** | **0.9826** | **0.9916 / 0.9896** |
| `hebbian_readout` | 0.8507 | 0.8809 | 0.9388 / 0.9388 |

Cosine to the *cortical*-only counterfactual, by contrast: **0.0021, −0.0772, −0.0025,
−0.0971** — zero, in every instrumental and untrained arm.

| arm | reflex share of drive variance | cortical | noise share of motor |
|---|---|---|---|
| untrained | 100.8% | 0.4% | 4.9% |
| instrumental | 100.3% / 100.5% | 0.7% / 0.6% | 2.0% |
| hebbian | 83.6% / 90.2% | 14.8% / 7.6% | 3.1% / 3.6% |

**Prediction 1 holds, strongly.** 0.98 against a threshold of 0.8, replicating to within
0.0004 across disjoint seed blocks.

**Prediction 2 holds.** Reflex share above 80%, cortical below 20%. The share exceeding
100% is not an error: reflex and cortical are *negatively* correlated, so `var(reflex)`
slightly exceeds `var(reflex + cortical)` — the learned pathway is partially cancelling
the arc's variation rather than adding to it.

**Prediction 3 is half wrong, and the half that fails matters.** I predicted `hebbian`
would show *low* alignment, and said that if so the two rules "fail for different reasons
and the tree should stop treating them as variants of one thing". Alignment under hebbian
is **0.85**, and **0.94 in the windows where the update actually lands**. The two rules
differ in degree, not in kind. They fail the same way.

**Prediction 4 holds.** Noise is 2–5% of the motor output's variance — real, and a small
minority. The triviality falsifier does not fire.

## 7. Interpretation

**The only direction the readout can be pushed is, to within 2%, the reflex arc's own.**

`Δcortical = m · (dz_slow · stub) · dz_motor`. Every term but `dz_motor` is a scalar, so
`dz_motor` *is* the update's direction in motor space — and it points at what the arc is
already doing, at cosine 0.98, rising to **0.99 in exactly the windows where the reward
fires and the update lands**.

So the rule writes "more of what she was already doing at a food patch". At a patch the
arc already pecks. Reinforcing "peck harder here" changes the *magnitude* of a tendency
that is already correct, and can never substitute a different action for it. The learned
pathway can rescale the innate policy. It cannot redirect it.

**This is the first measured mechanism for a claim the tree has held since E007.** H2b
says the learned pathway "cannot *initiate*, only modulate", and it has rested on
behavioural nulls and an arithmetic argument about reflex weights. It now has a direct
measurement of *why*, in the rule's own terms, that replicates across seed blocks.

**It also explains why the two things that ever changed behaviour were both
multiplicative.** E101's descending gate and E102's basal-ganglia gate both altered
behaviour — degenerately, but really — and both act by *multiplying* the arc rather than
adding to it. A multiplicative gate is the one intervention in this project that is not
constrained to `dz_motor`'s direction, which is exactly why it could do something the
additive readout could not. That was noticed at the time as a design intuition; it is now
a consequence of a measurement.

**And it explains `hebbian_readout`'s characteristic failure.** E055 recorded "every
calling channel elevated regardless of condition — a broken readout, not a targeted
policy". At cortical share 15% and cosine 0.85, that rule is still mostly amplifying the
arc, just with enough of its own contribution to blow the magnitude out.

**What this does not establish.** That amplifying the arc can never change behaviour.
Rescaling a tendency does change what a hen does — E101 proved that much. The claim is
narrower and specific: the *direction* of change is fixed to the arc, so no amount of
learning on this rule can make her do something the arc was not already doing more of.
For a hypothesis tree whose upper branches need "crouch **when you hear an alarm**" and
"call **when an audience is present**" — new pairings, not louder old ones — that is a
sufficient obstacle.

**Where this leaves the count.** Four explanations have now been offered for H2's null.
Three were upstream and all three are dead. This one is downstream, it survived its
falsifier, and it replicates. It is the first candidate in this arc that is still
standing at the end of the experiment that tested it.

## 8. Consequence

**No code changes.** Read-only probe, no flag, no mechanism, no default.

**`docs/hypothesis.md`.** H2b gains its first measured mechanism, cited to this
experiment, replacing an inference from reflex-weight arithmetic. H2's node records that
the standing explanation for its null is now downstream and specific: the update's
direction is the arc's.

**What follows from it, and what does not.** The obvious next move is a rule whose
postsynaptic factor is *not* the motor output — credit the cortical contribution alone,
or the exploration noise alone, so the update has a direction of its own. That is a real
experiment with a clear prediction. But it is also the fifth mechanism proposed against
this null, and the previous four were each proposed with the same confidence, so it should
be pre-registered with a falsifier that can end the line rather than extend it.

**Not adopted.** Any claim that this *is* the cause of H2's null. It is a sufficient
obstacle and a measured one; whether removing it produces learning is untested, and this
project has four consecutive demonstrations of what happens when that step is skipped.

### Follow-ups

1. **E110 — a postsynaptic factor with its own direction.** Trace the cortical drive, or
   the exploration noise, instead of the motor output. The direct consequence.
2. **The trained-flock mute** (backlog §5, open since E032) remains untouched and is now
   by a wide margin the oldest open item.
3. **E101/E102's permuted-gate control**, outstanding from
   [E107](E107-red-team-review-2026-08-24.md). E109 sharpens why it matters: if a
   multiplicative gate is the only thing in this project that can redirect behaviour, the
   claim that E101's gate learned a *selective policy* rather than a blunt one is load
   bearing.

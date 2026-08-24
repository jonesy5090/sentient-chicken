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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*

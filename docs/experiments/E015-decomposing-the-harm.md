# E015 — decomposing the harm between the two learned pathways

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
`REFUTED at this timescale`. This completes the ablation E014 left outstanding.

## 2. Question

E014 showed freezing the motor readout (`W_out`) recovers most of the harm, and
inferred that recurrent learning alone is "close to harmless". That arm never
finished. Is it actually neutral?

## 3. Result

6 seeds, 20 min. The fixed anchor was re-run in the same job and reproduced E014's
+0.036 / 5.6% exactly, so the two runs are directly comparable.

| condition | what learns | hunger change | harm vs fixed | fed % | synapses | \|W_out\| |
|---|---|---|---|---|---|---|
| fixed | nothing | +0.036 | — | 5.6 | 36,369 | 0.0399 |
| `eta_out=0` | recurrent only | +0.046 | **+0.010** | 5.6 | 28,383 | — |
| `eta=0` | readout only | +0.057 | **+0.021** | 5.0 | 36,369 | 0.0581 |
| full | both | +0.088 | **+0.052** | 4.9 | 30,109 | — |

## 4. Interpretation

**Three findings, and the first corrects E014.**

**1. Recurrent learning is not neutral — it is mildly harmful.** E014 inferred it was
"close to harmless" from an unfinished run. Measured, it costs +0.010. Small against
the readout's +0.021, but not zero, and the inference should not have been stated as
firmly as it was.

**2. The readout is roughly twice as harmful as the recurrent weights**, which is
consistent with E014's localisation and with H2d: the readout is the pathway that
carries a degenerate representation into the motor system.

**3. The two together are superadditive — and that is the new finding.** Expected
harm if the pathways were independent: +0.031. Measured: **+0.052**, two-thirds again
as much. Learning both at once is worse than the sum of learning either.

The natural reading is a **moving-target problem**. The readout learns a mapping from
pallial state to motor output while the pallial state is itself being rewritten
underneath it. Neither pathway can converge on the other, so the readout keeps
chasing a representation that keeps changing. That is a hypothesis about the
interaction, not a measured mechanism, and this project's record on such hypotheses
is poor enough that it should be labelled as one.

**4. Pruning is nearly free.** The recurrent-only condition prunes 22% of the
connectome (36,369 → 28,383) and costs +0.010. Whatever is wrong, **it is not the
loss of synapses** — which retires the last remnant of E013's original story.

## 5. Consequence

- **E014's outstanding item is closed**, and its inference corrected: recurrent
  learning is mildly harmful rather than neutral.
- **H2 unchanged**: `REFUTED at this timescale`.
- **The superadditivity is a new open question.** If it is a moving-target effect,
  then staging the learning — let the representation settle, then train the readout
  against it — should recover most of the loss. Cheap to test and directly actionable.
- **Retired for good**: the idea that connectome loss explains the behavioural harm.
  Three experiments have now pointed away from it and one measures it directly at
  +0.010 for a 22% loss.
- **H2d remains the critical path.** Both pathways are learning from, or into, a
  representation that cannot discriminate its inputs.

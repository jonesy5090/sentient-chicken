# E106 — an interneuron in the pallium and the motor stub

*Sections 1–5 written and committed before the 2×… arms were run. §2's diagnostic
(E106a) was run first, on purpose, and is labelled a diagnostic rather than
pre-registered.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2d**, and through it **H2b**, **H2c**, **H2f**, **H3**.
The first experiment to act on [E105](E105-decorrelating-readout.md)'s diagnosis.

---

## 2. Question

E105 established that the learned pathway's failure is not in the readout, the learning
rule, the reward or the gate, but in what the readout *reads*: the motor stub is **99.98%
a constant vector**, at hatch, and a reared `W_out` tracks its input's direction stability
to within 0.01 at every point of a gain sweep. Six mechanisms failed because all six were
aimed downstream of the problem.

The chain has four stages and only the first has ever been attacked:

| stage | direction stability | addressed by |
|---|---|---|
| observation | 0.6375 | — (the world genuinely varies) |
| sensory stub | 0.9707 | E103 diagnosed, E104 built the fix |
| pallium | 0.9934 | **nothing** |
| motor stub | 0.9930 | **nothing** |

### The diagnostic that had to come first

Before building a seventh mechanism: **is a positive result reachable?** If the
situation-specific signal has genuinely been destroyed rather than buried, no interneuron
can recover it. `scratchpad/e106a_is_the_signal_there.py`, 4 seeds, 30 min rearing —
subtract each population's own mean across units at each step, which is exactly what a
pooled inhibitory interneuron does and the ceiling on what one could achieve:

| population | as built | **mean subtracted** |
|---|---|---|
| observation | 0.6573 | — |
| sensory stub | 0.9708 | **0.6317** |
| pallium | 0.9927 | **0.7164** |
| motor stub | 0.9925 | **0.7443** |
| cortical output (reared `W_out`) | 0.9587 | **0.7699** |

**The signal is there.** The sensory stub with its mean removed is as varied as the
observation that produced it (0.632 against 0.657) — the relay destroys nothing, it
*buries*. Two recurrent stages later three quarters of the variation still survives
underneath, and the readout the project already grew would produce a state-dependent
output from it without being retrained.

So: **does a pooled inhibitory interneuron in the pallium and the motor stub expose it in
the running model, and does the flock survive having one?**

### Why this is not `balanced_ei`

[E072](experiments/E072-balanced-ei-and-h2d.md) already built a weight-level fix —
scaling each row's inhibition so the recurrent row sum is zero — and
[E077](experiments/E077-reread-balanced-ei.md) closed it as null for H2d separability.
It has **never been measured on this metric**, and there is a specific reason to expect it
to fail here that is worth testing rather than asserting: balancing the weights removes
the common component from the **current**, and the rate nonlinearity puts it back.
E104 §6b measured exactly that at the sensory relay — current DC 75.3%, rate DC 87.6%,
because every unit's resting bias is the same (−2.000, sd 0.000) so every unit sits at the
same point on the same sigmoid. `balanced_ei` is included as an arm because it is free,
because a null there is itself informative about *where* the fix has to act, and because
leaving an already-built mechanism untested next to a new one would be indefensible.

---

## 3. Prediction

1. **The interneuron works, and lands near the diagnostic's ceiling.** Motor-stub
   direction stability falls from 0.9925 to **below 0.85**, pallium likewise; cortical
   output falls below 0.90 for the first time in the project's history.
2. **`balanced_ei` does not move it.** It acts on the current and the nonlinearity
   restores the common mode. Predicted explicitly so that a null is a confirmation of the
   mechanism's location rather than a disappointment.
3. **The running model will not reach the post-hoc ceiling.** The diagnostic subtracted
   the mean from a trajectory the *unmodified* brain produced. Putting the interneuron in
   the loop changes the dynamics that generate the trajectory, and the recurrent gain
   (0.95) was tuned without it. I expect the measured stability to land **above** 0.7443,
   and would treat matching it exactly as suspicious.
4. **Behaviour is the real risk, and I am not confident here.** Removing common-mode
   recurrent drive from a network deliberately tuned near a saddle-node may simply
   quieten it. The ethogram and the flock's welfare are the things to watch, not the
   representation metric.

## 4. Falsifier

**Primary.** Motor-stub direction stability stays **at or above 0.90** at every strength.
The arithmetic in §2 says the signal is there, so a mechanism that subtracts the mean and
does not expose it would mean the recurrent dynamics regenerate the common mode as fast as
it is removed — which closes option A and leaves only a different rate code or acceptance
in the negative.

**Ceiling falsifier.** The representation moves but the **cortical output does not** —
stability stays ≥ 0.90 while the motor stub falls below 0.85. That is E104's failure
repeating one stage later and would say something downstream re-imposes the common mode.

**Degeneracy falsifier — the one I expect to be closest.** Any of: mean pallial rate falls
below 0.15 or rises above 0.95 (the network silenced or saturated rather than balanced);
mean |cortical| falls more than 50% against baseline; **any ethogram assay changes state**;
or the flock's welfare gets worse than baseline on hunger *and* predation together.
Removing the common mode by breaking the brain is not the result.

**Regression falsifier.** Inertness not bit-identical with the flag off, or the suite
fails.

## 5. Design

**`recurrent_lateral: float = 0.0`** in `CoopConfig`. A pooled inhibitory interneuron for
the pallium and one for the motor stub. Each receives from every unit in its own
population and projects to everything that population projects to, subtracting
`strength × mean(rate)` of its pool.

Implemented as a subtraction on the rate vector *as seen by projections* — which is
exactly a pooling interneuron and not a change to the rate code, since
`W r − λ·mean(r)·1 = (W − (λ/n)·1·1ᵀ) r` is a weight matrix with an inhibitory
broad-projecting unit in it. The neuron's own rate is untouched; what changes is what its
targets receive. Applied to the recurrent projection, to `W_out`, to `W_pred` and to
`W_str`, because an interneuron sits between a population and *everything* downstream, not
between it and a chosen one.

Pools are **PALLIUM** and **MOTOR** only. The sensory stub already has E104's
`sensory_lateral`, which acts one stage earlier on the afferent current; arm E tests the
two together.

**Arms**, matched seeds, 4 seeds, 30 min rearing, `hebbian_readout`,
`readout_scaling_strength=0.3`:

| arm | |
|---|---|
| A | baseline |
| B | `balanced_ei=True` |
| C | `recurrent_lateral=0.5` |
| D | `recurrent_lateral=1.0` |
| E | `recurrent_lateral=1.0` + `sensory_lateral=1.0` |

Measured: pallium and motor-stub direction stability and mean rate; cortical direction
stability and mean |cortical|; the full neonatal ethogram and the welfare metrics at the
best arm.

### Cost

~25 minutes for the arms, plus the assays.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*

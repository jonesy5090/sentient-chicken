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

### 6a. The arms — the primary falsifier does not fire

4 seeds, 30 min rearing, `hebbian_readout`, `readout_scaling_strength=0.3`.
`scratchpad/e106_recurrent_inhibition.py`.

| arm | pallium | motor stub | **cortical** | \|cort\| | pallial rate |
|---|---|---|---|---|---|
| A baseline | 0.9927 | 0.9925 | **0.9587** | 1.606 | 0.627 |
| B `balanced_ei` | 0.9889 | 0.9951 | **0.9912** | 0.099 | 0.132 |
| C interneuron 0.5 | 0.9661 | 0.9744 | **0.9748** | 0.201 | 0.334 |
| D interneuron 1.0 | 0.7105 | 0.7400 | **0.8428** | 0.020 | 0.242 |
| E + `sensory_lateral` | 0.6797 | 0.6733 | **0.5735** | 0.007 | 0.169 |

**The mechanism works.** Motor-stub direction stability falls 0.9925 → 0.7400 at full
strength and 0.6733 with the sensory relay included. And for the first time in this
project, **cortical direction stability moves**: 0.9587 → 0.8428 → **0.5735**. Six
previous mechanisms could not shift it off 0.95.

Prediction 1 holds. **Prediction 2 holds**: `balanced_ei` does not move direction
stability at all (0.9889 / 0.9951) — it removes the common component from the *current*
and the nonlinearity restores it, exactly as E104 §6b measured at the sensory relay. It
also quietens the network to a pallial rate of 0.132.

**Prediction 3 is wrong, and interestingly so.** I predicted the running model would land
*above* the post-hoc ceiling of 0.7443, because the diagnostic de-meaned a trajectory the
unmodified brain generated. It landed at 0.7400 and, with the sensory relay, well below at
0.6733. Putting the interneuron in the loop does not merely unmask the variation the old
dynamics produced — it changes the dynamics, and they produce more.

**The degeneracy falsifier fires, on magnitude.** Mean |cortical| collapses from 1.606 to
**0.020** at D and 0.007 at E — a 99% drop against a 50% bar. The pallial rate clears its
own bound (0.242, inside [0.15, 0.95]); `balanced_ei` does not (0.132).

### 6b. Replication on a fresh seed block

E021's rule. Seeds 4–7, against the 0–3 above. `scratchpad/e106c_replicate_and_control.py`.

| arm | pallium | motor stub | cortical | \|cort\| |
|---|---|---|---|---|
| baseline | 0.9926 | 0.9929 | 0.9745 | 1.718 |
| interneuron 1.0 | **0.6981** | **0.7450** | **0.7519** | 0.017 |

Both blocks clear the primary falsifier and both fire the degeneracy one. The motor stub
lands at 0.7400 and 0.7450 on disjoint seeds; the pallium at 0.7105 and 0.6981. This is
the most precisely replicating result in the project.

### 6c. Behaviour — the falsifier's other clauses do not fire

`scratchpad/e106b_behaviour.py`, then `e106c` on fresh seeds.

**Ethogram: no assay changes state.** 12 of 13 pass with the same registered xfail, at
baseline and at `recurrent_lateral=1.0` alike.

| | hunger | caught/dive |
|---|---|---|
| seeds 0–7, interneuron vs baseline | −0.1009, **t=−3.45** | −0.0092, t=−0.32 |
| seeds 8–15, interneuron vs baseline | −0.0613, t=−1.35 | −0.0599, t=−2.11 |

**The hunger result does not replicate**, and it is worth saying plainly that the first
block looked like the project's second behavioural win. It is noise at 8 seeds — the E021
pattern exactly, and the reason for the rule.

**And the control disposes of it regardless.** Under the interneuron the learned pathway
is 99% quieter, so a hen who simply *does less* would be the mundane explanation — the
same mechanism that made E101's gate a degenerate win. A hen reared with **no cortical
pathway at all** (`readout_scale=0`, `eta_out=0`) is the limit of that:

| contrast (hunger, lower is better) | | |
|---|---|---|
| interneuron vs baseline | −0.0613 | t=−1.35 |
| **silence control** vs baseline | −0.0656 | t=−1.56 |
| **interneuron vs silence control** | **+0.0044** | **t=+0.64** |

Indistinguishable. Whatever welfare difference exists is "the cortical pathway went
quiet", not "the representation got better". **No behavioural claim is made.**

One unplanned observation, reported because it is large. `vigour` — vocal energy, 1.0
rested, 0.0 calling flat out — reads **0.0000 at baseline and 0.9255 under the
interneuron**. The baseline flock calls itself to exhaustion. That is E055's "every
calling channel elevated regardless of condition" seen from the world side, and it is a
gain in face validity: real hens do not call continuously.

## 7. Interpretation

**The representation defect is fixed, and it took acting at the stage the diagnosis
named.** E103 found the cause at the sensory relay, E104 built the fix there, and E105
showed it did not survive the two recurrent stages downstream. Putting the same mechanism
in those two stages moves every number: pallium 0.993 → 0.70, motor stub 0.993 → 0.74,
and the readout's own output 0.96 → 0.75–0.84, replicating to within 0.005 on disjoint
seeds.

**`balanced_ei` failing here is a positive result about location.** It has been in the
tree since E072 and was closed as null for separability. Now there is a reason: it acts on
the current, and the common mode this model suffers from is in the *rate*, put there by
the nonlinearity on top of a uniform resting bias. **Balancing weights cannot fix a defect
that the sigmoid re-creates every step.** That is a general statement about this
architecture, not about one flag.

**What replaces the old problem.** The learned pathway can now say something and has
almost no voice: |cortical| 1.606 → 0.020. This is not a surprise in hindsight and the
arithmetic is simple — the common mode *was* the magnitude. A near-constant vector of
length 1.6 became a varying vector of length 0.02, and the readout's growth machinery was
calibrated against the former. `readout_scale` starts at 0.05 by design so the pallium has
to earn influence; with a hundred-fold smaller presynaptic signal, the rate at which it
can earn it falls with it.

**So E106 does not deliver behaviour and does not claim to.** It converts a pathway that
was loud and could not vary into one that varies and is inaudible. Whether the second is
better than the first is exactly the question the next experiment has to ask, and the
honest position today is that the model is *not* better off — no assay moved, no welfare
contrast survived replication, and the one that looked promising is fully explained by a
silence control.

**What would be self-deception here.** Calling this a success because the metric the last
six experiments chased finally moved. The metric moved; the behaviour did not. E100's
direction stability was always a proxy, adopted because it explained the nulls, and a
proxy that improves while the thing it proxies for does not is precisely the situation
this project has talked itself into before.

## 8. Consequence

**Adopted, off by default.** `recurrent_lateral=0.0`. Turning it on by default would
change the basis of every existing result while delivering no measured behavioural
benefit, which is the silent-comparison-basis change this project's conventions exist to
prevent.

**`docs/hypothesis.md`.** H2d's blocker is re-stated: the defect is in the rate, not the
current, and it is now demonstrably removable. E105's "architectural, possibly
unfixable" reading is narrowed — the representation was recoverable; what is unresolved
is the pathway's magnitude.

**The immediate follow-up, and the one thing that should happen next.** The readout now
reads a signal two orders of magnitude smaller than the one its learning rate was
calibrated against. `eta_out`, `readout_scale` and `readout_scaling_strength` were all
set against the old regime. Re-calibrating them under `recurrent_lateral` is a
single-variable experiment with a clear prediction and a clear falsifier, and it is the
first time in this project that the learned pathway has had anything worth amplifying.

**Not adopted.** `balanced_ei` remains closed, now with a mechanism for why rather than
just a null.

### Follow-ups

1. **E107: re-calibrate the readout under the interneuron.** The direct consequence
   above.
2. **The trained-flock mute** (backlog §5) is still untouched and still unaffected.
3. **`sensory_lateral` and `recurrent_lateral` together (arm E) are the strongest
   representation result** (cortical 0.5735) and the weakest signal (0.007). If E107
   works, arm E is where to re-run it.

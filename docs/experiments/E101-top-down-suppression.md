# E101 — can the higher brain overrule a reflex?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2**, **H2b**, **H2e**, and the architecture beneath all of them.
Successor to [E100](E100-does-the-readout-distil-the-reflex-arc.md).

---

## 2. Question

E100 found the learned pathway converges on a **fixed output direction** (0.62 untrained →
0.96 reared) and named that as a sufficient mechanism for essentially every null in the
tree: a fixed direction can rescale existing tendencies but cannot express "do X in
situation A, Y in situation B".

A design review then asked a sharper question — *can the higher brain overrule a reflex at
all?* Measured, on reared hens with a hawk directly overhead:

| | value |
|---|---|
| reflex drive to `M_CROUCH` | **+8.000** |
| cortical drive to `M_CROUCH` | **+1.286** (positive — it *adds* to crouching) |
| most negative cortical at crouch, 16 hens | **+0.367** — never negative |
| cortical range, all channels | **[−0.173, +2.150]** |
| fraction of cortical entries negative | **1.6%** |

**The answer is no, twice over.** The pathways meet by addition (`brain.py:87`,
`drive = reflex + cortical + b_motor`) with no gate, and in practice the learned pathway is
98.4% excitatory with a maximum opposition ~46× too small to counter an 8.0 reflex. Worse,
the one genuine top-down route is *forbidden* from suppressing by construction:
`brain.py:85` reads `clip(obs + pred_gain * relu(predicted))`, with the comment stating
*"relu, so association can only add percepts, never suppress real ones."*

**She can learn to imagine a hawk. She cannot learn to ignore one.**

This is not how vertebrate top-down control works. Development is substantially the forebrain
acquiring the ability to *suppress* brainstem reflexes; the avian substrate for that is the
arcopallium's descending projection and the basal ganglia's tonic-inhibition-then-release
architecture. The model has an arcopallium that `W_out` does not read from, and no basal
ganglia at all. Its reflex arc is a single matrix multiply with **no interposed stage for
anything to inhibit**.

**Does giving the higher brain a way to suppress produce targeted learning — and does
E100's fixed-direction collapse relax when it does?**

---

## 3. Prediction

Two mechanisms, both off by default.

**A — signed top-down perception.** Drop the relu so `W_pred` may subtract from the
observation the reflex arc reads. This is sensory gating: she learns to stop *perceiving*
what she should ignore.

**B — a learned multiplicative gate on the reflex output.** `drive = reflex * gate +
cortical`, with `gate` read from the motor stub through its own weights and squashed to
[0, 1]. This is descending gain control: the arc still fires, the forebrain decides how much
of it reaches the muscles.

1. **Either mechanism lets a hen suppress a reflex at all** — the measurement above becomes
   possible rather than 46× short. Necessary, not sufficient, and checked first.
2. **Direction stability falls.** E100's headline was 0.9587 reared under `hebbian_readout`.
   If the fixed-direction collapse is a *symptom* of having only an additive, near-positive
   channel, then adding a suppressive one should relax it. I predict **below 0.85** for at
   least one mechanism. This is the prediction I care about most.
3. **B beats A on behaviour, A beats B on safety.** A lets her mask real percepts, so she
   can learn to not-see a hawk; B leaves perception intact and gates the response.
4. **At least one mechanism costs something visible.** A hen who can suppress her own
   anti-predator reflex is a hen who can get killed by suppressing it. If neither arm shows
   any cost, suppression probably is not being used.

## 4. Falsifier

**Capability falsifier (gate, checked before anything else).** Neither mechanism produces a
cortical or gated drive capable of opposing the crouch reflex — i.e. the staged
hawk-overhead measurement still shows the reflex arriving at the muscles essentially
undiminished. Then the mechanism is not built correctly and no behavioural result from it
means anything.

**Primary.** Direction stability stays **at or above 0.90** in both mechanisms. E100's
fixed-direction finding would then be independent of additivity, and "the pathway can only
shout louder" is not the explanation for it.

**Safety falsifier — the one to watch.** Predation rate rises by more than 25% relative to
the fixed control in either arm. Suppression that gets hens eaten is a mechanism working
*too* well and an argument the model needs the basal ganglia's *selective* release rather
than a free gate.

**Inertness falsifier.** Any result changes with both mechanisms off. Asserted
bit-identical.

---

## 5. Design

### The two changes, both defaulting to off

**A — `pred_signed: bool = False`** in `PlasticConfig`. When set, `brain.py:85` becomes
`clip(obs + pred_gain * predicted, 0, 1)` — no relu, so a negative prediction masks a real
percept. The clip already prevents perceiving more vividly than reality; this lets her
perceive *less*.

**B — `reflex_gate: bool = False`** in `PlasticConfig`, with a new `W_gate` in
`BrainParams`. When set, `drive = reflex * sigmoid(W_gate @ motor_stub) + cortical`.
Initialised so `sigmoid(·) ≈ 1` at hatch — a newly hatched hen gates nothing, and the arc
reaches her muscles intact, which is the correct developmental starting point and keeps the
inertness falsifier meaningful. `W_gate` learns on the same rule and schedule as `W_out`.

The arcopallium is deliberately **not** re-wired here. Making it the descending route is
option C in the design review and belongs in its own experiment; changing two things at once
would confound the regression gate, which is E089's lesson.

### The measurements

1. **Capability** — the staged hawk-overhead probe above, per arm. Can the reflex be
   opposed?
2. **Direction stability** — E100's measure, per arm, untrained and reared.
3. **Behaviour** — the audience assay muted contrast (E098's repaired instrument), the
   ethogram, and free-running hunger and predation rate.

Four arms on matched seeds: **off** (control), **A**, **B**, **A+B**. 4 seeds, 30 min
rearing, `hebbian_readout` throughout since that is the rule producing the largest
behavioural change.

### Cost

~30 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*

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

### 6a. Capability — B works, A is inert

Staged hawk directly overhead, reared hens, 4 seeds:

| arm | reflex@crouch | cortical | M_CROUCH out | direction stability |
|---|---|---|---|---|
| off | 8.000 | 1.171 | 0.9985 | 0.9567 |
| **A** (signed perception) | **8.000** | 0.834 | 0.9977 | 0.9530 |
| **B** (reflex gate) | **4.756** | 1.283 | 0.8861 | 0.9209 |
| A+B | 4.877 | 1.176 | 0.8979 | **0.8946** |

**Capability gate clears via B**, which cuts the crouch reflex 41%. **A is inert** —
`reflex@crouch` stays at exactly 8.000, so signed perception is available and learning
never uses it. The `off` arm reproduces E100's 0.9587 at 0.9567.

**The primary falsifier FIRES.** I predicted direction stability below 0.85 for at least
one mechanism; the best is 0.8946, and both single mechanisms stay above 0.90.
**Additivity is not the main cause of E100's fixed-direction collapse** — that was my
hypothesis and it is largely wrong.

Note the saturation echo of E089: B cuts the *drive* 41% but the crouch *output* only
0.9985 → 0.8861, because the remaining drive is still deep in the sigmoid's flat region.

### 6b. Behaviour — a significant, replicated reduction in predation

2×2 {untrained, reared} × {no gate, gate}, 8 seeds per block, two disjoint blocks:

| cell | block 0–7 | block 8–15 |
|---|---|---|
| untrained, no gate | 0.1630 | 0.1579 |
| untrained, gate | 0.1582 | 0.1628 |
| reared, no gate | 0.1940 | 0.1757 |
| **reared, gate** | **0.1024** | **0.0855** |

| contrast (df=7, crit 2.365) | block 0–7 | block 8–15 |
|---|---|---|
| rearing effect (no gate) | +0.0310, t=+0.78 ns | +0.0179, t=+0.61 ns |
| **gate on an untrained brain** | −0.0048, t=−0.13 **ns** | +0.0049, t=+0.28 **ns** |
| **gate on a reared brain** | **−0.0917, t=−3.74 ✓** | **−0.0903, t=−3.04 ✓** |
| reared+gate vs untrained baseline | −0.0607, t=−1.83 ns | −0.0724, t=−3.36 ✓ |

**The effect replicates on disjoint seeds at near-identical size, and the null control
stays null in both blocks.** The gate does nothing until it has learned something — that
interaction is the evidence, because presence of the mechanism alone changes nothing.

**Safety falsifier clear**: predation *fell* rather than rising. **Cost is real**: hunger
0.498 → 0.596. A hen who gates her reflexes forages worse.

### 6c. What the gate actually learned — and it is degenerate

Gate value per motor channel after 30 min (1.00 = arc intact, untrained = 0.982):

| channel | gate | | channel | gate |
|---|---|---|---|---|
| TURN_R | **0.0992** | | FORWARD | 0.5314 |
| PECK | **0.1163** | | SCRATCH | 0.6384 |
| TURN_L | **0.1351** | | FLEE | 0.6716 |
| CALL_AERIAL | 0.4672 | | CALL_GAKEL | 0.6868 |
| CROUCH | 0.4756 | | CALL_FOOD | 0.7726 |
| CALL_GROUND | 0.6683 | | CALL_CONTACT | 0.9807 |

**It did not learn to suppress the crouch reflex. It learned to shut down almost the whole
arc** — turning to 0.10, pecking to 0.12 — sparing only the contact call.

The predation benefit therefore has a mundane mechanism: `actuation.py` sets
`mobility = 1 − crouch`, so a crouching hen **cannot move and stays inside the strike
radius**. A largely inert hen crouches less, drifts out, and is caught less often. She also
eats much less, which is the hunger cost.

## 7. Interpretation

**The architectural diagnosis was right and the fix works — but the rule uses it
degenerately.** The model genuinely had no way for the forebrain to overrule a reflex:
addition with no gate, a learned pathway 98.4% excitatory, peak opposition ~46× too small,
and the one top-down route relu'd so it could only add percepts. Mechanism B removes that,
and for the first time in this project **something a hen learned produces a significant,
twice-replicated behavioural improvement.**

**But it is a blanket shutdown, not selective control**, and that matters for what it means.
A free gate optimised on a broad signal finds one global setting — the same degeneracy E100
found in `W_out`'s fixed direction, reappearing in a new pathway. The hen has not learned
*when* to suppress a reflex; she has learned that suppressing most of them is, on balance,
survivable.

**And the improvement is partly an artefact of the crouch/mobility coupling.** Crouching
freezes her inside the strike radius, so it is a *badly calibrated* innate response in this
world — E026 flagged exactly this coupling as a metric hazard. The forebrain overruling it
is real learning, but what it discovered is closer to "this reflex is a liability" than to
anything about communication.

**A's inertness is separately informative.** Signed perception was available and unused —
`reflex@crouch` never moved off 8.000. So it is not that she cannot learn to mask a percept;
nothing in her experience pushes her to. That points at the teaching signal rather than the
wiring, and is a cheaper lead than any further architecture.

## 8. Consequence

**Adopted: `reflex_gate`, off by default.** It is the first mechanism in this project by
which the pallium can oppose the innate arc, and its behavioural benefit replicates.

**Not adopted: any claim that the hen learned to use information.** She learned that a
reflex was costing her. That is a real and previously impossible result, and it is not the
project's thesis.

**This is the argument for the basal ganglia (design-review option C), and it is now
empirical rather than aesthetic.** The failure mode of a *free* gate is exactly
indiscriminate suppression. The vertebrate answer to that is not a free gate — it is tonic
inhibition that is *selectively released*, which makes suppression a per-action decision
rather than a global one. E101 is the measurement that says a gate helps; it is also the
measurement that says an unselective gate is not enough.

**Open, and cheaper than C:** why does A go unused? And does the gate become selective if the
learning signal distinguishes situations — which is E100's unanswered "why does the direction
collapse" in another guise. Both should be settled before building a striatum.

# E107 — outside review, 2026-08-24: the readout never collapsed

**This is a diagnostic, not a pre-registered experiment.** An outside agent with none of
this conversation's context was pointed at the repo and asked the five standing
questions. Every finding below was then re-measured independently, with scripts written
from scratch rather than by patching the reviewer's. Where my numbers differ from theirs,
mine are given and the difference is stated.

---

## 1. Parent

All of `docs/hypothesis.md`. Two findings change published conclusions; one closes a line
of work; one proposed replacement does not survive re-measurement.

## 2. The headline: E100–E106 measured the wrong thing

**The `stability()` function used unchanged in E100, E103, E104, E105 and E106 pools all
sixteen hens before computing the mean direction.**

```python
def stability(a):                                   # a is (T, H, D)
    a = np.asarray(a).reshape(-1, a.shape[-1])      # <-- hens flattened in with time
    m = a.mean(0); m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=1) + 1e-12)).mean())
```

Every hen has her own `W_out` — `connectome.build` draws `(n_hens, MOTOR_DIM, n_motor)`
per hen — so each hen's cortical drive sits near her *own* direction. Averaging sixteen
private directions and scoring each hen against that average measures **how much hens
differ from each other**, not how much one hen's output varies with her situation. Every
write-up read it as the second.

### Re-measured, my own script (`scratchpad/e107_verify_pooling.py`), 4 seeds

| stage | POOLED | **PER-HEN** | between-hen |
|---|---|---|---|
| **untrained** | | | |
| observation | 0.6375 | 0.6580 | 0.9721 |
| sensory stub | 0.9707 | 0.9756 | 0.9953 |
| pallium | 0.9934 | 0.9997 | 0.9937 |
| motor stub | 0.9930 | 0.9998 | 0.9932 |
| **cortical** | **0.6193** | **0.9932** | **0.6308** |
| reflex | 0.8956 | 0.9026 | 0.9928 |
| **reared, hebbian** | | | |
| **cortical** | **0.9587** | **1.0000** | **0.9588** |
| motor stub | 0.9925 | 0.9998 | 0.9927 |
| **reared, instrumental** | | | |
| **cortical** | **0.9133** | **0.9993** | **0.9141** |

The pooled column reproduces E100/E103's published table to four decimal places, so this
is the same measurement and not a different one. The third column is the diagnosis: for
the cortical drive the pooled number tracks the **between-hen** figure almost exactly
(0.6193 vs 0.6308; 0.9587 vs 0.9588; 0.9133 vs 0.9141). **The pooled statistic *is* the
between-hen statistic.**

**Within a hen there is no collapse. Cortical direction stability is 0.9932 at hatch and
1.0000 after rearing — it moves by 0.007.**

What E100 actually discovered, correctly but under the wrong name, is that **rearing
under `hebbian_readout` makes sixteen hens converge on the same readout direction**
(between-hen 0.6308 → 0.9588). That is a real and slightly interesting fact about the
rule. It is not "training makes the learned pathway less state-dependent", and it says
nothing about whether one bird can express "do X in situation A, Y in B".

### E106's own result does not survive either

`scratchpad/e107_verify_e106.py`, same 4 seeds, E106's arms:

| arm | stage | POOLED | **PER-HEN** | between | \|cort\| |
|---|---|---|---|---|---|
| A baseline | pallium | 0.9927 | **0.9997** | 0.9931 | |
| | motor stub | 0.9925 | **0.9998** | 0.9927 | |
| | cortical | 0.9587 | **1.0000** | 0.9588 | 1.606 |
| D interneuron 1.0 | pallium | 0.7105 | **0.9592** | 0.7406 | |
| | motor stub | 0.7400 | **0.9651** | 0.7685 | |
| | cortical | 0.8428 | **0.9823** | 0.8587 | 0.020 |
| E + sensory | pallium | 0.6797 | **0.9030** | 0.7525 | |
| | motor stub | 0.6733 | **0.9262** | 0.7262 | |
| | cortical | 0.5735 | **0.9109** | 0.6187 | 0.007 |

**E106's primary falsifier, correctly computed, fires.** It was written as "motor-stub
direction stability stays at or above 0.90 at every strength → option A closed". Per hen
it is **0.9651** and **0.9262** — above 0.90 in every arm. E106 reported the falsifier
clearing on the pooled number.

The interneuron does move the representation — per hen the motor stub goes 0.9998 →
0.9651 → 0.9262, which is a real effect and replicated across arms. But it is roughly a
tenth of the size reported (0.9925 → 0.7400 pooled), and it does not reach the bar E106
set for itself in advance.

**And E106a's "ceiling" was the same artefact.** The diagnostic that motivated the whole
experiment — post-hoc de-meaning giving 0.7443 — used the same pooled statistic, so the
target E106 was aiming at was a between-hen quantity too.

## 3. What this costs, precisely

- **E100's mechanism is withdrawn.** "Training makes the learned pathway converge on a
  fixed output direction" is false within a hen. It was near-fixed at hatch (0.9932) and
  stayed there.
- **E101, E104, E105 and E106 each pre-registered a primary falsifier against this
  quantity.** Per hen the cortical number starts at 0.9932, so bars of "below 0.90" and
  "below 0.85" were unreachable by construction. Three of those falsifiers fired and each
  firing was read as evidence for building the next mechanism. **The pre-registration was
  honest; the quantity was not.**
- **E105's conclusion survives and is strengthened.** "The readout is a faithful map and
  its input is what is fixed" holds *more* strongly per hen: the motor stub is 0.9998, and
  E105's gain sweep — cortical stability tracking its input to within 0.01 — is an
  internal-consistency result unaffected by the pooling, since both sides used the same
  statistic.
- **E103 survives.** Pooled and per-hen agree closely for the observation, sensory stub,
  pallium and motor stub. "The internal representation is DC-dominated at hatch and
  training does not change it" is real, and per hen it is more extreme than published.
  What falls is the inference built on top — that this *caused* a collapse in the readout,
  because there was no collapse.
- **E106 is downgraded, not withdrawn.** The interneuron does what it says on the
  representation; the magnitude collapse (|cortical| 1.606 → 0.020) is real and was
  correctly reported; no behavioural claim was made and none needs retracting.

**Six mechanisms were built to explain a `reshape`.** The near-agreement of pooled and
per-hen at every *other* stage is why it survived seven experiments: the metric looked
sane everywhere except the one place a conclusion depended on it.

## 4. The reviewer's proposed replacement — **not adopted**

The review's second finding, ranked as the way forward, was that the reward's dominant
contingency is destroyed at the first synapse: `IDX_FOOD_ARRIVAL` carrying "at a feeder"
at AUC 0.992 in the observation and **0.528 at the sensory stub, 0.418 at the motor
stub** — below chance — so a covariance rule can never bind reward to feeding, and the
fix is to route that channel to a dedicated stub slice.

**The central number does not reproduce.** `scratchpad/e107_verify_routing.py`, ridge
decode trained on the first half of a trajectory, AUC on the held-out second half,
3 seeds:

| stage | AUC "at food" | AUC "hawk near" |
|---|---|---|
| observation | 0.966 | 0.935 |
| sensory stub | **0.670** | 0.908 |
| pallium | **0.784** | 0.848 |
| motor stub | **0.726** | 0.822 |
| cortical drive | 0.484 | 0.514 |
| reflex drive | 0.952 | 0.916 |

Robust across three label definitions (at a feeder now / at a feeder last step / within
1 m) and when each hen is decoded separately (motor stub 0.787 per hen). **There is a
real loss at the afferent projection — 0.966 → 0.670 — but the signal is well above
chance, and the pallium partially recovers it.** The proposed routing fix is aimed at a
defect I cannot measure.

What *does* replicate from that finding, and matters:

- **The reward is hunger-dominated.** I measure hunger **82.8%** of the teaching
  signal's variance (reviewer: 91.4%), cold 3.7%, strikes **1.9%**.
- **The vision food channels carry nothing.** AUC 0.479, and the mean `CLS_FOOD`
  activation is 0.072 at a feeder against 0.079 away — the channel does not distinguish
  standing on food from not. With `vision_range` 10 m and `peck_radius` 0.30 m in a 20 m
  coop, "food is visible" is true almost everywhere. **She finds food by one discovery
  pulse, not by seeing it.**
- **The cortical drive carries nothing about either food or hawks** (0.484, 0.514).
  Consistent with per-hen stability ~1.0: a fixed direction whose magnitude varies is one
  scalar, and that scalar is not about the world.

## 5. Confirmed, and being fixed

**`CLAUDE.md`'s first operational instruction is stale.** It says, in bold, "**Before
anything else runs: the reward is 87% `n_struck` at the H4 configuration**". E028 fixed
that; I measure the strike term at **1.9%**. This is the failure mode the same file names
— a quantity checked in the place it was moved *from*.

**E101 and E102's "untrained control" is inert by construction.** `W_gate` and `W_str`
are initialised to zeros and written only under `pc.enabled`; the gate is
`sigmoid(W @ stub + GATE_OPEN_BIAS)` with `GATE_OPEN_BIAS = 4.0`, so an untrained brain's
gate sits at **sigmoid(4.0) = 0.982**. The "untrained, gate" arm differs from "untrained,
no gate" by a **1.8% attenuation** of the reflex arc. Its null is guaranteed, so "the
interaction is the evidence" carries no information beyond the main effect. Both
experiments make that claim. The alternative that needed excluding — that *any* fixed
suppression of those channels helps, since `mobility = 1 - crouch` — remains open.

**The audience assay's arms differ in far more than audience.** `ABSENT = 1e4`, but
`actuation.py` clips positions into the coop, so parked flockmates are fenced back in.
Measured after the 300-step assay: **13.32 m minimum**, against `hear_range = 15.0 m`.
Fourteen observation channels differ by more than 0.05 between the arms, including an
audio channel going 0.0033 → **1.0000 (saturated)** and vision channels across five
classes at ~0.83. E096 accounted for the alarm-relay half; the visual and isolation
halves have never been ablated.

**Thirst is close to a dead subsystem, though not as dead as reported.** The reviewer
said no hen ever drinks and thirst contributes exactly 0.00%. Both overstate it: I count
356 and 127 drinking steps per 30 minutes at 16 hens, against ~69,800 feeding steps — a
ratio of roughly 1:200 — and I measure thirst at **11.6%** of reward variance, not 0%.
But mean thirst sits at **0.95** at 30 minutes, essentially pinned at maximum, so the
substantive point stands: the water loop barely functions.

**Pre-registration is real.** The reviewer diffed the first committed version of E100–E106
against the final, restricted to §§1–5, and found **zero** post-hoc edits except E105's
labelled §5a which says on its face that it was written before any measurement. They went
looking for quietly-revised predictions and did not find any. Worth stating plainly
alongside everything above.

**Multiplicity control does not exist**, and the point stands on arithmetic rather than
needing replication: `_report()` tests every condition against the first on two metrics,
and the escalate-and-pool practice (run a block; if it does not clear, run another and
test the pool) inflates the real α to roughly 8–11% against a nominal 5%.

## 6. Not adopted, and why

- **The routing fix** (§4). Its motivating measurement does not reproduce.
- **"The whole line of work is moot."** The review's framing is that H2's null has a
  single upstream cause. That was E105's framing too, with a different cause, and both
  are exactly the kind of unifying reinterpretation this project has been wrong about
  before. Recorded, not acted on.
- **Finding 7** (that `CLAUDE.md`'s "0.01 vs 0.87 aerial signal" is a staged single-hen
  best case and the live figure is ~5× rather than ~87×). Plausible and specific; **I did
  not re-measure it**, so it is recorded as unverified rather than adopted.

## 7. Consequence

1. **`run/metrics.py` created.** `direction_stability` (per hen, the one to use),
   `pooled_direction_stability` (named so nobody reaches for it by accident),
   `between_hen_alignment` (the diagnostic that separates them) and `dc_share`. Three
   guard tests, on a synthetic case built so the two statistics must disagree — because
   on real trajectories they agree everywhere except the one stage that mattered, which
   is precisely why this lasted seven experiments.
2. **`CLAUDE.md`'s 87% instruction corrected to 1.9%.**
3. **`docs/hypothesis.md`**: E100's mechanism struck; E101, E104, E105 and E106's rows
   annotated with the corrected falsifier calibration; E101/E102's untrained-control
   defect recorded on both.
4. **No seventh mechanism.** The next experiment on this line, if there is one, has to
   begin by stating which statistic it is testing and why that statistic answers the
   question.

### Follow-ups

- **The audience assay needs its three ablation arms** (flockmates deleted rather than
  parked; audience gagged; audience invisible) before any future result from it counts.
- **E101/E102's confound needs the permuted-gate control**: take a reared `W_gate`,
  permute its rows, re-run the 2×2. If predation still falls, the benefit is "less
  crouching", not "learned selective release".
- **Declare the contrast count in §5 and divide α by it**; require both blocks to clear
  independently rather than pooling after seeing the first.
- **The trained-flock mute** (backlog §5) is untouched by all of this and is still the
  oldest open item.

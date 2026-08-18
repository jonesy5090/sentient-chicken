# E055 — a non-reward-gated readout rule, tested on H2f's own falsifier

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2f** — the learning rule is the wrong *kind*, not merely wrongly routed or blocked.
`UNDER TEST`. H2f's own falsifier, stated in `docs/hypothesis.md`: "a rule closer to
Pavlovian association (e.g. sourced from `W_pred`, already architecturally positioned
for this per H2c) succeeds where the instrumental rule failed, on the same task and the
same scaffold." This is the first attempt at that rule.

## 2. Question

E036 found learning adds no audience-conditional alarm calling on top of a wired-in
comprehension scaffold (`S+L − S = −0.005 ± 0.002, t=2.25`), using the existing
reward-modulated three-factor rule on `W_out` — the only pathway that can move a motor
decision like "call more when someone is listening" (the fixed reflex arc has no
audience-gating on call production by design, and `W_pred`, the existing non-reward-gated
pathway, only ever augments *perception*, feeding into the reflex arc — it has no route
to a motor policy decision at all). **Does replacing the reward-modulation factor in the
readout update with a constant — turning it into unsupervised Hebbian/covariance
learning — produce the audience effect the reward-gated version did not, on the identical
task and scaffold?**

## 3. Prediction

**No confident directional prediction — registered as genuinely uncertain**, per this
session's established practice for H2c/H2f. There is a real mechanistic reason it might
work: reward requires the credit-assignment chain E036 §"why this belongs in the tree"
names (she calls, a flockmate hears and responds, the benefit returns through the kin
term) to close all the way through — several steps, several chances to fail. A
correlation-only rule needs only that calling and audience-presence *co-occur* in the
traces, a much shorter chain. There is an equally real reason it might not: H2d's
representational bottleneck (the pallium barely separates the relevant distinctions
regardless of which rule reads them) was never claimed to be fixed by this, and a
correlation rule with no reward check can just as easily learn something else entirely
that happens to correlate.

**What would be genuinely informative either way**: this is the first time any
non-instrumental rule has been tested on a *motor policy* question in this project, as
opposed to a *perceptual* one (`W_pred`'s existing use, H2c). A positive result would be
the first positive result of this project's entire learning-rule series and would need
immediate replication before being trusted, per this project's standing rule.

## 4. Falsifier

If `S+L(hebbian) − S` is not significantly positive, a non-reward-gated rule does not
rescue this task either — narrowing H2f from "maybe reward-gating is the problem" to
something the rule-kind swap alone does not fix (most likely H2d's representational
bottleneck, or a more basic absence of any exploitable audience-correlated signal in the
current pallial representation, tested by neither the hebbian rule nor anything before
it).

## 5. Design

**The fix**: `hen/plasticity.py`'s `PlasticConfig.hebbian_readout` (new, default
`False`). When `True`, `consolidate()`'s readout update drops the reward-modulation
factor `m`, replacing it with a constant — `dw_out = eta_out * dz_motor * dz_slow`
instead of `eta_out * m * dz_motor * dz_slow`. Everything else about the pathway is
unchanged: same covariance form (both traces are already deviations from their own slow
means, so the update can still go negative for anti-correlated pairs — reward no longer
gates the sign, but correlation still does), same Dale's law enforcement, same `w_max`
clip, same eligibility traces. `W` (recurrent) and `W_pred` (already non-reward-gated)
are untouched — this targets exactly the one pathway H2f's falsifier is about.

**New unit test** (`tests/test_plasticity.py`):
`test_hebbian_readout_ignores_reward_sign` — confirms directly, at the `consolidate()`
level, that identical traces under opposite-signed `m` produce an identical `W_out`
update when the flag is on, and different updates when it is off (i.e. the flag actually
does what it claims, not just what it's named).

**Task**: E036/E040's exact instrument (`run/audience.py`'s `_run_cell`,
`scaffold_2x2`'s `S` and `S+L` cells), unmodified. Only the `S+L` cell's `PlasticConfig`
changes: `hebbian_readout=True` added to the same `learn` config E036 used
(`enabled=True, growth_enabled=False, kin_audible=True, explore_sigma=0.6`) — everything
else, including the innate comprehension scaffold, held identical.

**Conditions**: `S` (scaffold, fixed — the required control) vs. `S+L-hebbian`
(scaffold, hebbian readout learning). Not re-running `N`/`N+L` or the original `S+L`:
those are already established (E036/E040) and this experiment's question is specifically
about the readout rule, not the scaffold's own effect.

**World**: 16 hens, `food_deplete_rate=0` (E040's clean setup, since the original E036
confound audit found this doesn't change the result but the clean version is the more
rigorous baseline going forward), `hawk_period_s` at `spec.DEFAULT_COOP`'s default (900s
— matching E036 exactly; this task is about the audience-effect mechanism, not predation
rate).

**Primary metric**: `alarm_effect` (audience-conditional alarm calling), paired
`S+L-hebbian − S`, matching E036's exact primary contrast definition.
**Secondary, exploratory**: `food_effect`, comprehension manipulation check (must stay
~0.19-0.25, confirming the scaffold itself is unaffected by the readout-rule change),
`|W_out|` growth as a sanity check that the hebbian rule is actually writing anything.

**Replicates**: 8 seeds, matching E036/E040's own count for this exact task.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e055_hebbian_readout.py --seeds 8 --minutes 30
```

## 6. Result

8 seeds, 30 min rearing, 16 hens, `food_deplete_rate=0`.

```
condition                                   audience  compreh.  strikes/hen   hunger  synapses
S   (scaffold, fixed)                         +0.065    0.1922       192.95    0.390     36319
S+L-hebbian (scaffold, hebbian readout)       +0.162    0.3439       185.40    0.728     35785

PRIMARY -- audience effect, S+L-hebbian - S:
  +0.0963 +/- 0.0366  t=2.63  threshold(df=7)=2.365  -> SIGNIFICANT
```

**On the registered primary metric alone, the falsifier does not fire — and that
reading would be wrong.** Two numbers in the table above are large enough to demand a
mechanism check before anything is trusted: hunger nearly doubled (0.390 → 0.728,
against H2's own established equilibrium of ~0.30), and comprehension jumped 79%
(0.192 → 0.344) despite the readout-rule change having no reason, on the stated theory,
to touch a `W_pred`-mediated metric at all. Per §3's registered standard — "would need
immediate replication before being trusted" — this triggered a diagnostic before
replication.

**Diagnostic** (`scratchpad/e055b_diagnose_hebbian_result.py`, 3 seeds, full breakdown
plus reflex/cortical drive magnitude):

```
 seed  alarm alone  alarm aud.  food alone food aud.  |reflex| |cortical|
    0       0.4376      0.7625      0.6863    0.8415    1.5335     4.0276
    1       0.3898      0.5240      0.6245    0.7443    1.5073     3.0115
    2       0.3018      0.3425      0.6788    0.8362    1.4903     3.0867
```

**This is unbounded readout growth overwhelming the reflex arc — the exact documented
failure mode, worse than the threshold that names it.** `PlasticConfig.eta_out`'s own
docstring records that at a gain producing 1.72× reflex magnitude, "cortical drive
overwhelms the innate arc... and behaviour gets worse." Here `|cortical|` runs
2.0–2.7× `|reflex|`. Every calling channel is elevated regardless of condition —
`alarm_alone` (0.30–0.44) is itself far above any innate baseline, `food_alone`
(0.62–0.68) more so — and audience and alone conditions rise together, not
differentially. A hen whose motor output is generally saturated is not exhibiting a
targeted "call more when someone is listening" policy; she is exhibiting a broken one.

## 7. Interpretation

**The primary contrast is a real, measured, statistically significant number, and it is
not evidence for H2f's falsifier.** It is evidence that `W_out` under this specific
implementation of Hebbian learning is unstable. `W` (the recurrent weights) has a
synaptic-scaling correction pulling its row sums back toward the innate connectome
(`consolidate()`, `correction = 1.0 + scaling_strength * (innate_row_sum/row_sum - 1)`)
— `W_out` has none. Under the reward-modulated rule this was masked: a reward-prediction
error averages toward zero over time (`baseline` tracks it), so growth without a
stabiliser was rarely tested at the extreme this removes. Stripping the reward gate
removes the one check that was implicitly keeping the readout's growth in a survivable
range, and nothing else in the pathway bounds it.

**This means H2f's specific falsifier — a rule closer to Pavlovian association
succeeding where the instrumental one failed, on a clean comparison — has still not
actually been tested.** This attempt confounds "removed the reward gate" with "removed
the only thing keeping the readout stable," and the significant result traces to the
second, not the first. A fair test needs the readout bounded by *something* that isn't
itself a reward signal — most directly, the same synaptic-scaling correction `W`
already has, extended to `W_out`.

## 8. Consequence

- **H2f stays `UNDER TEST`.** Neither confirmed nor refuted by this attempt — the
  instrument broke before the hypothesis could be tested, in the specific sense
  CLAUDE.md's "test the instrument before the hypothesis" section exists to catch.
- **A concrete, previously-unknown gap identified**: `W_out` has no synaptic-scaling
  correction, unlike `W`. This was never exercised under the reward-modulated rule
  because that rule's own zero-mean-over-time property happened to keep growth bounded
  in practice. Worth its own line in `docs/backlog.md` regardless of H2f's outcome —
  any future non-reward-gated experiment on this pathway needs it fixed first.
- **`hebbian_readout` ships as implemented** (the flag does exactly what its docstring
  and unit test claim: removes reward-gating, nothing else) — the defect is the
  *absence* of a stabiliser elsewhere, not a bug in this flag. Left in the codebase,
  off by default, for the corrected follow-up.
- **Not re-attempted in this experiment.** Adding readout scaling and re-running is a
  natural next step, but a different, second experiment (with its own prediction and
  falsifier) rather than a silent patch to this one's result.


# E045 — T1: the intake/risk trade-off across a capacity sweep

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H4** — an intact channel beats a shuffled one on a task requiring private
information. `SUPPORTED as written` (E030), checked and held against the
`food_deplete_rate` confound (E039). This experiment operationalizes `docs/backlog.md`
§3's **T1** (shared vigilance) properly for the first time: T1's own registered metric
(fed %, L vs C?) has been run three times before (E026, E028, E028b) plus a
depletion-controlled re-analysis this session, always as a **raw mean comparison**, and
always null. `docs/backlog.md` §2 specifies the actual intended test — a **capacity
sweep tracing intake-vs-risk curves**, "not raw intake" — which has never been run.

## 2. Question

H4 already establishes L is much safer than C? at the standard capacity (`caught/dive`:
E030's −0.044 pooled, E039's −0.077 at a fresh block). A raw fed% comparison at that
single capacity cannot distinguish "L achieves the same intake more safely" (a real,
if different, benefit) from "L simply forages less" (a real cost) — both are consistent
with a non-significant mean difference. **Does the intake/risk trade-off — traced across
several pallium capacities, not read off one point — show L dominating C? (safer with no
worse intake, or better on both axes), and does the picture change with capacity, per
`docs/backlog.md` §2's "where does the curve bend"?**

## 3. Prediction

**Primary.** At every capacity tested, L's `caught/dive` is lower than C?'s (replicating
H4's already-established result at each new capacity, not a fresh claim) — this part is
close to a manipulation check. **The live question**: whether L's fed% is significantly
*higher* than C?'s at any capacity, consistent with T1's original prediction, or whether
it stays statistically indistinguishable (as in the three prior single-capacity checks)
across the whole sweep, in which case the honest characterisation of the trade-off is
"same intake, less risk" rather than "more intake."

**Secondary, from `docs/backlog.md` §2 directly:** channel-driven separation between L
and C? (on either axis) should not simply grow monotonically with capacity — a genuine
possibility is that it plateaus or even narrows at high capacity if a larger, untrained
pallium's baseline interference (E002's ceiling finding: too much cortical influence
makes behaviour worse) starts to matter more than any channel effect.

## 4. Falsifier

**T1's original prediction (L forages more at matched risk) is not supported** if fed%
stays statistically indistinguishable between L and C? at every capacity tested — four
null replications (three prior single-point checks plus this sweep) would make "no
detectable intake benefit, only a safety benefit" the standing characterisation, not a
finding still awaiting a big-enough sample.

**A materially different, still-informative outcome**: if the *sign* of the fed%
difference flips across capacities (L worse at low capacity, better at high, or vice
versa) — that would be `docs/backlog.md`'s "where the curve bends" and the most
interesting possible result, whether or not any single point clears significance.

## 5. Design

`run/h4.py`'s existing machinery — `Condition`, `run_condition`, `_t_critical` — reused
directly, matching `scratchpad/e030.py`/`e039_h4_depletion.py`'s pattern. Two conditions
only: `L language` and `C? yoked`, both with `scaffold=True` (matching every existing
H4-lineage run for comparability — H4's registered ladder has always included the
auditory scaffold in these two conditions, not something introduced here).

**Capacity sweep**: `pallium_scale` ∈ {0.5, 1.0, 1.5, 2.0, 4.0} — `docs/backlog.md`
§2's own suggested levels, 1.5 being H4's existing standard (`EXPANDED`) so every prior
H4 result sits on this curve as one of its points.

**World**: 16 hens, hawk every 20 s, 10 minutes (H4's standard duration),
`food_deplete_rate=0` — required this time, not optional: fed % is exactly the metric
E037 found confounded by the default, and this experiment's whole point is to measure
it properly.

**Metrics, both tracked per cell**: `fed_rate` and `caught_itt` (intent-to-treat
caught/dive, H4's registered risk metric — unmovable denominator).

**Replicates**: 8 seeds per (capacity, condition) cell — this session's established
first-pass count, **not** a claim to replace H4's own 36-seed pooled headline; this
experiment is about the *shape* of the trade-off across capacity, which a first-pass
sweep can show, not about re-establishing significance at H4's existing standard
capacity (already established, repeatedly).

**Command:**
```bash
python -m scratchpad.e045_t1_pareto --seeds 8 --minutes 10 --hawk-period 20
```

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*

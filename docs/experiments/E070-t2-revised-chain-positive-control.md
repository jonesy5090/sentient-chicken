# E070 — T2-revised: whole-chain positive control

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**T2-revised** (`docs/backlog.md`), staging step 3. Both new mechanisms are built and
validated in isolation: the innate withdrawal response to hearing a gakel call
(mechanism 1, ethogram assay), and the shared allocentric map (mechanism 2, corr 1.0000
with topography preserved). This is the check that they compose into working behaviour
**before** any L vs. C? contrast is run.

E065 called for exactly this control, deferred it, and three experiments (E065, E066,
E068) were spent producing nulls that were each explained afterwards by a newly
discovered defect. E069 then established in a single sweep what all three were
circling. The rule in `CLAUDE.md` — *"if it cannot see a hand-wired success, it cannot
see a real one"* — is being applied first this time, deliberately.

## 2. Question

With a `W_pred` association **hand-planted** rather than learned — place-pattern P →
gakel channel — and `pred_gain` nonzero, does a hen avoid place P? And does she avoid
*only* P rather than becoming globally averse?

No learning anywhere. If a planted association does not produce avoidance, nothing
learned can, and T2-revised stops here.

## 3. Prediction

**Avoidance at P**: expected. Every link is now built and individually validated, and
the arithmetic is checkable in advance — a planted prediction drives `reflex_in` on the
gakel channel, which `_add_gakel_scaffold` wires to `M_FORWARD` and `M_PECK` at
`-1.5`, against `TONIC_FORWARD = 1.4` and `REST_BIAS = -2.5`. At `pred_gain` high
enough to push the predicted channel near 1.0, forward drive should fall
substantially, as the ethogram already measured for a *real* call (0.786 → 0.465).

**Selectivity**: the genuinely uncertain part, and the reason this is worth running
rather than deriving. The prediction is generated from *pallial* activity, not from the
place channel directly, and pallial states for nearby places overlap. Whether avoidance
stays localised to P or smears across the arena is not predictable from the wiring.

**Foraging intact away from P**: expected but not assured — see the hallucination
failure mode in §4.

## 4. Falsifier

- **No avoidance at P** at any `pred_gain`: the chain does not compose, and
  T2-revised is finished as designed regardless of what learning could do.
- **Avoidance everywhere** (no selectivity): the association is not referential, which
  is the property T2 exists to test. A flock that avoids the whole arena would still
  reduce sickness, so this failure mode would *pass* an aggregate metric while being
  scientifically worthless — exactly the trap E066's witnessed/testimony split was
  built to catch, and worth catching here first.
- **Hallucination**: a hen who over-predicts gakel calls perceives danger everywhere
  and stops foraging. Watch hunger and total food intake, not just avoidance. A
  "successful" avoidance that starves the flock is not a success.

## 5. Design

**No learning.** `PlasticConfig(enabled=False)` throughout — `W_pred` is written by
hand, not trained. This isolates "does the chain compose" from "can the chain be
learned", which are separate questions and were run together to everyone's cost in
E065–E068.

**Planting.** Choose one grid cell P. Set `W_pred` so that pallial activity
characteristic of P predicts the gakel audio channel at full amplitude. Concretely:
drive the network with the observation for "hen at P", record the resulting pallial
rates, and write those rates (normalised) into `W_pred`'s row for the gakel channel,
scaled to produce a prediction near 1.0. Everything else in `W_pred` stays zero, so
the planted association is the only one present.

**Conditions**, 4 seeds each, 20 minutes, 16 hens, `gakel_scaffold=True`,
`shared_place_map=True`:

| condition | `pred_gain` | purpose |
|---|---|---|
| control | 0.0 | prediction computed but not injected — the baseline |
| low | 0.5 | |
| mid | 1.0 | `PlasticConfig`'s own default |
| high | 2.0 | headroom, and where hallucination should appear first if anywhere |

**Measurements**:
- **occupancy at P** vs. control — the primary check. Fraction of hen-steps within one
  grid spacing of P's centre.
- **occupancy at a matched control cell P'** equidistant from arena centre — the
  selectivity check. Avoidance must be specific to P, not global.
- **total food intake and mean hunger** — the hallucination check.
- **mean forward drive** — mechanistic, to confirm the effect runs through the wired
  route rather than something incidental.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e070_chain_positive_control.py
```

## 6. Result

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._

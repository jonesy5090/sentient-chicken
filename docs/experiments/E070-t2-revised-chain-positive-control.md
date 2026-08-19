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

**The control failed, and diagnosed why — before any contrast was run.** No sweep was
needed; the chain was measured directly and does not compose.

**The planted prediction is not selective.** With place P → gakel written into `W_pred`
by hand and read back through the exact path `brain.py` uses:

| location | predicted gakel |
|---|---|
| P (the planted place) | +1.0000 |
| Q (a different place entirely) | +0.9637 |

3.6% selectivity. A hen would withdraw everywhere, not at P.

**Where the place information is lost** (cosine similarity between two *different*
places; 1.0 = indistinguishable):

| stage | similarity |
|---|---|
| observation, place block | **0.0000** — perfectly orthogonal; E063 works exactly as built |
| sensory stub | 0.94 |
| pallium | **0.9995** |

Two candidate explanations ruled out by measurement rather than argument:

- **Not an artefact of settling.** Identical at 1, 10, 50 and 300 settle steps, so it
  is not the recurrent network collapsing onto an attractor.
- **Not dilution by competing channels.** Carving a dedicated place-only slice of the
  sensory stub — the same trick `modality_segregated` already uses for audio, at
  10/21/32 units — leaves stub similarity at 0.91–0.94 and pallium at 0.9997. Giving
  place channels their own private stub does not help.

**The information is present. The readout cannot see it.** Centring the pallial states
across places leaves real structure — top singular values 0.294, 0.117, 0.097, 0.066 —
so places *are* linearly separable in the pallium. But the across-place variation is
**3.7% of the DC baseline** (mean pallial rate 0.2329, across-place std 0.0085), and
`brain.py`'s prediction is `W_pred @ (src * pred_src)` on **raw, uncentred** rates. The
DC term dominates the projection, which is exactly why P and Q predict alike.

**Two further defects found in this experiment's own design, both caught by the smoke
test:**

- `PlasticConfig(enabled=False)` silently disables the mechanism under test.
  `_one_step` passes `pred_from=ps.z_lag`, but the entire plasticity block sits behind
  `if not pc.enabled: return`, so `update_traces` never runs and `z_lag` stays at its
  initial zeros. `pred_gain=0.0` and `2.0` returned bit-identical results. "No
  learning" and "traces frozen" are not the same thing, and conflating them made the
  prediction identically zero.
- The primary metric had no room to move: occupancy at P was **0.0000** — the flock
  clumps and never visits an arbitrary grid cell, so avoidance could not have been
  detected regardless. `CLAUDE.md`'s fifth instrument check, failed.

## 6b. Correction (E071)

§7 below concludes "for place information the projection is fine, the readout is not."
**That is too strong, and [E071](E071-pred-centring.md) corrects it.** Centring the
readout — the fix §8 proposes — does help materially: the prediction goes from flat
across places (ratio 1.042) to varying by place. But the centred pallial states remain
correlated at 0.94–0.96, so a linear readout keyed to one place still responds strongly
to others; after `relu`, a distractor place drives *stronger* withdrawal than the
planted one. Both problems are real: an uncentred readout **and** genuinely poor
pallial separability. The residual is H2d/E017. Read §7 with that correction in place.

## 7. Interpretation

**This is E019's defect, in the one pathway that never received E019's fix.**
`plasticity.py` is explicit about the general form: rates are sigmoids, strictly
positive, so "without centring the outer product is dominated by the product of the two
means" — measured then as a rank-one update that "could only slide a constant offset."
The fix was applied to `W` and `W_out`, whose rules subtract each trace's own slow mean
(`z_fast_bar`, `z_slow_bar`, `z_motor_bar`). **`W_pred` has no equivalent** — neither
its learning (`d_pred = eta_pred * z_err * (z_lag * pred_src)`) nor its readout in
`brain.py` centres anything. It is the same defect, in the same shape, in the last place
it was not corrected.

That reframes the failure completely. T2-revised is not blocked by a missing mechanism
— both new mechanisms work as specified and pass their isolation tests. It is blocked
by a readout that cannot resolve a 3.7% signal riding on a DC baseline.

**This is not T2's problem alone.** `docs/backlog.md` has carried H2d/E017 as "still
the critical path — H2, H2b, H2c and H3 all trace back to it" and characterised it as a
projection problem. This measurement says something more specific and more tractable:
at least for place information, the signal survives the projection intact and is lost at
an uncentred readout. Those call for different fixes, and the second is far cheaper.

**The positive control did exactly its job.** Five minutes of diagnostics established
what would otherwise have been a contrast returning a null, an explanation constructed
for it, and another experiment. That is the E065–E068 sequence precisely, and it did not
happen this time because the control ran first.

## 8. Consequence

**T2-revised is paused, not abandoned.** Mechanisms 1 and 2 are built, tested and
correct; they stay. The chain cannot be tested until the prediction readout can resolve
place.

**Next, and it is worth doing on its own merits regardless of T2**: centre `W_pred`'s
source before projection, mirroring what `W` and `W_out` already do — a running mean of
pallial rate subtracted from `src`, the direct analogue of `z_slow_bar`. This needs its
own pre-registration and a guard test, because it is a core change to `brain.py` and
touches the one pathway H2c and H3 also depend on. If it works, it plausibly unblocks
more than T2.

**Do not re-run E070 until that lands.** Re-running the same control against the same
uncentred readout would reproduce the same 0.96, and the temptation to read a marginal
improvement as progress is exactly what this experiment exists to prevent.

**Two design corrections to carry forward** when it is re-run: use
`PlasticConfig(enabled=True)` with all learning rates and `scaling_strength` set to zero
(traces live, weights frozen) rather than `enabled=False`; and place food at P and at a
matched control cell so occupancy has a baseline to fall from — which also tests T2's
"food there should be avoided too" step directly rather than separately.

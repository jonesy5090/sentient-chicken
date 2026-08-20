# E082 — T2-revised's whole-chain positive control, with a discriminative plant

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**T2-revised** (`docs/backlog.md`), staging step 3 — redone.
[E070](E070-t2-revised-chain-positive-control.md) ran this control and concluded the
chain does not compose. [E081](E081-separability-vs-decodability.md) showed that
conclusion was an artefact of the plant: E070 wrote a **matched filter** into `W_pred`
(copy P's pallial pattern, normalise), which scores **18.8%** at "am I at P" — below
chance — where a **discriminant** on the identical states scores **84.6%**.

## 2. Question

With the association planted **discriminatively** rather than as a matched filter, and
`pred_gain` nonzero, does a hen avoid place P — and *only* P?

Still no learning. This is a positive control: if a correctly-planted association does
not produce selective avoidance, nothing learned will, and T2-revised stops for a reason
that is about the chain rather than about my plant.

## 3. Prediction

**Avoidance at P: expected**, more confidently than E070 predicted it, because the
readout is now known to carry the signal at 84.6% rather than 18.8%. The remaining
uncertainty is whether decodability at that level converts into a large enough
`predicted` value to move `M_FORWARD`/`M_PECK` through `relu` and the reflex arc.

**Selectivity: the real question, and genuinely open.** 84.6% is well short of the 98.8%
the hawk-vs-call contrast supports, and 15% error on a signal that gates withdrawal could
still produce avoidance smeared across the arena. E070's falsifier for this stands
unchanged and is the one I expect to bind if anything does.

**Foraging intact away from P: expected but not assured** — the hallucination failure
mode from E070 §4 applies identically.

## 4. Falsifier

- **No avoidance at P** at any `pred_gain`: the chain does not compose, now on a fair
  test, and T2-revised is finished as designed.
- **Avoidance everywhere**: the association is not referential, which is the property T2
  exists to test. Note this failure mode would still *reduce sickness* and so would pass
  an aggregate metric while being worthless — the trap E066's witnessed/testimony split
  was built to catch.
- **Hallucination**: watch hunger and forward drive, not just occupancy. A "successful"
  avoidance that starves the flock is not a success.

## 5. Design

Carries forward **both** design corrections E070 recorded for its own re-run, since they
were the two things that made it uninterpretable beyond the plant:

1. **Traces live, weights frozen.** `PlasticConfig(enabled=True)` with every learning
   rate and `scaling_strength` at zero — *not* `enabled=False`, which short-circuits the
   plasticity block so `ps.z_lag` never updates and the prediction is identically zero.
   E070 ran with `pred_gain` 0.0 and 2.0 returning bit-identical results for that reason.
2. **Food at P and at a matched control cell P′.** E070's occupancy metric read 0.0000 —
   the flock clumps and never visits an arbitrary grid cell, so avoidance could not have
   been detected regardless. Placing food at both gives occupancy a baseline to fall
   from, and tests T2's "food there should be avoided too" step directly rather than
   separately.

**Planting.** Settle the network at P and at several other cells, take
`mean(P) − mean(elsewhere)` over pallial rates as the discriminant direction, and write
that into `W_pred`'s gakel row, scaled so the prediction at P lands near 1.0. Everything
else in `W_pred` stays zero.

**Conditions**: 4 seeds, 20 min, 16 hens, `gakel_scaffold=True`, `shared_place_map=True`,
`place_cells_enabled=True` (T2 opts in, per E076), `contamination_enabled=False` (this
control is about the planted association, not about sickness occurring).
`pred_gain` ∈ {0.0, 0.5, 1.0, 2.0}.

**Measured**: occupancy at P, occupancy at P′ (selectivity), mean hunger and forward
drive (hallucination), and mean `predicted` on the gakel channel (did the plant fire).

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e082_chain_control_redone.py
```

## 6. Result

**First run was invalid and is recorded rather than discarded.** I planted against raw
`z_lag` after a 300-step (3 s) settle, but the runtime readout uses `z_lag − z_lag_bar`
with `baseline_tau_s = 20 s`, so at 3 s the bar was still ≈0. The plant was scaled against
one signal and read back through another: `pred@gakel` came out at **0.04** instead of
~1.0, and the sweep measured nothing. **This is the identical timescale error E071
documented — and E071's own stated fix was "report the convergence ratio rather than
assume it", which I did not do.** Repeated two experiments after writing it down.

Corrected: 300 s tour-settle (15× `baseline_tau_s`), plant against the centred signal,
and a pre-flight that prints the prediction at P per seed.

**Pre-flight**: predicted gakel at P = **1.000, 1.000, 1.000, 1.000** across seeds. Live
`pred@gakel` runs 0.86–0.96, so the plant fires strongly throughout.

4 seeds, 20 min, 16 hens, no learning:

| `pred_gain` | occupancy P | occupancy P′ | selectivity | hunger | fwd drive | pred@gakel |
|---|---|---|---|---|---|---|
| 0.0 | 0.4501 | 0.1937 | −0.2564 | 0.427 | 0.622 | 0.9615 |
| 0.5 | 0.4640 | 0.1940 | −0.2700 | 0.424 | 0.584 | 0.8677 |
| 1.0 | 0.4372 | 0.2386 | −0.1987 | 0.431 | 0.575 | 0.8710 |
| 2.0 | 0.4339 | 0.2187 | −0.2152 | 0.437 | 0.519 | 0.9042 |

## 7. Interpretation

**The first falsifier fires: no avoidance at P, at any gain.** Occupancy runs
0.4501 → 0.4640 → 0.4372 → 0.4339 — a drift of about 3%, non-monotonic, against a
baseline of 45%.

**But the chain conducts, and that is new.** The plant fires at 0.86–0.96 live, and
**forward drive falls 17%** (0.622 → 0.519) as gain rises. Every link works: place →
pallium → discriminant → `W_pred` → `relu` → `reflex_in` → the gakel scaffold → motor.
E070 could not establish any of that, because its plant never fired.

**The failure is in mechanism 1's response type, and it is diagnosable.**
`_add_gakel_scaffold` suppresses `M_FORWARD` and `M_PECK`. `actuation.py` computes
`speed = mobility * (fwd * walk_speed + flee * flee_speed)` — so suppressing forward
drive makes a hen **move less**. A hen already at P who slows down **stays at P**. The
anchor produces lingering where avoidance requires leaving.

**And I argued myself into it explicitly.** `_add_gakel_scaffold`'s docstring records
"*No crouch or flee. This is bad food, not a predator... borrowing the anti-predator
response would make the two call classes behaviourally indistinguishable*". That
reasoning was right about not borrowing crouch — and I then wired a response that is
functionally a freeze anyway. Suppressing locomotion *is* the anti-predator template,
arrived at by a different route.

**The other two falsifiers did not fire**, which is worth stating: hunger is flat
(0.427 → 0.437) and occupancy at P′ is flat, so there is no hallucination and no
smearing. The +0.76 distractor leak seen in pre-flight did not produce arena-wide
avoidance. Selectivity was the risk I flagged as most likely to bind; it did not.

## 8. Consequence

**T2-revised is not blocked by representation, plumbing, or selectivity. It is blocked by
one wrong reflex.** That is a much better position than E070 left it in, and a
specifically actionable one.

**Mechanism 1 needs redesigning so the response produces *leaving*, not *stopping*.**
Candidates, in rough order of how well they fit the biology already cited:
1. **Suppress `M_PECK` only.** A hen who will not eat at P but still wanders normally
   leaves by ordinary foraging. Minimal, and directly matches "food there should be
   avoided" without touching locomotion.
2. **Suppress peck and *raise* forward drive.** Explicitly "move on" — closer to what an
   animal declining a patch actually does.
3. **A turn component.** Needs a direction the audio channel does not carry; the place
   channel is allocentric and the model exposes no heading, so this is the expensive
   option and should not be first.

Option 1 is the cheapest and the most conservative, and its ethogram assay already exists
(the peck half of `withdraw_on_hearing_a_gakel_call` measured 0.989 → 0.954 — correct
direction, near ceiling). Try it before anything more elaborate.

**Do not re-run the L vs C? contrast until a redesigned anchor passes this control.**
The chain now demonstrably conducts, so a contrast run against the current anchor would
measure a real signal driving a behaviour that cannot produce the outcome — a null with
a known cause, which is the category of experiment this project has already run four
times for T2.

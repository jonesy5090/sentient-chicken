# E096 — red-team review, 2026-08: what replicated and what did not

**This is a diagnostic, not a pre-registered experiment.** An agent with no context on this
project reviewed the repo against the five standing questions in
`.claude/skills/red-team/`. Its findings are recorded below with each one marked
**adopted**, **not adopted**, or **open**, and every adopted finding independently
re-measured in this repo with my own script. Where my number differs from the reviewer's,
both are given.

---

## 1. Parent hypothesis

Multiple. Touches **H2f** (status change), **H4**, **T1**, **H2c/H2d/H3** (via E081), and
the statistical machinery under all of them.

---

## 2. Adopted — H2f's audience effect is mostly the audience's *calls*, not their presence

**Reviewer's claim:** `run/audience.py:_staged` places 15 flockmates at 2 m for
`audience=True` and at `ABSENT` otherwise, with the staged hawk at `HAWK_DISTANCE = 7.0` —
which puts it 5–9 m from the audience, inside `vision_range`. They see it and alarm-call.
So "audience present" also drives the focal hen's aerial channel. Reviewer measured 64% of
the audience-specific effect carried by audio.

**Re-measured independently** (`scratchpad/e096_h2f_audio_ablation.py`, 3 seeds, 30 min
rearing at E057's configuration, the *same reared brain* assayed twice — audio muted at
test only, so the audience is still present, still seen, still counted by `IDX_ISOLATION`
and `CLS_FLOCKMATE`):

| condition | alarm alone | alarm audience | audience-specific DiD |
|---|---|---|---|
| audio intact | 0.4378 | 0.7356 | **+0.2723** |
| audio muted at test | 0.4144 | 0.4987 | **+0.0577** |

**21% survives muting.** My figure is *stronger* than the reviewer's — 79% audio-carried
against their 64%. The intact DiD (+0.2723) reproduces E057's published +0.232 to the
degree three seeds allow.

**Why this is not merely a staging quibble.** `hen/innate.py:259-263` deliberately declines
to wire a call relay, giving this reason: *"A relay makes the acoustic environment
self-driving… which would confound the audience assay — the quantity being measured is
precisely whether a hen calls more when others are present."* The innate wiring refused to
build the confound. **Learning is free to build it, and the assay cannot tell the two
apart.**

**And the tree already half-knew.** E074 recorded H2f's food-channel control *firing* at
+0.1054, t=10.04 — 47% the size of the effect it exists to control for — and the node
carries that as "an unresolved caveat about its own control". E096 supplies the mechanism:
the control is null in E057 partly because the food staging gives the audience no matched
auditory manipulation, while the alarm staging does.

**Consequence: H2f moves from `SUPPORTED` to `SUPPORTED, CONFOUNDED`.** Not withdrawn — a
real, twice-replicated behavioural change remains, and 21% of it survives the ablation. But
the claim it was licensed to make, *audience-conditional* calling, is not separable from
*heard-a-call-so-called* on this assay.

## 3. Adopted — H2f was scored against a rule its own falsifier did not describe

**Verified documentary, `docs/hypothesis.md:1118-1121`.** H2f's falsifier reads:

> *a rule closer to Pavlovian association (**e.g. sourced from `W_pred`**, already
> architecturally positioned for this per H2c) succeeds where the instrumental rule failed,
> on the same task and the same scaffold.*

What was implemented and scored against it is `hebbian_readout`, which sets the modulator to
a constant for `W_out` (`hen/plasticity.py:479`). Both factors in that update are same-time
traces — `tau_motor` 0.10, `tau_slow` 0.20. **`W_pred` (`tau_lag` 1.5) is the only pathway
in the codebase with a genuine cue→outcome lag, and it has never been run on this task.**

The two rules share exactly one property — not reward-gated — and the falsifier's
distinguishing content, Pavlovian *association*, is the one they do not share. The node's
own text says "absence of any attempt is not evidence either way; this node opens the work
rather than closing it."

**Consequence: E097 runs the rule the falsifier actually names.**

## 4. Adopted — `_t_critical` was more permissive than correct on every pooled test

**Verified and fixed.** `run/experiment.py` returned **1.96 for all df > 30**, with a
comment asserting "erring high is the safe direction". t(35) = 2.030, so 1.96 errs *low* —
more permissive, not less. H4's and T1's headline results are 36-seed pooled tests at
df=35.

E030 quoted 2.030 in prose and got it right, so **no published conclusion moves**; the
harness that printed the verdict did not. Table extended to df 120, fallback made
conservative, guard test added asserting the returned value is never below the true
threshold and stays monotone in df.

## 5. Adopted — README claimed 13/13 assays; the suite gives 12

**Verified and fixed.** `README.md:301` said "13 out of 13 behavioural assays pass". The
shipped `python -m run.probes` gives **12/13** — `run/probes.py` never passes E090's adopted
`gakel_peck_weight` / `hunger_peck_weight`, so the gakel assay runs at defaults and remains
the registered `EXPECTED_FAILURES` entry. I introduced that README line during E090. It now
states 12 of 13 and explains why the thirteenth is a deliberate registered failure.

The reviewer also noted E091 §6 says "Ethogram 12/13 at adopted weights", which contradicts
the tree's "13/13". The tree is right and E091's line is wrong: monkeypatching the weights
in gives 13/13. E091 is corrected.

## 6. Open — findings I have not re-measured, recorded rather than acted on

Each is plausible and none has been independently verified in this repo. **None has been
propagated to the tree.**

- **E029 ran on seeds 48–59; E028's null came from 36–47.** Reviewer recomputed block A
  t=1.42, block B t=4.75, block C t=1.10 — so H4's only positive control demonstrated
  detectability on a block where the effect exists, and licensed reading a null from one
  where it does not. Cheap to settle: `run/poscontrol.py --seed-offset 36`.
- **T1's "Pareto improvement" is asymmetric.** Pooling made risk significant (t=3.60);
  the same pooling on intake gives −6%, t=1.55, negative in two of three blocks.
- **E081's 98.8% is a parked number.** Reviewer measured 68–80% live. More seriously:
  hawk-visible-without-a-call occurs on **0.003–0.006%** of hen-steps, so the H2d/H2c
  contrast is posed on a state the environment barely produces.
- **`ABSENT` flockmates are clipped into the arena** at 13.8 m, inside `hear_range` 15.0 —
  so "no audience" is "distant audience".
- **`C− capacity` differs in two variables** (`run/h4.py:61-63`), so H0's "at any capacity"
  clause has no valid test.
- **`observability` does not match its docstring**; with `place_cells_enabled=False`, 50 of
  138 channels are identically zero and `W_pred` is trained to predict them.
- **`pred_err` is computed against raw `predicted` while behaviour reads
  `pred_gain * relu(predicted)`** — the rule regresses a different quantity than the one
  that reaches perception.

## 7. Not adopted — the reviewer's central structural claim, pending test

The reviewer argues that `hebbian_readout` trains `W_out` to reconstruct the reflex arc's
own output (since `dz_motor` traces the final post-sigmoid motor vector, which at
cortical/reflex 0.03–0.10 *is* the reflex arc), and therefore that E058/E059/E069's
"the rule amplifies innate anchors and cannot build new contingencies" is a **theorem
derivable in one line** rather than an inductive finding from five experiments.

This is the most interesting claim in the review and it is **not adopted**, because it is
argued from the source and not measured. It makes a sharp prediction — that `W_out`'s
learned output should correlate strongly with the innate arc's output — which is cheap to
test and is not tested here. Recorded so it is not lost.

## 8. Consequence

- **H2f: `SUPPORTED` → `SUPPORTED, CONFOUNDED`.** The behavioural change is real; the
  *audience-conditional* reading is not separable from a learned relay on this assay.
- **Two fixes shipped with guards** (`_t_critical`, README).
- **E097 next: `W_pred` on the audience task**, the rule H2f's falsifier names and which has
  never been run. If a genuinely Pavlovian rule produces an audience effect that *survives
  muting*, that is the result H2f was reaching for and does not yet have.
- **T2's instrument work stops here.** Four whole-chain controls, four invalidations. The
  reviewer's recommendation — that the two experiments worth more than the whole arc are
  `W_pred` on the audience task and the trained-flock mute (backlog §5, open since E032) —
  is adopted.

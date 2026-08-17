# E036 — E018 re-run: does an innate auditory reflex unblock learned usage?

> **Re-run, not a fresh design.** [E018](E018-innate-auditory-reflex.md) was aborted mid-run
> when an external review found the auditory channel it depends on carried no information
> at `n_hens=16` — the instrument was broken, not the hypothesis tested. E018 §8: *"Re-run
> after defect 1 is fixed, unchanged in design. Sections 1–5 stand as registered."* That
> defect was fixed in [E019](E019-three-verified-defects.md); the E/I bug and gain
> re-baseline ([E023](E023-ei-fix-and-rebaseline.md)) happened since too. Sections 1–5
> below are E018's, unedited, reproduced here for a self-contained record. This file
> follows this project's convention of a new number for a re-run on a repaired instrument
> (E026→E028, E028→E029/E030), keeping E018 as the historical record of the defect.

## 1. Parent hypothesis

Primary: **[H3](../hypothesis.md#h3)** — learned usage reproduces the audience effect
without being programmed. `UNDER TEST`, two nulls (E005, E006), blocked by H2b.

Secondary: **[H2b](../hypothesis.md#h2b)** — the learning rule cannot acquire behaviours
outside the innate repertoire. This tests H2b's stated mechanism, not H2b itself.

## 2. Question

*(E018 §2, unchanged.)* E006 and E007 explained their nulls the same way: the chain H3
needs — *she calls → a flockmate hears and responds → the flockmate avoids a strike → the
benefit returns through the kin term* — breaks at step two, because `hen/innate.py` wires
no response to hearing a call. Real chicks don't learn that response from scratch either;
naive chicks already respond differentially to fear calls, and the learned part is
association off an already-arousing stimulus. **If step two is closed by construction,
does the rest of the chain close on its own?**

## 3. Prediction

*(E018 §3 plus its addendum, unchanged — including the addendum's corrected number and
the innate audience-effect floor it found.)*

1. **Primary.** `A(S+L) − A(S) > 0`, significant at two-tailed p<0.05, 8 matched seeds.
2. **Manipulation check, not a result.** Comprehension ≈0 unscaffolded, ≈0.19 scaffolded
   (corrected from the original mis-derived 0.25 — E018's addendum, confirmed again in
   this file's own smoke test below at 0.1899).
3. **Secondary, exploratory.** Strikes/hen fall in S vs N by 10–30%, innate route.
4. **Secondary, exploratory.** Hunger rises in S vs N — context-blind crouching costs
   foraging time.

**Confidence, restated from E018:** low on prediction 1 specifically; it depends on a
learning rule H2e/H2d work has independently found weak reasons to trust for acquiring
anything beyond retiming existing behaviour.

**The floor moves, per E018's addendum.** `S`'s audience effect is not expected near
zero — the scaffold's peck/scratch suppression opens an innate head-up route (hear call →
stop pecking → see the hawk herself at 7 m → weight-8.0 visual reflex fires the alarm
call). E018's smoke test measured this at S=+0.066; this file's own smoke test (below)
measured +0.063 on the current codebase. **The primary must clear this floor, not zero.**

## 4. Falsifier

*(E018 §4, unchanged.)* If `A(S+L) − A(S) ≈ 0`, E006's/E007's "nothing responds to calls"
explanation is false in its consequence even though correct in its diagnosis, and the
unnamed alternative — the rule is the wrong *kind*, instrumental where the biology is
Pavlovian — is promoted from speculation to leading and needs its own node. Against H2b: a
positive result weakens it: a null strengthens it and narrows its mechanism to "can't
acquire from an unmet precondition" rather than "can't acquire at all."

**Not a falsifier:** any result in condition S alone (see the design rule below).

## 5. Design

*(E018 §5, unchanged — reproduced for completeness.)*

**The one rule the whole design exists to enforce:** anything measurable in
scaffold-without-learning (S) is wired, not learned. The primary is always `S+L − S`,
never `S+L − N`. Full 2×2 — no scaffold × no learning (N), scaffold × no learning (S,
"the control that matters"), no scaffold × learning (N+L, current known null), scaffold ×
learning (S+L, the test) — matched seeds, genome, coop, predator arrivals.
`explore_sigma` explicit and equal in all four.

**Scaffold weights** (`hen/innate.py`, `auditory_scaffold=True`): aerial alarm heard →
crouch +1.5, peck/scratch −1.5; ground alarm heard → flee +1.5, peck/scratch −1.5. Fixed
a priori, not tuned on any metric here. Explicitly **not** wired: no call relay, no
posture/audience/context-dependence, nothing on food/contact channels.

**Primary metric:** the audience effect index from `run/audience.py`'s existing Evans &
Marler assay, paired across seeds, two-tailed t against `run/experiment.py`'s
`_t_critical()` table.

**Replicates:** 8 matched seeds (E018's deliberate compromise; extend to 16 only if the
primary lands between p=0.05 and p=0.15, per E018's pre-registered response — not report
the 8-seed result as a trend).

**Command:** `python -m run.audience --minutes 30 --seeds 8 --scaffold-2x2`

### Instrument check before committing to the full run

A 2-seed, 2-minute smoke test on the current codebase (this session, before launching the
full run) confirms the manipulation check passes: comprehension bare −0.0003 (expected
~0), comprehension scaffold **0.1899** (expected ~0.19, matching E018's corrected
addendum almost exactly) — the scaffold is wired as specified on the post-E019/E023
connectome. It also reproduced E018's own unpredicted finding: S's audience effect
(innate route) measured **+0.063**, matching E018's own smoke test (+0.066) closely. The
instrument and the innate floor both replicate before the real run starts.

## 6. Result

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*

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

8 matched seeds, 16 hens, 30 min rearing, full 2×2. Wall clock 1095 s.

```
condition                   audience  compreh.  strikes/hen   hunger  synapses
--------------------------------------------------------------------------------
N   (bare, fixed)             +0.003   -0.0001       257.21    0.496     36319
S   (scaffold, fixed)         +0.066    0.1921       341.70    0.510     36319
N+L (bare, learning)          +0.001   -0.0001       545.48    0.490     35744
S+L (scaffold, learning)      +0.061    0.1895       533.80    0.498     35708

PRIMARY: audience effect, S+L - S     -0.005 +/- 0.002 SE  t=2.25  need 2.37  NOT SIGNIFICANT
  (wrong sign: predicted positive, measured negative)

MANIPULATION CHECK: comprehension bare -0.0001 (expected ~0)
                    comprehension scaffold 0.1921 (expected ~0.19)  -- clean

SECONDARY, exploratory:
  strikes/hen, S - N        +84.48 +/- 80.96 SE  t=1.04  not significant
  hunger, S - N             +0.014 +/- 0.008 SE  t=1.71  not significant
  audience effect, S+L - N+L +0.060 +/- 0.001 SE  SIGNIFICANT (confounded on purpose)
```

**Manipulation check is clean, at both durations now.** 0.1921 at 30 minutes, 0.1899 in
this file's own 2-minute smoke test, both matching the addendum's corrected prediction of
0.1893. The scaffold is wired as specified and stable across durations.

**The primary misses, in the wrong direction, and is close enough to the threshold to
name exactly.** `S+L − S = −0.005 ± 0.002, t=2.25` against a threshold of 2.37 at 7 df —
learning does not add to the scaffold's own audience effect; if anything it subtracts a
little. This is not "suggestive of a positive effect that needs more seeds" — the sign is
already wrong, and E018 §5 pre-committed that no result in condition S alone is a
falsifier, but this is `S+L − S`, exactly the registered primary, not S alone.

**The secondary "strikes/hen" metric is not trustworthy, and this was knowable before the
run.** `run/audience.py`'s `_run_cell` reads `summary.struck`, which sums `w.n_struck` —
the same **per-step exposure-contact count** E026's ablation and E027's measurement
already found confounded (up to 87.3% of reward variance, and a 15–17× spread from
positional luck rather than behaviour), which is exactly why H4's lineage moved to
event-anchored `n_strike_events` / `caught-per-dive` in E026–E028. `run/audience.py`
predates that fix and was never updated to use `n_strike_events`, which exists on `WorldState`
but is not threaded into `Summary` at all. The raw magnitudes here (257–545 "strikes" per
hen over 30 minutes, at a 900 s hawk period that permits at most ~2 dives) confirm it: this
is counting exposure-steps, not catches. **Prediction 3 (strikes fall 10–30% in S) cannot
be evaluated with this instrument** — not "null", genuinely uninterpretable, the same
distinction E018 itself insisted on for its own predecessors.

## 7. Interpretation

**The falsifier fires.** E018 §4: *"If A(S+L) − A(S) ≈ 0, E006's and E007's stated
explanation is false."* Measured: −0.005, indistinguishable from zero and wrong-signed
relative to the registered prediction. E006 and E007 were right about **where** the chain
breaks (nothing responded to calls) but wrong about the **consequence** of fixing it: with
comprehension supplied by construction — not learned, not discovered by exploration, just
present — the learned, audience-contingent extra calling that H3 predicts still does not
appear. The scaffold gives her a reason to be caught listening; it does not give her a
reason to call more when someone is listening to *her*, and thirty minutes of a working
three-factor rule does not find one either.

**This promotes E017's unnamed alternative from speculation to the leading
explanation, exactly as E018 pre-committed it would.** The rule is reward-modulated
three-factor — instrumental conditioning: act, get rewarded, strengthen. The biology this
node is trying to reproduce (Curio's mobbing-chain work, naive birds acquiring alarm
responses by observation with no reward and no action of their own) is closer to
Pavlovian / observational learning. `W_pred`, the top-down associative pathway H2c
explored and found blocked by H2d's representational bottleneck, is architecturally
closer to that mechanism than `W_out` is. **Every routing fix tried so far (E002, E007,
E008, E009) assumed the right learning rule in the wrong place; this result says the rule
itself may be the wrong kind, independent of where H2d's bottleneck eventually gets
fixed.**

**H2b is narrowed, not weakened.** Its claim — the rule cannot acquire a behaviour outside
what the reflex arc already produces — predicted this null and predicted it more
specifically than before: E018 supplied the missing precondition (a response to *hear*),
which had been H2b's *stated mechanism* for why H3 fails. With that precondition met and
the null persisting, "missing a foothold" is no longer sufficient as the whole
explanation. What's left standing is closer to: the rule can retime an existing
stimulus-response pairing (the scaffold gave her exactly one to retime) but does not use
retiming to build the *specific new contingency* — call more with an audience present —
that H3 needs, even when every ingredient for that contingency to be learnable is present
in the world.

## 8. Consequence

- **H3 gets a third null and its blocking condition changes.** Previously "blocked by
  H2b" meant *the precondition for testing H3 doesn't exist*. That precondition now
  exists (the scaffold), and H3 still fails. H3 stays `UNDER TEST`, but the open question
  is no longer "can the chain even close" — it can, mechanically — it's "does the rule
  build contingent behaviour from a closed chain," which is the same question the new
  node below asks more directly.
- **New node opened: H2f — the learning rule may be the wrong *kind*.** Promoted from
  E017's unnamed speculation to `UNDER TEST` by this falsifier firing exactly as
  pre-registered. See `docs/hypothesis.md`.
- **The innate audience-effect pathway (S vs N, entirely unlearned) replicates a third
  time**: E018's original smoke test (+0.066), this file's own smoke test (+0.063), and
  now the full 8-seed run (+0.066) all agree closely. This is a real, robust, and purely
  innate finding — real chickens do raise their heads at alarm calls, and the model now
  reproduces the *consequence* of that (an apparent audience effect with zero learning
  involved) reliably. It must continue to not be reported as evidence for H3 or for
  anything learned.
- **`run/audience.py`'s strikes metric needs fixing before it is used again**: thread
  `w.n_strike_events` into `Summary` (it already exists on `WorldState` and already
  exists in `Summary` nowhere), or drop the per-hen strikes secondary in favour of an
  event-anchored one matching `run/h4.py`'s `caught/dive`. Filed here rather than fixed
  as part of this run, since E018's own rule is that instrument repairs are a separate
  change from the experiment that discovers the need for one.
- **E018 itself is now fully resolved** — its `ABORTED, no result` status is superseded
  by this file, and its sections 1–5 (reproduced here) no longer need re-registering for
  any future work in this area.

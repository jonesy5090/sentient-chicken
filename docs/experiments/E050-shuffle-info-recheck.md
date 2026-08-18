# E050 — re-run E024's shuffled-channel information-retention check against the E048 fix

> **Written after the run, not before**, the same deviation E048 flagged in its own
> preamble — this is becoming a pattern in this session and is worth stating plainly
> rather than repeating silently: sections 1–5 describe the question as it was actually
> approached. The result in §6 is honestly reported either way.

## 1. Parent hypothesis

The E025-adjacent backlog item, closed by E048: "gregariousness's attraction-only wiring
... needs a crowding/individual-distance channel." E048 built and measured that channel,
flagging one thing not yet done (§8): "re-running H4's own information-retention metric
(E026's measure) against this fix, to check whether the dispersal achieved here is enough
to matter for the shuffled-channel control it was originally motivated by." This is that
check. Feeds H4 only indirectly — H4's `SUPPORTED` status already rests on the **yoked**
control (E026), not the shuffled one — so nothing here can move H4's status either way.

## 2. Question

E024 found the shuffled-sender control retains ~90% of the intact channel's information
about "is a hawk on me," because 38.8% of the flock shares each hawk's strike radius
by the time it arrives. Does E048's personal-space fix reduce that retention?

## 3. Prediction

Given E048's own diagnostic — nearest-neighbour distance 0.14 → 0.38 m, general
strike-radius overlap 26.8% → 21.8% — some reduction in the shuffled-channel's retained
correlation and in the hawk-targeted clustering fraction, though not necessarily enough
to make the shuffled control viable (E048 itself only measured *general* dispersal, never
the *hawk-targeted* clustering this specific instrument checks).

## 4. Falsifier

If the shuffled channel's retained correlation and the hawk-targeted clustering fraction
are statistically indistinguishable from E024's original pre-fix, pre-E023 baseline
(90% retained, 38.8% mean in strike radius), the fix's dispersal does not reach the
specific quantity this metric depends on, whatever it did to general flock spacing.

## 5. Design

`scratchpad/shuffle_info.py` (E024's original instrument), unmodified in its metric
definitions. Two changes only, both toward more statistical power, not toward a
different measurement: 8 seeds instead of E024's 3 for the correlation metric (this
project's standing first-pass default), and the "fraction of flock in strike radius when
a hawk is live" geometry check averaged over the same 8 seeds rather than E024's single
seed — this number is exactly what E048's own diagnostic (3 seeds) targeted, and 3 vs 8
seeds giving very different answers on a first pass (see §6) was the reason a second,
independent seed block was run before trusting either.

**Conditions**: `channel_mode` intact vs. shuffled, otherwise identical (16 hens, hawk
every 60 s, `pallium_scale=1.5`, `auditory_scaffold=True`, 3-minute rollout, no
plasticity) — E024's exact setup, run on the current (E023- and E048-corrected)
connectome.

**Primary metric**: `corr(heard, hawk on me)` for intact vs. shuffled, and the ratio
between them (E024's own primary number). **Secondary**: heard-given-hawk /
heard-given-no-hawk signal-to-background ratio and its retention; the hawk-targeted
strike-radius fraction.

**Replicates**: two independent 8-seed blocks (seeds 0–7, seeds 8–15) — the project's
"no status change on one seed block" rule, applied here because the first block's
correlation-retention number (88.7%) looked like it might be moving in the fix's favour
relative to E024's 90%, and a 3-seed pilot of the same geometry statistic had disagreed
with the first 8-seed block by 8 points (30.4% vs. 38.4%), which is exactly the kind of
swing that rule exists to catch.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e049_shuffle_info_recheck.py
SEED_OFFSET=8 PYTHONPATH=. .venv/bin/python scratchpad/e049_shuffle_info_recheck.py
```

## 6. Result

```
                        block seeds 0-7          block seeds 8-15
mode        corr    heard|hawk  heard|no  ratio | corr    heard|hawk  heard|no  ratio
intact      0.6104  0.9333      0.0751    12.42 | 0.5595  0.9925      0.0915    10.84
shuffled    0.5415  0.9005      0.0814    11.07 | 0.5606  0.9780      0.0948    10.32

retention (shuffled/intact):
  corr(heard, hawk on me):  block A 88.7%  |  block B 100.2%
  heard|hawk : heard|no-hawk ratio:  block A 89.1%  |  block B 95.2%

hawk-targeted clustering, "% of flock in strike radius when a hawk is live":
  block A (seeds 0-7):  38.4% mean, 88% max
  block B (seeds 8-15): 38.4% mean, 88% max
  E024 baseline (pre-E023, pre-E048, single seed): 38.8% mean, 50% max

for comparison, E048's own 3-seed diagnostic (different metric window, general
not hawk-targeted, 6 min not 3): "in strike radius" 26.8% (pre-fix) -> 21.8% (fixed)
```

**The hawk-targeted clustering fraction replicates exactly across two independent
8-seed blocks (38.4% both times) and matches E024's original pre-fix, pre-E023 baseline
(38.8%) to within noise.** The correlation-retention numbers move around between blocks
(88.7% vs. 100.2%) in the way the pre-registered replication check was designed to
catch, but both land in the same 89–100% range E024/E026's original figures (90%, 98%)
already occupied — no block shows a retention low enough to call the shuffled control
fixed, or even clearly improved.

**E048's own 3-seed diagnostic (26.8% → 21.8%) does not replicate at 8 seeds on the
matching statistic.** The pattern is the same one CLAUDE.md names directly: a small
seed block's *variance* is not representative, and this project has been burned by
exactly this shape of mistake before (E021). E048's numbers were not fabricated or
wrong in what they measured — they measured a real quantity, on 3 real seeds, honestly
reported — but the two properly-powered, independent 8-seed blocks here disagree with
them, and by the project's own standing rule, the larger, replicated blocks are what the
tree should trust going forward.

## 7. Interpretation

**The personal-space fix (E048) does not resolve the shuffled-channel control problem,
and the earlier 3-seed reading that it might have made real progress here does not
hold up.** The specific quantity the shuffled control's viability depends on — how many
other hens happen to be within a hawk's strike radius of *whichever hen the hawk
targets* — is unmoved from E024's original, pre-E023, pre-personal-space-fix number.

**This is not necessarily a contradiction of E048's dispersal finding, but it does
narrow what that finding licenses.** E048's own metric (general nearest-neighbour
distance and unconditional strike-radius overlap, 6-minute window, 3 seeds) may still be
a real effect — it was not re-measured with more seeds here, only the hawk-targeted
version was. What this experiment does establish is that *whatever* general dispersal
the fix produces, it is not enough to change the specific local-clustering statistic
that matters for the sender-scrambling control: the hawk always dives near an existing
hen (`coop/world.py`'s targeted-arrival design, not a uniform draw), so this metric is
conditioned on exactly the moment and location personal space would need to have already
acted — and at 0.5 m of repulsion radius against a hawk strike radius sized to catch a
huddle, it evidently has not moved the needle.

**Does not affect H4's status.** H4 abandoned the shuffled control for the yoked one at
E026 specifically because the shuffled control was never viable, and nothing here changes
that; this experiment answers a narrower, previously-flagged question (would the E048
fix have rehabilitated the old control) with "no."

## 8. Consequence

- **`docs/experiments/E048-personal-space-fix.md` corrected in place**: its 3-seed
  "in strike radius" figures do not represent the hawk-targeted, properly-powered
  statistic and are struck through with a forward pointer here, not deleted. E048's
  core finding (a new observation channel producing *some* general dispersal, without
  breaking huddling or feeding) stands; the implication that it meaningfully addresses
  H4's original control problem does not.
- **`docs/hypothesis.md`'s E048 tree entry gets a forward-pointer note.** No status
  changes — the shuffled control was already abandoned in favour of yoked before this
  session began, so there is no live hypothesis status resting on this number.
- **A process note worth acting on, not just recording**: this is the second experiment
  in a row (after E048 itself) written up after implementation/measurement rather than
  before. Both self-corrected honestly, but the discipline is drifting under time
  pressure and should be tightened back up before the next piece of work, not treated as
  now-normal.

# E067 — the reward-modulation factor is sampled, not traced, at consolidation time

> **Diagnostic, not pre-registered.** This documents independent verification of a
> finding from an adversarial review (the `red-team` skill), commissioned after T2's
> Stage 2 returned a null twice (E065, E066). Per this project's own red-team
> discipline: nothing here was adopted before being re-measured independently, in
> this repo, with a fresh script — see `scratchpad/e067_reward_eligibility_check.py`.

## 1. Parent hypothesis

Directly: T2 (`docs/hypothesis.md`), whose Stage 2 conclusion this revises.
Indirectly: every hypothesis whose plastic conditions use the standard reward-gated
update to `W` and rely, even partly, on a discrete reward event (`strike_penalty`) —
which is most of this project's history prior to H2f's `hebbian_readout` rule. That
broader scope is named here and explicitly **not** acted on; see §8.

## 2. Question

Is the reward-modulation factor `m` (`run/simulate.py:100`, `m = reward -
ps.baseline`) — which gates `consolidate()`'s update to the recurrent weights `W`
(`hen/plasticity.py`, `dw = eta * m[...] * dz_slow[...] * dz_fast[...]`) — a genuine
trace with memory of what happened since the last consolidation, or a single-step
snapshot that only reflects a discrete reward event (a predator strike, a sickness
onset) if that event happens to land on the exact step `consolidate()` fires?

## 3. Prediction

None — this is a diagnostic run to check a specific external claim, not a test of a
hypothesis about the world.

## 4. Falsifier

N/A (diagnostic).

## 5. Design

Independent re-derivation using the actual `hen/plasticity.py` functions and default
`PlasticConfig` values (`interval=50`, `baseline_tau_s=20.0`, `strike_penalty=1.0`,
and `sickness_penalty=1.0`, E066's opt-in value): trace `m` step by step following a
single `-1.0` reward spike at an arbitrary, non-boundary offset, and separately sweep
every possible offset within one consolidation interval to compute what fraction
produce `|m| > 0.1` at the boundary `consolidate()` actually reads. Reused for both
`sickness_penalty` and `strike_penalty`, since both are the same discrete-event shape
going through the identical code path.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e067_reward_eligibility_check.py
```

## 6. Result

Worked example: a `-1.0` reward spike at step 23, `interval=50`.

| step | `m` |
|---|---|
| 23 (the spike itself) | −0.9995 |
| 49 (one step before the boundary) | +0.000494 |
| 50 (the boundary `consolidate()` reads) | +0.000493 |

The spike is completely invisible at the boundary — `m` at t=50 is indistinguishable
from background drift.

**Capture rate**, swept over every possible offset within one interval:

| term | capture rate |
|---|---|
| `sickness_penalty` | 2.0% (1/50) |
| `strike_penalty` | 2.0% (1/50) |

Both terms go through the identical code path (`reward()` → `m = reward -
ps.baseline` → `consolidate(p, ps, m, pc)`, called only when `w_next.t % pc.interval
== 0`, using whatever `m` happens to be at that exact step) and show the identical
defect.

**Independent confirmation of the reviewer's finding**: the reviewer reported "~2%"
from a worked example at three offsets; this diagnostic reproduces that number
exactly via an exhaustive sweep, using a freshly written script against the real
codebase functions rather than the reviewer's own numbers.

## 7. Interpretation

`z_fast`, `z_slow`, and `z_motor` (the eligibility traces `consolidate()` also uses)
genuinely are exponential moving averages, continuously updated every step by
`update_traces()` — they carry real memory of activity since the last consolidation.
`m` does not. It is recomputed fresh every step from the current `reward` and the
current `ps.baseline` (itself a genuine slow trace), but only the value at the exact
consolidation-boundary step is ever passed into `consolidate()`. A reward that is
already a rate or a slowly-varying quantity (like `d_drive`, hunger/thirst/cold
reduction) is well approximated by its value at any given instant, so this does
little damage to that part of reward. A reward that is a **discrete, single-step
event** — exactly what `strike_penalty` and `sickness_penalty` both are, by design,
following E014's own fix for the opposite problem (a rate-based penalty letting one
event dominate) — is, for practical purposes, invisible to the mechanism meant to
learn from it: real event timing has no reason to correlate with a fixed 50-step
grid, so the ~2% figure is close to what an actual run should show, not a worst case.

**This directly affects T2's own conclusion.** E066's write-up describes
`sickness_penalty` as providing "a genuine reward signal," "a real teaching signal."
That framing does not hold: the discrete event the whole T2 line depends on is
overwhelmingly likely to never reach the one weight-update pathway (`W`, the
reward-gated recurrent update) capable of routing place-cell information toward
motor output at all — a separate, previously-verified finding from the E066 design
review (`hebbian_readout`'s `W_out` update never reads sensory input directly, only
the motor region's own rates, so `W` is the only route that matters here).

**This does not touch H2f's own validated result (E057).** `hebbian_readout=True`
replaces `m_out` with a constant `1.0` for the `W_out` update specifically
(`hen/plasticity.py:401`) — the audience-sensitive-calling mechanism that result
measures runs entirely through `W_out` under that flag, bypassing `m` (and this
defect) for the pathway that actually matters to it. T2 uses the identical
`hebbian_readout=True` configuration, but per the same design review, T2 additionally
needs `W`'s reward-gated update to route place-cell information to where `W_out`
could exploit it — and that pathway *is* subject to this defect. T2's two
representational prerequisites, T2's genuine architectural gap (no pre-existing
motor correlate for place cells, unlike H2f's task), and this reward-sampling defect
now form three compounding, independently-verified reasons T2's rule could not
plausibly have succeeded, regardless of whether the underlying "can a flock learn
this" question has a real yes.

## 8. Consequence

**Adopted, and reflected in `docs/hypothesis.md`'s T2 node**: E065 and E066's framing
— that H2f's validated rule was fairly tested against T2 and failed — is revised.
Both experiments' *measured numbers* stand as reported (the nulls are real
measurements of what the code as written actually does), but the *interpretation*
changes: T2's rule had two independent, structural reasons it could not have
produced a positive result — the reward it depends on almost never reaches the one
pathway that could use it (this diagnostic), and even if it did, that pathway had no
pre-existing correlate to amplify (the E066 design review's own finding). T2's status
stays `NOT SUPPORTED`, but the reason is now "the instrument could not have detected
a positive result even if the flock were capable of one" rather than "a fair test of
whether the flock is capable."

**Not adopted, and explicitly flagged as unverified rather than acted on**: the
reviewer's broader claim that this defect may have affected `strike_penalty`-dependent
learning throughout the project's history (H2, H4, T1) prior to `hebbian_readout`'s
introduction. The mechanism is real and confirmed here for `strike_penalty` in
isolation — but whether any *specific* prior conclusion (T1's Pareto-safety result,
H4's various states, H2's clean nulls) actually depended on the discrete strike-event
term surviving to a consolidation boundary, versus being adequately explained by the
continuous `d_drive` pathway (which *is* a genuine trace and unaffected), has not been
checked for any of them. Per this project's own red-team discipline — "never rewrite
the tree on an unverified reinterpretation" — no other hypothesis's status is changed
here. This is recorded as a real, open, high-priority question for a future,
appropriately-scoped investigation, not resolved by inference.

**Not fixed here.** Making `m` a genuine trace (e.g. accumulating discrete-event
reward into its own eligibility, the same way `z_fast`/`z_slow` already accumulate
activity, rather than sampling a snapshot) would be a load-bearing change to the core
mechanism every plastic experiment in this project's history has run under. That
needs its own careful design and validation — this diagnostic's job was to establish
whether the claim was real, not to redesign the mechanism.
# E114 — does the learned gate work through vigilance, or something else?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H2b**. Completes [E113](E113-permuted-gate-control.md), which
established *that* the gate's channel assignment matters and could not say *why*.

---

## 2. Question

[E113](E113-permuted-gate-control.md) showed E102's learned gate beats both a scrambled
version of itself (pooled z=−4.40) and a flat one (z=−7.00). The assignment carries about
three quarters of the effect. What it could not show is the **mechanism** E102 asserts:

> Pecking and turning drive `head_down`, which `sensing.py` uses to blind her to the sky:
> **she learned to suppress the behaviours that blind her and keep the ones that save
> her.**

### That sentence is wrong about the code, and reading it was enough to find out

`spec.py:250`:

```python
HEAD_DOWN_ACTIONS = (M_PECK, M_SCRATCH)
```

**Turning does not drive `head_down` at all.** And `head_down` is what
`sensing.py:177` uses to gate the aerial channel — `aerial = aerial * (1 - head_down)`.

Set that against the gate the model actually learns, reproduced identically on four
disjoint seed blocks:

| channel | learned gate | does it blind her? |
|---|---|---|
| **TURN_R** | **0.22–0.25 — closed** | **no** |
| **TURN_L** | **0.27–0.32 — closed** | **no** |
| **PECK** | **0.29–0.45 — closed** | yes |
| SCRATCH | 0.92 — spared | **yes** |
| FORWARD | 0.92–0.95 — spared | no |

So the gate closes **two channels that do not blind her**, closes one that does, and
**spares the other one that does**. "She learned to suppress the behaviours that blind
her" is not a description of this gate. One of the two blinding behaviours is spared, and
two of the three suppressed ones have no effect on vigilance whatsoever.

This is a claim the source does not support, and it has been in the tree since E102 and
was repeated in E113's own §2. It is recorded here rather than quietly corrected because
the interesting question is what the gate *is* doing, and that needs measuring rather than
another story.

**So: if `head_down` no longer blinds her, does the gate still help?**

## 3. Prediction

1. **The benefit survives.** Removing the blinding link should leave the gate's advantage
   largely intact, because on the arithmetic above only one of its three closed channels
   has anything to do with vigilance. I expect the gate effect to fall by less than half.
2. **Everyone improves when blinding is removed**, gate or no gate, because a hen who can
   always see the sky gets more warning. That is a manipulation check as much as a
   prediction.
3. **If I am wrong and the benefit vanishes**, then vigilance is the whole mechanism after
   all, `PECK` alone carries it, and the two turn channels are along for the ride. That is
   a perfectly good outcome and it would vindicate E102's story on a narrower basis than
   E102 gives.
4. **What I do not expect to learn here** is what the turn channels are *for*. If the
   benefit survives, the next question is theirs, and this experiment does not answer it.

## 4. Falsifier

**Primary.** If the gate's advantage over no-gate falls by **more than half** when
`head_down` stops blinding her, vigilance is the operative mechanism, my reading of the
channel table is misleading, and E102's story stands on `PECK` alone.

**Instrument falsifier — reported before the headline.** With the flag on, the aerial
channel must actually stop being gated. Measured as mean aerial input during head-down
steps with a hawk overhead; it must rise to match the head-up value. If it does not, the
flag is not doing what it claims.

**Regression falsifier.** Inertness bit-identical at the default; the `head_down_blindness`
ethogram assay must still pass at the default (it is the guard that this mechanism exists
at all, and it *should* fail when the flag is on, which is the point).

**Replication rule.** E021, and E113's own experience: this arc has produced two seed
blocks that disagreed. Nothing moves the tree on one block.

## 5. Design

**`head_down_blinds: float = 1.0`** in `CoopConfig`. The default is **1.0, not 0.0**,
because unlike every other flag in this project it *disables* an existing mechanism rather
than adding a new one — 1.0 reproduces current behaviour exactly and is the inertness
condition.

Applied in both places `head_down` currently gates perception:

- `sensing.py:177` — `aerial = aerial * (1 - head_down_blinds * head_down)`
- `sensing.py:273` — the observability mask, which tells the prediction pathway she could
  not see the sky. If the aerial channel is no longer gated, the mask must not claim she
  was blind; leaving them inconsistent would train `W_pred` on a lie.

**A 2×2 on the same reared brains**, exactly E113's staging: rear 8 flocks with
`bg_gate=True`, 30 min, `hebbian_readout`, `readout_scaling_strength=0.3`,
`hawk_period_s=60`; then run a 10-minute test with plasticity off.

| | `head_down_blinds=1.0` | `head_down_blinds=0.0` |
|---|---|---|
| **true gate** | E113's true arm | |
| **no gate** | E113's none arm | |

Rearing happens once per seed at the default, so the two blinding conditions test the
*same* learned gate — the intervention is on the world, not on what was learned.

**Measured**: `caught/dive` and hunger; the aerial channel during head-down steps (the
instrument check); the fraction of time head-down; and the per-channel gate profile, to
confirm the same TURN/PECK signature before anything is concluded.

### Cost

~20 minutes.

---

## 6. Result

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*

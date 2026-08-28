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

### 6a. The pre-registered design floored, and its falsifier fired for a bad reason

`scratchpad/e114_head_down.py`, 8 seeds.

| blinds | gate | caught/dive | head-down | aerial while head-down |
|---|---|---|---|---|
| 1.0 | on | 0.1182 | 0.672 | 0.1588 |
| 1.0 | off | 0.1955 | 0.807 | 0.1092 |
| **0.0** | **on** | **0.0000** | 0.675 | 0.5251 |
| **0.0** | **off** | **0.0000** | 0.816 | 0.5821 |

The instrument falsifier cleared — the aerial channel rises from 0.109 to 0.582 against a
head-up reference of 0.579, so the flag does exactly what it claims. **And the experiment
is void anyway**: with blinding removed, `caught/dive` is **exactly 0.0000 in both arms**.
There is no room for a gate effect to appear in.

The script printed "gate effect retained: −0%" and "vigilance IS the mechanism". **That is
not a finding, it is a floored dependent variable**, and my primary falsifier fired on it.
`CLAUDE.md`'s rule 5 — "does the dependent variable have room to move in both
directions?" — names this exactly, and I wrote the falsifier without applying the rule to
it.

### 6b. Graded blinding — mostly floored too

`scratchpad/e114b_graded.py`, same seeds, rearing once per seed so every cell tests the
same learned gate.

| `head_down_blinds` | gate on | gate off | gate effect | t |
|---|---|---|---|---|
| 1.00 | 0.1182 | 0.1955 | −0.0773 | −2.38 |
| 0.75 | 0.0329 | 0.0875 | −0.0546 | −4.70 |
| 0.50 | 0.0000 | **0.0094** | −0.0094 | −1.52 |
| 0.25 | 0.0000 | **0.0000** | +0.0000 | — |
| 0.00 | 0.0000 | **0.0000** | +0.0000 | — |

**Only two settings are usable**, and the baseline more than halves between them, so an
absolute-effect comparison across them is confounded. The manipulation is too coarse to
answer the question.

**But it establishes something solid in its own right: head-down blinding is the dominant
determinant of predation in this world.** Reducing it by a quarter more than halves
baseline predation (0.1955 → 0.0875); halving it essentially abolishes predation (0.0094).
P(caught) is close to being a restatement of "was she head-down when the hawk came".

### 6c. The question, asked properly — and answered

The world was the wrong thing to manipulate. E102's claim is about **which channels** the
gate closes, and that is answerable inside the brain with the world untouched: open the
learned gate's rows selectively and see which ones carry the benefit.
`scratchpad/e114c_which_channels.py`, two disjoint seed blocks.

| arm | caught/dive (0–7 / 8–15) | vs none | t | **share of full effect** |
|---|---|---|---|---|
| learned (all 12) | 0.1182 / 0.1191 | −0.0773 / −0.0732 | −2.38 / −2.69 | 100% |
| **PECK only** *(the blinding one)* | 0.1544 / 0.1359 | −0.0411 / −0.0564 | −2.24 / −3.19 | **53% / 77%** |
| **TURNS only** *(neither blinds)* | **0.0542 / 0.0472** | **−0.1413 / −0.1451** | **−6.69 / −5.15** | **183% / 198%** |
| PECK+TURNS | 0.0535 / 0.0486 | −0.1421 / −0.1436 | −6.08 / −4.86 | 184% / 196% |
| + SCRATCH | 0.0510 / 0.0394 | −0.1445 / −0.1529 | −7.28 / −5.61 | 187% / 209% |
| none | 0.1955 / 0.1923 | — | — | — |

**The two turn channels — neither of which affects vigilance — carry roughly twice the
benefit of the entire learned gate.** Suppressing `PECK`, the one closed channel that does
blind her, carries about half to three quarters. Replicated on disjoint blocks with the
numbers nearly unchanged.

**And the learned gate is substantially worse than a lesioned version of itself.**
TURNS-only beats the full gate by −0.064 and −0.072. The rule found the right channels and
then added incidental suppression of the other nine — all sitting at 0.92–0.99, including
`CROUCH` and `FLEE` — which costs back about half the benefit.

## 7. Interpretation

**E102's mechanism claim is refuted.** "She learned to suppress the behaviours that blind
her" fails twice over: the source says only `PECK` and `SCRATCH` blind her, the gate closes
`PECK` and spares `SCRATCH`, and when the channels are separated the benefit is carried
almost entirely by the two channels that have **nothing to do with vigilance**.

**What the gate actually does is a locomotion effect.** Suppressing left and right turning
while leaving `FORWARD` open makes a hen travel in straighter lines, so she leaves the
strike radius instead of milling inside it. That is mundane, it is not about seeing the
sky, and it is what the numbers say.

**This explains E113's most puzzling result rather than contradicting it.** E113 found a
uniform gate matched in mean level gives *no* benefit while the structured one does, and I
could not say why. Now I can: uniform suppression damps `FORWARD` along with everything
else, cancelling exactly the locomotion advantage that turn-suppression creates. **The
structure matters because the useful structure is specifically "suppress turning, keep
going forward"** — and a flat gate cannot express that. E113's selectivity finding stands
and now has a mechanism; only E102's account of *what* was selected was wrong.

**A finding about the learning rule, not just this gate.** The rule identified the useful
channels and could not stop there — its incidental 1–8% suppression of the remaining nine
costs back roughly half of what it gained. What it learned is real, structured, and
strictly worse than a hand-lesioned version of itself. That is a more precise statement of
this project's central null than "learning does nothing": here learning does something
useful and something harmful at once, and cannot separate them.

**What this experiment cost me, methodologically.** Two designs before one worked. The
first floored its own metric with the falsifier already written; the second was too coarse
to resolve anything. Both failed the same way — I manipulated **the world** when the claim
was about **the brain**. The third asked the question where the claim lived, needed no new
flag, and answered it in twenty minutes.

## 8. Consequence

**`head_down_blinds` is adopted, off by default (1.0 = the shipped model).** It is not
recommended for use: E114b shows the metric floors below 0.75, so it cannot support a
clean contrast. It stays because it produced the finding in §6b about blinding dominating
predation, and because deleting it would make that unreproducible.

**`docs/hypothesis.md`.** E102's mechanism sentence is struck: the gate does not work
through vigilance. H2b records that the gate's benefit is a locomotion effect carried by
turn suppression, that E113's selectivity finding stands with this as its mechanism, and
that the learned gate underperforms a lesioned version of itself by about half.

**Not adopted.** Any claim that the gate improves vigilance *as its mechanism*. It does
reduce head-down time (0.807 → 0.672), and that is real; it is not what the predation
benefit runs through.

### Follow-ups

1. **Why turning specifically?** The obvious reading is straighter travel out of the
   strike radius, and it is untested. A direct measurement — path straightness, and time
   from hawk onset to leaving the radius — would settle it and is cheap.
2. **The rule cannot stop at the useful channels.** Whether a sparser update, or a
   threshold on the striatal drive, recovers the lesioned gate's performance is a real
   question and would be the first mechanism proposed against a *measured* deficit rather
   than a suspected one.
3. **The trained-flock mute** (backlog §5, open since E032) remains untouched.

# E014 — a units error in the reward, and what it does to H2

> **Pre-registered.** Sections 1–5 committed before the contrast was run.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
Currently `REFUTED at this timescale` on [E013](E013-clean-test-of-h2.md).

## 2. What E013 got wrong about its own result

E013's finding stands as measured: learning made hens significantly worse and
destroyed a large fraction of the connectome. **Its explanation of why was wrong.**

E013 claimed a near-zero-mean random walk plus an irreversible zero floor — "a
ratchet". Three measurements disconfirmed that:

| claim | measurement | verdict |
|---|---|---|
| weights random-walk to zero | mean \|w\| *rose* 0.064 → 0.076 | wrong |
| inhibitory weights erode first | E and I survive equally (82.8% / 82.6%) | wrong |
| synaptic scaling drives it | ablating it: 30,217 → 31,099 survivors | wrong |

The tell was seed variance: survival was 83%, 86%, **25%**, **50%** across four seeds.
A ratchet is systematic; this was not.

## 3. The actual cause

Reward was computed as

```python
own = d_drive * reward_scale - struck * strike_penalty / cfg.dt
```

**Being caught is a discrete event, not a rate, and it must not be divided by `dt`.**
At dt=0.01 a single strike contributed **−100** to that hen's reward — roughly 150x
what the drive terms contribute (~0.6), and far beyond what a baseline tracking over
20 s can absorb. The modulator then slams every eligible synapse at once.

Per-seed, before the fix:

| seed | survive | \|reward\| max | mean \|w\| | strikes |
|---|---|---|---|---|
| 0 | 83% | 0.087 | 0.077 | 0 |
| 1 | 86% | 0.091 | 0.075 | 0 |
| 2 | **25%** | **11.85** | 0.255 | 7638 |
| 3 | **50%** | **7.92** | 0.124 | 4932 |

Ablating the strike term restored seed 2 to 83% and seed 3 to 82%; seed 0, which
happened to take zero strikes, was unaffected. **The erosion tracked strikes exactly.**

After removing the `/ dt`:

| seed | survive | \|reward\| max | mean \|w\| | strikes |
|---|---|---|---|---|
| 0 | 83% | 0.087 | 0.077 | 0 |
| 1 | 86% | 0.091 | 0.075 | 0 |
| 2 | 82% | 0.080 | 0.078 | 8562 |
| 3 | 81% | 0.240 | 0.081 | 7850 |

Seeds 2 and 3 still take thousands of strikes and no longer blow up.

## 4. Prediction

**The contrast is re-run with the fix and nothing else changed.**

Genuinely uncertain, and worth stating plainly after being wrong about the mechanism
three times in a row. The bug clearly explains the *connectome destruction*. Whether
learning now *helps* is a separate question: removing a harm is not the same as
producing a benefit, and H2 has never been supported outside the saturated regime.

- **If learning now beats the control** (t > 2.23), H2 comes off `REFUTED` and E013's
  negative result is attributed to the bug.
- **If it is null**, H2 sits at "does no harm, does no good" — the erosion is
  explained but the hypothesis is not rescued.
- **If it is still significantly worse**, the bug was incidental and something else is
  wrong.

The middle outcome is the most likely.

## 5. Design

Identical to E013 in every respect except the units fix. Four conditions, exploration
stated per condition, 12 matched seeds, 20 min, gain 0.70, vigour budget.

- **Command**: `python -m run.experiment --minutes 20 --seeds 12`

## 6. Result

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._

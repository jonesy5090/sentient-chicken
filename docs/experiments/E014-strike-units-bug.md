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

12 matched seeds, 20 min, identical to E013 except the units fix.

| condition | hunger change | fed % | synapses | (E013) |
|---|---|---|---|---|
| fixed (innate only) | +0.018 | 6.2 | 36,373 | +0.018 |
| noise only | +0.014 | 6.2 | 36,373 | +0.014 |
| learning, no growth | **+0.082** | 4.8 | **30,058** | +0.081 / 19,088 |
| learning + growth | +0.063 | 5.3 | 54,521 | +0.095 / 40,731 |

```
learning, no growth  +0.064 +/- 0.018 SE   t=3.46   SIGNIFICANT, WORSE
learning + growth    +0.045 +/- 0.019 SE   t=2.38   SIGNIFICANT, WORSE
noise only           -0.004 +/- 0.013 SE   t=0.32   noise
```

**The third pre-registered outcome.** The connectome recovered — 19,088 → 30,058
surviving synapses — and the behaviour did not move at all: +0.081 → +0.082.

### Follow-up ablation: where does the harm come from?

6 seeds, freezing one learned pathway at a time:

| condition | hunger change | fed % | synapses |
|---|---|---|---|
| fixed | +0.036 | 5.6 | 36,369 |
| learning (full) | +0.088 | 4.9 | 30,109 |
| **learning, `W_out` frozen** | **+0.046** | **5.6** | 28,383 |

*(a fourth condition freezing the recurrent weights did not finish inside the time
budget and is outstanding)*

**Freezing the motor readout removes most of the harm** and returns feeding to the
control's rate — while recurrent learning continues, and prunes just as much
(28,383 synapses).

## 7. Interpretation

**The units bug was real and is fixed, and it was not the cause of the behavioural
harm.** Two symptoms that E013 treated as one thing are dissociated: the connectome
destruction tracked strikes and is gone; the behavioural harm tracked neither and
remains, unchanged to three decimal places.

That is the second time E013's mechanism story has been wrong, and it is worth being
blunt about the pattern: *pruning and harm looked causally linked because they
appeared together, and they are not.*

**The harm is in the learned readout.** Freezing `W_out` recovers most of it. Learning
the recurrent weights — including whatever pruning that entails — is close to
harmless. So it is not plasticity as such, nor structural change, nor exploration
(t=0.32). It is specifically the growing cortical influence over the motor output.

**This connects H2 to [H2d](E009-lagged-pallial-association.md), and the chain is
coherent.** The pallium cannot represent distinctions — states for "heard an alarm"
and "saw a hawk" differ by under 1% of mean rate. `eta_out` then grows a readout
*from that uninformative representation*, so the cortical pathway comes to transmit
structured, state-dependent noise into a motor system that was already competent.
E002 measured this ceiling from the other side: cortical influence that is not
well-trained makes behaviour worse.

Exploration noise is harmless because it is zero-mean and uncorrelated. A learned
readout from a degenerate representation is neither.

**The prediction that follows:** fixing the representation (H2d) should make learning
stop being harmful *before* it makes learning helpful. That is testable and is the
right next step — but it is a prediction, and this project's record on mechanism
predictions is 1-for-4, so it gets measured rather than assumed.

## 8. Consequence

- **H2 stays `REFUTED at this timescale`.** The re-test did not rescue it.
- **The units fix stands on its own merits** — a discrete event scaled by 1/dt was
  simply wrong, and the connectome recovery is real even though the behaviour did not
  follow.
- **H2d is promoted to the critical path.** It is now implicated in H2's failure, not
  just H2b's and H2c's. Every open problem in the project traces back to a pallium
  that cannot tell its inputs apart.
- **Outstanding**: the recurrent-frozen arm of the ablation, to confirm that recurrent
  learning alone is genuinely neutral rather than mildly harmful.
- **A guard is warranted**: nothing tests that reward components stay within an order
  of magnitude of each other. A single strike contributing 150x the drive terms should
  have been caught by construction, not by an eight-experiment detour.

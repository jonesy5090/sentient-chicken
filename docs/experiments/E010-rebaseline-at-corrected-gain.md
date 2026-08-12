# E010 — re-baselining: does H2 survive a non-saturated network?

> # ⚠ INVALID — CONFOUNDED
>
> **This experiment varied two things, not one, and its conclusion does not follow.**
>
> The condition labelled "fixed (innate only)" was constructed as
> `PlasticConfig(enabled=False)`, which inherits `explore_sigma=0.6` — motor noise
> added in [E007](E007-exploration-does-not-rescue-comprehension.md), *after*
> [E004](E004-replication-at-twelve-seeds.md) ran. The noise is applied before the
> `enabled` check in `run/simulate.py`, so **every condition here was running with
> exploration noise, including the control**, while E004's conditions had none.
>
> So E010 compared *(new gain + noise)* against *(old gain, no noise)* and attributed
> the entire collapse to the gain. That is not supported by the run.
>
> The tell was in [E011](E011-retune-the-readout.md): the fixed control did not improve
> as `readout_scale` fell toward zero. At `readout_scale = 0.005` the cortical term is
> ~0, so the gain *cannot* affect a fixed hen at all — yet she was still doing badly.
> Something other than the gain had to be responsible.
>
> **Superseded by [E012](E012-corrected-phase-1-contrast.md)**, which states
> exploration explicitly in every condition and adds a noise-only control to separate
> what exploration costs from what learning earns. `run/experiment.py` now names
> `explore_sigma` in every condition, and two tests in `tests/test_plasticity.py`
> fail if a "fixed" control ever carries noise again.
>
> **What survives from E010:** the gain sweep and the reasoning for choosing 0.70
> (§3) stand — those were measured directly and do not depend on the contrast. The
> H2 downgrade does *not* stand on this evidence and is re-decided by E012.
>
> Left in place unedited below, because a confounded experiment that gets quietly
> corrected teaches nothing.


> **Pre-registered.** Sections 1–5 written and committed while the run was executing.

## 1. Parent hypothesis

**H2** — three-factor plasticity produces measurable behavioural improvement.
Re-testing an already-`SUPPORTED` hypothesis, because
[E009](E009-lagged-pallial-association.md) showed the evidence for it was gathered
under a defect.

## 2. Question

E004 supported H2 at t=3.93 with the network running saturated: mean pallial rate
0.83, deep in the flat part of the sigmoid. E009 found that and fixed the operating
point (gain 0.9 → 0.70, mean rate 0.27). **Does the result hold?**

This is not a formality. Drive regulation apparently only needs coarse modulation,
which a saturated network can supply, so it is entirely possible that H2's support
came from the one kind of learning that survives a bad operating point.

## 3. Choosing the gain

Relative separability of two percepts — "heard an alarm call" versus "saw a hawk" —
in pallial state, averaged over genomes:

| gain | mean pallial rate | separability, % of mean rate |
|---|---|---|
| 0.60 | 0.212 | 3.3% |
| 0.70 | 0.271 | **7.5%** ← chosen |
| 0.75 | 0.349 | 14.2% ← measured optimum |
| 0.78 | 0.497 | 6.2% |
| 0.90 | 0.830 | 0.9% ← the old default |

**0.70 rather than the 0.75 optimum, deliberately.** The peak sits ~0.03 from a
transition — by 0.78 the mean rate has jumped to 0.50 and separability has collapsed.
Weights move during learning, so a value that must be held to two decimal places is
not a value to build on. 0.70 keeps 8x the old separability, and its mean rate is
tight across genomes (0.26–0.28) where 0.78's is not (0.42–0.59).

Worth recording: separability varies enormously between genomes at any gain
(3.5%–25.5% at 0.70). That is individual variation in how well a hen's wiring
separates her world — interesting rather than defective, but it means contrasts need
replicates and single-seed results mean little.

## 4. Prediction

**H2 holds, and plausibly strengthens.** A network that can represent distinctions
should learn at least as well as one that cannot. Stated as: learning-without-growth
beats the fixed control on within-run hunger change by more than t=2.20 (11 df).

**Secondary:** the growth condition stays the weaker of the two, as in E001, E003 and
E004.

**If H2 weakens or vanishes**, that is the more interesting result — it would mean the
supported finding depended on the saturated regime, and would need explaining rather
than explaining away.

## 5. Design

Byte-identical to [E004](E004-replication-at-twelve-seeds.md) except for the gain:
same conditions, same 12 seeds, same coops, same 20 min, same metric, same threshold.

- **Ethogram re-checked first**: 7/7 assays still pass at gain 0.70, and head-down
  blindness strengthened (aerial 0.63 vs 0.57 head-up; crouch 0.92 vs 0.85). The
  innate layer is unaffected by the change, as it should be — the reflex arc does not
  pass through the recurrent network.
- **Command**: `python -m run.experiment --minutes 20 --seeds 12`

## 6. Result

12 matched seeds, 20 min, 16 hens. Wall clock 62 min.

| condition | hunger early | hunger late | change | fed % | exposure | synapses |
|---|---|---|---|---|---|---|
| fixed (innate only) | 0.422 | 0.655 | +0.234 | 5.6 | 4533 | 36373 |
| learning, no growth | 0.414 | 0.645 | +0.231 | 6.1 | 3597 | 21797 |
| learning + growth | 0.420 | 0.639 | +0.219 | 6.6 | 5316 | 40344 |

```
hunger change   learning, no growth   -0.002 +/- 0.028 SE   t=0.08   noise
hunger change   learning + growth     -0.014 +/- 0.036 SE   t=0.41   noise
exposure        learning, no growth    -936  +/- 1567 SE    t=0.60   noise
exposure        learning + growth      +783  +/- 1415 SE    t=0.55   noise
```

Against E004 at the old gain, same seeds, same everything else:

| | E004 (gain 0.9) | E010 (gain 0.70) |
|---|---|---|
| learning, no growth | −0.063 ± 0.016, **t=3.93** | −0.002 ± 0.028, **t=0.08** |
| fixed control, hunger late | 0.330 | **0.655** |
| learning, hunger late | 0.272 | **0.645** |

## 7. Interpretation

**H2 does not survive the re-baselining.** The effect collapses from t=3.93 to t=0.08.
The prediction in §4 was wrong, and the alternative it named — that a weakened result
would be the more interesting one — is what happened.

**Every hen also got much worse at foraging.** Hunger at the end of a run roughly
doubled, from ~0.33 to ~0.65, in *both* conditions including the fixed control. That
is the key to reading this: the change did not selectively remove learning, it
degraded the whole flock's behaviour and took the learning signal with it.

**The mechanism, and it is E002's finding wearing a different hat.** At gain 0.9 the
pallium was saturated, so its output was near-constant, so the cortical readout acted
as a fixed *bias* on the motor drive — harmless. At 0.70 the pallium is responsive,
so an *untrained* readout injects genuine variability into the motor drive, and that
disrupts reflex-driven foraging. E002 already established that cortical influence
which is not well-trained makes behaviour worse. Saturation had been accidentally
protecting behaviour by keeping the pallium's output constant.

So the two knobs are coupled in a way nobody planned: **`readout_scale` and `eta_out`
were tuned by E002 against a saturated network,** and they are now wrong. The
correct reading of E010 is not "learning does not work" but "the readout parameters
are stale".

**What this does to H2.** Its status has to come down. E004 was a real, replicated,
pre-registered result — but it was obtained in a regime we now have measured reason
to believe is defective, and it does not transfer. `SUPPORTED at gain 0.9` and `null
at gain 0.70` are both true, and the honest summary is that H2 is open again.

That is uncomfortable and it is the correct call. A result that only holds in a
regime where the network cannot represent distinctions is not a result about
learning.

**Learning is still doing something.** Directionally, the no-growth condition feeds
more (6.1% vs 5.6%) and takes 21% less predator exposure (3597 vs 4533). Neither is
significant, but neither points the wrong way. There is a signal under the noise.

**One measurement caveat**, stated so it is not used as an excuse: hunger now ends at
0.65 rather than 0.33, closer to the clip at 1.0, so the metric has less room and may
be compressing. It is not *at* the ceiling, so this is a partial contributor at most.

## 8. Consequence

- **H2 downgraded `SUPPORTED` → `UNDER TEST`.** Both results recorded, both true, and
  the scope of E004's finding is now known to be narrower than it looked.
- **The gain change stands.** E009's measurement of the saturation problem is not in
  doubt, and reverting to a regime where a hen cannot tell a call from a hawk would
  trade a real capability for a metric. The fix is to re-tune what depended on it.
- **E011**: redo E002's readout sweep at gain 0.70. `readout_scale` and `eta_out` were
  both chosen against a saturated network. Likely direction: a *smaller* initial
  readout, so an untrained pallium disrupts less, with `eta_out` free to grow it as
  competence develops.
- **Then re-run H2 again.** Only after the readout is re-tuned does a null mean
  anything about learning.
- **Standing note reinforced**: parameters tuned under a defect inherit the defect.
  E002's sweep was careful and correct at the time, and it silently encoded an
  assumption about the operating point that outlived its validity.

# E028 — repairing the instrument E027 condemned

> **Repair-and-re-run, not a hypothesis test.** Sections 1–5 describe fixes whose
> acceptance criteria were fixed before each was measured. The re-run in §6 uses seeds
> **36–47**, which no prior experiment has touched.

## 1. Parent hypothesis

**H4**, downgraded to `UNDER TEST` by [E027](E027-third-review-verified.md). Nothing here
tests H4; it makes H4 testable again.

## 2. Question

E027 confirmed that the reward at the H4 configuration was 87.3% "was I just caught",
that the risk metric's denominator moved up to 63% with the treatment, that Dale's law
was violated on the only pathway reaching a muscle, and that the registered contrasts
were never computed. Can each be fixed, and does the ladder then say something
different?

## 3. Phase A — the teaching signal

### A1. Strikes are events, not rates

`n_struck` incremented **every step** of contact, so one catch during a 12 s dive was
worth up to ~1000 against per-step drive terms of 0.1–0.5. E014 found the original form
(a strike divided by `dt`), removed the `/dt`, and left the per-step accumulation. E022
filed the remainder as owed; nobody verified it for six experiments.

`n_strike_events` counts rising edges — which works for both predators, where
`hit_this_dive` is hawk-specific and belongs to the metric. The reward reads it.
`n_struck` survives for diagnostics and for reading old logs.

| share of reward variance, hawk every 20 s | before | after |
|---|---|---|
| `n_struck` / `n_strike_events` | **87.3%** | **0.2%** |

### A2. The guard is split, because parametrising it was not enough

The guard ran only at `hawk_period_s=900`, where **0 of 3000 steps** contain a strike.
Parametrising it over the H4 rate looked like the fix and was not: a 30 s window at
`hawk_period_s=20` *also* contains no strike, so it would have passed for exactly the
same vacuous reason. 100 s is the shortest window measured to contain one.

The new test therefore asserts two things — that the strike term is small, **and that
strikes actually happened**. A guard that cannot fail is not a guard.

### A3. Dale's law on `W_out`, and the fix that broke the bird

Violated at initialisation and under learning: **0 of 48 columns compliant, all 48
mixed**, 10 source neurons inhibitory. Found by E022, filed "verified — adopt", dropped
from the action list for four experiments.

**The obvious fix broke the flock, and the measurement caught it within one run.**
Signing a rectified draw (`|normal| * dale`) leaves every motor channel with a large
positive DC bias, because the stub is 80% excitatory. Measured immediately after:

| | before A3 | naive A3 | balanced A3 |
|---|---|---|---|
| strike-steps in 3000 (hawk 20 s) | 1000 | **0** | — |
| hunger share of reward variance | 43.1% | **91.6%** | 38.5% |
| Dale-compliant `W_out` columns | 0/48 | 48/48 | **48/48** |
| net DC drive per motor channel | ~0 | large | **1.8e-07** |

The hens crouched permanently — safe, blind and starving. The recurrent path never
showed this because synaptic scaling renormalises its row sums; the readout has no such
correction. Fixed by scaling the inhibitory group to match the excitatory one per (hen,
motor channel), which makes the untrained readout contribute exactly zero DC. That is
the stated design intent anyway — the pallium must *earn* influence — now true by
construction rather than by luck.

**This is a re-baselining** in the E010/E023 sense: every number measured before it is
un-comparable, including E027's own verification block.

Behaviour after: **7/7 ethogram assays pass**, fed % 3.04 against E026's 3.06, crouch
0.35.

`test_being_caught_is_aversive` failed on the old field, which is the rule working. It
now also asserts the contact counter does **not** reach the reward, so it cannot pass
on a dead field.

## 4. Phase B — the metric

### B1. Intent to treat

`caught/dive` over every (hen, dive) pair. The denominator is fixed by `hawk_period_s`,
run length and flock size; no behaviour can reach it. Measured across seven conditions:

```
condition        dives    vs C?   blind risk    vs C?
N  natural         139     0.0%         13.0   -25.0%
C- capacity        139     0.0%         13.3   -23.1%
C0 severed         139     0.0%         13.3   -23.1%
C? yoked           139     0.0%         17.3     0.0%
Cs self-only       139     0.0%          7.3   -57.7%
L  language        139     0.0%         18.3    +5.8%
Lx lesioned        139     0.0%         26.0   +50.0%
```

`dives` is flat to 0.0%. `blind risk` — E026's denominator — spans **−57.7% to +50.0%**.
The table now prints this above the headline every run, instead of it having to be
inferred two experiments later.

**The numerator was wrong too, and the smoke run is what found it.** `n_caught` is gated
on `at_risk`, sampled at dive onset, so a hen who walked into the radius mid-dive and was
taken was never counted. A 1-minute run showed dozens of contact steps with `caught` at
exactly **zero**. `n_caught_any` counts being caught during the dive at all.

### B2. The lesion is a standing rung

`Lx lesioned` — L with `W_out` zeroed — is now permanent. If it matches L, the pallium is
not in the causal path. On the smoke run it already reported **+0.009 ± 0.010, t=0.87**:
E027's finding announcing itself rather than waiting for an outside reviewer.

`C−` is kept and labelled: with plasticity off in every condition the capacity control is
**vacuous by construction**, so H0's "at any capacity" clause is untested until a
learning rule works. Stated in the output so it is not silently re-dropped.

### B3. The contrasts that were registered

- **L vs C?** is the headline, per `docs/backlog.md` §1. E026 reported everything against
  deaf and never computed it.
- **Pooled counts beside paired means.** E026 quoted the mean-of-ratios (−0.198) in prose
  while pooled over the same events was −0.150.
- **T1's registered metric** — food intake at matched risk — printed unconditionally,
  with E026's null recorded inline.
- **`_t_critical` carries every df from 1 to 30.** It had no entry for 11 and broke the
  10-vs-12 tie by dict order, returning 2.228 where E026's prose said 2.201.

## 5. Throughput

Measured on this machine at the pre-change commit, because the README's figure was taken
on different hardware and this container does not reach it either way:

| 512 neurons, 16 hens | real-time factor |
|---|---|
| pre-change (E027 commit) | 20.7× |
| after A+B | 18.7× |
| README, different machine | 33.6× |

About 10%, at the edge of what a single benchmark run separates from noise. Recorded
rather than dismissed; `CLAUDE.md` treats throughput as a correctness constraint.

## 6. Result

Seeds **36–47**, 12 seeds × 10 min × 7 conditions, on the re-baselined genome.

```
condition        fed %  caught/dive   dives  at risk  blind risk  caught/event
N  natural        2.69        0.137     305     50.0        27.8         0.482
C- capacity       2.91        0.118     305     39.2        23.2         0.452
C0 severed        2.91        0.118     305     39.2        23.2         0.452
C? yoked          2.62        0.115     305     39.6        18.9         0.404
Cs self-only      2.72        0.113     305     52.6        26.4         0.386
L  language       2.67        0.086     305     48.4        28.0         0.262
Lx lesioned       2.68        0.080     305     55.2        30.3         0.261
```

### The headline, on a denominator the treatment cannot move

**L vs C? — the registered contrast — is −0.029 ± 0.020, t=1.42. Not significant**
(needs 2.20 at 11 df). Pooled over events: 0.084 vs 0.112, difference −0.028, which
agrees with the paired estimate.

**And the metric E026 used still says otherwise.** On `caught/event` — the confounded
one — the same 12 seeds give **−0.142 ± 0.061, SIGNIFICANT**. Two metrics, one dataset,
opposite verdicts. The difference is entirely the denominator: `dives` is flat to 0.0%
across all seven conditions, while `blind risk` ranges from C?'s 18.9 to Lx's 30.3, a
**60% spread**.

That is the whole of E027's finding, reproduced deliberately rather than discovered
afterwards: **the previous headline was substantially an artefact of a denominator that
moved with the treatment.**

### The pallium is not in the causal path — replicated

**L vs Lx: +0.006 ± 0.013, t=0.47, noise.** Lesioning `W_out` entirely — every route
from 512 simulated neurons to a muscle — changes nothing. If anything the lesioned flock
is marginally *better* (0.080 vs 0.086), consistent with E002's ceiling and with E027's
finding that removing an untrained random projection helps.

E027 measured this on the old genome with 8 seeds and no repaired metric. It replicates
on fresh seeds, a re-baselined brain and an unmovable denominator.

### T1's registered metric is null for the third time

**fed % +0.053 ± 0.087, t=0.61.** `docs/backlog.md` §3 predicted L forages *more* than C?
at equal risk. E026: 3.06 vs 3.07, then 2.54 vs 2.41. Now 2.67 vs 2.62. Three blocks,
no effect.

### The rest of the ladder

| contrast | caught/dive | verdict |
|---|---|---|
| L vs C− (capacity) | −0.032 ± 0.015 | t=2.18, just under 2.20 |
| L vs C0 (severed) | −0.032 ± 0.015 | t=2.18, just under 2.20 |
| L vs Cs (self-only) | −0.027 ± 0.013 | t=2.00 |
| L vs N (natural) | −0.051 ± 0.012 | **SIGNIFICANT** |
| **L vs Lx (lesioned)** | **+0.006 ± 0.013** | **noise** |

The pattern is coherent: L beats every condition lacking present-tense information by
roughly the same 0.03, and C? sits with C0 and C− rather than with L — which is what a
working yoked control should do. **L vs N is the only significant contrast, and it is
also the least clean**: N differs from L in capacity *and* scaffold, so it varies two
things at once and cannot carry a claim.

## 7. Consequence

- **H4 stays `UNDER TEST`, and now for a second, independent reason.** E027 withdrew the
  claim that it was a result about the *brain*. E028 finds that on a denominator the
  treatment cannot move, the registered contrast does not reach significance at 12
  seeds either. The direction is consistent (L lowest of every informed condition,
  −0.029) and the effect may well be real and small — but "supported" was not what the
  data said once the instrument was fixed.
- **The honest summary of E026 is now: a real but small channel effect, reported at
  roughly 5× its size** by a metric whose denominator moved with the treatment and an
  estimator that weighted sparse seeds equally. −0.198 became −0.029.
- **A positive control is now unavoidable**, and it is the same argument as `CLAUDE.md`
  check 6 in reverse: before reporting that a 12-seed contrast on the ITT metric is
  null, show the metric detecting an effect of known size. Without that, this null is as
  uninformative as the ones it replaced. **This is the top of the queue.**
- **Recorded, not fixed:** `C−` and `C0` come out byte-identical again despite the
  `call_vigour_drain=0` patch meant to separate them, so the energetic cost of calling is
  inert. Needs its own measurement.
- **Recorded, not fixed:** over a 100 s window at the H4 configuration, hunger carries
  **95.9%** of the reward variance (43% over 30 s). That is not the defect this
  experiment addressed and it is not obvious it is a defect — hunger arguably *should*
  dominate a foraging reward — but nobody has measured it.
- Still owed from E027: E025 has no file, and `docs/backlog.md` §7-'s item 0 should be
  demoted, since the dispersal diagnosis it rests on was wrong.

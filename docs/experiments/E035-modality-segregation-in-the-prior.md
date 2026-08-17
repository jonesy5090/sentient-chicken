# E035 — modality segregation moved into the connectivity prior, and does more capacity help

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2d** — the pallium does not form separable representations of distinct stimuli.
`SUPPORTED as a limitation`, reprioritised upward by
[E034](E034-h2d-remeasured.md): the hawk-vs-alarm-call contrast this node depends on
occurs on 11.9% of hen-steps in the live coop, not the near-hypothetical situation that
had demoted it.

## 2. Question

E017 and E034 both measured modality segregation ("give audition its own slice of the
sensory stub and its own pallial target — Field L kept apart from the entopallium, as in
a real bird") via a hand-cut post-hoc script (`scratchpad/why_pallium_collapses.py`) that
zeroes connections on an already-built connectome. `docs/backlog.md` §7a explicitly asked
for this to be done differently: **"via the connectivity prior in `regions.py`, not a
slice."** Two questions:

**(a)** Implemented properly — as a `connectome.build()` option driven by the same
random-generation machinery as everything else, rather than post-hoc surgery — does it
reproduce the same number (1.45×, E034)? This is a implementation-correctness check as
much as a scientific one: if the "proper" version disagrees with the probe, one of the
two has a bug.

**(b)** Now that trying a different partition size costs one keyword argument instead of
a rewritten script, does giving audition **more** of the stub and pallium buy more
separability, or does the benefit saturate? E034's ceiling framing ("1.45× against a
~14–17× loss") implicitly assumes 1/6 is a reasonable allocation; nobody has checked
whether 1/3 or 1/2 does better, worse, or the same.

## 3. Prediction

**(a)** The structural implementation reproduces 1.45× within measurement noise (6
genomes, same spread as E017/E023/E034 — typically ±0.02–0.03 on this metric). This
should hold almost exactly, since the two implementations perform the same masking
operation on the same random draws; a large discrepancy means one is wrong.

**(b)** Separability rises with `aud_fraction` but sub-linearly and does not close the
gap: more of the pallium dedicated to audition means fewer fan-in-diluting inputs per
Field-L neuron (more units sharing the same ~4 informative channels), which should help,
but the mechanism E017 identified — fan-in dilution at a single feedforward projection —
is not fixed by resizing, only by how concentrated the informative signal is within
whichever slice receives it. No prediction on the exact numbers; this is genuinely open.

## 4. Falsifier

**(a)** The structural and ad hoc implementations disagree by more than genome-to-genome
noise (a factor of ~1.3× or more) — one of the two has a bug, and this needs resolving
before either number is trusted.

**(b)** Separability is flat or falls as `aud_fraction` increases — would mean the
mechanism is not simply "concentration of informative input," and the fan-in dilution
story needs revision.

## 5. Design

`connectome.build(modality_segregated=True, aud_fraction=f)` for `f` in {1/6, 1/3, 1/2},
against the `modality_segregated=False` baseline, all at the current default connectome
(gain 0.95, E/I-corrected). Same settle-and-separate probe as E009/E017/E023/E034: hawk
overhead vs. flockmate's aerial call, presented alone, matched amplitude, held 2s to
settle, read at the pallium as RMS difference over mean rate. 6 genomes per condition,
matching E017's replicate count.

**Command:** `python -m scratchpad.e035_modality`

## 6. Result

**First pass, 6 genomes (seeds 0–5), matching E017/E023/E034's replicate count exactly:**

```
intact (mixed)                0.0735 +- 0.0279
segregated, 1/6 (structural)  0.0654 +- 0.0258     (0.89x intact)
```

**This already misses prediction (a) badly** — 0.89× (worse than intact), not 1.45×.
Diagnosed rather than dismissed: the structural implementation computes fan-in *after*
the cross-modal cut, so Field L's surviving synapses are correctly re-normalised for
their true (reduced) in-degree — the codebase's standard convention
(`w_raw = ... * gain / sqrt(fan_in)`), applied consistently. E017/E034's post-hoc probe
built a fully-connected, fully-normalised connectome first and only *then* zeroed the
cross-modal entries, leaving the survivors under-driven relative to their true fan-in —
never re-normalised. Reproducing that exact method (`build_posthoc` in the script) on
the *same keys* as the structural version confirmed the mean rate of the Field-L slice
drops further under the old method than the properly-normalised one (checked directly:
mean rate under a call, intact 0.30–0.33 vs. structural 0.26–0.29 — a real but modest
drop from re-normalisation alone).

**Second pass, properly paired, 12 genomes (seeds 0–11):**

```
seed   intact  structural  posthoc
0      0.1290    0.0836    0.0668
1      0.0764    0.1082    0.1370
2      0.0591    0.0744    0.0897
3      0.0612    0.0507    0.1179
4      0.0387    0.0385    0.0799
5      0.0766    0.0369    0.1389
6      0.0528    0.1435    0.0607
7      0.1319    0.1504    0.0997
8      0.2206    0.1516    0.0571
9      0.0749    0.0752    0.0561
10     0.0837    0.0725    0.0562
11     0.0411    0.0546    0.0519

intact       0.0872 +- 0.0492   (1.00x)
structural   0.0867 +- 0.0404   (0.99x)
posthoc      0.0843 +- 0.0308   (0.97x)

paired, n=12, threshold t=2.201 (df=11):
  structural - intact     -0.0005 +/- 0.0119   t=0.04   not significant
  posthoc    - intact     -0.0028 +/- 0.0187   t=0.15   not significant
  posthoc    - structural -0.0023 +/- 0.0167   t=0.14   not significant
```

**Every pairwise comparison is null.** Individual genome values range from 0.039 to
0.221 — nearly 6× spread — which is the same genome-to-genome variability E009 already
flagged ("separability varies 3.5%–25.5% between genomes at a fixed gain") and which
neither E017, E023, nor E034 treated as a warning about sample size, because each
reported a ratio-of-means across 6 genomes rather than a paired per-genome contrast.

**Part B, fraction sweep (structural, properly normalised), 6 genomes — reported but not
trusted, for the reason above:**

```
1/6    0.0654 +- 0.0258
1/3    0.0888 +- 0.0277
1/2    0.0548 +- 0.0250
```

Non-monotonic, and given Part A's demonstration that 6-genome, unpaired samples of this
exact quantity cannot distinguish 0.89× from 1.45× from 0.99×, this sweep is not
informative as run. Recorded rather than discarded, per this project's convention, but
not to be cited as a finding.

## 7. Interpretation

**Prediction (a) is falsified, and not for the reason anticipated.** §4 predicted that
disagreement between the structural and ad hoc methods would mean "one of the two has a
bug." Neither does. The disagreement is real and explained: the ad hoc probe leaves
segregated neurons under-driven relative to their true fan-in, a genuine confound
between "anatomical segregation" and "reduced total input drive" that a properly
re-normalised implementation removes. Once removed, on a properly *paired* 12-genome
sample, **modality segregation has no measurable effect on pallial separability at all**
— not 1.45×, not 2.06×, not the 0.89× this run first appeared to show either.

**This is a bigger correction than "the number moves."** E017's 2.06× and E034's 1.45×
were each individually treated as confirming the same qualitative story (segregation
helps, partially) because they agreed in direction on two independent 6-genome samples.
Both were unpaired ratio-of-means on a quantity whose per-genome spread (0.039–0.221 in
this very sample) is large enough that agreement in direction between two 6-genome draws
is not strong evidence — this run's own first pass (6 genomes, 0.89×) disagreed with
both of them in direction, on the *same* underlying method family. **The project's own
standing rule — "no status changes on one seed block," instituted after E021 — applies
here to genomes, not seeds, and was not being followed for this specific measurement
across E017/E023/E034.**

**What survives:** the *localisation* finding (E017/E034: loss concentrated at the
sensory→pallium projection, recurrence not the cause) rests on stage-by-stage
measurements *within* a single connectome (comparing sensory/pallium/arcopallium/motor
separability on the same genomes), which is a paired, within-genome comparison by
construction — that part is on firmer footing than the Field-L segregation ablation
specifically, which compares *across* connectome variants and needs the between-genome
pairing this run finally applied.

## 8. Consequence

- **Correct, not quietly edit, E017's 2.06× and E034's 1.45×.** Both experiment files
  keep their original numbers with a forward pointer to this correction, following
  [E018](E018-innate-auditory-reflex.md)'s and [E032](E032-causal-efficacy.md) §6's
  precedent of recording an error rather than removing it.
- **`docs/hypothesis.md`'s H2d section needs its "cite 1.45×, not 2.06×" line struck** —
  neither number should be cited going forward. Modality segregation, as measured so
  far, has no established effect on separability in either direction.
- **The connectivity-prior implementation stays** (`connectome.build(modality_segregated
  =True, aud_fraction=...)`) — it is more correct than the probe it replaces (properly
  fan-in-normalised, tested, reusable) and the negative result is only trustworthy
  *because* this implementation exists and made a proper paired comparison cheap. The
  ad hoc script (`scratchpad/why_pallium_collapses.py`) should not be used for this
  measurement again.
- **The backlog item "modality-segregated afferents" should be downgraded**, not
  promoted. It was carried as a partial, real fix (worth ~1.45–2.06×) against H2d's
  ~14–17× loss. It is now an *untested* fix again — worse than that, the one clean test
  it has had (this one, paired, n=12) found nothing.
- **Standing process gap worth fixing**: any future representational/structural
  diagnostic on this connectome (stage separability, ablations, sweeps) should use a
  paired per-genome contrast with an explicit t-test against a threshold, exactly as
  `run/experiment.py` already does for behavioural contrasts, rather than reporting
  ratio-of-means across an unpaired genome sample. `scratchpad/e035_modality.py` is the
  template.
- **H2d itself is untouched by this** — the representational bottleneck (sensory stub
  separates cleanly, pallium does not, loss at the feedforward projection) is not in
  question. What is now open again is whether *anything* fixes it; modality
  segregation was the one concrete candidate with a measured number, and the number is
  gone.

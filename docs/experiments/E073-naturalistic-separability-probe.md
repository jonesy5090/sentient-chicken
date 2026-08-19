# E073 — is H2d's probe measuring the right thing?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2d**. Not a mechanism test — a test of the *instrument* every H2d number since E009
has been taken with.

## 2. Question

[E072](E072-balanced-ei-and-h2d.md) found the same intervention (balanced E/I) leaves
the classic separability metric flat (0.90×, t=0.74, paired 12 genomes) while taking
place-to-place pallial correlation from 0.9807 to 0.7520. The proposed reason is that
the two probes feed the network very different input:

- The settle-and-separate probe (E009/E017/E023/E034/E035/E041) injects **one channel
  at amplitude 1.0 into an otherwise-zero observation**. Almost no common mode exists,
  so removing common mode can do almost nothing.
- Naturalistic observations from `sensing.observe` have dozens of channels active and a
  large common mode.

Does H2d's separability metric behave differently under naturalistic input than under
hand-injected sparse input?

## 3. Prediction

**Under naturalistic input, `balanced_ei` improves separability; under sparse injection
it does not** (the latter already measured, E072). If both probes agree, E072's
discrepancy is about place coding specifically rather than about input statistics, and
H2d's measurement history is unaffected.

No prediction on absolute separability values under the naturalistic probe — they are
not expected to be comparable to the E009 series, which is part of the point.

## 4. Falsifier

If naturalistic and sparse probes give the same verdict on `balanced_ei` (paired, 12
genomes), the probe-statistics explanation is wrong. E072's place result would then need
a different explanation and should not be used to question the H2d series.

## 5. Design

**The same contrast E072 ran, under a different probe.** Everything else held fixed:
12 genomes, paired, `balanced_ei` on vs off, same settle procedure and duration.

**Naturalistic stimulus pair.** The hawk/call contrast, staged in a real coop rather
than hand-injected: a hen with a hawk overhead and no call audible, versus the same hen
with a flockmate alarm-calling and no hawk visible. Both observations come from
`sensing.observe` on a staged `World`, so every other channel (vision, interoception,
somatic, place) carries its normal live values. This is the same *contrast* the E009
series uses — only the surrounding input statistics change, which isolates the variable
under test.

**Both probes run in the same script on the same genomes**, so the comparison between
probes is itself paired and cannot be attributed to genome sampling.

**Reported**: separability under each probe, for each condition, plus the paired
contrast within each probe, plus mean pallial rate as a sanity check that the
naturalistic probe is not simply driving the network into a different operating regime.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e073_naturalistic_probe.py
```

## 6. Result

**Input statistics of the two probes:**

| stimulus | nonzero channels | mean value |
|---|---|---|
| sparse hawk / call | **1** | 0.0072 |
| naturalistic hawk / call | **39** | 0.0547 |

**Separability, 12 genomes, paired within each probe (threshold t=2.201):**

| probe | baseline | `balanced_ei` | paired contrast |
|---|---|---|---|
| sparse (E009 series) | 0.0961 | 0.0867 | −0.0094 ± 0.0127, t=0.74, **not significant** (0.90×) |
| naturalistic | **0.0365** | 0.0776 | **+0.0411 ± 0.0071, t=5.75, SIGNIFICANT (2.13×)** |

**Mean pallial rate — the number that decides which probe to believe:**

| condition | mean pallial rate |
|---|---|
| sparse probe | 0.2724 |
| naturalistic probe | 0.6019 |
| **live rollout** (16 hens, 5 min, hawk every 20 s, 3 seeds) | **0.7288** |
| live rollout, `balanced_ei` | 0.1209 |

## 7. Interpretation

**The prediction holds: the two probes give opposite verdicts on the same intervention,
on the same genomes, paired.** `balanced_ei` is null under sparse injection and a
significant 2.13× under naturalistic input.

**The live measurement decides which probe is representative, and it is not the one
H2d's history used.** Live operation runs the pallium at 0.7288. The naturalistic probe
reaches 0.6019; the sparse probe only 0.2724 — it under-drives the network by roughly
2.7× relative to the regime a hen actually inhabits.

**That has a consequence for E009's original diagnosis that nobody has drawn.** E009
identified saturation: *"the network runs saturated (mean pallial rate 0.83, where the
sigmoid slope is ~0.12)"*. E023 then re-ran across a gain sweep, reported mean rate
0.189 at the new default, and the saturation concern left the record as handled. But
that 0.189 was measured on the sparse probe. **In live operation the network is still
saturated — 0.7288.** The gain re-baselining fixed saturation only in a regime the hen
is never in. `balanced_ei` brings live rate to 0.1209, out of the compressive region for
the first time.

**Baseline separability under naturalistic input is 0.0365 — 2.6× worse than the sparse
probe reports (0.0961).** So H2d is not merely mis-measured in its response to
interventions; its headline severity has been understated throughout.

**What this does not establish.** 2.13× is a real improvement against a 14.5–17× loss —
a dent, not a fix, and the same order as the modality-segregation figures that did not
survive E035. The naturalistic probe is staged (four hens, hand-placed) rather than
sampled from live rollouts, so it is *closer* to live than the sparse probe, not
identical to it. And this measures representation only: nothing here shows any
downstream behaviour or learning improves.

## 8. Consequence

**`balanced_ei` is revalidated as a live candidate**, and E072's null on it should be
read as a null *of the sparse probe*, not of the intervention. E072's own §7 flagged
the discrepancy and declined to lean on it pending exactly this test; that was the right
call and the test has now come back.

**The H2d measurement series needs re-reading, not discarding.** E009, E017, E023, E034,
E035 and E041 are all internally valid — their numbers are correct for the probe they
used. What changes is the scope of their conclusions: they characterise separability in
a low-drive regime, and are specifically blind to interventions that act on common-mode
drive. E041's density finding in particular (~2× at full connectivity, monotonic) should
be re-measured naturalistically before being adopted or dismissed, since it may look
quite different out of saturation.

**Adoption gate, not yet cleared.** `balanced_ei` changes live pallial rate from 0.73 to
0.12, which is a large shift in operating point. Before it becomes a default it needs:
the full ethogram re-run with it on (the reflex arc is separate from `W`, so innate
behaviour is expected to be unaffected — expected, not verified); a throughput check;
and a check on whether any existing plastic result depends on the saturated regime. It
stays off by default until those land.

**Next**, in order: re-run the ethogram and throughput with `balanced_ei=True`; then
re-measure E041's density sweep under the naturalistic probe, since the two interventions
may compose and density was never tested out of saturation.

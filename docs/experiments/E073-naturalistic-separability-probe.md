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

_Not yet run._

## 7. Interpretation

_Pending §6._

## 8. Consequence

_Pending §6._

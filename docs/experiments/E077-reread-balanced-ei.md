# E077 — re-reading `balanced_ei` against the corrected baseline: the 2.13× was my own artefact

> **Diagnostic.** E076 flipped `place_cells_enabled` and `contamination_enabled` to
> `False`. That invalidates the baseline E073 measured on, so E073's conclusions have to
> be re-taken rather than assumed to carry.

## 1. Parent hypothesis

**H2d**, and the `balanced_ei` intervention E072–E074 pursued.

## 2. Question

[E073](E073-naturalistic-separability-probe.md) reported `balanced_ei` giving **2.13×
separability (t=5.75, paired 12 genomes)** under a naturalistic probe, against a null
under the sparse probe — and used that discrepancy to argue H2d's whole measurement
series was taken in an unrepresentative regime.

But that naturalistic probe fed `sensing.observe`, which at the time included E063's
place-cell block at **25.1% of all observation drive, always on** — precisely the
common-mode term `balanced_ei` acts on. The measurement may have been of my own
addition rather than of anything about the pallium.

Does the 2.13× survive with the block off?

## 3. Result

Same script, same 12 genomes, same pairing — only `DEFAULT_COOP`'s corrected flags
differ:

| probe | baseline | `balanced_ei` | paired contrast |
|---|---|---|---|
| sparse (E009 series) | 0.0961 | 0.0867 | −0.0094, t=0.74, null (0.90×) |
| naturalistic, **E073** (place on) | 0.0365 | 0.0776 | **+0.0411, t=5.75, 2.13×** |
| naturalistic, **E077** (place off) | **0.0814** | 0.0856 | **+0.0041, t=0.35, null (1.05×)** |

Supporting numbers: naturalistic stimuli drop from 39 to 14 nonzero channels; mean
pallial rate under the naturalistic probe drops 0.6019 → 0.4602.

## 4. Interpretation

**`balanced_ei` does not improve separability under either probe.** E072's null was
correct. E073's positive was an artefact of a 25%-of-drive common-mode block that I had
added two experiments earlier and that had no business being in a baseline.

**Two of E073's four claims fall, and two stand.**

| E073 claim | verdict |
|---|---|
| `balanced_ei` gives 2.13× under naturalistic input | **falls** — 1.05×, t=0.35 with the block off |
| H2d's severity understated ~2.6× (0.0365 vs 0.0961) | **falls** — 0.0814 vs 0.0961 is 1.2×, not 2.6× |
| The E009 probe under-drives the pallium vs live | **stands** — live 0.6907 (E076) vs sparse 0.2724 |
| E009's saturation was never actually fixed | **stands** — live 0.6907 is still deeply saturated |

The surviving pair is the more important half and is independent of the place block:
E023's gain re-baselining reported mean rate 0.189 on the sparse probe, and live
operation has never left saturation. That remains true and remains unaddressed.

**What this costs.** E074's entire adoption gate is moot — there is nothing to adopt.
`balanced_ei` stays in the codebase as a tested, documented, non-improving intervention,
which is a legitimate thing for it to be.

## 5. Consequence

**This is the third distinct harm traced to one un-opted-in addition.** E063's place
block has now been shown to (i) break H2f's food control, (ii) manufacture a false
positive for `balanced_ei` that survived two follow-up experiments, and (iii) inflate
the apparent gap between probe and live drive. All three were invisible until something
forced a comparison against a baseline without it.

The failure mode is worth naming precisely, because "add an opt-in flag" understates it.
The block was not merely *on*; it was **large, constant, and in the shared observation**,
so it acted like a change to the operating point rather than like an extra input. Any
future channel with those properties deserves the same scepticism regardless of whether
it is opt-in.

**H2d is back where E072 left it**, with the balanced-E/I branch closed:
- E041's density result (~2× at full connectivity, monotonic, no optimum found) remains
  the only intervention with a measured positive effect on H2d's own metric — and it has
  never been re-measured naturalistically, which is now the obvious next test.
- E035's warning stands: on this quantity, with ~6× genome-to-genome spread, only
  paired designs against an uncontaminated baseline are worth anything.

**Correction propagated into E073 in place**, rather than leaving its headline standing
with a footnote elsewhere.

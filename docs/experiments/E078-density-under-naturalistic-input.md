# E078 — does E041's density result survive a naturalistic probe?

> **Pre-registered.** Sections 1–5 written and committed before the run.

## 1. Parent hypothesis

**H2d**. After [E077](E077-reread-balanced-ei.md) closed the balanced-E/I branch,
**E041's density finding is the only intervention with a measured positive effect on
H2d's own metric** — and it has never been measured under anything but the sparse
E009-series probe.

## 2. Question

E041 swept `sensory_pallium_density` on a paired 12-genome design and found separability
falling as density falls (t=4.08–5.37 across four reductions) and *rising* monotonically
as it rises, with **no optimum found up to full connectivity (~2× the default)**. It was
recorded as "promising, checked, not adopted".

Its stated mechanism was: at random-sparse density, **too few pallial units get any
connection to the one or two informative channels at all**.

That mechanism is specific to sparse input. The E009-series probe injects a *single*
channel at amplitude 1.0; a naturalistic observation has 14 nonzero channels. If the
mechanism is right, spreading information across many more channels should make random
sparse sampling far less costly — because a unit that misses one informative channel is
likely to catch another.

Does the density effect survive naturalistic input, and at what size?

## 3. Prediction

**The effect should weaken substantially under naturalistic input**, for the reason
above: E041's mechanism depends on informative channels being rare enough to miss, and
naturalistically they are not. Stated numerically before running: the density ratio
(1.00 vs 0.30) should be **materially below E041's ~2×** on the naturalistic probe, while
reproducing near ~2× on the sparse probe.

**Direction should stay positive** in both. Nothing in the mechanism predicts denser
being actively harmful.

If instead the effect is *undiminished* naturalistically, E041's mechanism is wrong
about *why* density helps, even though the effect itself is real — which would matter,
because the mechanism is what makes "no optimum found up to 1.0" sound safe to act on.

## 4. Falsifier

If density's effect is null or reversed on the naturalistic probe, E041's finding does
not transfer to the regime a hen actually inhabits, and H2d has **no** remaining
intervention with a positive effect — which is a materially worse position than the tree
currently records and should be stated plainly.

## 5. Design

Both probes, same genomes, paired — the design E073 used, so probe-to-probe comparison
is itself paired and cannot be attributed to genome sampling. **Run on E076's corrected
baseline** (`place_cells_enabled=False`, `contamination_enabled=False`), which is the
whole reason this is worth redoing.

- **Densities**: 0.15, 0.30 (default), 0.60, 1.00 — brackets the default both ways and
  reaches the full connectivity E041 found best.
- **Metric**: `pallial_sep` verbatim from E041 (`RMS(hawk − call) / mean|rest|` over
  pallial units), so numbers stay comparable to the E009/E017/E023/E034/E035/E041 series.
- **12 genomes**, paired, per E035's finding that unpaired ratio-of-means on this
  quantity produced both a false positive and a false negative.
- **Reported**: separability and mean pallial rate per density per probe, plus the paired
  contrast of each density against the 0.30 default within each probe.

Mean rate is reported because density changes total input drive — if separability tracks
drive rather than density as such, that shows up here rather than being discovered two
experiments later.

**Command:**
```bash
PYTHONPATH=. .venv/bin/python scratchpad/e078_density_naturalistic.py
```

## 6. Result

12 genomes, paired, threshold t=2.201. Sparse probe has 1 nonzero channel, naturalistic 14.

| probe | density | separability | mean pallial rate |
|---|---|---|---|
| sparse | 0.15 | 0.0656 | 0.2648 |
| sparse | **0.30** | **0.0961** | 0.2724 |
| sparse | 0.60 | 0.1448 | 0.2848 |
| sparse | 1.00 | 0.2035 | 0.3040 |
| naturalistic | 0.15 | 0.0698 | 0.3786 |
| naturalistic | **0.30** | **0.0814** | 0.4602 |
| naturalistic | 0.60 | 0.0729 | 0.5714 |
| naturalistic | 1.00 | 0.0581 | 0.6745 |

Paired against the 0.30 default:

| probe | density | Δ ± SE | t | | ratio |
|---|---|---|---|---|---|
| sparse | 0.15 | −0.0304 ± 0.0060 | 5.06 | significant | 0.68× |
| sparse | 0.60 | +0.0487 ± 0.0109 | 4.46 | significant | **1.51×** |
| sparse | 1.00 | +0.1074 ± 0.0264 | 4.07 | significant | **2.12×** |
| naturalistic | 0.15 | −0.0116 ± 0.0064 | 1.83 | null | 0.86× |
| naturalistic | 0.60 | −0.0085 ± 0.0032 | 2.70 | significant | **0.90×** |
| naturalistic | 1.00 | −0.0234 ± 0.0044 | 5.37 | significant | **0.71×** |

## 7. Interpretation

**E041 reproduces exactly on its own probe — 2.12× at full connectivity against E041's
~2×.** That is worth stating first: this is not a failure to replicate, and the sparse
measurement was sound. It is a failure to *transfer*.

**Naturalistically the effect reverses.** Denser connectivity is significantly *worse*,
and my pre-registered prediction was wrong on direction — §3 said "direction should stay
positive in both" and "nothing in the mechanism predicts denser being actively harmful".
It is harmful.

**The mechanism is saturation, and the mean-rate column shows it.** Under sparse input,
raising density from 0.30 to 1.00 barely moves drive (0.2724 → 0.3040), so extra
connections buy more units a view of the one informative channel at no cost. Under
naturalistic input the same change drives the network from 0.4602 to **0.6745** — into
the compressive region, where differences get squashed faster than extra connections
reveal them.

That ties directly to the one surviving E073 finding: live operation runs at 0.6907.
Density 1.00 naturalistically lands at 0.6745 — i.e. **E041's recommended direction moves
the network toward exactly the saturated regime where separability is worst.**

**There is an optimum naturalistically, and the current default is at it.** E041 reported
"no optimum found all the way to full connectivity". Under naturalistic input separability
peaks at 0.30 — the shipped default — falling significantly above it and non-significantly
below (0.15 is 0.86×, t=1.83). That is a reassuring result about the current
configuration and a direct contradiction of "more connections is simply better,
everywhere tested".

**The falsifier fires, and §4 committed to saying so plainly: H2d now has no intervention
with a positive effect under naturalistic input.** The full list, all paired and all on
this metric: E/I identity fix null (E023), modality segregation null (E035),
`balanced_ei` null (E077), density **reversed** (here). H2d's position is materially
worse than the tree recorded.

## 8. Consequence

**E041's density finding is re-scoped, not withdrawn.** It is correct for sparse,
artificial stimuli and does not transfer to the input statistics a hen actually receives.
It should not be adopted — the tree's "promising, checked, not adopted" now reads
"checked twice, and adopting it would make things worse."

**`sensory_pallium_density=0.30` is validated as roughly optimal** for the regime that
matters, which is more than it had before.

**Saturation is the binding constraint, and it is now the only live lead.** Every
intervention tried has either not touched drive (density, segregation, E/I identity) or
touched it without helping (`balanced_ei`, which cut live rate 0.73 → 0.12 and gave
1.05×, null). That pair is the puzzle worth attacking: reducing drive alone does not
help, and increasing it actively hurts.

**Concrete next test, and it is cheap.** E023 swept recurrent `gain` and set the default
from that sweep — **on the sparse probe**, where the network sits at 0.27 and is not
saturated at all. Nobody has swept gain under naturalistic input. Given that separability
peaks where drive is moderate, and that both too-dense and too-saturated hurt, the gain
default may be badly placed for live operation in the same way the density
recommendation was. That is the same class of error this session has now found four
times, and it is one sweep away from being answered.

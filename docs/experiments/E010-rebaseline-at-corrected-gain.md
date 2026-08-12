# E010 — re-baselining: does H2 survive a non-saturated network?

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

_Pending._

## 7. Interpretation

_Pending._

## 8. Consequence

_Pending._

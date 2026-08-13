# E023 — inhibition in the pallium, and the third gain re-baseline

> **Re-baselining, recorded as an experiment because it invalidates every prior number.**
> The fix follows E022 3a. The gain sweep was pre-declared in the sense that the
> prediction below was written and pushed (commit `fe1f1b6`) before the numbers existed.

## 1. Parent hypothesis

H2d directly; everything else through the connectome.

## 2. The defect

`hen/connectome.py` assigned excitatory/inhibitory identity by **flat index** over a
**region-ordered** array. Regions are contiguous, so the 80% cut landed mid-arcopallium:

| region | before | after |
|---|---|---|
| sensory | 100% exc | 79.7% |
| **pallium** | **100% exc** | **80.1%** |
| hippocampus | 100% exc | 80.0% |
| arcopallium | 20.8% | 79.2% |
| hypothalamus | **0% exc** | 81.2% |
| motor stub | **0% exc** | 79.2% |

The 256-unit recurrent pool that is the entire subject of this project contained **no
inhibitory neurons**. Every drive entered through a wholly inhibitory hypothalamus and
`W_out` read from a wholly inhibitory motor stub.

Now stratified within each region, shuffled from the mask key (which neurons are
interneurons is genetic and shared by the flock, not developmental noise). Verified
deterministic and genome-varying.

## 3. Prediction, written before the sweep

From commit `fe1f1b6`: *"If the saddle-node reading is right the operating point should
now be stable over a much wider band. If the knife edge survives inhibition, the E/I bug
was real but was not the cause of the gain sensitivity, and I was wrong about the most
important claim adopted from the review."*

The review additionally predicted **1.4× better separability** at comparable mean rate.

## 4. Result

6 genomes per gain, same settle-and-separate probe as E009/E017.

| gain | mean pallial rate | genome sd | separability |
|---|---|---|---|
| 0.60 | 0.172 | 0.001 | 3.8% |
| 0.70 | 0.189 | 0.002 | 4.5% |
| 0.85 | 0.228 | 0.004 | 5.8% |
| **0.95** | **0.276** | **0.009** | **7.4%** |
| 1.00 | 0.320 | 0.016 | 9.4% |
| 1.05 | 0.415 | 0.041 | 9.4% |
| 1.10 | 0.603 | 0.061 | 4.5% |
| 1.40 | 0.916 | 0.008 | 1.0% |

Against the old, purely-excitatory network: 0.60 → rate 0.212/sep 3.3%; 0.70 →
0.271/7.5%; 0.75 → 0.349/**14.2%**; 0.78 → 0.497/6.2%; 0.90 → 0.830/0.9%.

**Stability: prediction confirmed, emphatically.** The old network bifurcated between
0.75 and 0.78 — mean rate 0.35 → 0.50 — and was dead by 0.90. This one climbs smoothly
from 0.60 to 1.00 with genome spread under 0.02 and holds together past 1.05. The usable
band is about **four times wider**.

**Separability: the review's prediction failed, and so did any hope this fixes H2d.**
The peak is 9.4% against the old peak of 14.2% — *lower*. Usable-to-usable, which is the
fair comparison since E010 explicitly refused the 14.2% for sitting 0.03 from a cliff:
**7.4% at the new default against 7.5% at the old one.** Identical to within noise. The
predicted 1.4× improvement is absent.

## 5. Interpretation

**The fix bought robustness, not representation.** That is a real gain and a smaller one
than the review advertised. A gain parameter that no longer has to be held to two decimal
places is worth having — E010, E011 and E012 were all spent circling a saddle-node that
should not have existed. But **H2d is untouched**: a hen still barely distinguishes a
heard alarm call from a seen hawk, at ~7% of mean rate either way.

Anyone reading the review's "this is the highest-value structural change on the board"
should read it alongside this: the defect was real, the diagnosis of *why the gain was
brittle* was right, and the claimed representational payoff did not appear. Two of three.

**New default 0.95**, chosen by E010's rule rather than by peak-chasing: leave margin,
because learning moves weights. 0.95 sits ~0.12 below the transition where the old 0.70
sat ~0.08 below its own, and its genome spread is 3.3% of mean rate against 10% at 1.05.
Deliberately not 1.00, which is measurably better on separability and closer to the edge
— that is the trade E010 got right and it would be perverse to unlearn it here.

## 6. Consequence

**Every number in the tree measured before this commit is on a different network** — a
brain with no inhibition in its pallium, at a gain chosen for that brain. Specifically
invalidated as *comparable*:

- E004's t=3.93 (already single-block and unreplicated)
- E017's separability figures, including the 17× fan-in loss and the 2.06× Field L result
- E019's three defect measurements and the fix verification
- E020 and E021's pooled H2 null, +0.011 ± 0.012

**None of those results is now known to be wrong.** They are unrepeated on the corrected
connectome, which is a different claim, and the tree should say so rather than quietly
carrying them forward.

**What needs re-running, in order:**

1. **H2, on fresh seeds.** The pooled null is the tree's current headline for H2 and it
   describes a network that no longer exists.
2. **E017's fan-in diagnosis.** The 17× loss was measured on a pallium that could not
   inhibit anything; whether the loss is still fan-in dilution is now open.
3. **E019's audibility check**, which is cheap and guards the project's most expensive
   past mistake.

**Not re-run, deliberately:** E013, E015 and E016, which are already superseded, and
E011, which has no result to re-run.

# E017 — where along the pathway is "hawk" lost against "alarm call"?

> **Diagnostic, not a pre-registered test.** This measures the connectome as already
> built rather than contrasting conditions, so sections 3 and 4 are weaker than the
> template demands. It is recorded because it corrects a stated mechanism in H2d and
> because two of the three things it measured came out against my prediction.

## 1. Parent hypothesis

[H2d](../hypothesis.md#h2d) — the pallium does not form separable representations of
distinct stimuli. H2d is the blocker behind H2b, H2c, and everything from H3 down.

## 2. Question

E009 established *that* the pallial states for "saw a hawk" and "heard an aerial alarm
call" are nearly identical. It attributed this to two causes: saturation, and the two
percepts landing on overlapping random subsets of the sensory stub. Saturation was
fixed in E010 by dropping the gain to 0.70. Separability is still ~6%. So: which
synapse actually loses it?

## 3. Prediction

Stated before running. The stub is 64 units carrying 53 exteroceptive channels at 30%
density, so the two percepts should share a large fraction of their target units, and
**most of the loss should already be present at the sensory stub** — the second half of
H2d's stated mechanism. Whatever survives should then be diluted further by pallial
recurrence mixing it with everything else.

## 4. Falsifier

If the stub separates the two percepts cleanly, H2d's stated mechanism ("overlapping
random subsets of the sensory stub") is wrong and must be rewritten, whatever else is
true about the pallium.

## 5. Design

Two matched percepts of equal magnitude presented alone: aerial channel at 1.0 (hawk
directly overhead) versus the aerial-call audio channel at 1.0 (a flockmate calling at
full amplitude, heard adjacent). Nothing else differs — no vision, no drives, no
somatic input. Held for 2 s of chicken time (200 steps at dt=10 ms) so the network
settles, then read at each stage.

**Primary metric:** RMS difference between the two settled state vectors, divided by
mean activity at that stage under a null (all-zero) observation — so stages of
different size and different operating rate are comparable. Same relative-separability
metric E009's gain sweep used.

**Replicates:** 6 genomes. E009 found separability varies 3.5%–25.5% between genomes at
a fixed gain, so single-seed numbers are meaningless here.

**Ablations**, to localise the loss once found:
- pallial recurrence zeroed (afferents intact)
- auditory afferents routed to their own sixth of the sensory stub, projecting to
  their own disjoint sixth of the pallium — a crude stand-in for Field L, the avian
  primary auditory area, being an anatomically separate pallial target from the visual
  entopallium

Reproduce: `scratchpad/modality_mixing.py`, `where_it_collapses.py`,
`why_pallium_collapses.py` at this commit.

## 6. Result

Separability of "saw hawk" against "heard alarm", as a multiple of mean activity:

| stage | separability |
|---|---|
| sensory stub | **1.055 ± 0.221** |
| pallium | 0.062 ± 0.012 |
| arcopallium | 0.059 ± 0.013 |
| motor stub | 0.045 ± 0.012 |

At the stub the two percepts share only **9.4% ± 2.7%** of their target units, with a
cosine similarity of 0.245 ± 0.127. They are close to orthogonal.

The 17× loss happens in the single sensory → pallium projection. Ablations:

| pallium variant | separability | vs intact |
|---|---|---|
| intact | 0.062 ± 0.012 | 1.00× |
| pallial recurrence removed | 0.049 ± 0.009 | **0.79×** |
| auditory afferents segregated ("Field L") | 0.128 ± 0.028 | **2.06×** |

Separately, the innate arc's response to each percept, read straight off
`reflex_matrix()`:

| motor channel | hawk seen | alarm heard |
|---|---|---|
| crouch | 8.00 | **0.00** |
| call_aerial | 7.00 | **0.00** |
| forward | −6.00 | **0.00** |
| *(all eight others)* | 0.00 | **0.00** |

The auditory channels drive nothing. Every entry in that column is zero by
construction.

## 7. Interpretation

**H2d's stated mechanism is wrong in its second half and must be corrected.** The
sensory stub does not blur the two percepts — it separates them about as well as
anything could, at 1.055, and they barely share units. A random sparse projection into
64 dimensions is a good separator, which in hindsight is what random projections are
for. The claim that they "project onto overlapping random subsets of the sensory stub"
was inferred, not measured, and it is false.

**The loss is the feedforward projection, not the recurrence.** Removing pallial
recurrence entirely makes separability slightly *worse*, not better, so mixing by
recurrent connections is not the cause either — the second thing predicted here and the
second one wrong. What remains is dilution by fan-in: each pallial unit sums ~19 stub
inputs, of which one or two carry the distinction and the rest sit at baseline,
so a clean difference becomes a small perturbation on a large common-mode drive.

**Structured wiring helps, and does not close the gap.** Giving audition its own stub
partition and its own pallial target — which is the arrangement a real bird has, with
auditory input arriving in Field L via nucleus ovoidalis while vision arrives in the
entopallium via nucleus rotundus, two anatomically separate thalamic relays — doubles
separability. That is a real effect and it is in the right direction, but 2.06× applied
to a 17× loss reaches 0.128, still an order of magnitude below what the stub already
had. Anatomical segregation is worth doing and is not on its own the fix.

**The finding that matters most was not what this set out to measure.** There is no
innate auditory reflex at all. `hen/innate.py` wires production of all four calls and
no response to hearing any of them, on the reasoning that comprehension is learned. But
comprehension being learned does not mean it is learned *from nothing*: parentally
naive chicks already show a differential response to conspecific fear calls, and Curio's
mobbing work shows the learned part is second-order conditioning off a call that is
already arousing. The model asks reward-modulated plasticity to discover, by trial and
error, a link that nature supplies as a reflex and then refines. That is a much harder
problem than the one the biology poses, and we set it by accident.

**What this does not license.** It does not show that adding an innate auditory reflex
would produce comprehension — that is a separate experiment with its own falsifier, and
my record on predicting these mechanisms is 1-for-7 after today. It does not measure
anything under learning; all numbers here are at hatch. And the Field L ablation is a
crude probe, not an anatomy: it segregates by fiat rather than by a connectivity prior.

## 8. Consequence

- **H2d rewritten.** The "overlapping subsets of the sensory stub" mechanism is struck;
  the loss is localised to fan-in dilution at the sensory → pallium projection, with
  recurrence explicitly ruled out.
- **New backlog item: an innate auditory reflex arc.** Weak crouch on hearing an aerial
  alarm, weak flee/vigilance on hearing a ground alarm, both well below the visual
  reflex weights so they scaffold rather than solve. This also removes the exploration
  problem E006 and E007 hit — she cannot learn to crouch at a call she has never once
  crouched at.
- **New backlog item: modality-segregated afferents**, with the connectivity prior
  doing the work rather than a hand-cut slice. Measured 2.06× and cheap.
- **Open question, not yet an item:** the plasticity rule is instrumental
  (reward-modulated three-factor), and the mechanism the biology points at is Pavlovian
  (association off an innately arousing stimulus). `W_pred` is closer to the right
  machinery than `W_out` is. Needs its own hypothesis node before anything is built.

# E094 — the `strike_penalty` audit: which recorded conclusions actually move?

*Sections 1–5 written and committed before anything was run.*

---

## 1. Parent hypothesis

Not a single node. This audits **H2, H4 and T1** against a defect confirmed in
[E067](E067-reward-eligibility-sampling-defect.md), and it is the backlog's longest-open
high-priority item — raised at E067 and deferred through roughly twenty-five experiments.

---

## 2. Question

E067 confirmed, via adversarial review and independent re-verification, that `m` — the
factor gating `consolidate()`'s update to the recurrent weights `W` — was **sampled at the
exact consolidation-boundary step rather than traced**. A discrete single-step reward event
therefore reached `consolidate()` on only **~2% of occurrences**.

Two reward terms are discrete and exposed: `struck` (used since ~E014, throughout H2, H4
and T1) and `sick_onset` (T2, E066). `d_drive` is continuous and is a genuine trace —
unaffected.

`CLAUDE.md` records `strike_penalty` as **87.3% of reward variance at the H4
configuration** (hawk every 20 s), against 0.0% at the 900 s default the guard test runs
at, where no hawk arrives at all.

**E067 established the mechanism and explicitly did not adopt any reinterpretation.** Its
own wording: *"whether T1's Pareto-safety finding, any of H4's states, or H2's own clean
nulls actually depended on the discrete strike-event term surviving to a boundary — versus
being adequately explained by the continuous `d_drive` pathway — has not been checked for
any of them."*

**So the question is not whether the defect is real. It is which conclusions move.**

The fix already exists and is a flag: `legacy_m_sampling=True` reproduces the defective
behaviour, `False` uses `m_acc`. Every experiment before E067 ran defective; every one
after ran fixed. The audit is a controlled comparison, not a rebuild.

---

## 3. Prediction

**Part A — screening.** In a configuration where strikes are frequent, fixing the sampling
changes the recurrent weights measurably: `|W|` drift differs by **≥5%** between
`legacy_m_sampling` True and False. If it does not, no downstream conclusion can move and
the audit closes there.

**Part B — the three hypotheses.** Conditional on A:

1. **H4 does not move.** It is `SUPPORTED` but E027 established the effect **survives
   lesioning `W_out` entirely** — it is a result about two hand-set reflex weights, not
   about learning. A defect in the learning rule's credit assignment should not touch a
   conclusion that does not depend on learning.
2. **H2's null does not become an effect.** H2 is a clean null on foraging (E020/E021).
   The defect *suppressed* teaching signal, so fixing it could in principle reveal
   learning — but E068 measured the sickness term at ~0.007% of reinforcement, and the
   strike term is only large at hawk periods H2 does not run at.
3. **T1 is the one most likely to move**, because its finding is about vigilance under
   predation, which is exactly the regime where the discrete strike term dominates.

I hold prediction 1 most firmly and 3 least. Recording that ordering in advance because a
defect that has been outstanding this long invites the conclusion that it must matter.

## 4. Falsifier

**Screening falsifier (Part A).** `|W|` drift differs by <5% between the two settings at
the H4 configuration. Then the defect is real, confirmed, and **inconsequential for every
recorded result** — which is a legitimate and useful outcome, and the audit records it and
closes.

**H4 falsifier.** The H4 contrast changes sign or loses significance under the fix. That
would move a `SUPPORTED` node and require the H4 section rewritten.

**H2 falsifier.** The foraging null becomes a significant effect under the fix, on both a
first and a fresh seed block. One block does not move a status (E021).

**T1 falsifier.** The Pareto-safety finding changes direction.

**Scope falsifier.** Any conclusion moves that I have not listed here. The audit's value is
its completeness, and discovering a fourth affected node means the enumeration was wrong
and needs redoing rather than patching.

---

## 5. Design

### Part A — screening first, because it can close the audit cheaply

`|W|` drift over a fixed run at the **H4 configuration** (`hawk_period_s=20`, where the
strike term carries 87.3% of reward variance), with `legacy_m_sampling` True and False, 8
seeds, matched world keys. This is the configuration where the defect is *most* exposed;
if it does not move the weights here it moves them nowhere.

Reported alongside: the **number of discrete strike events per run** and how many landed
on a consolidation boundary, to confirm E067's ~2% figure in this harness rather than
inheriting it.

Uses the `w_norm` diagnostic added in E090 — mean `|W|` over live synapses. `|W_out|` is
the wrong instrument here for the reason that fix recorded: under `hebbian_readout` it
drifts whether or not a reward arrived.

### Part B — only the hypotheses Part A implicates

Each re-run as its own matched-seed contrast at the configuration its original experiment
used, `legacy_m_sampling` as the only variable. Any that clears must then replicate on a
fresh seed block before a status moves.

### What this audit is not

It is **not** a re-run of every experiment since E014. The defect suppressed a term; the
question is whether any *conclusion* rested on it. A null that was null because nothing was
learned stays null whether or not the teaching signal arrived.

### Cost

Part A ~15 minutes. Part B is open-ended and gated on A.

---

## 6. Result

### Part A — the screening falsifier fires, and the audit closes

8 seeds, 30 simulated minutes, H4 configuration (hawk every 20 s), `legacy_m_sampling`
True against False on matched world keys.

| seed | strike events | on a consolidation boundary | `|W|` legacy | `|W|` fixed | diff |
|---|---|---|---|---|---|
| 0 | 3479 | 70 | −0.005323 | −0.005265 | +1.1% |
| 1 | 3940 | 70 | −0.005316 | −0.005359 | −0.8% |
| 2 | 7514 | 155 | −0.005439 | −0.005426 | +0.2% |
| 3 | 3841 | 77 | −0.005260 | −0.005267 | −0.1% |
| 4 | 2384 | 49 | −0.005235 | −0.005255 | −0.4% |
| 5 | 8748 | 177 | −0.005303 | −0.005314 | −0.2% |
| 6 | 3854 | 78 | −0.005515 | −0.005549 | −0.6% |
| 7 | 1782 | 49 | −0.005260 | −0.005262 | −0.1% |

**The defect is confirmed independently, at E067's rate.** 725 of 35,542 strike events land
on a consolidation boundary — **2.0%**. E067 obtained ~2% by an exhaustive sweep over
timing offsets; this counted events in a running flock. Same quantity, different method,
same answer.

**And fixing it moves the weights by 0.1%.** `|W|` drift: legacy −0.005331, fixed
−0.005337. Paired difference −0.000006 ± 0.000011, **t=−0.52** against t(7)=2.365 — not
significant. The pre-registered screening falsifier (<5%) fires, so **Part B does not
run.**

### 6b. Direction check — added because the pre-registered screen was too weak

*Post-hoc, and it makes the result stronger rather than weaker, which is why it is worth
separating from the screen it supplements.*

`|W|` is a **magnitude**. Two conditions can show identical drift while moving different
synapses in different directions, and the pre-registered screen could not have detected
that. Cosine similarity between the two conditions' per-synapse weight *changes*:

| seed | cos(ΔW legacy, ΔW fixed) | max elementwise diff |
|---|---|---|
| 0 | 0.983614 | 3.60e-01 |
| 1 | 0.982547 | 4.49e-01 |
| 2 | 0.979304 | 2.31e-01 |
| 3 | 0.986432 | 1.94e-01 |
| **mean** | **0.982974** | |

**98.3% aligned — and not 1.0.** The residual is real: individual synapses differ by up to
0.36, which is far larger than the mean drift of 0.005. So the discrete term *does* move
individual synapses noticeably, and those movements **cancel in aggregate**.

## 7. Interpretation

**A term can dominate reward *variance* and still not move the weights.** That is the
finding, and it dissolves an alarm that has stood for twenty-five experiments.
`CLAUDE.md` records `strike_penalty` at 87.3% of reward variance at this configuration, and
the natural inference — that a defect suppressing 98% of it must have distorted everything
downstream — is wrong. `consolidate()` multiplies the modulator by eligibility traces, so a
spike arriving with no particular eligibility pattern produces **undirected perturbation**
that averages out. The direction check shows exactly that shape: large per-synapse
differences, 98.3% aggregate alignment.

**This is E069's finding from the other side.** E069 measured a thousandfold
`sickness_penalty` sweep producing "undirected perturbation, not behaviour" — the signal
reached the weights (`|W−W₀|` rose 26%) and changed nothing. E094 measures the converse:
removing 98% of a discrete signal also changes nothing. Both say the same thing about this
rule — **discrete reward events are not what moves `W`.** The continuous `d_drive` pathway
is, and it is a genuine trace that the defect never touched.

**What this does not establish, stated plainly.** The 1.7% residual is not zero, and no
behavioural re-run was performed — Part B was gated on Part A and Part A did not implicate
it. So the claim is "the weight trajectory is 98.3% identical and its magnitude is
statistically indistinguishable", **not** "no behavioural conclusion could possibly have
differed". Those are different, and the second would need H2, H4 and T1 re-run under the
flag. The judgement that 98.3% alignment does not warrant that expense is a judgement, and
it is recorded as one.

**The confidence ordering in §3 was not tested**, because Part A closed. H4, H2 and T1 are
all unimplicated by the same measurement rather than individually cleared.

## 8. Consequence

**The audit closes: the defect is real, confirmed at E067's rate by an independent method,
and inconsequential for every recorded conclusion.** The backlog's longest-open
high-priority item is resolved after twenty-five experiments — and resolved as a null,
which is the outcome the staging was designed to reach cheaply if it were true.

**`legacy_m_sampling` stays** as a bisection tool. E075 used it; it costs nothing and it is
now the thing that made this audit a fifteen-minute comparison rather than a rebuild.

**Recorded for the next audit of this shape: screen on direction, not just magnitude.** The
pre-registered screen was `|W|` drift, which is a scalar summary that cannot distinguish
"nothing changed" from "the same amount changed elsewhere". It happened to be adequate
here because the direction check agreed. It would not have been adequate if the fix had
redistributed weight, and I would have closed the audit on a number that could not have
told me.

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

*To be filled after the run.*

## 7. Interpretation

*To be filled after the run.*

## 8. Consequence

*To be filled after the run.*

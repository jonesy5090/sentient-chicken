# E099 — correcting E098: the environment *does* supply the asymmetry

**Diagnostic and correction, not a pre-registered experiment.** It exists because E098
propagated a claim into `docs/hypothesis.md` that does not survive independent
re-measurement — mine, of my own merged conclusion.

---

## 1. Parent hypothesis

`docs/hypothesis.md` → **H1a** (the environment creates an information asymmetry the flock
cannot resolve individually), and by consequence **H2f**, **H3** and **H0**.

---

## 2. What E098 claimed, and why it is wrong

E098 §6b reported that of 18.3 hawk events per run, roughly **0.3** occurred while the focal
hen was blind, and concluded:

> *"In ~1.6% of hawk events does a call tell her anything she cannot already see… That is
> the environment failing to instantiate the thing the whole project is about."*

I merged that into the tree. **It does not replicate**, and it should have looked wrong on
its face: H1a records hens head-down **64%** of the time, and a figure of 1.6% cannot sit
beside that.

## 3. Measured, three ways, free-running flock (16 hens, hawk/60 s, 3 seeds × 30 min)

| question | answer |
|---|---|
| blind at the hawk's **onset step**, while a call sounds | **0%** (0 / 73 events) |
| blind at **some point during the dive**, while a call sounds | **90.4%** (66 / 73) |
| fraction of **call-time hen-steps** in which she is blind | **47.0%** |

These are three different questions and E098 answered the first while reporting it as the
third. The first is **near-zero by construction**: at the onset step nobody has called yet,
so "blind *and* a call is sounding" cannot be true. It measures the simultaneity of two
events one of which causes the other.

**The correct statement: in 90% of hawk events there is a moment when she is blind and a
flockmate is calling, and across all call-time she is blind 47% of the time.**

## 4. Interpretation

**The environment supplies the asymmetry.** H1a is not merely `SUPPORTED` on the head-down
gate in isolation — the gate actually produces informative calls at a high rate in a
free-running flock. E098's conclusion that "the environment fails to instantiate the thing
the whole project is about" is **withdrawn**.

**Which makes the learning results harder, not easier.** E098 offered the environment as an
explanation for why no rule has produced a targeted behavioural change. That escape is now
closed: **the rules fail while the contingency is available 47% of the time.** `W_pred` at
+0.0815 and indistinguishable from its own gain-0 control, and `hebbian_readout` at 78% call
relay, are results about the rules, not about a world that never gave them a chance.

**The red-team's related finding is untouched and still stands.** It measured
*hawk-visible-with-no-call* at 0.003% of hen-steps — a different claim, about whether the
H2d/H2c *discrimination* contrast (tell a seen hawk from a heard call) occurs. It does not.
That the asymmetry exists does not mean the two referents are ever separable.

**How this happened, since it is the third time in this arc.** I delegated a measurement,
received a number, and propagated it without re-measuring — in an experiment whose own §6a
records me shipping a no-op repair for want of asking one more question. The red-team
skill's central rule is *verify before you act; the reviewer will be wrong about some of
it*, and a subagent is a reviewer. The figure was checkable in four minutes and I checked it
only after merging.

## 5. Consequence

- **E098 §6b's headline and §7's environment conclusion are struck**, with the correct
  figures in place. The rest of E098 — the no-op repair, the completed fix, Part B's
  falsifier verdicts — is unaffected and stands.
- **H1a strengthens**: the head-down gate demonstrably produces informative calls, not just
  blindness.
- **H2f's negative result strengthens**: no environmental excuse remains for it.
- **Standing correction**: a delegated measurement is a claim, not a fact. Re-measure before
  it enters the tree, not after.

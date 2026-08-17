# Handover — session ending 2026-08-17

Written for whoever picks this up next. `CLAUDE.md` is the operating manual and
`docs/hypothesis.md` is the authority on status; this file covers **what changed in this
session, what is mid-flight, and the traps that cost time.**

---

## 1. Where the project actually is

**H4 is `SUPPORTED`, and it is not a result about the brain.** That sentence is the whole
state of play and both halves matter.

- A flock hearing its own present tense is caught **−0.044 ± 0.012 (t=3.60)** less often
  per hen per hawk than one hearing identical calls shifted in time. 36 seeds, three
  independent blocks, threshold declared and committed *before* the third block ran
  ([E030](experiments/E030-third-block-replication.md)).
- **Lesioning `W_out` entirely changes nothing** (+0.010, t=0.46, twice on fresh seeds).
  H0's subject is *a neural model of a chicken*; the pallium contributes nothing
  measurable. **H0 is not satisfied and cannot be until a learning rule works.**

**H2's null has lost its standing.** [E031](experiments/E031-the-credit-window-is-not-the-blocker.md)
found an untrained `W_out` at 0×, 1× and 10× gain all indistinguishable on `fed %`
(t=0.68, t=0.54) on a metric that detects a halved peck reflex at t=4.32. So "the rule
doesn't learn" and "the route from the rule to the metric is inert" are not currently
distinguishable — see **H2e**.

**The credit window is dead.** It was the leading explanation for H2 since E022 and was
top of the queue. Measured: a hen feeds every **0.3 s**, reward moves every step, and two
thirds of the peak peck–reward correlation sits at **lag 0**, inside the rule's 0.2 s
window. **Do not sweep `tau_slow`** — it would fix a problem that does not exist.

---

## 2. Mid-flight: E032, the causal efficacy test

**Pre-registered in [`docs/experiments/E032-causal-efficacy.md`](experiments/E032-causal-efficacy.md),
committed before the run. Read it before touching anything.**

This is the experiment that decides H2e, and it is `docs/backlog.md` §5's causal-efficacy
check, unrun for 31 experiments. Rear a flock, then fork: both test branches continue from
the **identical** end-of-rearing state, so lesioning `W_out` is the only difference.

**Primary quantity, fixed in advance:**

```
(trained: intact − lesioned)  −  (fixed: intact − lesioned)
```

The fixed pair prices the lesion itself; anything above it is what learning bought.
Threshold **t = 2.201** at 11 df.

- **H2e predicts ~0.** Lesioning a trained readout costs no more than a random one.
- **A significantly negative interaction falsifies H2e** and hands H2's null back to being
  a fact about the learning rule. That is the outcome worth hoping for.

### To resume

```bash
python -m scratchpad.e032 --seeds 12 --rear 20 --test 5 --budget 100000
```

Per-cell results are cached in `scratchpad/e032_cache.json`; completed cells are skipped,
so it resumes safely after any kill. **Roughly 6 minutes per cell** — plastic rearing
consolidates the full `(H,N,N)` weight tensor every 50 steps, so it is far slower than the
fixed-weight benchmark suggests.

### State at handover

**11 of 24 cells**, all `trained`. **None of the 12 `fixed` control cells has run**, and
without them there is no test — "lesioning a trained readout costs nothing" is
uninterpretable unless you also know what lesioning an untrained one costs.

Interim, 10 trained cells — **not a result**, and must not be reported as one:

```
manipulation check |dW_out|/|W_out| = 0.092   (gate >0.05: PASSES)
intact   fed %  13.911
lesioned fed %  13.798
drop            +0.113 +/- 0.116   t=0.97
per-seed        [+0.15 -0.23 -0.17 +0.46 -0.32 -0.12 +0.19 +0.90 +0.25 +0.02]
```

The manipulation check passing matters: this is **not** E001's frozen readout. Training
genuinely moves `W_out` by ~9%.

---

## 3. The environment trap that shaped this session — and should not affect you

This session ran in a **cloud container that is reclaimed whenever the session goes idle
between turns.** It killed three long runs (E029 twice, E032 once). Everything grew a
per-cell JSON cache as a result, and long runs had to be driven in foreground chunks
inside a single turn.

**In a local container none of this applies** — just run to completion. But keep the
caches; they cost nothing and they are why no work was lost.

Two related traps worth not repeating:

- **`pgrep -f <name>` from inside an `sh -c` whose own command line contains `<name>`
  matches itself.** This produced three false "still running" reports, and later a watcher
  loop that could never terminate. Wait on a **PID** (`kill -0 $PID`), not a name.
- **`nohup ... &` did not survive the launching tool call**; `setsid nohup ... &` did.

---

## 4. Traps specific to this codebase, learned the hard way

**Test the instrument before the hypothesis.** This is `CLAUDE.md`'s central rule and it
earned its place four more times this session:

| what was believed | what was true |
|---|---|
| the metric's denominator can't move (stated in 4 files) | it moved up to **63%** across conditions |
| the headline effect was −0.198 | **−0.044** once the denominator was fixed and estimators matched |
| the reward is drive-reduction | **87.3%** of its variance was "was I just caught" |
| the credit window blocks H2 | reward arrives every **0.3 s**; lag-0 correlation is 2/3 of peak |

**Two guards now exist because parametrising a config was not enough.** A 30 s window at
`hawk_period_s=20` contains *no strike*, so the reward guard would have passed vacuously
at the very configuration it was re-pointed at. It now **asserts strikes actually
occurred** before checking the share. Apply the same suspicion to any new guard.

**`scaffold_gain` exists for positive controls** (`innate.reflex_matrix` →
`connectome.build` → `Condition`). Gain 1.0 is the hen; anything higher is a deliberately
exaggerated bird for testing whether a measurement can see anything. **Never let it carry
a biological claim.** Before reporting any null on a metric, plant an effect and check the
metric finds it — E029 is the template.

**`Lx lesioned` is a permanent rung in `run/h4.py`.** If a result survives zeroing
`W_out`, the ladder now says so on every run instead of waiting for an outside reviewer.

**The Dale fix broke the flock on the first attempt.** Signing a rectified draw gave every
motor channel a positive DC bias (the stub is 80% excitatory) and hens crouched
permanently. The shipped version balances inhibition against excitation *per motor
channel* so the untrained readout contributes exactly zero DC. If you touch `W_out`
initialisation, re-check that.

---

## 5. What to do next, in order

1. **Finish E032.** 13 cells, ~80 minutes. It is pre-registered; do not change the design
   or the seed count to make it cheaper. If compute forces a change, record it as a
   declared deviation.
2. **If H2e survives** (interaction ~0): the pathway is inert whether trained or not, H2
   is a fact about the *architecture*, and the answer is **E007's unresolved question** —
   multiplicative gating of the reflex arc instead of additive competition into one motor
   drive. That is a core architectural change and needs its own node.
3. **If H2e is falsified** (trained flocks lose more): the pathway works once structured,
   H2's null returns to being about the rule, and the attribution ladder (four things
   changed between E013 and E020) becomes worth running.
4. **Owed regardless:** E025 still has no experiment file — the oldest process debt.
   E004's t=3.93 and E016's staging result are both single-block and both still cited.

**Do not start T2** (the poisoned feeder). It needs the signal to carry *which* feeder,
and production is hardwired to stimulus classes. It is further away than the tree implies,
and `docs/ethics.md` now says it gets its own review before it is built, because it is the
first design here whose purpose is to make hens worse off.

---

## 6. Housekeeping

- **Branch:** `claude/chicken-neural-language-lmtpdi`. PRs #1 and #2 are merged; a merged
  PR cannot take new work, so restart the branch from master after each merge
  (`git fetch origin master && git checkout -B <branch> origin/master`).
- **Tests:** 53 pass. `python -m pytest tests/ -q` takes ~7 minutes.
- **Assays:** `python -m run.probes` — 7/7, and the fastest sanity check after any change
  to the connectome or reflex arc.
- **Throughput:** `python -m bench.envelope --sweep`. The README's 33.6× was measured on
  different hardware; this container managed ~19–21×. Treat a drop as a defect, but
  measure the *same commit* on the *same machine* before believing one.
- **The two skills matter.** `/status` answers where the project is from the files rather
  than from memory. `/red-team` buys an outside reader and has now overturned a headline
  result twice — it is the highest-value thing in the repo when work feels stuck.

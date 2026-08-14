# E008 — does top-down association produce comprehension?

> **Pre-registered**: the prediction and falsifiers were written into H2c in
> `docs/hypothesis.md` and committed *before* the architecture was implemented.

## 1. Parent hypothesis

**H2c** — a learned cue can recruit an innate response via top-down association.

## 2. Question

E007 established that the learned pathway cannot supply the +2.50 of motor drive
needed to initiate crouching — it manages 0.002. H2c proposes routing learning to the
*sensory representation* instead: hearing an alarm reconstructs the percept of a hawk,
and the innate reflex (weight 8.0) does the rest, which needs only ~0.3 in sensory
units rather than 2.50 in motor units.

## 3. Prediction (from H2c, committed before implementation)

Comprehension rises above zero with the top-down projection and stays at zero without
it. The no-cue baseline must not rise — a pathway that writes into a hen's own senses
can make her perceive things that are not there, and that is the failure mode.

## 4. Result

30 min rearing, single seed, exploration on:

| condition | comprehension | crouch, no cue | \|W_pred\| |
|---|---|---|---|
| additive only (E007) | −0.0000 | 0.0778 | 0.0000 |
| top-down association | −0.0006 | 0.0704 | **0.0500** (at cap) |

**Comprehension did not emerge.** The projection is learning — `|W_pred|` saturates
its per-synapse cap — so the machinery runs. It simply does not learn the association
we need. The no-cue baseline did not rise, so at least the hallucination failure mode
did not occur.

## 5. Diagnosis — the predictor is autoencoding, not predicting

The rule as implemented is

```
predicted(t) = W_pred @ rate(t),   trained toward obs(t)
```

That is an **autoencoder**: it learns to reconstruct the current observation from the
current brain state. And the current brain state, during a hawk event, is dominated by
the hawk percept itself. So the association it forms is *"when the brain is in
hawk-state, predict hawk"* — circular, and useless for recovering the percept from a
call.

For Pavlovian association the rule has to map a **cue to a later outcome**:

```
predicted(t) = W_pred @ rate(t - delta),   trained toward obs(t)
```

Then it learns what *precedes* the percept rather than what accompanies it. The
existing `z_slow` trace (tau 0.2 s) provides some lag, but the instantaneous rate
dominates it, so the circular term wins.

There is a second, compounding problem worth noting before the next attempt.
Co-occurrence data is very thin: hawks arrive on a ~900 s schedule and dive for 12 s,
so a 30-minute rearing contains roughly 24 s of hawk presence, of which only the
head-up fraction (~36%) is usable — about **9 seconds of training signal**. Any fix
should be tested at raised predator density first, so that a null means the mechanism
is wrong rather than that the data was absent.

## 6. Consequence

- **H2c stays `NOT STARTED`.** The mechanism has not had a fair test yet: this run
  tested an autoencoder, not an associator, so it says nothing about the hypothesis.
- **Architecture kept**, opt-in behind `pred_enabled=False`. It is wired, tested and
  inert by default, so it changes no existing result.
- **E009**: source the prediction from a *lagged* trace rather than the instantaneous
  rate, and run at raised predator density so the co-occurrence data exists.
- **Standing method note, now earned three times over**: measure the mechanism before
  the behaviour. E002 (is the readout moving?), E007 (what drive is available vs
  needed?) and this diagnosis all took minutes and each settled a question that
  behavioural runs had been circling for hours.
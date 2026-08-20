# E081 — H2d measures distance, not decodability, and the difference is the whole story

> **Diagnostic.** Prompted by asking where T2 actually stands after E080, and tracing
> its blocker back through E070 to H2d.

## 1. Parent hypothesis

**H2d**, and through it **T2**. E080 left H2d with no identified mechanism and six closed
interventions. T2-revised was blocked at [E070](E070-t2-revised-chain-positive-control.md),
whose blocker was pallial separability — i.e. H2d.

## 2. Question

H2d's metric, unchanged since E009, is
`pallial_sep = RMS(hawk − call) / mean|rest|`. That is a **distance**. Every conclusion in
the H2d series rests on it being small (~7–11%).

But distance is not what any downstream mechanism needs. `W_pred` and `W_out` are
**linear readouts**. What they need is **linear decodability** — and two state
distributions can be highly correlated, tiny in RMS distance, and still perfectly
separable by a hyperplane. Nothing in the series has ever measured that.

Separately: E070 planted its place→gakel association as a **matched filter** (copy P's
pallial pattern, normalise). A correlational rule converges to something matched-filter
shaped; a **delta rule converges toward a discriminant**. Those are different directions,
and on correlated data they perform very differently.

## 3. Method

**Decodability of the H2d contrast.** Six genomes. For each, 40 noisy hawk observations
and 40 noisy call observations (σ=0.02 on the observation), settled to pallial states.
Report H2d's own metric, the correlation between mean states, and **held-out accuracy of
a linear readout** trained on half and tested on the other half.

**Place, matched filter vs discriminant.** Four genomes, five grid cells, 24 samples per
place with 0.35 m positional jitter. Two readouts, both linear, both held-out:
matched filter (`mean(P)`) and discriminant (`mean(P) − mean(elsewhere)`). Scored as
balanced classification accuracy for "at P" vs "anywhere else".

Accuracy rather than a projection ratio, deliberately: a discriminant direction is
near-zero-mean, so projections onto it are small numbers of either sign and their ratio
is unstable — the same cancellation artefact that made E071's "0.180" misleading. A first
pass here made exactly that mistake and was discarded.

## 4. Result

**The H2d contrast (hawk vs alarm call):**

| measure | value |
|---|---|
| H2d's `pallial_sep` | 0.1113 |
| correlation between mean states | **0.9928** |
| **held-out linear decoding accuracy** | **98.8%** |

**Place (at P vs anywhere else):**

| readout | held-out accuracy |
|---|---|
| matched filter — *what E070 planted* | **18.8%** |
| discriminant | **84.6%** |

## 5. Interpretation

**The pallium separates these stimuli almost perfectly. H2d has been measuring the wrong
quantity for its entire history.** States that are 0.9928 correlated and 0.11 "separable"
support 98.8% held-out linear decoding. Both facts are true; only the second bears on
whether a readout can use the representation.

**This does not make the H2d series wrong — it makes it about something else.** E009,
E017, E023, E034, E035, E041, E072, E077–E080 all measured `pallial_sep` correctly. What
they measured is the *distance* between mean states, which is small and which no
intervention moves. What none of them measured is whether the distinction is *available*,
and it is.

**E070's blocker was my own plant, not the network.** A matched filter scores **18.8%** —
below chance, i.e. anti-correlated with the thing it was meant to detect — while a
discriminant on the same states scores 84.6%. E070 concluded "the chain does not
compose"; what did not compose was the readout I planted into it. That conclusion, and
the T2-revised pause built on it, do not survive.

**Why this is more than a metric quibble.** It relocates the problem from
*representation* to *rule type*. Correlational rules (Hebbian, covariance — what `W_out`
uses under `hebbian_readout`) converge toward matched-filter-like directions, which are
poor discriminators on highly correlated inputs. Delta rules (what `W_pred` uses) converge
toward discriminants. The information is there; whether a given rule finds it is a
separate question that has never been asked, because H2d was believed to have foreclosed
it.

**Caveats, and they matter.** These are *supervised* discriminants fit with labels the
network does not have. `W_pred`'s delta rule does have a target — the actual observation —
so it is not implausible that it finds a similar direction, but that is an argument, not a
measurement. 84.6% is also well below 98.8%: place is harder than hawk-versus-call, and
jitter degrades it. **Nothing here shows any rule in this codebase actually learns these
discriminants.** It shows the information is available to be learned, which is precisely
what H2d was taken to have ruled out.

## 6. Consequence

**H2d's status needs restating, not deleting.** "The pallium does not form separable
representations of distinct stimuli" is false as written. The accurate statement is: *the
mean pallial states for distinct stimuli are very close in distance, and no structural
intervention moves that — but the stimuli are nonetheless linearly decodable at high
accuracy.* Downstream hypotheses were blocked on the first clause when only the second is
load-bearing.

**T2-revised is unblocked, and this is the answer to "where does this leave T2".** Its
chain was paused at E070 on a result that was an artefact. The next step is E070 re-run
with a discriminative plant — the whole-chain positive control, done correctly. If a
discriminative place→gakel association produces selective avoidance, T2-revised proceeds
to its L vs C? contrast for the first time.

**The instrumental T2 remains dead** (E069: no `sickness_penalty` magnitude produces
learned avoidance). Nothing here revives it. T2's route is the associative one.

**Worth stating plainly**: this is the fifth time this session that a blocking result
turned out to be about the instrument. It is also the second where the instrument was one
I built (E070's plant, after E060/E063's defaults). The measurement that settled it took
minutes and could have been run at any point in the last thirty experiments.

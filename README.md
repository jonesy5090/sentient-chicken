# sentient-chicken

A neural model of a hen in a coop, built to find out what happens when you give the
bird language.

This is **phase 0**: the coop, a measured performance envelope, and a hen that is
convincingly newly hatched — reflexive, stupid, and not yet learning anything.
Nothing here is plastic. That is the point: you cannot tell whether an animal has
learned something without first knowing exactly what it was born with.

## The premise, and its honest form

A real chicken has ~290 million neurons, and **78% of them are in the cerebellum and
optic tectum** — motor control and vision. The telencephalon, the part that thinks,
is only 66 million. So a simplified environment really does let you refuse to pay for
the expensive part of a bird.

That is a budget argument, not a biological one. Synapses are not fungible; you
cannot free up cerebellar tissue and spend it on syntax. What this project actually
does is **replace 224 million neurons of tectum and cerebellum with a 59-dimensional
observation vector and 11 motor channels**, and spend the savings on a pallium.

| structure | real chicken | here |
|---|---|---|
| optic tectum | 42 M neurons | 12 angular bins × 4 classes |
| cerebellum | 182 M neurons | 11 motor scalars |
| telencephalon | 66 M neurons | 400 units of pallium, hippocampus, arcopallium |

## Measured envelope

Developmental wall-clock time, not memory, is the binding constraint: a hen learns
her surroundings over days and her rank over weeks, so a simulation that is not many
times faster than reality is useless.

Measured on a 4-core CPU container (`python -m bench.envelope --sweep`):

```
 neurons   hens      steps/s   real-time     W (MB)   GB/s read    hrs/day
     128     16        8,761       87.6x        1.0         9.2       0.27
     256     16        5,853       58.5x        4.2        24.5       0.41
     512     16        3,360       33.6x       16.8        56.4       0.71
    1024     16          733        7.3x       67.1        49.2       3.28
```

Note the `GB/s read` column plateauing around 50–56. With per-hen weight matrices the
recurrent update is a batched matrix-vector product, which reads H×N×N weights per
step and does one multiply-add per weight read. **It is memory-bandwidth bound, not
FLOP bound**, which is why the affordable neuron count lands in the hundreds rather
than the thousands. A machine with more memory bandwidth buys neurons roughly
linearly; more cores does not.

The default 512-neuron hen runs at **~30× real time**, so one chicken-day takes about
48 minutes and a 30-day rearing runs overnight.

## What a hen is born with

Behaviour comes from two pathways running in parallel to the motor output:

- **reflex arc** — `obs → motor`, fixed, innate, the brainstem/tectal shortcut
- **cortical** — `obs → sensory → pallium → … → motor stub → motor`, plastic from phase 1

At hatch the cortical readout is scaled to near-silence, so the bird is almost
entirely reflexive. The pallium is wired and running; it simply has nothing to say
yet.

Call *production* is innate, and this is not a modelling convenience — Konishi (1963)
deafened day-old chicks and they developed the normal repertoire anyway. Chickens are
vocal non-learners. What is deliberately **not** wired in is the audience effect: real
cockerels alarm- and food-call far more readily when a hen is watching. That is
*usage*, which is learned, so it is left as a prediction for phase 2/3 to reproduce —
and a way for the model to be wrong.

## The head-down gate

The most important line in the codebase is in `coop/sensing.py`:

```python
aerial = aerial * (1.0 - w.head_down)   # the vigilance/foraging trade-off
```

A hen with her beak in the litter cannot see a hawk. Signalling only evolves when the
receiver lacks information the sender has, so without this asymmetry no alarm call
would ever be worth making, however many neurons the bird is given. **The pressure for
language lives in the environment, not the brain.**

It works. From `python -m run.probes`, one hen and one hawk, moments apart:

```
head-down blindness   aerial seen 0.01 pecking vs 0.57 head-up;
                      crouch 0.06 vs 0.85 (116/283 steps)
```

And across an hour of flock time, hens spend **~64% of their time head-down** — blind
to the sky, and dependent on a flockmate who is not.

## Running it

```bash
pip install -r requirements.txt

python -m bench.envelope --sweep            # measure your machine, size the brain
python -m run.probes                        # neonatal ethogram, 7 assays
python -m run.lifetime --minutes 60          # rear a fixed, innate flock
python -m run.lifetime --minutes 60 --plastic  # ...with learning on
python -m run.experiment --minutes 30 --seeds 4  # matched-seed A/B between conditions
python -m pytest tests/ -q
```

## Layout

```
coop/    spec.py       sensory/motor contract -- the interface everything keys off
         world.py      state, reset, dynamics, predators
         sensing.py    world -> 59-dim observation (the tectum analogue)
         actuation.py  11 motor channels -> movement (the cerebellum analogue)
hen/     regions.py    region sizes, time constants, connectivity priors
         connectome.py innate mask, Dale's law, weight init
         neurons.py    continuous-time rate units
         innate.py     the fixed reflex arc: what a chick is born knowing
         brain.py      assembly, reflex + cortical pathways
         plasticity.py three-factor learning, structural growth
run/     simulate.py   the closed loop, world and brain in one compiled scan
         probes.py     neonatal ethogram assays
         lifetime.py   developmental runs
         experiment.py matched-seed contrasts between conditions
bench/   envelope.py   measure the machine, print affordable brain sizes
docs/    hypothesis.md the hypothesis tree -- start here
         backlog.md    proposed work, not started
         ethics.md     moral standing, tripwires, review cadence
         experiments/  one file per experiment, from TEMPLATE.md
```

Two design decisions are load-bearing and expensive to reverse:

**The environment lives inside the same compiled scan as the brain.** A NumPy world
driving a JAX brain would pay a host round-trip every 10 ms and forfeit one to two
orders of magnitude.

**Weights are dense with a boolean mask, not sparse.** At 14% density dense BLAS wins
comfortably, and it makes phase 1's structural growth trivial: growing a connection
is flipping a mask bit and initialising a weight, with no reallocation and no
recompilation.

## Status

**Phase 0 complete**: 7/7 ethogram assays. **Phase 1 learning: positive but not yet
significant.** 32/32 tests.

A hen that learns now regulates herself better than a genome-matched, coop-matched hen
that cannot: mean hunger *falls* across a run (0.321 → 0.295) where the fixed control
*rises* (0.306 → 0.370), with feeding rate up from 5.2% to 6.5% of timesteps. That
took three experiments — a null ([E001](docs/experiments/E001-does-plasticity-help.md)),
a diagnostic that found the cortical readout frozen so nothing the pallium learned
could reach a muscle ([E002](docs/experiments/E002-can-the-pallium-reach-a-muscle.md)),
and a rerun with only that fixed ([E003](docs/experiments/E003-does-the-fixed-readout-rescue-learning.md)).

**It has not cleared significance** — t=2.50 against a 3.18 threshold at four seeds,
p≈0.09. E003 also caught a bug in the analysis rather than the simulation: the harness
had been using a 2-SE threshold, which at n=4 would have manufactured a result.
Replication at 12 seeds is the next step.

An unanticipated finding: **structural growth is the weaker condition in both runs**,
ending with nearly twice the synapses. Continuous rewiring may be destabilising what
is learned, which inverts the naive expectation that more plasticity is better.

Not yet built: spatial memory, social hierarchy, the language channel.

The experiment this is all building toward — whether a flock with a communication
channel can reach a goal a capacity-matched flock without one cannot — is specified
in [`docs/backlog.md`](docs/backlog.md). The standing hypothesis tree is
[`docs/hypothesis.md`](docs/hypothesis.md); every experiment names a parent node
there and every result comes back to change one.

**For a current status**, run the `status` skill (`/status`), which answers three
standing questions from the docs: what we are trying to achieve, what has been built
versus what has actually been *proven*, and what is next. That distinction is the
whole point — this project has thousands of synapses restructuring themselves with no
demonstrated behavioural effect, and a status report that blurs the two is worse than
none.

Whether any of this is morally acceptable — the model is a representation of a living
animal, and the headline experiment poisons some of them on purpose — is argued in
[`docs/ethics.md`](docs/ethics.md), along with six concrete tripwires that would stop
the work.

## Sources

Chicken regional neuron counts: [Frontiers in Neuroanatomy
2022](https://www.frontiersin.org/journals/neuroanatomy/articles/10.3389/fnana.2022.1048261/full) ·
[Olkowicz et al., PNAS 2016](https://www.pnas.org/doi/10.1073/pnas.1517131113).
Vocal non-learning: [Konishi
1963](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1439-0310.1963.tb01156.x).
Referential alarm calls and audience effects: [Evans & Marler
1988](https://pubmed.ncbi.nlm.nih.gov/3396311/) ·
[Anim. Behav. 1983](https://www.sciencedirect.com/science/article/abs/pii/S0003347283711589).
GPU-resident environment precedent:
[XLand-MiniGrid](https://github.com/dunnolab/xland-minigrid).

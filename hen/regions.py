"""Region allocation for the hen's brain.

Sizes mirror real chicken neuroanatomy at roughly 10^-5 scale, with one deliberate
distortion: the tectum and cerebellum analogues are stubs. In a real chicken those
two structures hold 78% of all neurons. Here they are 22% of a very small brain,
because the coop is simple enough that they do not need to be more.

Total is deliberately small. See `bench/envelope.py` -- with per-hen dense weights the
simulation is memory-bandwidth bound, not FLOP bound, so the neuron count that
sustains 50x real time on a CPU is in the hundreds, not the thousands.
"""

from typing import NamedTuple

SENSORY = 0     # tectum analogue: feedforward, mostly fixed
PALLIUM = 1     # nido-/mesopallium: recurrent, the "cortex", where plasticity will live
HIPPOCAMPUS = 2  # birds have a real one; place and spatial memory
ARCOPALLIUM = 3  # fear and valence
HYPOTHALAMUS = 4  # drive nuclei; slow time constants
# The subpallium (E115). Absent from this model until now, and the structure a
# vertebrate uses to SELECT among competing actions -- which is the capability E109,
# E112 and E114 each measured the model as lacking, from three different directions.
#
# These sit BEFORE the motor stub, and that ordering is load-bearing rather than
# aesthetic: `brain.step` reads the readout's presynaptic population as
# `rate(x)[:, -n_motor:]`, the LAST n_motor units. Appending the subpallium after motor
# would have silently pointed the readout at the pallidum. It would also have been
# invisible at the default, where both regions are size 0 -- exactly the class of bug
# that survives because the guard runs at the configuration where it cannot appear.
# Anatomically this is also the right order: the subpallium sits between the forebrain
# and the motor output, which is where it is in the loop.
STRIATUM = 5    # GABAergic, near-silent, bursts; the plastic corticostriatal target
PALLIDUM = 6    # GABAergic and TONICALLY ACTIVE; its inhibition is what gets released
MOTOR = 7       # cerebellum analogue: stub. MUST REMAIN LAST -- see above.

N_REGIONS = 8

REGION_NAMES = (
    "sensory", "pallium", "hippocampus", "arcopallium", "hypothalamus",
    "striatum", "pallidum", "motor",
)


class Regions(NamedTuple):
    sensory: int = 64
    pallium: int = 256
    hippocampus: int = 80
    arcopallium: int = 48
    hypothalamus: int = 16
    # Both default to ZERO, so `total` stays 512 and every result recorded before E115
    # is untouched. `Regions(striatum=64, pallidum=32)` builds the loop.
    striatum: int = 0
    pallidum: int = 0
    motor: int = 48          # keep last: `brain.step` reads `rate(x)[:, -n_motor:]`

    @property
    def sizes(self):
        return (self.sensory, self.pallium, self.hippocampus,
                self.arcopallium, self.hypothalamus,
                self.striatum, self.pallidum, self.motor)

    @property
    def total(self) -> int:
        return sum(self.sizes)

    def bounds(self, region: int):
        """(start, stop) neuron indices for a region."""
        sizes = self.sizes
        start = sum(sizes[:region])
        return start, start + sizes[region]

    def scaled(self, factor: float) -> "Regions":
        """Proportionally resized brain, for sweeps and benchmarking."""
        return Regions(*(max(4, int(round(s * factor))) for s in self.sizes))

    def with_pallium(self, factor: float) -> "Regions":
        """Resize the pallium alone, leaving sensory and motor stubs fixed.

        This is what the H4 capacity ladder needs and `scaled` is not. Scaling the
        whole brain changes the sensory and motor interface widths too, so a
        capacity control would differ from the language condition in how much of the
        world it can see and how finely it can move -- two more things at once, in a
        design whose entire purpose is to vary one.
        """
        return self._replace(pallium=max(4, int(round(self.pallium * factor))))


DEFAULT_REGIONS = Regions()

# Membrane time constants per region, seconds. Hypothalamic drive nuclei integrate
# over seconds rather than milliseconds -- they are the slow internal state that the
# phase 4 language channel will eventually read from, which is what would make any
# emergent signal grounded rather than decorative.
REGION_TAU = (0.05, 0.05, 0.08, 0.05, 2.00,
              0.05,    # striatum
              0.03,    # pallidum -- fast, so tonic inhibition tracks striatal bursts
              0.04)    # motor

# Connection probability from source region (row) to target region (column).
# Roughly follows avian pallial circuitry: sensory drives pallium and arcopallium,
# pallium is densely recurrent and projects to motor, arcopallium gates motor
# directly (the fear shortcut), hypothalamus modulates broadly.
# The last two rows and columns are the subpallial loop (E115): pallium excites
# striatum, striatum inhibits pallidum, pallidum's tonic inhibition of the motor stub is
# what a striatal burst LIFTS. Two inhibitory steps, so selection is disinhibition rather
# than excitation. Striatal collaterals (striatum -> striatum) are inhibitory by
# construction, which gives competition among candidate actions for free.
REGION_CONNECTIVITY = (
    #  sens  pall  hipp  arco  hypo  stri  pall' motor      <- target
    (0.05, 0.30, 0.20, 0.25, 0.05, 0.15, 0.00, 0.10),   # sensory
    (0.02, 0.15, 0.20, 0.15, 0.05, 0.30, 0.00, 0.20),   # pallium
    (0.00, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.05),   # hippocampus
    (0.00, 0.15, 0.05, 0.10, 0.15, 0.00, 0.00, 0.25),   # arcopallium
    (0.00, 0.10, 0.05, 0.20, 0.10, 0.00, 0.00, 0.10),   # hypothalamus
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.40, 0.00),   # striatum
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.40),   # pallidum
    (0.00, 0.05, 0.00, 0.02, 0.02, 0.00, 0.00, 0.15),   # motor
)

# Size of the modality-segregated auditory pathway (the Field L / nucleus ovoidalis
# analogue), as a fraction of the sensory stub and of the pallium respectively. Off by
# default (`connectome.build(modality_segregated=True)` turns it on) -- a minority slice
# of each region is carved out to receive auditory input and project to its own pallial
# target exclusively, instead of the default fully-mixed sensory->pallium projection.
#
# 1/6 matches the crude hand-cut probe this replaces (E017's `why_pallium_collapses.py`,
# re-run on the corrected connectome in E034), kept as the default so results are
# directly comparable: that probe measured 1.45x separability of "saw hawk" vs "heard
# alarm" at this same partition size, real but well short of closing a ~14-17x loss.
AUD_FRACTION = 1 / 6

# Fraction of neurons that are excitatory, **within every region**. Dale's law is
# enforced in connectome.py: a neuron's outgoing weights all share its sign.
#
# The "within every region" is the whole point and it was not true until E022. The
# identity used to be assigned by flat index over a region-ordered array, so the 80%
# cut landed in the middle of the arcopallium and segregated excitation from
# inhibition *by region*: sensory, pallium and hippocampus came out 100% excitatory,
# hypothalamus and the motor stub 100% inhibitory. A 256-unit recurrent pool with no
# inhibition in it is why the gain had to be held to two decimal places.
#
# Real avian pallium is roughly 20-30% GABAergic throughout, so mixing within each
# region is both the fix and the biologically faithful reading.
EXCITATORY_FRACTION = 0.8

# Per-region override of the fraction above. `None` means "use EXCITATORY_FRACTION".
#
# The striatum and the pallidum are **100% inhibitory**, because both are GABAergic.
# This is the same SHAPE as the bug E022 found -- there, E/I was assigned by flat index
# over a region-ordered array, so whole regions came out entirely one sign and a 256-unit
# recurrent pallium ended up with no inhibition in it at all. That was an artefact. This
# is the anatomy, and the difference is that it is written down here as a deliberate
# per-region fact rather than falling out of an indexing accident.
REGION_EXCITATORY = (None, None, None, None, None, 0.0, 0.0, None)

# Resting bias per region. Everything outside the subpallium is -2.0, which is what the
# whole brain used before E115, so the default connectome is unchanged.
#
# The two new values are the circuit's whole point. A striatal cell is near-silent and
# fires in bursts; a pallidal cell fires tonically and its targets live under constant
# suppression. Get these wrong and the loop is a pair of ordinary populations wearing
# anatomical names.
REGION_BIAS = (-2.0, -2.0, -2.0, -2.0, -2.0,
               -4.0,    # striatum: sigmoid(-4.0) = 0.018 at rest
               +1.5,    # pallidum: sigmoid(+1.5) = 0.82 at rest, tonically active
               -2.0)    # motor

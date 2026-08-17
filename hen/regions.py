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
MOTOR = 5       # cerebellum analogue: stub

N_REGIONS = 6

REGION_NAMES = (
    "sensory", "pallium", "hippocampus", "arcopallium", "hypothalamus", "motor",
)


class Regions(NamedTuple):
    sensory: int = 64
    pallium: int = 256
    hippocampus: int = 80
    arcopallium: int = 48
    hypothalamus: int = 16
    motor: int = 48

    @property
    def sizes(self):
        return (self.sensory, self.pallium, self.hippocampus,
                self.arcopallium, self.hypothalamus, self.motor)

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
REGION_TAU = (0.05, 0.05, 0.08, 0.05, 2.00, 0.04)

# Connection probability from source region (row) to target region (column).
# Roughly follows avian pallial circuitry: sensory drives pallium and arcopallium,
# pallium is densely recurrent and projects to motor, arcopallium gates motor
# directly (the fear shortcut), hypothalamus modulates broadly.
REGION_CONNECTIVITY = (
    #  sens  pall  hipp  arco  hypo  motor      <- target
    (0.05, 0.30, 0.20, 0.25, 0.05, 0.10),   # sensory
    (0.02, 0.15, 0.20, 0.15, 0.05, 0.20),   # pallium
    (0.00, 0.20, 0.15, 0.10, 0.05, 0.05),   # hippocampus
    (0.00, 0.15, 0.05, 0.10, 0.15, 0.25),   # arcopallium
    (0.00, 0.10, 0.05, 0.20, 0.10, 0.10),   # hypothalamus
    (0.00, 0.05, 0.00, 0.02, 0.02, 0.15),   # motor
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

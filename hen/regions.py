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

# Fraction of neurons that are excitatory. Dale's law is enforced in connectome.py:
# a neuron's outgoing weights all share its sign.
EXCITATORY_FRACTION = 0.8

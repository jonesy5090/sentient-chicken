"""Shared measurement helpers, with the definitions written down.

This module exists because of E107. A statistic called `direction stability` was
copy-pasted between seven experiment scripts (E100 through E106) and pooled sixteen
hens before taking the mean direction. Every hen has her own `W_out`, so the pooled
number answered "how much do hens differ from each other" while every write-up read it
as "how much does one hen's output vary with her situation". The two answers differ by
0.37: pooled 0.6193, per hen 0.9932, on the same trajectories.

The lesson is not that the formula was hard. It is that a statistic living in seven
copies of a scratch script has no single place to be wrong in, and therefore no single
place to be fixed. Anything a conclusion rests on belongs here, with its axes named.
"""

import numpy as np


def _one(a: np.ndarray) -> float:
    """Direction stability of a single (T, D) trajectory.

    Mean cosine between each step's vector and the trajectory's own mean direction.
    1.0 means one fixed pattern whose magnitude alone varies; lower means the vector
    genuinely points in different directions at different times.
    """
    a = np.asarray(a, dtype=np.float64)
    a = a[np.linalg.norm(a, axis=-1) > 1e-8]
    if len(a) == 0:
        return float("nan")
    m = a.mean(0)
    m /= np.linalg.norm(m) + 1e-12
    return float(((a @ m) / (np.linalg.norm(a, axis=-1) + 1e-12)).mean())


def direction_stability(a: np.ndarray) -> float:
    """Within-hen direction stability, averaged over hens. `a` is (T, H, D).

    **This is the one to use.** Each hen is scored against her own mean direction, so
    the number means "does this hen's output vary with her situation" -- which is what
    every hypothesis in this project is actually about.
    """
    a = np.asarray(a)
    return float(np.mean([_one(a[:, h]) for h in range(a.shape[1])]))


def pooled_direction_stability(a: np.ndarray) -> float:
    """The E100-E106 statistic: hens flattened in with time. **Do not use for a claim
    about an individual brain.**

    Kept so those experiments reproduce, and named so nobody reaches for it by accident.
    For any quantity that is per-hen -- `W_out`'s output above all -- this is dominated
    by `between_hen_alignment` below rather than by anything within a hen.
    """
    a = np.asarray(a)
    return _one(a.reshape(-1, a.shape[-1]))


def between_hen_alignment(a: np.ndarray) -> float:
    """How alike different hens' mean directions are. `a` is (T, H, D).

    The diagnostic that separates the two above. When `pooled_direction_stability`
    tracks this number rather than `direction_stability`, the pooled statistic is
    measuring the flock's homogeneity and not any hen's behaviour.
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.stack([a[:, h].mean(0) for h in range(a.shape[1])])
    m /= np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12
    g = m.mean(0)
    g /= np.linalg.norm(g) + 1e-12
    return float(np.mean(m @ g))


def dc_share(a: np.ndarray) -> float:
    """||mean over time|| / mean ||x||, per hen, averaged. `a` is (T, H, D).

    Note `numpy.linalg.norm`'s second positional argument is `ord`, not `axis`. Passing
    -1 there silently computes a different norm; it produced a reported DC share of
    0.14% in one E106 diagnostic where the true figure was 99.98%.
    """
    a = np.asarray(a, dtype=np.float64)
    out = []
    for h in range(a.shape[1]):
        mh = a[:, h].mean(0)
        out.append(np.linalg.norm(mh)
                   / max(float(np.mean(np.linalg.norm(a[:, h], axis=-1))), 1e-12))
    return float(np.mean(out))

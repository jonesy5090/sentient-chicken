import numpy as np
from hen import regions
r = regions.DEFAULT_REGIONS
n = r.total; n_exc = int(round(regions.EXCITATORY_FRACTION*n))
dale = np.where(np.arange(n) < n_exc, 1.0, -1.0)
print(f"N={n}, n_exc={n_exc} (cut at index {n_exc})")
for i, name in enumerate(regions.REGION_NAMES):
    lo, hi = r.bounds(i)
    e = int(np.sum(dale[lo:hi] > 0)); tot = hi - lo
    print(f"  {name:<14} idx {lo:>3}-{hi:<3}  excitatory {e:>3}/{tot:<3} = {100*e/tot:5.1f}%")

# E040 — does E036's falsifier survive without food depletion?

> **Pre-registered.** Sections 1–5 written and committed before the run starts.

## 1. Parent hypothesis

**H2f** — opened by E036's falsifier firing (learning did not add a contingent audience
effect on top of an innate comprehension scaffold). This audit closes the last
unverified leg of the `food_deplete_rate` audit trail (H2: E037, H2e: E038, H4: E039).

## 2. Question

E036 used 30-minute rearing at `n_hens=16`, longer even than H2's 20-minute harness, with
no `food_deplete_rate` override. `docs/backlog.md` had reasoned this was lower-risk than
E032/E033 because E036's primary metric comes from a short, deterministic staged assay
run *after* rearing rather than from the depleting world directly, and because its own
data already showed stability across a 15× duration range (2-min smoke test comprehension
0.1899 vs. the 30-min full run's 0.1921). That is evidence, not a direct check of the
kind E038/E039 ran. **Does the registered primary (`S+L − S` on the audience effect)
hold at `food_deplete_rate=0`?**

## 3. Prediction

Weak prior toward "yes, holds," for the structural reason already given (metric computed
outside the depleting world) and the existing duration-stability evidence. Registered as
weak, not confident — the same class of argument was tried and found insufficient for
E032/E033.

## 4. Falsifier

If `S+L − S` reverses sign or the manipulation check (comprehension index) moves
substantially from its established ~0.19, that would mean rearing-phase depletion
affects the *learned state* enough to matter even though the assay itself doesn't touch
the depleting world — and H2f's falsifier would need re-examining on a clean run.
Holding (same sign, comparable magnitude) closes the audit trail for good.

## 5. Design

`run/audience.py --scaffold-2x2`, unchanged design, `--food-deplete-rate 0.0`, same 8
seeds (0–7) and 30 minutes as E036 for direct comparability. Added
`--food-deplete-rate` to the script's CLI for this (previously unparametrised).

**Command:**
```bash
python -m run.audience --minutes 30 --seeds 8 --scaffold-2x2 --food-deplete-rate 0.0
```

## 6. Result

8 matched seeds, 16 hens, 30 min rearing, `food_deplete_rate=0`. Wall clock 1096 s.

```
condition                   audience  compreh.  strikes/hen   hunger  synapses
--------------------------------------------------------------------------------
N   (bare, fixed)             +0.003   -0.0001       432.45    0.334     36319
S   (scaffold, fixed)         +0.066    0.1921       384.81    0.321     36319
N+L (bare, learning)          +0.001   -0.0001       595.45    0.310     35722
S+L (scaffold, learning)      +0.061    0.1894       417.77    0.329     35724

PRIMARY: audience effect, S+L - S     -0.005 +/- 0.003 SE  t=1.53  need 2.37  NOT SIGNIFICANT
  (E036's depleted-world result: -0.005 +/- 0.002, t=2.25)

MANIPULATION CHECK: comprehension scaffold 0.1921 (E036: 0.1921 and 0.1894 — identical)
```

**The falsifier does not fire.** The mean is identical to E036's to three decimal places
(−0.005 both), same sign, comparable SE, not significant in either world. The
manipulation check and the innate `S` floor (+0.066, matching E036 and this file's own
earlier smoke test) both reproduce exactly.

## 7. Interpretation

**H2f's result is robust to the confound that broke H2e, joining H4.** Unlike E032/E033
(sign reversed, status changed), E036's finding holds essentially unchanged with
`food_deplete_rate` removed. The structural reasoning in §2 — the primary metric comes
from a short staged assay computed after rearing, not from the depleting world directly
— appears to be the right explanation this time, and now has a direct check behind it
rather than only the duration-stability argument.

## 8. Consequence

- **H2f's status and evidence are unchanged.** No correction needed to E018/E036.
- **The `food_deplete_rate` audit trail is now complete for all four results it was
  raised against**: H2 (E037, confound real, number corrected), H2e (E038, confound
  real, status reverted), H4 (E039, confound present but not consequential), H2f (E040,
  confound present but not consequential). Two affected, two not — checking each was
  the only way to know which.
- **`docs/backlog.md`'s audit item can close.**

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

*Pending — filled in after the run, not before.*

## 7. Interpretation

*Pending §6.*

## 8. Consequence

*Pending §6.*

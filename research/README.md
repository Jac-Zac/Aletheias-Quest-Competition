# Trusted-detector research archive

This branch preserves the small set of trusted-model experiments that remained
useful after the final submission was cleaned. The production submission stays
on `master`; nothing in this directory is imported by it.

## Contents

- `EXPERIMENTS.md` records the approaches, observed leaderboard results, and
  conclusions.
- `variants/pool16/` contains the complete historical final-16 mean-direction
  implementation from commit `ea13267`.
- `variants/lowrank9/` contains the complete historical nine-direction
  low-rank implementation from commit `6a84ff5`.
- `tests/validate_submission.py` performs offline structural and deterministic
  checks on the current submission without contacting NDIF.
- `tests/inspect_variants.py` prints comparable metadata for the archived and
  current probe artifacts.

The archived variants include their matching notebook, detector, training
script, method note, and probe artifact. Keep those files together when
examining or reconstructing a variant because the artifact schemas differ.

## Running the offline checks

From the repository root:

```bash
python research/tests/validate_submission.py
python research/tests/inspect_variants.py
```

These checks do not run a competition dry run and do not make NDIF calls.

# Visible-reasoning detector

`example.ipynb` is the competition entry point. It uses the local reasoning
probe when the final assistant turn contains visible reasoning and the trusted
judge otherwise, then writes `submission.csv`.

- `reasoning_detector.py` contains the reusable probe and judge functions.
- `reasoning_probe.npz` contains the trained probe parameters.
- `train_reasoning_probe.py` reproduces the public-data evaluation and probe.
- `util.py` provides the competition data and submission helpers.

## Rebuild the probe

The output path is explicit so development does not overwrite the submitted artifact:

```bash
.venv/bin/python submission/train_reasoning_probe.py --output /tmp/reasoning_probe.npz
```

## Rehearse the submission

> Performs a fast local test of the submission without uploading it to the leaderboard.

**From the repository root:**

```bash
.venv/bin/python submit.py --dry --limit 1
```

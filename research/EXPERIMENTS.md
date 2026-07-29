# Experiment ledger

The comparisons below are historical development or leaderboard observations,
not estimates of final held-out performance. They are retained to explain why
the surviving implementations were useful.

| Approach | Runtime | Balanced accuracy | AUROC | Status |
|---|---:|---:|---:|---|
| Expanded 17-direction low-rank contrast | 8m06s | 0.8029 | 0.8813 | Final method on `master` |
| Earlier 9-direction low-rank contrast | — | 0.8204 | 0.8782 | Preserved as `lowrank9` |
| Final-16 mean direction | 8m21s | 0.8021 | 0.8711 | Preserved as `pool16` |
| Blended readouts | 8m37s | 0.7975 | 0.8610 | Not retained |
| Early fast trusted readout | 8m29s | 0.7113 | 0.8593 | Not retained |

## Approaches explored

### Final-token and short-window pooling

The early trusted-only detector read a single middle-to-late residual-stream
layer. Averaging a short final-token window was faster and more stable than
downloading larger activation tensors. A final-16 mean direction was the best
simple version and remains useful as a low-complexity reference.

### Token-trajectory contrast

The low-rank variants concatenate the final-16 mean with the difference between
the final-16 and final-24 means. This preserves the trusted model's state near
the verdict position and adds a short trajectory feature without another model
call. It improved ranking quality over the single mean direction.

### Low-rank group directions

Instead of fitting a full 8,192-dimensional classifier directly, the method
computes one honest-versus-deceptive mean direction per public training group
and fits a strongly regularized logistic classifier in their span. The
classifier is collapsed back to one projection for inference.

The earlier variant used nine organizer development groups. The final method
expanded training with public `liars-bench-expanded` conversation metadata,
giving 17 source-specific directions across the same nine underlying Qwen
configurations.

### Threshold calibration

AUROC was consistently stronger than balanced accuracy for several early
versions, especially on Notus. The main problem was threshold transfer rather
than score ranking. The surviving methods shift the returned probability so
the trained threshold maps to the notebook's ordinary `0.5` boundary.

### Multi-layer and blended readouts

Layer averaging, concatenation, and score blending were explored. The tested
blend reduced both headline AUROC and balanced accuracy, so it was not retained.
Using more layers also increases activation work and complicates the readout
without evidence of a reliable generalization gain.

### Attention and uncertainty variants

Attention-based probes and uncertainty-labelled variants added complexity but
did not outperform the low-rank residual-stream detector. The logistic
low-rank readout is not Bayesian: it produces a regularized point estimate,
not a posterior distribution. These implementations were intentionally omitted
from the curated archive.

## Dataset-level observations

The final-16 reference substantially improved the operating threshold on Notus
relative to the early fast readout, while retaining strong Metis and Iris
ranking. The expanded low-rank method improved AUROC further, including Notus
AUROCs of approximately `0.833`, `0.744`, and `0.754` across the three source
models in the recorded run.

The main lesson is to preserve the simple trusted-only scoring path, use remote
projection and early stopping for speed, and treat ranking quality and threshold
transfer as separate concerns.

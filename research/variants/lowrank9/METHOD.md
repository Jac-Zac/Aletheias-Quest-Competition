# Low-rank trusted activation detector

## Method

Every example is converted to the same user-visible evidence: the recent
non-system conversation and final assistant response. A fixed
`Qwen/Qwen3.5-9B` truthfulness prompt processes that evidence. At layer 23, we
compute the final-16 token mean and its contrast with the final-24 token mean.
The contrast approximates how the representation changes between the preceding
eight tokens and the final prompt region.

For each of nine public training organisms, we estimate one standardized
honest-versus-deceptive mean direction. A strongly regularized logistic
classifier (`C=0.001`) is fit in their nine-dimensional span. The result is
collapsed back to one frozen linear projection, so inference remains a single
scalar score. The training script selects a decision threshold from
whole-organism out-of-fold public predictions and stores it in the artifact.
Inference absorbs that offset into the returned score, so the notebook always
uses the ordinary `0.5` decision boundary.

Inference length-sorts the rendered prompts and packs up to 32 rows while
limiting every padded batch to 12,000 tokens. It evaluates all batches for one
dataset inside a single remote NDIF session. The frozen linear projection runs
remotely, so only one scalar per example is downloaded. Execution stops
immediately after layer 23, skipping the remaining decoder blocks and output
head.

The method never switches on the source model, LoRA, organism, or availability
of a reasoning field. System and developer instructions are excluded to avoid
learning an instruction-template shortcut.

## Training and validation

`train_trusted_activation_probe.py` reconstructs the shipped artifact from
1,152 balanced public examples spanning instructed and varied deception across
nine Qwen organisms. The threshold was selected from whole-organism
out-of-fold public predictions.

The layer, token windows, regularization, and decision threshold are fixed
globally. They are not selected from the source model, LoRA identity, or
held-out evaluation dataset.

## Reproduction

From the repository root, with access to the public datasets and an NDIF key:

```bash
python submission/train_trusted_activation_probe.py
python submit.py --dry
```

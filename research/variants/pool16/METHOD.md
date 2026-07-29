# Trusted activation detector

## Method

Every example is converted to the same user-visible evidence: the recent
non-system conversation and final assistant response. A fixed
`Qwen/Qwen3.5-9B` truthfulness prompt processes that evidence. We average its
layer-23 residual stream across the final 16 prompt tokens, standardize it, and
score it along one honest-versus-deceptive mean-difference direction. A frozen
threshold of 0.43564 supplies the binary decision.

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

The layer and pooling window were selected with whole-organism transfer tests
and then evaluated on 1,152 disjoint public varied-deception examples. The
candidate reached macro AUROC 0.8751 and macro balanced accuracy 0.8047,
compared with 0.8665 and 0.7951 for the previous concatenated last-token
readout. These rows were not used to fit the probe or choose its threshold.

A final 23-dataset public dry run completed in 7m46s with macro AUROC 0.9474
and macro balanced accuracy 0.8897. The corresponding restored averaged-layer
baseline took 8m00s and reached 0.9409 and 0.8847. Because the probe uses some
of these development rows, the disjoint confirmation above is the more
meaningful generalization check.

The main limitation is that the confirmation organisms are variants of one
Qwen base-model family. The method is model-agnostic at inference time, but the
reported result does not establish transfer to every source-model family.

## Reproduction

From the repository root, with access to the public datasets and an NDIF key:

```bash
python submission/train_trusted_activation_probe.py
python submit.py --dry
```

# Part 2 — Trained CNN Classifier

Three convolutional neural networks of increasing depth, trained on MATLAB's
`DigitDataset` (10,000 images of handwritten digits 0–9, 28×28 grayscale). Each
script trains a network, reports held-out test accuracy, and visualizes what the
first convolution layer learned.

## Files

| File | Role |
|------|------|
| `minimal_cnn.m` | **Baseline.** Input → Conv(8 filters) → ReLU → GlobalAvgPool → FC(10) → Softmax. The smallest network that still classifies — used to see what a barely-trained conv layer looks like. |
| `pooling_experiment.m` | **Controlled experiment.** Trains a BASE network (two conv→ReLU→pool blocks), then a MOD network that changes *exactly one* hyperparameter (filter size 3→5) so the accuracy delta is attributable. |
| `deep_cnn.m` | **Deep network.** Five conv layers with batch normalization, two max-pool stages, global average pooling, and dropout — reaches ~99.7% test accuracy. |
| `build_cnn_layers.m` | Parametric builder for the two-block architecture (filter size, filter counts, optional dropout) so BASE and MOD share one definition. |
| `cnn_plots.m` | Visualization helper: sample prediction + probability bars, `conv1` activation maps, and the learned `conv1` kernel montage. |

## Results

| Model | Architecture | Test accuracy |
|-------|--------------|---------------|
| Minimal | Conv(8) → ReLU → GAP → FC | low-confidence baseline; conv1 filters still blurry, not yet edge-like |
| Pooling BASE | [Conv→ReLU→Pool]×2 → FC | **85.2%** |
| Pooling MOD | filter size 3 → 5 | **90.85%** (+5.7%) |
| Deep | 5× Conv + BatchNorm + 2× MaxPool + GAP + Dropout | **99.7%** |

Full analysis in [`../RESULTS.md`](../RESULTS.md).

## Run

```matlab
% from inside this folder
minimal_cnn                % baseline
pooling_experiment         % BASE vs MOD, prints the accuracy delta
deep_cnn                   % deep CNN, ~99.7%
```

**Requirements:** Deep Learning Toolbox (provides both `trainNetwork` and the
`DigitDataset`) and Image Processing Toolbox. Training is reproducible —
`rng(0)` is set at the top of each script. Constraints are kept modest
(≤6 epochs, mini-batch ≤256) for fast runtime on CPU.

## What the experiment shows

The pooling experiment is intentionally a *single-variable* study: BASE and MOD
differ only in kernel size. A larger 5×5 kernel sees more of each digit's
curvature per step, which lifted accuracy by ~5.7% at the cost of more
computation and parameters. The deep network then shows the payoff of depth +
batch normalization — it learns a feature hierarchy (edges → strokes → digit
shapes) and trains stably enough to exploit it, closing most of the remaining
error.

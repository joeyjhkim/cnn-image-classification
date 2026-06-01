# Results & Analysis

Author: Jaehyeok (Joey) Kim

This document consolidates the problem each part solves, the approach, and the
measured results. Accuracy figures are on a held-out 20% test split of MATLAB's
`DigitDataset` (10,000 grayscale 28×28 handwritten digits, 0–9).

> Figures (training curves, conv1 filter montages, activation maps, prediction
> charts) are produced by the scripts at runtime. To regenerate them, run the
> scripts in [`part2-cnn-classifier/`](part2-cnn-classifier) — each ends with a
> `cnn_plots(...)` call that renders the prediction, activation maps, and learned
> filters. Drop the saved images into a `figures/` folder and they can be linked
> here.

---

## Part 1 — Convolution from scratch

**Problem.** Implement 2D convolution by hand and use it to build the classical
image filters, with no reliance on built-in convolution for the core operation.

**Approach.**
- `correlation2D_BW` performs zero-padded 2D correlation as an explicit
  sliding-window sum (works in `double` to avoid `uint8` overflow).
- `correlation2D_Color` applies it independently to each RGB channel.
- `makeGaussianKernel` builds a normalized `k×k` Gaussian kernel.
- `filters_and_edges` drives every demo: manual correlation blur, the basic
  3×3 filters (identity, box blur, sharpen), edge detection (Prewitt, Sobel,
  Roberts, Laplacian), multi-scale Gaussian smoothing, and a low-pass/high-pass
  hybrid image.

**Result (qualitative).** The gradient operators recover clean edges; the
Gaussian kernel produces smooth multi-scale blur; the hybrid image reads as one
face up close and the other from a distance. The takeaway that motivates Part 2:
a hand-picked kernel (e.g. the Sobel matrix) and a learned CNN kernel are the
*same operation* — the only difference is who chooses the weights.

---

## Part 2 — Trained CNN classifier

### Minimal CNN (`minimal_cnn.m`)

**Problem.** Train the smallest CNN that still classifies, to see what a
barely-trained convolution layer looks like.

**Architecture.** `Input → Conv(8) → ReLU → GlobalAvgPool → FC(10) → Softmax`.

**Result.** A working but low-confidence baseline — predicted-probability bars
are spread across several digits rather than spiking on one, and the conv1
filters are still blurry and smooth rather than sharp edge/corner detectors.
This is the expected starting point before depth and training do their work.

### Pooling + controlled experiment (`pooling_experiment.m`)

**Problem.** Add pooling and a second convolution block, then change *exactly
one* hyperparameter so any accuracy change is attributable rather than
coincidental.

**Architecture.** `Input → [Conv → ReLU → Pool] × 2 → FC(10) → Softmax`, built
parametrically by `build_cnn_layers` so BASE and MOD come from one definition.

**Controlled change.** Convolution kernel size 3×3 → 5×5, everything else fixed.

| Network | Kernel | Test accuracy |
|---------|--------|---------------|
| BASE | 3×3 | **85.2%** |
| MOD | 5×5 | **90.85%** (+5.7%) |

**Why it improved.** A 5×5 kernel has a larger receptive field per step, so it
captures whole curves of a digit instead of only short edges — which suits the
rounded shapes of handwritten numbers. Training was slightly slower and the
larger kernel adds parameters (more capacity, higher overfitting risk), but
here the extra capacity paid off.

### Deep CNN (`deep_cnn.m`)

**Problem.** Design a deeper network and see how far accuracy can go.

**Architecture.**
`Input → [Conv→BN→ReLU]×2 → Pool → [Conv→BN→ReLU]×2 → Pool → Conv→BN→ReLU →
GlobalAvgPool → Dropout(0.3) → FC(10) → Softmax`
(five convolution layers, batch normalization throughout, two max-pool stages).

**Result.** **99.7%** test accuracy.

**What helped most.**
1. **More convolution layers** — the network learns a feature hierarchy, from
   simple edges to strokes to whole digit shapes.
2. **Batch normalization** — keeps activations stable during training, so the
   deeper network trains faster and more consistently.

**Tradeoff.** Longer training time than the shallower models, in exchange for
markedly higher accuracy; the larger model also carries more overfitting risk,
which the dropout layer and batch normalization help control.

---

## Summary

| Model | Architecture | Test accuracy |
|-------|--------------|---------------|
| Minimal | Conv(8) → ReLU → GAP → FC | low-confidence baseline |
| Pooling BASE | [Conv→ReLU→Pool]×2 → FC | 85.2% |
| Pooling MOD | same, 5×5 kernel | 90.85% |
| Deep | 5× Conv + BatchNorm + 2× MaxPool + GAP + Dropout | 99.7% |

The progression shows the two levers that mattered most: a larger receptive
field (kernel size) and depth + batch normalization.

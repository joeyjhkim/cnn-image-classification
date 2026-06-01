# CNN Image Classification — From Manual Convolution to a Trained Deep Network

A two-part MATLAB project that builds up convolutional neural networks from
first principles. **Part 1** implements 2D convolution, Gaussian smoothing, and
classical edge detectors *by hand* — the exact operation a CNN learns to apply.
**Part 2** then trains real CNNs (MATLAB Deep Learning Toolbox) on a
handwritten-digit dataset, progressing from a single-filter classifier to a deep
batch-normalized network that reaches **99.7% test accuracy**.

By Jaehyeok (Joey) Kim.

---

## The arc

| Stage | What it does | Key idea |
|-------|--------------|----------|
| **Part 1 — Convolution from scratch** | Hand-coded 2D correlation with zero-padding, normalized Gaussian kernels, and classical filters (identity, box blur, sharpen, Sobel / Prewitt / Roberts / Laplacian edge detection), plus a hybrid-image demo. | A convolution is a small kernel slid across an image. CNNs *learn* these kernels instead of hand-designing them. |
| **Part 2 — Trained CNN classifier** | Three CNNs of increasing depth trained on the 28×28 `DigitDataset`, with visualization of predictions, conv1 activation maps, and learned filters. | Stacking conv → ReLU → pool blocks and letting backprop learn the kernels. |

The point of putting these side by side: Part 1's `correlation2D_BW.m` and the
`conv1` layer in Part 2 do the *same math*. The only difference is who chooses
the kernel weights — you, or gradient descent.

---

## Results

**Part 1 — classical filters** (qualitative): Sobel/Prewitt/Roberts recover
gradient edges; the Gaussian kernel produces clean multi-scale blur; the
low-pass/high-pass split produces a Monroe↔Einstein hybrid image.

**Part 2 — learned CNNs** (test accuracy on held-out 20% split):

| Model | Architecture | Test accuracy |
|-------|--------------|---------------|
| Minimal | Input → Conv(8) → ReLU → GlobalAvgPool → FC → Softmax | baseline — diffuse, low-confidence predictions; conv1 filters still blurry |
| Pooling BASE | Input → [Conv→ReLU→Pool]×2 → FC | **85.2%** |
| Pooling MOD | same, filter size 3 → 5 (single controlled change) | **90.85%** (+5.7%) |
| Deep | 5 conv layers + BatchNorm + ReLU + 2× MaxPool + GlobalAvgPool + Dropout | **99.7%** |

The pooling experiment is deliberately *controlled* — exactly one hyperparameter
changes (kernel size) so the accuracy delta is attributable. The deep network
shows what depth + batch normalization buys you. Full write-up in
[`RESULTS.md`](RESULTS.md).

---

## Repository layout

```
cnn-image-classification/
├── RESULTS.md                        # consolidated results & analysis
├── part1-convolution-from-scratch/   # manual convolution & filters
│   ├── correlation2D_BW.m            #   hand-coded 2D correlation (grayscale)
│   ├── correlation2D_Color.m         #   per-channel wrapper for color images
│   ├── makeGaussianKernel.m          #   normalized k×k Gaussian kernel
│   ├── filters_and_edges.m           #   driver: blur, sharpen, edge detect, hybrid
│   └── images/                       #   marilyn.bmp, einstein.bmp (hybrid demo)
│
└── part2-cnn-classifier/             # trained CNNs
    ├── minimal_cnn.m                 #   single-conv baseline classifier
    ├── pooling_experiment.m          #   BASE vs MOD controlled experiment
    ├── deep_cnn.m                    #   deep batch-normalized CNN (99.7%)
    ├── build_cnn_layers.m            #   parametric layer-array builder
    └── cnn_plots.m                   #   prediction / activation / filter viz
```

---

## Running it

**Requirements:** MATLAB (R2021b+). Part 2 additionally needs the
**Deep Learning Toolbox** and **Image Processing Toolbox**. The digit dataset
(`DigitDataset`) ships with the Deep Learning Toolbox — no external download.
The Part 2 scripts use the classic `trainNetwork`/`classificationLayer` API,
which still runs on current releases.

```matlab
% Part 1 — run from inside part1-convolution-from-scratch/
filters_and_edges          % produces blur / sharpen / edge / hybrid figures

% Part 2 — run from inside part2-cnn-classifier/
minimal_cnn                % trains the baseline, prints test accuracy
pooling_experiment         % trains BASE + MOD, prints accuracy delta
deep_cnn                   % trains the deep CNN, prints ~99.7% accuracy
```

Each Part 2 script opens MATLAB's training-progress plot, then renders the
sample prediction, the `conv1` activation maps, and the learned `conv1` filters.

---

## What I took away

- A convolution layer is not magic — it is the `correlation2D_BW.m` loop with
  the kernel weights treated as trainable parameters.
- **Controlled experiments matter:** changing only the filter size (3→5) and
  holding everything else fixed is what makes the +5.7% improvement
  interpretable rather than a coincidence.
- **Depth + batch normalization** is what moved the network from ~85% to 99.7%:
  more conv layers learn a hierarchy (edges → strokes → digit shapes), and
  batch norm keeps training stable enough to actually use that depth.

# Part 1 — Convolution From Scratch

Hand-coded 2D convolution and the classical image filters that convolution
makes possible. No `conv2`, no toolbox shortcuts for the core operation — the
correlation is a literal double loop so the math is visible.

## Files

| File | Role |
|------|------|
| `correlation2D_BW.m` | Core operation: 2D correlation on a single channel with zero-padding, implemented as an explicit sliding-window sum. |
| `correlation2D_Color.m` | Applies `correlation2D_BW` independently to each RGB channel. |
| `makeGaussianKernel.m` | Builds a normalized `size × size` Gaussian kernel (sums to 1) for a given σ. |
| `filters_and_edges.m` | Driver script — runs every demo below. |
| `images/` | `marilyn.bmp`, `einstein.bmp` for the hybrid-image demo. |

## What the driver demonstrates

1. **Manual correlation** — blur a color image using the hand-coded kernel loop.
2. **Basic 3×3 filters** — identity, box blur, and sharpen.
3. **Edge detection** — Prewitt, Sobel, and Roberts cross gradients (X, Y, and magnitude), plus the Laplacian.
4. **Gaussian smoothing** — multi-scale blur (σ = 1, 2, 5) from `makeGaussianKernel`.
5. **Hybrid image (extra credit)** — low-pass one image, high-pass another, sum them: the result reads as one face up close and the other from a distance.

## Run

```matlab
% from inside this folder, so the relative image paths resolve
filters_and_edges
```

Uses MATLAB's built-in sample images (`peppers`, `coins`, `cameraman`,
`sherlock`) where possible; only the hybrid-image inputs are bundled in
`images/`.

## Why it's here

`correlation2D_BW.m` is the same operation a CNN `convolution2dLayer` performs
in the forward pass — the difference is that here *I* pick the kernel
(e.g. the Sobel matrix), whereas in [Part 2](../part2-cnn-classifier) the kernel
weights are learned by gradient descent.

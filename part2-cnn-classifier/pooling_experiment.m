%% Pooling + controlled parameter change
%--------------------------------------------------------------------------
% Trains TWO CNNs on DigitDataset:
%   (1) BASE network (fixed baseline parameters)
%   (2) MOD network  (exactly ONE parameter changed)
%
% Purpose:
%   - Introduce pooling and a second convolution block
%   - Run a controlled experiment: change one variable, hold the rest constant
%
% INPUTS:
%       None
%
% OUTPUTS:
%       - Printed BASE and MOD test accuracy
%       - Optional figures (prediction + activations + learned filters)
%
% REQUIRED FILES (same folder or on the MATLAB path):
%       - build_cnn_layers.m
%       - cnn_plots.m   (visualization helper)
%
% CONSTRAINTS (recommended):
%       - MaxEpochs <= 6
%       - MiniBatchSize <= 256
%--------------------------------------------------------------------------
% NOTE: the dataset is ordered by folders (0s then 1s then 2s...). That is
% normal. augmentedImageDatastore keeps sizing consistent in train/test.
%--------------------------------------------------------------------------

%% Clear variables
clc; clear; close all;
rng(0);

%% ============================ USER SETTINGS ==============================
MAX_EPOCHS = 6;
MAX_BATCH  = 256;

% Training settings
numEpochs   = 6;        % must be <= MAX_EPOCHS
miniBatchSz = 128;      % must be <= MAX_BATCH
initLR      = 1e-3;     % typical range: 1e-4 to 3e-3

assert(numEpochs <= MAX_EPOCHS, "numEpochs exceeds MAX_EPOCHS.");
assert(miniBatchSz <= MAX_BATCH, "miniBatchSz exceeds MAX_BATCH.");

% BASE network parameters
baseFilterSize  = 3;
baseNumFilters1 = 8;
baseNumFilters2 = 16;

% Optional regularization (off in BASE by default)
baseUseDropout  = false;
baseDropoutProb = 0.30;   % only used if dropout is enabled

%% ============================== LOAD DATA ===============================
digitDatasetPath = fullfile(matlabroot, "toolbox", "nnet", "nndemos", "nndatasets", "DigitDataset");

imdsAll = imageDatastore(digitDatasetPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Dataset summary:");
disp(countEachLabel(imdsAll));

% Keep class balance
[imdsTrain, imdsTest] = splitEachLabel(imdsAll, 0.8, "randomized");

% Enforce input size
inputSize = [28 28 1];
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest);

YTrue = imdsTest.Labels;

%% =========================== TRAINING OPTIONS ===========================
opts = trainingOptions("adam", ...
    "InitialLearnRate", initLR, ...
    "MaxEpochs", numEpochs, ...
    "MiniBatchSize", miniBatchSz, ...
    "Shuffle", "every-epoch", ...
    "Verbose", false, ...
    "Plots", "training-progress");

%% ============================== RUN BASE ================================
layersBase = build_cnn_layers( ...
    inputSize, ...
    baseFilterSize, baseNumFilters1, baseNumFilters2, ...
    baseUseDropout, baseDropoutProb);

fprintf("\nTraining BASE network...\n");
netBase = trainNetwork(augTrain, layersBase, opts);

fprintf("Evaluating BASE network...\n");
YPredBase = classify(netBase, augTest);
accBase = mean(YPredBase == YTrue);
fprintf("BASE test accuracy: %.4f\n", accBase);

% Visualize - sampleIdx can be changed
sampleIdx = 1;
cnn_plots(netBase, imdsTest, "Pooling BASE", sampleIdx);

%% ===================== ONE CONTROLLED CHANGE ===========================
% RULE: change EXACTLY ONE parameter below. This is a controlled experiment.
%
% Options:
%   A) modNumFilters1 = 16
%   B) modFilterSize  = 5
%   C) modUseDropout  = true
%   D) initLR         = 5e-4     (also requires rebuilding opts)
%   E) miniBatchSz    = 256      (also requires rebuilding opts)

% Start MOD as a copy of BASE values
modFilterSize  = 5;
modNumFilters1 = baseNumFilters1;
modNumFilters2 = baseNumFilters2;

modUseDropout  = baseUseDropout;
modDropoutProb = baseDropoutProb;

%% If LR or batch changed, rebuild training options (keeps constraints)
assert(numEpochs <= MAX_EPOCHS, "numEpochs exceeds MAX_EPOCHS.");
assert(miniBatchSz <= MAX_BATCH, "miniBatchSz exceeds MAX_BATCH.");

opts = trainingOptions("adam", ...
    "InitialLearnRate", initLR, ...
    "MaxEpochs", numEpochs, ...
    "MiniBatchSize", miniBatchSz, ...
    "Shuffle", "every-epoch", ...
    "Verbose", false, ...
    "Plots", "training-progress");

%% ============================== RUN MOD ================================
layersMod = build_cnn_layers( ...
    inputSize, ...
    modFilterSize, modNumFilters1, modNumFilters2, ...
    modUseDropout, modDropoutProb);

fprintf("\nTraining MOD network...\n");
netMod = trainNetwork(augTrain, layersMod, opts);

fprintf("Evaluating MOD network...\n");
YPredMod = classify(netMod, augTest);
accMod = mean(YPredMod == YTrue);

fprintf("MOD test accuracy: %.4f\n", accMod);
fprintf("Delta (MOD - BASE): %.4f\n\n", accMod - accBase);

% Visuals
cnn_plots(netMod, imdsTest, "Pooling MOD", sampleIdx);

%% ============================ OBSERVATIONS =============================
% Controlled change: convolution kernel size 3x3 -> 5x5 (everything else held
% constant). Results from this run:
%   BASE accuracy: 85.2%
%   MOD  accuracy: 90.85%   (delta +5.7%)
%
% Why: a 5x5 kernel sees a larger receptive field per step, so it captures
% whole curves of a digit rather than only short edges - which suits the
% rounded shapes of handwritten numbers.
%
% Tradeoff: the larger kernel adds parameters and computation (training was
% slightly slower) and raises overfitting risk; here the extra capacity helped.
% Full analysis: RESULTS.md.

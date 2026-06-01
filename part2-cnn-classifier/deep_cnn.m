%% Deep CNN (custom architecture)
%--------------------------------------------------------------------------
% A deeper, batch-normalized CNN for digit classification.
%
% Purpose:
%   - Chain conv/BN/ReLU/pool blocks into a deeper network.
%   - Observe how architecture choices affect accuracy and training behavior.
%
% INPUTS:
%       None
%
% OUTPUTS:
%       - Printed test accuracy
%       - Training progress plot (loss/accuracy vs time)
%       - Visualization figures (prediction + activations + learned filters)
%
% REQUIRED FILES:
%       - cnn_plots.m
%
% CONSTRAINTS:
%       - MaxEpochs <= 6
%       - MiniBatchSize <= 256
%--------------------------------------------------------------------------
% NOTES:
%   - The network must end with:
%         fullyConnectedLayer(10) -> softmaxLayer -> classificationLayer
%   - The first convolution layer is named "conv1" so cnn_plots can
%     visualize its filters and activations.
%--------------------------------------------------------------------------

%% Clear variables
clc; clear; close all;
rng(0);

%% ============================ USER SETTINGS ==============================
MAX_EPOCHS = 6;
MAX_BATCH  = 256;

numEpochs   = 6;        % must be <= MAX_EPOCHS
miniBatchSz = 128;      % must be <= MAX_BATCH
initLR      = 1e-3;     % typical range: 1e-4 to 3e-3

assert(numEpochs <= MAX_EPOCHS, "numEpochs exceeds MAX_EPOCHS.");
assert(miniBatchSz <= MAX_BATCH, "miniBatchSz exceeds MAX_BATCH.");

%% ============================== LOAD DATA ===============================
digitDatasetPath = fullfile(matlabroot, ...
    "toolbox", "nnet", "nndemos", "nndatasets", "DigitDataset");

imdsAll = imageDatastore(digitDatasetPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Dataset summary:");
disp(countEachLabel(imdsAll));

[imdsTrain, imdsTest] = splitEachLabel(imdsAll, 0.8, "randomized");

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

%% ============================ MODEL DEFINITION =========================
% Three conv blocks with batch normalization, two pooling stages, global
% average pooling, and dropout before the classifier head.
layers = [
    imageInputLayer(inputSize, "Normalization","zerocenter", "Name","input")

    % Block 1 (learn simple edges/strokes)
    convolution2dLayer(3, 16, "Padding","same", "Name","conv1")
    batchNormalizationLayer("Name","bn1")
    reluLayer("Name","relu1")
    convolution2dLayer(3, 16, "Padding","same", "Name","conv1b")
    batchNormalizationLayer("Name","bn1b")
    reluLayer("Name","relu1b")
    maxPooling2dLayer(2, "Stride",2, "Name","pool1")

    % Block 2 (learn bigger shapes)
    convolution2dLayer(3, 32, "Padding","same", "Name","conv2")
    batchNormalizationLayer("Name","bn2")
    reluLayer("Name","relu2")
    convolution2dLayer(3, 32, "Padding","same", "Name","conv2b")
    batchNormalizationLayer("Name","bn2b")
    reluLayer("Name","relu2b")
    maxPooling2dLayer(2, "Stride",2, "Name","pool2")

    % Block 3 (combine features)
    convolution2dLayer(3, 64, "Padding","same", "Name","conv3")
    batchNormalizationLayer("Name","bn3")
    reluLayer("Name","relu3")

    % Reduce to a compact vector
    globalAveragePooling2dLayer("Name","gap")

    % Reduce overfitting
    dropoutLayer(0.30, "Name","dropout")

    % Classifier head
    fullyConnectedLayer(10, "Name","fc_out")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classOutput")
];

%% ============================== TRAIN MODEL =============================
fprintf("\nTraining deep CNN...\n");
net = trainNetwork(augTrain, layers, opts);

%% ============================== EVALUATE ================================
YPred = classify(net, augTest);
acc = mean(YPred == YTrue);
fprintf("\nDeep CNN test accuracy: %.4f\n", acc);

%% ============================== VISUALIZE ===============================
% Choose the test image index you want to display.
sampleIdx = 1;
cnn_plots(net, imdsTest, "Deep CNN", sampleIdx);

%% Minimal CNN classifier
%--------------------------------------------------------------------------
% Trains a minimal CNN classifier on the MATLAB DigitDataset and visualizes:
%   (1) Sample test prediction + probability bar chart
%   (2) conv1 activation maps (feature maps)
%   (3) conv1 learned filters (kernel weights)
%
% INPUTS:
%       None
%
% OUTPUTS:
%       - Trained network (net) in workspace
%       - Printed test accuracy
%       - Figures for prediction, activations, and learned filters
%--------------------------------------------------------------------------
% OTHER INFORMATION:
%       - Dataset: MATLAB DigitDataset (28x28 grayscale digits)
%       - Deep Learning Toolbox required
%       - Helper file required in same folder:
%             cnn_plots.m
%--------------------------------------------------------------------------

%% Clear variables
clc; clear; close all;

%% Reproducibility
rng(0);

%% Define training parameters
% Keep these small for a fast first pass.
numEpochs   = 3;
miniBatchSz = 128;
initLR      = 1e-3;

%% Load dataset
% DigitDataset ships inside the MATLAB installation.
digitDatasetPath = fullfile(matlabroot, "toolbox", "nnet", "nndemos", "nndatasets", "DigitDataset");
imdsAll = imageDatastore(digitDatasetPath, ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

disp("Dataset summary:");
disp(countEachLabel(imdsAll));

%% Split dataset into train/test
% splitEachLabel preserves class balance.
[imdsTrain, imdsTest] = splitEachLabel(imdsAll, 0.8, "randomized");

%% Preprocess (resize + channel formatting)
% The CNN expects a fixed input size. DigitDataset is already 28x28 grayscale,
% but we enforce consistency using augmentedImageDatastore.
inputSize = [28 28 1];
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augTest  = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% Define CNN architecture (minimal classifier)
% Architecture:
%   Input -> Conv -> ReLU -> GlobalAvgPool -> FC(10) -> Softmax -> ClassOutput
%
% Notes:
%   - convolution2dLayer contains the learned kernels (filters)
%   - globalAveragePooling2dLayer converts feature maps into a vector
filterSize = 3;
numFilters = 8;

layers = [
    imageInputLayer(inputSize, "Name","input")

    convolution2dLayer(filterSize, numFilters, ...
        "Padding","same", ...
        "Name","conv1")

    reluLayer("Name","relu1")

    globalAveragePooling2dLayer("Name","gap")

    fullyConnectedLayer(10, "Name","fc")

    softmaxLayer("Name","softmax")

    classificationLayer("Name","classoutput")
];

%% Set training options
% "training-progress" shows loss/accuracy during training.
opts = trainingOptions("adam", ...
    "InitialLearnRate", initLR, ...
    "MaxEpochs", numEpochs, ...
    "MiniBatchSize", miniBatchSz, ...
    "Shuffle", "every-epoch", ...
    "Verbose", false, ...
    "Plots", "training-progress");

%% Train network
fprintf("Training minimal CNN (epochs=%d, batch=%d)\n", numEpochs, miniBatchSz);
net = trainNetwork(augTrain, layers, opts);

%% Evaluate on test set
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;

acc = mean(YPred == YTrue);
fprintf("Minimal CNN test accuracy: %.4f\n", acc);

%% Visualize prediction + activations + learned filters
% cnn_plots() requires cnn_plots.m in the same folder or on the path.
% Change sampleIdx to visualize a different test image.
sampleIdx = 1;
cnn_plots(net, imdsTest, "Minimal CNN", sampleIdx);

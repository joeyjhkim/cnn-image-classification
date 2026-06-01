function layers = build_cnn_layers(inputSize, filterSize, numFilters1, numFilters2, useDropout, dropoutProb)
% build_cnn_layers
%--------------------------------------------------------------------------
% Builds a two-block convolutional layer array (Conv-ReLU-Pool x2).
%
% PURPOSE:
%   A readable CNN definition that can be inspected layer by layer. Used by
%   pooling_experiment.m to build the BASE and MOD networks from one source.
%
% INPUTS:
%   inputSize    - [H W C] network input size (e.g., [28 28 1])
%   filterSize   - convolution kernel size (e.g., 3 means 3x3)
%   numFilters1  - number of filters in conv1
%   numFilters2  - number of filters in conv2
%   useDropout   - true/false (adds dropout before fully-connected layer)
%   dropoutProb  - dropout probability in [0,1] (used only if useDropout = true)
%
% OUTPUTS:
%   layers       - Layer array for trainNetwork()
%
% NOTES:
%   Architecture:
%     Input -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> (Dropout) -> FC -> Softmax -> Class
%--------------------------------------------------------------------------

% ---------------------------- Input checking -----------------------------
arguments
    inputSize (1,3) double
    filterSize (1,1) double {mustBeInteger, mustBePositive}
    numFilters1 (1,1) double {mustBeInteger, mustBePositive}
    numFilters2 (1,1) double {mustBeInteger, mustBePositive}
    useDropout (1,1) logical = false
    dropoutProb (1,1) double = 0.3
end

if useDropout
    if dropoutProb <= 0 || dropoutProb >= 1
        error("dropoutProb must be in (0,1) when useDropout is true.");
    end
end

%% ------------------------------ Layer list -------------------------------
layers = [
    imageInputLayer(inputSize, "Name","input")

    convolution2dLayer(filterSize, numFilters1, ...
        "Padding","same", ...
        "Name","conv1")
    reluLayer("Name","relu1")
    maxPooling2dLayer(2, "Stride",2, "Name","pool1")

    convolution2dLayer(filterSize, numFilters2, ...
        "Padding","same", ...
        "Name","conv2")
    reluLayer("Name","relu2")
    maxPooling2dLayer(2, "Stride",2, "Name","pool2")
];

% Optional dropout
if useDropout
    layers = [layers
        dropoutLayer(dropoutProb, "Name","dropout")
    ];
end

% Classifier head
layers = [layers
    fullyConnectedLayer(10, "Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
];

end

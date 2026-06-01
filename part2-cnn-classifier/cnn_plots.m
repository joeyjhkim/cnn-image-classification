%% cnn_plots
%--------------------------------------------------------------------------
% Standardized CNN visualizations shared by minimal_cnn / pooling_experiment
% / deep_cnn.
%
% INPUTS:
%       net       = trained CNN (SeriesNetwork / DAGNetwork / dlnetwork)
%       imdsTest  = test imageDatastore (images + categorical labels)
%       tag       = string used for figure names/titles (e.g., "Pooling BASE")
%       sampleIdx = (optional) index of test image to visualize
%
% OUTPUTS:
%       None (figures only)
%--------------------------------------------------------------------------
% This function calls subfunctions to visualize:
%   (1) prediction + probability bars
%   (2) conv1 activation maps (feature maps)
%   (3) conv1 learned filters (kernel weights)
% If the network has no conv layer named "conv1", the conv1 plots are skipped.
%--------------------------------------------------------------------------
function cnn_plots(net, imdsTest, tag, sampleIdx)

    if nargin < 4 || isempty(sampleIdx)
        sampleIdx = 1;
    end

    % (1) Sample input + true/predicted labels + probability bars
    showPrediction(net, imdsTest, tag, sampleIdx);

    % (2) conv1 activation maps (feature maps)
    showConv1Activations(net, imdsTest, tag, sampleIdx);

    % (3) conv1 learned filters montage (kernel weights)
    showConv1Filters(net, tag);

end


%% Local helper: Prediction + probability bars
function showPrediction(net, imdsTest, tag, sampleIdx)
    I = readimage(imdsTest, sampleIdx);
    trueLabel = imdsTest.Labels(sampleIdx);


    Iin = single(I);

    if ndims(Iin) == 2
        Iin = reshape(Iin, [size(Iin,1) size(Iin,2) 1]);
    end

    scores = predict(net, Iin);
    classes = net.Layers(end).Classes;
    predLabel = classify(net, Iin);

    figure("Name", tag + " - Prediction", "Position", [100, 100, 800, 400]); % Set a wider figure size

    % Subplot 1: The actual image
    subplot(1,2,1);
    imshow(I, []); % [] scales display to min/max of the image
    title("True: " + string(trueLabel) + " | Pred: " + string(predLabel));

    % Subplot 2: The probability bars
    subplot(1,2,2);
    bar(scores);
    xticks(1:10);
    xticklabels(string(classes));
    xlabel("Class");
    ylabel("Probability");
    title("Predicted Probabilities");
    grid on;
end



%% Local helper: conv1 activation maps
function showConv1Activations(net, imdsTest, tag, sampleIdx)

    % Find the first layer named "conv1"
    idx = find(arrayfun(@(L) isprop(L,"Name") && strcmp(L.Name,"conv1"), net.Layers), 1);
    if isempty(idx)
        warning("No conv1 layer found. Skipping conv1 activations.");
        return;
    end

    % Read test sample
    I = readimage(imdsTest, sampleIdx);
    label = imdsTest.Labels(sampleIdx);

    % Convert to single and enforce channel dimension
    I = im2single(I);
    if ndims(I) == 2
        Iin = reshape(I, [size(I,1) size(I,2) 1]);
    else
        Iin = I;
    end

    % Activation maps for conv1
    A = activations(net, Iin, "conv1");     % H x W x numFilters

    % Display first 9 feature maps
    numMapsToShow = min(9, size(A,3));
    figure("Name", tag + " - conv1 Activations");
    tiledlayout(3,3, "Padding","compact", "TileSpacing","compact");

    for k = 1:numMapsToShow
        nexttile;
        imagesc(A(:,:,k));
        axis image off;
        title("Map " + k);
    end

    sgtitle(tag + ": conv1 activation maps | True label = " + string(label));

end


%% Local helper: conv1 learned filter montage
function showConv1Filters(net, tag)

    % Find the first layer named "conv1"
    idx = find(arrayfun(@(L) isprop(L,"Name") && strcmp(L.Name,"conv1"), net.Layers), 1);
    if isempty(idx)
        warning("No conv1 layer found. Skipping conv1 filter montage.");
        return;
    end

    % Extract learned kernel weights: [k x k x channels x numFilters]
    W = net.Layers(idx).Weights;

    % Visualize up to 16 filters (channel 1 for grayscale)
    numToShow = min(16, size(W,4));
    filters = cell(1, numToShow);

    for i = 1:numToShow
        w = W(:,:,1,i);

        % Normalize each filter for display only
        w = (w - min(w(:))) / (max(w(:)) - min(w(:)) + eps);

        filters{i} = w;
    end

    figure("Name", tag + " - conv1 Filters");
    montage(filters, "Size", [ceil(numToShow/4) 4]);
    title(tag + ": conv1 learned filters (kernel weights)");

end

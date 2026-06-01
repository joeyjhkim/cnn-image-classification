function I_out = correlation2D_BW(I, h)
    % Manually apply a 2D convolution to a single channel (i.e. a
    % black and white image).  Please use zero padding for this function (you
    % can use the padarray function to help you out with this).
    %
    % Inputs:
    %   I - grayscale image (MxN)
    %   h - filter kernel (kxk), assumed odd dimensions
    %
    % Output:
    %   I_out - filtered image, same size as I

    [m, n] = size(I);          % image size
    [k, l] = size(h);          % kernel size (should be odd)

 % Work in double for correct math (avoid uint8 overflow/rounding)
    I_d = double(I);
    h_d = double(h);

    % Kernel "radius" (padding size)
    pad_r = floor(k / 2);
    pad_c = floor(l / 2);

    % Zero padding
    I_pad = padarray(I_d, [pad_r, pad_c], 0, 'both');

    % Allocate output
    I_out = zeros(m, n);

    % Correlation: NO flipping of h
    for i = 1:m
        for j = 1:n
            region = I_pad(i : i+k-1, j : j+l-1);
            I_out(i, j) = sum(sum(region .* h_d));
        end
    end

end

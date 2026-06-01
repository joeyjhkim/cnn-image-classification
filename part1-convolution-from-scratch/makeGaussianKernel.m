function h = makeGaussianKernel(size, sigma)
    % Create a normalized 2D Gaussian filter. Please make sure that you
    % keep in mind the indexing of the filter so that your gaussian is
    % centered properlly.
    %   
    %
    % Inputs:
    %   size  - size of filter (odd number, e.g. 3, 5, 7)
    %   sigma - standard deviation of Gaussian
    %
    % Output:
    %   h     - size x size Gaussian kernel (sums to 1)

    c = floor(size/2);
    [x, y] = meshgrid(-c:c, -c:c);

    h = exp(-(x.^2 + y.^2) / (2*sigma^2));
    

    % Normalize so sum = 1
    h = h / sum(h(:));

end

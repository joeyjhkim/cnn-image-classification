%% PART 4 - Manually Apply 2D Correlation
% For this part, you will modify the correlation2D_BW.m file to mannually
% perform a correlation on a single color channel of an image. The
% correlation2D_Color function will then use your code and apply it to a
% color image. 

I = imread('peppers.png');
I = im2double(I);  % convert to double for filtering
imshow(I)
title('Original Image')
h = fspecial('average', [9 9]);  % 9x9 average (mean) blur kernel
img = correlation2D_Color(I, h);
figure;
imshow(img)
title('Filtered Image using Correlation');

%% PART 5 - Basic Filters
% For this part, you will be writing out several different 3x3 filters.
% Specifically, you will apply the Identity, Mean Blur, Sharpen.  Please
% fill in h_identity, h_box, and h_sharp
% Load image
I = imread('coins.png');
I = im2double(I);  % convert to double for filtering

figure;
subplot(2,2,1); imshow(I); title('Original Image');

% 2.1 Identity Filter
h_identity = [0 0 0;
              0 1 0;
              0 0 0];

I_identity = imfilter(I, h_identity, 'replicate');
subplot(2,2,2); imshow(I_identity); title('Identity');

% 2.2 Box Blur (Mean Filter)
h_box = (1/9) * ones(3,3);

I_box = imfilter(I, h_box, 'replicate');
subplot(2,2,3); imshow(I_box); title('Box Blur');

% 2.3 Sharpen Filter
h_sharp = [ 0 -1  0;
           -1  5 -1;
            0 -1  0];

I_sharp = imfilter(I, h_sharp, 'replicate');
subplot(2,2,4); imshow(I_sharp); title('Sharpen');

%% Part 6 - Edge Detection
% You will perform several different forms of edge detection.  Please fill
% in pewitt_x, preqitt_y, sobel_x, sobel_y, roberts_x, roberts_y, and
% laplacian

% Load image
I = im2double(imread('cameraman.tif'));

figure;
imshow(I); title('Original Image');

% --- 1. Prewitt ---
prewitt_x = [-1 0 1;
             -1 0 1;
             -1 0 1];

prewitt_y = [-1 -1 -1;
              0  0  0;
              1  1  1];

Ix_p = imfilter(I, prewitt_x, 'replicate');
Iy_p = imfilter(I, prewitt_y, 'replicate');
I_prewitt = sqrt(Ix_p.^2 + Iy_p.^2);

% --- 2. Sobel ---
sobel_x = [-1 0 1;
           -2 0 2;
           -1 0 1];

sobel_y = [-1 -2 -1;
            0  0  0;
            1  2  1];

Ix_s = imfilter(I, sobel_x, 'replicate');
Iy_s = imfilter(I, sobel_y, 'replicate');
I_sobel = sqrt(Ix_s.^2 + Iy_s.^2);

% --- 3. Roberts ---
roberts_x = [1 0;
             0 -1];

roberts_y = [0 1;
            -1 0];

Ix_r = imfilter(I, roberts_x, 'replicate');
Iy_r = imfilter(I, roberts_y, 'replicate');
I_roberts = sqrt(Ix_r.^2 + Iy_r.^2);

figure();

% Prewitt
subplot(3,3,1); imshow(Ix_p,[]); title('Prewitt X');
subplot(3,3,2); imshow(Iy_p,[]); title('Prewitt Y');
subplot(3,3,3); imshow(I_prewitt,[]); title('Prewitt Mag');

% Sobel
subplot(3,3,4); imshow(Ix_s,[]); title('Sobel X');
subplot(3,3,5); imshow(Iy_s,[]); title('Sobel Y');
subplot(3,3,6); imshow(I_sobel,[]); title('Sobel Mag');

% Roberts
subplot(3,3,7); imshow(Ix_r,[]); title('Roberts X');
subplot(3,3,8); imshow(Iy_r,[]); title('Roberts Y');
subplot(3,3,9); imshow(I_roberts,[]); title('Roberts Mag');

%% PART 7
% For this part, you will modify the correlation2D_BW.m file to create a
% normalized Gaussian kernal of size k by k, and with some variance sigma.
I = imread('sherlock.jpg');
I = im2double(I);

h_small = makeGaussianKernel(7, 1);   % small blur
h_med   = makeGaussianKernel(7, 2);   % medium blur
h_large = makeGaussianKernel(15, 5);  % strong blur (larger kernel)

I_small = imfilter(I, h_small, 'replicate');
I_med   = imfilter(I, h_med, 'replicate');
I_large = imfilter(I, h_large, 'replicate');

figure;
subplot(2,2,1); imshow(I); title('Original');
subplot(2,2,2); imshow(I_small); title('\sigma = 1');
subplot(2,2,3); imshow(I_med);   title('\sigma = 2');
subplot(2,2,4); imshow(I_large); title('\sigma = 5');

%% PART 3 - Hybrid Images - EXTRA CREDIT
% For this part, you will create a hybrid image of Marilyn Monroe and
% Albert Einstein. To do so, you will pull the low frequencies of the
% Marilyn Monroe image, and pull the high frequencies of the Einstein
% image, then combine the two together.  
% 
% Please use a Gaussian function for both the bluring and the sharpening, 
% then save the blurred Marilyn photo
% in the "low_frequencies" variable, the sharpened version of Einstein in
% the "high_frequencies" variable, and then the hybrid image in the
% "hybrid" variable. It may help you to add a constant alpha to vary how
% much of the high frequiencies you see in the final hybrid image

I1 = im2double(imread('images/marilyn.bmp'));
I2 = im2double(imread('images/einstein.bmp'));

% Choose Gaussian blur parameters
k = 15;        % kernel size (odd)
sigma = 3;     % blur strength
alpha = 1.0;   % how strong to keep high frequencies

h = makeGaussianKernel(k, sigma);

% Low-pass Marilyn (blurred)
low_frequencies = imfilter(I1, h, 'replicate');

% High-pass Einstein: original - blurred
I2_low = imfilter(I2, h, 'replicate');
high_frequencies = I2 - I2_low;

% Combine to form hybrid
hybrid = low_frequencies + alpha * high_frequencies;

figure;
subplot(2,2,1); imshow(I1); title('Image 1');
subplot(2,2,2); imshow(I2); title('Image 2');
subplot(2,2,3); imshow(low_frequencies); title('Low-pass (I1)');
subplot(2,2,4); imshow(high_frequencies+0.5); title('High-pass (I2)');
figure; imshow(hybrid,[]); title('Hybrid Image');

% Show hybrid image at full size
figure;
imshow(hybrid,[]); title('Hybrid Image (full size)');

% Resize hybrid image to emphasize low-frequency image
hybrid_small = imresize(hybrid, 0.2);  % shrink to 20% size
figure;
imshow(hybrid_small,[]); title('Hybrid Image (resized)');
function I_out = correlation2D_Color(I, h)
    I = im2double(I);   % ensure double for math
    I_out = zeros(size(I));   % allocate output
    
    % Process each channel separately
    for c = 1:3
        I_out(:,:,c) = correlation2D_BW(I(:,:,c), h);  % reuse grayscale function
    end
end
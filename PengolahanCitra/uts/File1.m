pkg load image;
% Baca gambar
gambar_rgb = imread('D:\cp\citra2\Biner,Grey,RGB.jpeg');
% Konversi ke grayscale
gambar_gray = rgb2gray(gambar_rgb);
% Konversi ke biner dengan ambang otomatis
gambar_gray = rgb2gray(gambar_rgb); % konversi grayscale
threshold = graythresh(gambar_gray); % cari nilai ambang otomatis
gambar_biner = gambar_gray > threshold * 255; % manual thresholding
% Menampilkan ketiganya dalam satu figure
figure;
subplot(1, 3, 1);
imshow(gambar_rgb);
title('Gambar Asli (RGB)');
subplot(1, 3, 2);
imshow(gambar_gray);
title('Grayscale');
subplot(1, 3, 3);
imshow(gambar_biner);
title('Citra Biner');

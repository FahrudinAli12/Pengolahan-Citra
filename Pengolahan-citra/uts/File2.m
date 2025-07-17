pkg load image;
img = imread('D:\cp\citra2\Twirl.jpg');
gray = rgb2gray(img);
level_list = [2, 4, 8];
figure;
for i = 1:length(level_list);
jumlah_level = level_list(i);
interval = 256 / jumlah_level;
kuantisasi = floor(double(gray) / interval) * interval;
kuantisasi = uint8(kuantisasi);
subplot(2, 3, i);
imshow(kuantisasi);
title([num2str(jumlah_level), ' Level']);
subplot (2,3,i+3);
imhist(kuantisasi);
title('Histogram');
end

pkg load image;
gambar = imread('D:\cp\citra2\Salt_and_Pepper_Noise_koenigsegg.png'); % Ganti dengan nama file kamu
% Mengubah ke grayscale jika masih RGB
if ndims(gambar) == 3
gambar_gray = rgb2gray(gambar);
else
 gambar_gray = gambar;
end
% Filter median 3x3
hasil_median = medfilt2(gambar_gray, [3 3]);
% Filter rata-rata (mean) 3x3
kernel_mean = ones(3, 3) / 9;
hasil_mean = imfilter(gambar_gray, kernel_mean);
% Menampilkan hasil perbandingan dalam satu figure
figure;
subplot(2, 2, 1);
imshow(gambar_gray);
title('Citra Asli (Grayscale)');
subplot(2, 2, 2);
imshow(hasil_median);
title('Hasil Filter Median 3x3');
subplot(2, 2, 3);
imshow(hasil_mean);
title('Hasil Filter Rata-rata 3x3');

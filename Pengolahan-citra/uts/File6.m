gambar = imread('D:\cp\citra2\Kertas.jpeg');
% Mengubah ke grayscale jika berwarna
if ndims(gambar) == 3
 gambar_gray = rgb2gray(gambar);
else
 gambar_gray = gambar;
end
% Menerapkan filter rata-rata (Low-Pass) 3x3
h = ones(3, 3) / 9;
blurred = imfilter(double(gambar_gray), h);
% Menentukan faktor boosting (A > 1, misal A = 1.5 atau 2)
A = 2.2;
% High-Boost Filtering: A*original - blurred
high_boost = A * double(gambar_gray) - blurred;
% Konversi hasil ke uint8 agar bisa ditampilkan
high_boost_uint8 = uint8(high_boost);
% Menampilkan hasil
figure;
subplot(1, 2, 1);
imshow(gambar_gray);
title('Citra Asli (Grayscale)');
subplot(1, 2, 2);
imshow(high_boost_uint8);
title(['Hasil High-Boost (A = ' num2str(A) ')']);

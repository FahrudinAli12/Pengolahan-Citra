pkg load image;
gambar = imread('D:\cp\citra2\FotoDlmRuang.jpg');
% Memisahkan channel warna
R = gambar(:, :, 1);
G = gambar(:, :, 2);
B = gambar(:, :, 3);
% Ekualisasi histogram pada setiap channel
R_eq = histeq(R);
G_eq = histeq(G);
B_eq = histeq(B);
% Menggabungkan kembali jadi citra RGB
gambar_eq_rgb = cat(3, R_eq, G_eq, B_eq);
% Menampilkan hasil sebelum dan sesudah
figure;
subplot(2, 2, 1);
imshow(gambar);
title('Citra Asli (RGB)');
subplot(2, 2, 2);
imshow(gambar_eq_rgb);
title('Citra Setelah Ekualisasi (RGB)');
% Menampilkan histogram
subplot(2, 2, 3);
imhist(gambar);
title('Histogram Asli');
subplot(2, 2, 4);
imhist(gambar_eq_rgb);
title('Histogram Sesudah Ekualisasi');

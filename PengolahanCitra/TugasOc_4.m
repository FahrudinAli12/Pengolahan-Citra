% Memuat paket image
pkg load image;

% Membaca citra
img = imread('D:\cp\citra2\Biner,Grey,RGB.jpeg'); % Ganti 'gambar.jpg' dengan nama file citra Anda

% Menampilkan citra asli
figure;
imshow(img);
title('Citra Asli');

% Flip Vertikal (pencerminan atas-bawah)
img_flip_vert = flipud(img);
figure;
imshow(img_flip_vert);
title('Flip Vertikal');

% Flip Horizontal (pencerminan kiri-kanan)
img_flip_horiz = fliplr(img);
figure;
imshow(img_flip_horiz);
title('Flip Horizontal');

% Pencerminan Diagonal (Transpose)
img_transpose = permute(img, [2 1 3]);
figure;
imshow(img_transpose);
title('Pencerminan Diagonal (Transpose)');

% Rotasi 90 derajat searah jarum jam
img_rot_90 = imrotate(img, -90); % Sudut negatif untuk rotasi searah jarum jam
figure;
imshow(img_rot_90);
title('Rotasi 90° CW');

% Rotasi 90 derajat berlawanan arah jarum jam
img_rot_90_ccw = imrotate(img, 90); % Sudut positif untuk rotasi berlawanan arah jarum jam
figure;
imshow(img_rot_90_ccw);
title('Rotasi 90° CCW');

% Rotasi 180 derajat
img_rot_180 = imrotate(img, 180);
figure;
imshow(img_rot_180);
title('Rotasi 180°');

% Scaling: Memperbesar citra 1.5x menggunakan interpolasi bicubic (default)
img_scale_up = imresize(img, 1.5);
figure;
imshow(img_scale_up);
title('Scaling: Perbesar 1.5x');

% Scaling: Mengecilkan citra menjadi 50% menggunakan interpolasi bilinear
img_scale_down = imresize(img, 0.5, 'bilinear');
figure;
imshow(img_scale_down);
title('Scaling: Perkecil 50% (Bilinear)');

% Scaling: Mengubah ukuran citra menjadi 300x400 piksel
img_resize_fixed = imresize(img, [300 400]);
figure;
imshow(img_resize_fixed);
title('Scaling: Ukuran Tetap 300x400');


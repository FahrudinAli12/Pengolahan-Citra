%Tugas praktikum2%
pkg load image;
%histogram%
img = imread ('D:\fahruXsam\Christian-Von-Koenigsegg_KOENIGSEGG-AUTOMOTIVE (1).jpg');
img = im2uint8(img);

figure;
imhist (img);

%1. pisahkan channel RGB%
R= img(:,:,1);
G = img(:,:,2);
B =img (:,:,3);

%tampilkan Histogram untuk masing-masing Channel%
figure;
subplot (3,1,1);
imhist (R);
title('Histogram merah');
xlim([0 255]);

subplot (3,1,2);
imhist (G);
title('Histogram hijau');
xlim([0 255]);

subplot (3,1,3);
imhist (B);
title('Histogram biru');
xlim([0 255]);

%2. meningkatkan kecerahan%
imgray = rgb2gray(img);
cerah = imgray +100;
figure;
subplot(2,2,2),imshow(cerah),title '1. Penigkatan Kecerahan';
subplot(2,2,4),imhist(cerah), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';


%3. meregangkan kontras%
figure;
kont = 1.8*imgray;
subplot(2,2,2),imshow(kont),title '2. keregangan kontras';
subplot(2,2,4),imhist(kont), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';

%4. kombinasi kecerahan dan kontras%
c = imgray - 35;
k = c * 1.9;
figure ;
subplot(2,2,2),imshow(k),title '3. kom Kecerahan dan kontras';
subplot(2,2,4),imhist(k), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';


%5. membalik citra%
figure;
balik = 255 - img;
subplot(2,2,2),imshow(balik),title '4. membalik citra';
subplot(2,2,4),imhist(balik), title 'histogram';
subplot(2,2,1),imshow(img),title 'gambar asli';
subplot(2,2,3),imhist(img), title 'histogram';

%6. pemetaan nonlinier%
figure;
meta = log(1+double(imgray));
meta2 = im2uint8(mat2gray(meta));
subplot(2,2,2),imshow(meta2),title '5. Pemetaan non linier';
subplot(2,2,4),imhist(meta2), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';

%7. pemotongan aras keabuan%
batas_bawah = 0; %menentukan rentang aras keabuan yang akan dipotong
batas_atas = 50;

nilai_baru = 255; %nilai intensitas baru untuk piksel dalam rentang

citra_baru = citra_gray; %konversi nilai citra
indeks = (citra_gray >= batas_bawah) & (citra_gray <= batas_atas);
citra_baru(indeks) = nilai_baru;

figure;
subplot(2,2,2),imshow(meta2),title '6. pemotongan aras keabuan';
subplot(2,2,4),imhist(meta2), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';

%8. Equalisasi Histogram

Img = rgb2gray(img);
Img = uint8(img);

[jum_baris, jum_kolom] = size(Img);
L = 256;
Histog = zeros(L, 1);

% Hitung histogram
for baris = 1:jum_baris
    for kolom = 1:jum_kolom
        pixel_val = Img(baris, kolom);
        Histog(pixel_val + 1) = Histog(pixel_val + 1) + 1;
    end
end

% Hitung transformasi kumulatif (CDF)
alpha = (L - 1) / (jum_baris * jum_kolom);
C = zeros(L, 1);           % Inisialisasi CDF
C(1) = round(alpha * Histog(1));
for i = 2:L
    C(i) = C(i-1) + round(alpha * Histog(i));
end

% Terapkan transformasi ke gambar
Hasil = zeros(jum_baris, jum_kolom);  % Inisialisasi array hasil
for baris = 1:jum_baris
    for kolom = 1:jum_kolom
        pixel_val = Img(baris, kolom);
        Hasil(baris, kolom) = C(pixel_val + 1);
    end
end

Hasil = uint8(Hasil);

##imshow(Hasil);   % Tampilkan gambar hasil
subplot(2,2,2),imshow(Hasil),title '%8. Equalisasi Histogram';
subplot(2,2,4),imhist(Hasil), title 'histogram';
subplot(2,2,1),imshow(imgray),title 'gambar asli';
subplot(2,2,3),imhist(imgray), title 'histogram';

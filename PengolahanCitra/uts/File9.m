clc;
F = imread('D:\cp\citra2\Parkiran.jpg');
F = rgb2gray(F);
kernel_size = 5; % ukuran kernel Gaussian harus ganjil
sigma = 2.0; % standar deviasi Gaussian
function [G] = konvolusi_gaussian(F, kernel_size, sigma)
% kernel Gaussian 1D
m2 = floor(kernel_size / 2);
x = -m2:m2;
H = exp(-(x.^2) / (2*sigma^2));
H = H / sum(H); % Normalisasi
Hkol = H; % Filter vertikal (1D horizontal)
Hbrs = H; % Filter horizontal (1D horizontal)
[tinggi_f, lebar_f] = size(F);
F2 = double(F);
T = F2;
% Konvolusi vertikal dengan Hkol
for y = m2+1 : tinggi_f - m2
for x = 1 : lebar_f
jum = 0;
for p = -m2 : m2
jum = jum + Hkol(p + m2 + 1) * F2(y - p, x);
end
T(y, x) = jum;
end
end
% Konvolusi horizontal dengan Hbrs
G = zeros(size(F2)); % Inisialisasi output
for y = 1 : tinggi_f
for x = m2+1 : lebar_f - m2
jum = 0;
for p = -m2 : m2
jum = jum + Hbrs(p + m2 + 1) * T(y, x - p);
end
G(y, x) = jum;
end
end
G = uint8(G);
end
G = konvolusi_gaussian(F, kernel_size, sigma);
subplot(1,2,1); imshow(G); title('Gaussian');
subplot(1,2,2); imshow(F); title('Citra Asli');

clc;
berkas =('D:\cp\citra2\Geometri.jpg');
F = imread(berkas);
img = rgb2gray(F);
F = double(img);
function G = tbilin(F, a1, a2, a3, a4, b1, b2, b3, b4)
[tinggi, lebar] = size(F);
G = zeros(tinggi, lebar); % Inisialisasi output
for y = 1 : tinggi
for x = 1 : lebar
x2 = a1 * x + a2 * y + a3 * x * y + a4;
y2 = b1 * x + b2 * y + b3 * x * y + b4;
if (x2 >= 1) && (x2 <= lebar - 1) && (y2 >= 1) && (y2 <= tinggi - 1)
p = floor(y2);
q = floor(x2);
a = y2 - p;
b = x2 - q;
% Interpolasi bilinear
intensitas = (1 - a) * ((1 - b) * F(p, q) + b * F(p, q + 1)) + a * ((1 -
b) * F(p + 1, q) + b * F(p + 1, q + 1));
G(y, x) = intensitas;
else
G(y, x) = 0;
end
end
end
G = uint8(G);
end
%acuan setting
##a1 = besarin gambar
##a2 = miring kanan kiri
##a3 = melengkkung horizontal
##a4 = geser kanan kiri (pixel)
##b1 = miring atas bawah
##b2 = memanjang keatas
##b3 = melengkung vertikal
##b4 = geser atas bawah (pixel)
% fungsi bilinier
G = tbilin(F, 0.3,0,0,140, 0,1,0,0); % fungsi biliniear
figure;
subplot (1,2,1); imshow (G); title ('Biliniear');
subplot (1,2,2); imshow (img); title ('Gambar Asli');

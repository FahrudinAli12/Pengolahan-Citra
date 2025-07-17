clc;
F = imread('D:\cp\citra2\Twirl.jpg');
function G = ripple(F, ax, ay, tx, ty)
% RIPPLE Berfungsi untuk melakukan transformasi 'ripple'.
dimensi = size(F);
tinggi = dimensi(1);
lebar = dimensi(2);
for y=1 : tinggi
for x=1 : lebar
x2 = x + ax * sin(2 * pi * y / tx);
y2 = y + ay * sin(2 * pi * x / ty);
if (x2>=1) && (x2<=lebar) && ...
(y2>=1) && (y2<=tinggi)
% Lakukan interpolasi bilinear
p = floor(y2);
q = floor(x2);
a = y2-p;
b = x2-q;
if (floor(x2)==lebar) || ...
(floor(y2) == tinggi)
G(y, x) = F(floor(y2), floor(x2));
else
intensitas = (1-a)*((1-b)*F(p,q) + ...
b * F(p, q+1)) + ...
a *((1-b)* F(p+1, q) + ...
b * F(p+1, q+1));
G(y, x) = intensitas;
end
else
G(y, x) = 0;
end
end
end
G = uint8(G);
end
G = ripple(F,10,15,120, 250);
subplot(1,2,1); imshow(G); title('Ripple');
subplot(1,2,2); imshow(F); title('Cirta Asli');

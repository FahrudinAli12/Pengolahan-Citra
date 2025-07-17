F = rgb2gray(imread('D:\cp\citra2\papan.jpeg'));
function G = taffine(F, a11, a12, a21, a22, tx, ty)
[tinggi, lebar] = size(F);
for y=1 : tinggi
for x=1 : lebar
x2 = a11 * x + a12 * y + tx;
y2 = a21 * x + a22 * y + ty;
if (x2>=1) && (x2<=lebar) && ...
(y2>=1) && (y2<=tinggi)
% interpolasi bilinear
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
##parameter : G = taffine(F, a11, a12, a21, a22, tx, ty)
##a11, a22 = skala dan rotasi
##a12,a21 = rotasi dan shear
##tx, ty = translasi
rad = pi/6;
G = taffine(F, 2 * cos(rad) ,sin(rad) ,-sin(rad), 2 * cos(rad), -30,-
50);
subplot(1,2,1); imshow (F); title ('citra asli');
subplot(1,2,2); imshow (G); title ('affine');

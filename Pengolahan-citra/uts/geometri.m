clc;
clear;

pkg load image; % pastikan package image aktif

% Baca gambar
berkas = ('D:\cp\citra2\Geometri.jpg');
F = imread(berkas);
img = rgb2gray(F);  % Konversi ke grayscale
F = double(img);    % Ubah ke tipe double untuk perhitungan

% Fungsi transformasi bilinear + ripple
function G = tbilin(F, a1, a2, a3, a4, b1, b2, b3, b4, amp, freq)
  [tinggi, lebar] = size(F);
  G = zeros(tinggi, lebar); % Inisialisasi output

  for y = 1 : tinggi
    for x = 1 : lebar
      % Transformasi bilinear
      x2 = a1 * x + a2 * y + a3 * x * y + a4;
      y2 = b1 * x + b2 * y + b3 * x * y + b4;

      % Ripple effect
      x2 = x2 + amp * sin(2 * pi * y / freq); % horizontal ripple
      y2 = y2 + amp * sin(2 * pi * x / freq); % vertical ripple

      % Cek batas citra
      if (x2 >= 1) && (x2 <= lebar - 1) && (y2 >= 1) && (y2 <= tinggi - 1)
        p = floor(y2);
        q = floor(x2);
        a = y2 - p;
        b = x2 - q;

        % Interpolasi bilinear
        intensitas = (1 - a) * ((1 - b) * F(p, q) + b * F(p, q + 1)) + ...
                     a * ((1 - b) * F(p + 1, q) + b * F(p + 1, q + 1));
        G(y, x) = intensitas;
      else
        G(y, x) = 0;
      end
    end
  end

  G = uint8(G); % Kembalikan ke tipe uint8
end

% Parameter transformasi
a1 = 0.3; a2 = 0; a3 = 0; a4 = 140;
b1 = 0; b2 = 1; b3 = 0; b4 = 0;
amplitudo = 10;  % amplitudo ripple
frekuensi = 330;  % frekuensi ripple

% Jalankan transformasi
G = tbilin(F, a1, a2, a3, a4, b1, b2, b3, b4, amplitudo, frekuensi);

% --- Tampilkan hasil gambar ---
figure;
subplot(2,2,1); imshow(img); title('Gambar Asli');
subplot(2,2,2); imshow(G); title('Setelah Transformasi + Ripple');

% --- Tampilkan histogram ---
subplot(2,2,3); imhist(img); title('Histogram Gambar Asli');
subplot(2,2,4); imhist(G); title('Histogram Gambar Hasil');


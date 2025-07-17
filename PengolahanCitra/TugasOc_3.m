%Praktikum 3%
%NB : 1 figure 5 derau untuk 1 filter%


% Baca gambar
pkg load image;

Image1 = imread('D:\fahruXsam\Quantization_Noise_koenigsegg.png');
Image2 = imread('D:\fahruXsam\Poisson_Noise_koenigsegg.png');
Image3 = imread('D:\fahruXsam\Speckle_Noise_koenigsegg.png');
Image4 = imread('D:\fahruXsam\Salt_and_Pepper_Noise_koenigsegg.png');
Image5 = imread('D:\fahruXsam\Gaussian_Noise_koenigsegg.png');

% Menyimpan gambar
image_asli = {Image1, Image2, Image3, Image4, Image5};
namaGambar = {
  'Uniform',
  'Poisson',
  'Speckle',
  'SaltPepper',
  'Gaussian'
};

%1. filter batas%
for i = 1:5
    img = image_asli{i};
    if ndims(img) == 3
        img = rgb2gray(img);
        Img = uint8(img);
    end
    [tinggi, lebar] = size(img);
    G = img;
    for baris = 2:tinggi-1
        for kolom = 2:lebar-1
            neighbor = [img(baris-1, kolom-1), img(baris-1, kolom), img(baris-1, kolom+1), ...
                        img(baris, kolom-1)                       , img(baris, kolom+1), ...
                        img(baris+1, kolom-1), img(baris+1, kolom), img(baris+1, kolom+1)];
            minPiksel = min(neighbor);
            maksPiksel = max(neighbor);
            if img(baris, kolom) < minPiksel
                G(baris, kolom) = minPiksel;
            elseif img(baris, kolom) > maksPiksel
                G(baris, kolom) = maksPiksel;
            end
        end
    end

    figure(i); % Figure 1–5
    subplot(2,2,1); imshow(img); title(['Noise ' namaGambar{i}]);
    subplot(2,2,2); imshow(G); title('Filter Batas');
    subplot(2,2,3); imhist(img); title('Histogram Sebelum');
    subplot(2,2,4); imhist(G); title('Histogram Sesudah');
end

%2. filter pererataan%
for i = 1:5
    img = image_asli{i};
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    [tinggi, lebar] = size(img);
    G = zeros(tinggi, lebar);
    img_double = double(img);
    for baris = 2:tinggi-1
        for kolom = 2:lebar-1
            blok = img_double(baris-1:baris+1, kolom-1:kolom+1);
            G(baris, kolom) = mean(blok(:));
        end
    end
    G = uint8(G);

    figure(i + 5); % Figure 6–10
    subplot(2,2,1); imshow(img); title(['Noise ' namaGambar{i}]);
    subplot(2,2,2); imshow(G); title('Filter Pererataan');
    subplot(2,2,3); imhist(img); title('Histogram Sebelum');
    subplot(2,2,4); imhist(G); title('Histogram Sesudah');
end

%3. filter median%
for i = 1:5
    img = image_asli{i};
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    [tinggi, lebar] = size(img);
    G = img;

    for baris = 2:tinggi-1
        for kolom = 2:lebar-1
            blok = [
                img(baris-1, kolom-1), img(baris-1, kolom), img(baris-1, kolom+1), ...
                img(baris, kolom-1),   img(baris, kolom),   img(baris, kolom+1), ...
                img(baris+1, kolom-1), img(baris+1, kolom), img(baris+1, kolom+1)
            ];
            blok = sort(blok);
            G(baris, kolom) = blok(5);  % Median
        end
    end

    figure(i + 10); % Figure 11–15
    subplot(2,2,1); imshow(img); title(['Noise ' namaGambar{i}]);
    subplot(2,2,2); imshow(G); title('Filter Median');
    subplot(2,2,3); imhist(img); title('Histogram Sebelum');
    subplot(2,2,4); imhist(G); title('Histogram Sesudah');
end



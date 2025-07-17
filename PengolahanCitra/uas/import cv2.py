# --- Import Library ---

import cv2 #menyediakan metode LBP (tekstur) dan HOG (bentuk).
import os #untuk menelusuri folder dataset.
import numpy as np #manipulasi array fitur.
import tkinter as tk #antarmuka pengguna.
from tkinter import filedialog, Label, Button #
from PIL import Image, ImageTk #menampilkan gambar di GUI.
from skimage.feature import local_binary_pattern, hog #menyediakan metode LBP (tekstur) dan HOG (bentuk).
from scipy.spatial import distance #menghitung cosine similarity antar fitur.

# --- Ekstraksi Fitur ---

# Warna adalah fitur penting dalam membedakan jenis rambut (misalnya rambut pirang, hitam, cokelat).
# HSV lebih baik dari RGB untuk persepsi manusia karena memisah "warna" (H) dan "intensitas/cerah" (V)
def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#Tekstur menunjukkan pola lokal — misalnya halus, kasar, keriting.
#LBP untuk membedakan permukaan rambut yang keriting, lurus, atau bergelombang.
def texture_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

#Bentuk/struktur rambut sering mencerminkan arah garis dan pola melengkung (keriting, lurus).
#HOG menangkap kontur dan orientasi tepi.
def shape_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return hog_features

#Tepi membantu menangkap batas dan garis tajam dalam gambar rambut.
#Canny menghasilkan peta tepi yang tajam.
def edge_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

#Kombinasi fitur visual memberi hasil pencocokan yang lebih akurat.
#Tiap fitur punya kekuatan sendiri:
#Warna: membedakan warna rambut
#Tekstur: membedakan pola rambut
#Bentuk: mengamati alur atau kontur
#Tepi: melihat batas tajam antar helai
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Gagal membaca gambar:", image_path)
        return np.zeros(1000)
    image = cv2.resize(image, (256, 256))
    color = color_histogram(image)
    texture = texture_feature(image)
    shape = shape_feature(image)
    edge = edge_histogram(image)
    combined = np.hstack([color, texture, shape, edge])
    return combined

#Bisa membaca struktur folder bertingkat (misalnya: dataset/keriting/img1.jpg).
#Label otomatis diambil dari nama folder → bagus untuk klasifikasi.
def build_dataset_features(dataset_path):
    features, filenames, labels = [], [], []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                feat = extract_features(path)
                label = os.path.basename(root)
                features.append(feat)
                filenames.append(path)
                labels.append(label)
    print(f"📂 Total gambar dimuat: {len(filenames)}")
    return np.array(features), filenames, labels

#Cocok untuk membandingkan arah vektor, bukan hanya panjangnya.
#Cocok jika datanya dinormalisasi (seperti histogram yang dijadikan proporsi).
#Skor antara 0–1: makin dekat ke 1 = makin mirip.
def search_similar(query_feat, dataset_features, filenames, labels, top_n=5):
    similarities = [1 - distance.cosine(query_feat, feat) for feat in dataset_features]
    sorted_indices = np.argsort(similarities)[::-1][:top_n]
    return [(filenames[i], labels[i], similarities[i]) for i in sorted_indices]

# --- GUI CBIR ---
class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR Jenis Rambut - Versi Akurat")

        # Ganti path dataset sesuai dengan lokasi folder dataset kamu
        self.dataset_path = "D:/fahruXsam/cbir_rambut/datasethair"
        self.dataset_features, self.dataset_filenames, self.dataset_labels = build_dataset_features(self.dataset_path)

        self.result_images = []

        self.label = Label(root, text="Pilih Gambar Rambut (Query)")
        self.label.pack(pady=10)

        self.btn_upload = Button(root, text="Upload Gambar", command=self.upload_image)
        self.btn_upload.pack()

        self.canvas = tk.Canvas(root, width=256, height=256)
        self.canvas.pack(pady=10)

        self.result_label = Label(root, text="Hasil Pencocokan:")
        self.result_label.pack(pady=5)

        self.results_frame = tk.Frame(root)
        self.results_frame.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        print("📤 Gambar Query:", file_path)

        # Tampilkan gambar query
        img = Image.open(file_path).resize((256, 256))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)

        # Ekstrak fitur dan cari gambar terdekat
        query_feat = extract_features(file_path)
        print("📈 Panjang fitur query:", len(query_feat))

        results = search_similar(query_feat, self.dataset_features, self.dataset_filenames, self.dataset_labels)

        if not results:
            print("❌ Tidak ada hasil yang ditemukan.")
            return

        # Bersihkan hasil sebelumnya
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.result_images.clear()

        print("\n✅ Hasil Pencocokan:")
        for i, (filename, label, score) in enumerate(results):
            print(f"✔️ {label} - {filename} - Kemiripan: {score:.4f}")
            img = cv2.imread(filename)
            if img is None:
                print(f"⚠️ Gagal memuat gambar hasil: {filename}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (120, 120))
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)

            panel = tk.Label(self.results_frame, image=img_tk)
            panel.image = img_tk
            panel.grid(row=0, column=i, padx=5)
            self.result_images.append(img_tk)

            label_text = tk.Label(self.results_frame, text=f"{label}\nSkor: {score:.2f}")
            label_text.grid(row=1, column=i)

# --- Jalankan Aplikasi ---
if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()

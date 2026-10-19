# Sesi 6: Mengklasifikasikan Penyebab Langsung Peringatan Deforestasi dari Citra Satelit dengan AI

**Nama:** (Rheby Ersa Monica) | **NPM:** (G1A023016) | Prodi Informatika, Universitas Bengkulu

Tugas Informatics Practitioner Lecture Series, *"Attributing direct drivers to deforestation alerts with satellite imagery and AI"*
(Bart Slagter, Wageningen University & Research; 19 Oktober 2026).

## Latar belakang
Sistem peringatan deforestasi memakai citra radar (Sentinel-1) dan optik (Sentinel-2) untuk mendeteksi gangguan hutan hampir tiap minggu pada resolusi 10-30 m.
Kelemahannya: peringatan itu tidak menjelaskan **penyebabnya**. Pemateri memakai *convolutional neural network* (CNN) pada tumpukan citra Sentinel-1 dan Sentinel-2
untuk mengklasifikasikan **penyebab langsung** tiap peringatan (pertanian skala kecil/besar, jalan, penebangan selektif, pertambangan, kebakaran, gangguan alami).
Studi kasus ini mensimulasikan masalah itu dan menerapkan konsep-konsep kuncinya.

> **Pertanyaan:** seberapa baik AI dapat menentukan penyebab langsung perubahan hutan dari citra radar dan optik, bagaimana awan memengaruhinya, dan bagaimana mengevaluasinya secara jujur?

## Data: SIMULASI (bukan citra satelit asli)
Kode membuat patch citra 24x24 piksel dengan 6 pita (2 radar VV/VH + 4 optik biru, hijau, merah, NIR) untuk 7 kelas penyebab, lengkap dengan speckle radar, awan, dan kasus ambigu
(kebakaran di dataran banjir; banjir yang sinyal radarnya teredam kanopi). Pola dan nilai pantulan adalah **asumsi saya**, terinspirasi penjelasan materi. Karena itu angka hasilnya
**hanya memperagakan konsep** dan tidak boleh dipakai sebagai klaim akurasi pada citra nyata. Contoh citra ada di `output/01_contoh_citra_simulasi.png`.

## Alur
| Modul | Isi | Kaitan dengan materi |
|---|---|---|
| 1. Simulasi citra | 7 kelas, radar + optik, dengan/tanpa awan | Sentinel-1 menembus awan, Sentinel-2 terhalang awan |
| 2. Metode konvensional | Random Forest pada fitur buatan tangan (statistik pita, NDVI, ukuran dan bentuk area gangguan) | Benchmark dengan metode konvensional dulu |
| 3. CNN mini | Konvolusi 3x3 + pooling + softmax, ditulis dengan NumPy, termasuk uji turunan | CNN pada citra multi-pita |
| 4. Evaluasi | Akurasi vs macro-F1, F1 per kelas, confusion matrix | Akurasi didominasi kelas pertanian; macro-F1 lebih jujur |
| 5. Fusi sensor dan awan | Radar saja, optik saja, gabungan; uji tanpa awan vs berawan | Model memakai radar saat optik berawan |
| 6. Label "tidak yakin" | Menolak prediksi berkeyakinan rendah | Label "unconfident" pada peta keluaran |
| 7. Contoh latih terarah | Menambah contoh kebakaran dan gangguan alami | Retraining dengan lebih banyak contoh kelas yang sulit |

## Cara menjalankan
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python case_study_forest.py
# opsi: --epochs 30 --seed 42
```
Waktu proses sekitar 1-2 menit. Hasil (grafik, CSV, `ringkasan.md`) ada di folder `output/`.

## Hasil (seed 42, data simulasi)
- **Akurasi vs macro-F1:** baseline "tebak mayoritas" akurasi 0,28 tetapi macro-F1 0,06. Random Forest: akurasi 0,85, macro-F1 0,84. CNN mini: akurasi 0,75, macro-F1 0,74.
  Pada simulasi ini selisih akurasi dan macro-F1 kecil; selisihnya makin besar jika satu kelas jauh lebih dominan.
- **Kelas tersulit:** jalan, penebangan selektif, dan gangguan alami (F1 Random Forest 0,73-0,76; CNN 0,57-0,61). Pertambangan dan pertanian skala besar paling mudah.
- **Fusi sensor (macro-F1 Random Forest):** optik saja 0,93 tanpa awan tetapi turun ke 0,33 saat berawan; radar saja 0,70 dan 0,68; gabungan 0,92 dan 0,69.
  Jadi gabungan hampir sama dengan sensor terbaik pada tiap kondisi dan tidak runtuh saat berawan. CNN mini (gabungan): 0,78 tanpa awan dan 0,68 berawan.
- **Label "tidak yakin" (ambang 0,70):** Random Forest menerima 68% prediksi dengan akurasi 99%; CNN mini menerima 55% dengan akurasi 92%.
- **Contoh latih terarah:** menambah 150 contoh kebakaran dan gangguan alami hampir tidak mengubah F1 (kebakaran 0,850 menjadi 0,856; alami 0,766 menjadi 0,767; rata-rata 3 pengulangan).
  Manfaat yang disebut pemateri pada data nyata tidak terlihat jelas di simulasi ini.
- **Uji turunan CNN:** galat relatif sekitar 2e-7 (turunan analitik sesuai turunan numerik).

## Kaitan dengan materi dan perbedaannya
**Diambil dari materi:** tujuh kelas penyebab langsung, masukan radar + optik sebagai citra multi-pita, kekuatan dan kelemahan radar vs optik, benchmark metode konvensional sebelum deep learning,
akurasi total vs macro-F1, kebakaran vs banjir yang mudah tertukar, label "tidak yakin", dan penambahan contoh latih untuk kelas sulit.

**Berbeda dari materi:**
- Tidak memakai citra Sentinel nyata maupun data peringatan (misalnya RADD atau GLAD) dan tidak memakai peta global; seluruh citra adalah simulasi.
- CNN pemateri jauh lebih dalam dan dilatih dengan contoh nyata hasil pelabelan manual; CNN di sini hanya 1 lapisan konvolusi (NumPy) untuk tujuan belajar. Untuk penelitian sungguhan pakai PyTorch atau TensorFlow.
- Analisis tren, urutan penyebab (misalnya jalan lalu penebangan lalu kebakaran), musiman, serta penilaian akurasi per wilayah tidak dikerjakan.
- Keterlambatan klasifikasi satu bulan dan perubahan penyebab di kemudian hari tidak dimodelkan.

## Keterbatasan
- **Data simulasi:** hasil bergantung pada asumsi simulator. Fitur buatan tangan dirancang dengan mengetahui cara simulasi dibuat, sehingga Random Forest diuntungkan; kemenangannya atas CNN mini
  **tidak** membuktikan bahwa metode konvensional lebih baik pada citra nyata.
- CNN mini dangkal dan hanya dilatih 30 epoch tanpa penyetelan hiperparameter.
- Hanya penyebab langsung yang bisa dikenali; penyebab yang lebih rinci (misalnya arang, penebangan tradisional di antara pertanian kecil) membutuhkan data lapangan.
- Ambang "tidak yakin" dan pembagian data dipilih sederhana dan belum divalidasi silang.

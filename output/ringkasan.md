# Ringkasan Hasil (dibuat otomatis; DATA SIMULASI, bukan citra satelit asli)

- **Akurasi vs macro-F1:** baseline akurasi 0.28 tapi macro-F1 0.06; RF 0.85/0.84; CNN 0.75/0.75
- **Fusi sensor dan awan:** macro-F1 berawan: radar 0.68, optik 0.33, gabungan 0.69
- **Label tidak yakin:** ambang 0,70: Random Forest 68% diterima, akurasi 99%; CNN mini 55% diterima, akurasi 91%
- **Retraining terarah:** F1 kebakaran 0.85 -> 0.86; F1 alami 0.77 -> 0.77; macro-F1 0.84 -> 0.85 (rata-rata 3 pengulangan)

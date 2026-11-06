# Demo analisis sentimen dan deteksi sarkasme opini MBG

Kerangka demo semhas Higen Putra Perangin Angin, NIM 211401108, S1 Ilmu Komputer, Universitas Sumatera Utara. Mengikuti rancangan yang diberikan untuk Subbab 3.14.

**Status: dummy deterministik. Belum menggunakan checkpoint IndoBERTweet.** Label dan probabilitas hanya simulasi untuk menguji aplikasi, bukan hasil penelitian, bukan ukuran akurasi, dan bukan penilaian resmi MBG. Delapan baris dalam `sample_input.csv` merupakan contoh sintetis, bukan unggahan X atau laporan kejadian nyata.

## Menjalankan

Python 3.12 digunakan untuk pengujian lokal.

```bash
cd demo-web
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false --theme.primaryColor "#8A5B0A"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false --theme.primaryColor '#8A5B0A'
```

Alternatif aktivasi Windows: `.venv\Scripts\activate` pada cmd atau `.\.venv\Scripts\Activate.ps1` pada PowerShell. Buka alamat lokal yang dicetak Streamlit, biasanya `http://127.0.0.1:8501`. Tekan Ctrl+C untuk menghentikan. Alamat loopback membatasi akses ke komputer sendiri. Telemetri Streamlit dinonaktifkan pada perintah tersebut.

## Struktur

```text
demo-web/
  app.py
  preprocess.py
  model_loader.py
  requirements.txt
  sample_input.csv
  README.md
  models/
    .gitkeep
```

`.venv` dan `__pycache__` adalah artefak lingkungan lokal, bukan modul aplikasi. Tidak ada login, database, scraping, training, atau deploy cloud.

## Cara memakai

- Pilih **Teks tunggal**. Isi teks, tambahkan unggahan induk bila tersedia, lalu tekan **Prediksi**. Dua keluaran menampilkan label dan probabilitas semua kelas. Expander menampilkan input setelah pra-pemrosesan.
- Pilih **Unggah CSV**. Gunakan UTF-8, pemisah koma, header `text` dan `parent_text` opsional. Unggah `sample_input.csv`, tekan **Prediksi CSV**, periksa tabel dan distribusi, lalu **Unduh CSV hasil**.
- Pilihan tema terang/gelap tersedia melalui menu pengaturan bawaan Streamlit. UI tidak menyisipkan logo, ilustrasi, statistik penelitian, atau identitas rekaan.

Disclaimer tampil di area utama dan sidebar: “Hasil merupakan prediksi model penelitian, bukan penilaian resmi terhadap program MBG.” Saat dummy aktif, peringatan simulasi juga tampil di atas input dan penamaan keluaran.

## Pra-pemrosesan dan batas input

Urutan `preprocess_text`: konversi string (`None` menjadi kosong), lowercase, mention menjadi `@USER`, URL menjadi `HTTPURL`, `emoji.demojize` bahasa Inggris, normalisasi spasi. Placeholder tetap kapital karena diganti sesudah lowercase. `preprocess_pair` menerapkan urutan yang sama pada unggahan induk.

Kesamaan dengan pelatihan **belum terverifikasi** karena kode pelatihan belum diberikan. Sebelum checkpoint dipasang, bandingkan hasil per teks, regex, bahasa/nama emoji, versi dependensi, tokenizer, panjang token, dan aturan pasangan input dengan pipeline eksperimen.

Batas operasional demo: 10.000 karakter per kolom setelah pra-pemrosesan, dengan pesan pemotongan. Batas ini bukan panjang token IndoBERTweet dan harus diganti/diselaraskan dengan strategi truncation tokenizer saat model nyata terpasang. CSV dibatasi 10 MiB; jumlah baris tidak dipotong diam-diam.

## Integritas CSV

- Isi dan urutan kolom asli dipertahankan. Kolom `text_preprocessed` dan `parent_text_preprocessed` mencatat input yang dipakai, termasuk jika dipotong.
- `NA`, `NULL`, dan angka seperti `001` tetap teks literal. Sel kosong tetap kosong.
- Baris `text` kosong tetap disertakan dengan `prediction_status=dilewati`; label dan probabilitas kosong, bukan nol atau prediksi rekaan.
- Distribusi hanya menggunakan baris `berhasil`. Baris gagal/dilewati tidak menjadi pembagi persentase.
- Header duplikat, kolom keluaran yang bertabrakan, jumlah kolom tidak cocok, encoding rusak, dan kutip CSV tidak lengkap ditolak dengan pesan.
- Keluaran menambah label, lima kolom probabilitas, teks pra-pemrosesan, status, catatan, dan `model_status=dummy_simulasi`. File unduhan memakai UTF-8 BOM untuk membantu pembacaan di Excel.
- CSV mempertahankan teks asli. Untuk data tak tepercaya, impor melalui **Data > From Text/CSV** dan set kolom teks sebagai **Text**. Jangan membuka atau menjalankan teks yang diawali `=`, `+`, `-`, atau `@` sebagai rumus. Tanda kutip CSV bukan perlindungan formula injection.

## Mengganti dummy dengan model penelitian

1. Sediakan checkpoint, tokenizer, pemetaan label, dan konfigurasi dari eksperimen terpilih dalam `models/`, atau set environment variable `MODEL_DIR`. Path default dihitung relatif terhadap `model_loader.py`, bukan lokasi Windows absolut.
2. Implementasikan adapter dalam `model_loader.py`: shared IndoBERTweet encoder, head sentimen tiga kelas, head sarkasme dua kelas; muat pada CPU, `eval()`, dan `inference_mode()`. Terapkan softmax pada masing-masing logits dan pertahankan kontrak empat key.
3. Pilih checkpoint M1 berdasarkan hasil eksperimen yang nyata; penggunaan `parent_text` mengikuti A2. Jangan menggabungkan input secara sembarang. Jika model sentimen saja terpilih, sarkasme harus ditandai tidak tersedia dengan penyesuaian kontrak/UI, bukan dibuatkan prediksi palsu.
4. Set `USE_DUMMY = False` hanya setelah adapter siap. Saat ini pilihan tersebut sengaja menghasilkan error jelas, tidak diam-diam kembali ke dummy. Restart server agar cache prediktor dibersihkan.
5. Validasi ulang preprocessing dan seluruh uji fungsional. Jangan mengklaim skor evaluasi yang belum diukur.

`get_model()` di `app.py` memakai `st.cache_resource`; model digunakan kembali pada rerun. Dummy tidak memerlukan GPU, unduhan bobot, atau akses jaringan untuk inferensi. Dependensi `torch` dan `transformers` disediakan untuk integrasi berikutnya, bukan bukti model sudah digunakan.

## Uji manual black-box

1. Buka aplikasi. Pastikan judul, dua mode, status dummy, dan disclaimer terlihat. Klik **Prediksi** tanpa teks; peringatan muncul tanpa hasil baru.
2. Masukkan teks dengan mention, URL, dan emoji. Tambahkan induk. Prediksi dua kali; hasil tetap sama. Buka expander dan periksa placeholder serta token emoji.
3. Masukkan teks lebih dari 10.000 karakter. Pastikan ada pesan pemotongan dan aplikasi tetap berfungsi.
4. Unggah `sample_input.csv`. Prediksi seluruh delapan baris. Pastikan label, lima probabilitas, status dummy, dan kedua distribusi muncul.
5. Coba CSV tanpa `text`, file kosong, header saja, kutip tidak lengkap, serta baris teks kosong. Input rusak ditolak; baris kosong dipertahankan dan tidak diprediksi.
6. Unduh hasil. Impor sebagai teks di Excel. Periksa jumlah/urutan baris, teks asli, sel kosong, dan jumlah probabilitas tiap tugas mendekati satu untuk baris berhasil.
7. Uji layar sempit, tema terang/gelap, keyboard Tab/Enter, pembuka sidebar, dan unduhan. Jangan menyimpulkan akurasi model dari dummy.

## Keputusan tampilan

Arah: alat sidang akademik yang mengutamakan input, provenance, dan hasil. ENERGY 1 / RHYTHM 1 / MOTION 1.

- Judul serif membantu pemisahan judul akademik dari formulir; badan Segoe UI/Arial dipilih untuk keterbacaan dan ketersediaan lokal.
- Accent amber gelap menandai aksi utama; bukan dekorasi menyeluruh. Tema bawaan dipertahankan agar kontrol terang/gelap tetap berfungsi.
- Sidebar dipakai karena spesifikasi menempatkan dua mode dan status model di sana, bukan untuk meniru dashboard.
- Dua blok hasil setara mencerminkan dua tugas klasifikasi; pada layar sempit keduanya bertumpuk. Tidak ada statistik pengisi atau animasi dekoratif.
- Warning simulasi tetap dekat input. Disclaimer di area utama tetap terlihat ketika sidebar tertutup pada ponsel.

## Status kerangka

Repo ini hanya kerangka web. Model IndoBERTweet MTL belum dilatih; setelah checkpoint tersedia, ikuti bagian *Mengganti dummy dengan model penelitian* di atas. Folder `models/` sengaja tidak ikut di-commit karena ukuran checkpoint besar.

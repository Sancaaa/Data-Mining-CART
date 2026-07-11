# Tugas Data Mining — Implementasi K-Means, Apriori, dan CART

Repositori ini berisi implementasi empat model data mining beserta pra-pemrosesan dan penelusuran (_tracing_) perhitungannya, yang dikerjakan untuk mata kuliah **Data Mining (A)**. Seluruh algoritma inti (K-Means, Apriori, Classification Tree, dan Regression Tree) **diimplementasikan secara manual** dari dasar teori; pustaka `scikit-learn` hanya digunakan untuk menghitung metrik evaluasi dan pembagian data latih/uji.

## Identitas

| Keterangan | Isi |
| --- | --- |
| Mata Kuliah | Data Mining (A) |
| Dosen Pengampu | Dr. Eng. I Putu Agung Bayupati, S.T., M.T. |
| Program Studi | Sarjana Teknologi Informasi |
| Fakultas / Universitas | Teknik — Universitas Udayana |
| Tahun | 2026 |

**Anggota Kelompok**

| Nama | NIM |
| --- | --- |
| Anak Agung Narendera Sancaya | 2405551038 |
| Anak Agung Gde Putra Purnama | 2405551172 |
| Marcell Christian Santoso | 2405551153 |

## Ringkasan

Empat model diuji pada dua skala data: **mikro** (subset kecil untuk menelusuri perhitungan langkah demi langkah) dan **makro** (dataset penuh untuk mengukur kinerja). Pemetaan model terhadap teknik data mining:

| Model | Teknik | Dataset | Target/Output |
| --- | --- | --- | --- |
| K-Means | Clustering | Instagram Visits (499 baris) | Segmentasi 3 cluster |
| Apriori | Asosiasi | Grocery Transactions (500 transaksi) | Aturan asosiasi |
| Classification Tree (CART) | Klasifikasi | SDSS Galaxy DR18 | subclass (STARFORMING/STARBURST) |
| Regression Tree (CART) | Estimasi | SDSS Galaxy DR18 | redshift (numerik) |

## Struktur Repositori

```
tugas-datamining-kelompok1/
├── README.md
├── requirements.txt
├── kmeans/
│   ├── kmeans_makro.py                     # implementasi lengkap: preprocessing, Elbow, training, evaluasi
│   ├── kmeans_trace.py                      # tracing manual step-by-step (12 baris pertama)
│   └── Instagram visits clustering.csv      # dataset
├── apriori/
│   ├── apriori_makro.py                     # implementasi lengkap: preprocessing, frequent itemset, rules, evaluasi
│   ├── apriori_trace.py                     # tracing manual step-by-step (15 transaksi pertama)
│   └── grocery_transactions.csv             # dataset
└── cart/
    ├── preprocessing.ipynb                  # pra-pemrosesan data SDSS (cleaning, feature selection, sampling)
    ├── classificationTree.ipynb             # Classification Tree (Gini Impurity)
    ├── regressionTree.ipynb                 # Regression Tree (Sum of Squared Residuals)
    ├── dataMikro.csv                        # dataset mikro CART (10 baris)
    ├── DataClass.csv                        # dataset makro klasifikasi (10.000 baris, hasil preprocessing)
    ├── DataReg.csv                          # dataset makro regresi (10.000 baris, hasil preprocessing)
    └── sdss_100k_galaxy_form_burst.csv      # data mentah SDSS (100.000 baris, ±39 MB)
```

## Cara Menjalankan

**1. Instalasi dependensi**

```bash
pip install -r requirements.txt
```

**2. K-Means**

```bash
cd kmeans
python kmeans_trace.py     # menampilkan tracing manual (normalisasi, jarak, update centroid per iterasi)
python kmeans_makro.py     # Elbow Method + training + metrik evaluasi (+ menampilkan grafik)
```

**3. Apriori**

```bash
cd apriori
python apriori_trace.py    # menampilkan tracing manual (L1, L2, L3, pembangkitan rules)
python apriori_makro.py    # frequent itemset + association rules + evaluasi pada data uji
```

**4. CART** — buka file `.ipynb` di dalam folder `cart/` menggunakan Jupyter Notebook / JupyterLab / VS Code, lalu jalankan sel secara berurutan:

```bash
cd cart
jupyter notebook
```

- `preprocessing.ipynb` menghasilkan `DataClass.csv` dan `DataReg.csv` dari data mentah SDSS.
- `classificationTree.ipynb` dan `regressionTree.ipynb` melatih model pada dataset makro dan mikro.

> **Catatan:** notebook CART sudah menyimpan output hasil eksekusi, sehingga hasilnya dapat dilihat langsung tanpa menjalankan ulang.

## Penjelasan Singkat Tiap Model

### K-Means (`kmeans/`)
Clustering berbasis jarak _Euclidean_. Data dinormalisasi dengan _Min-Max scaling_, jumlah cluster optimal ($K=3$) ditentukan melalui **Elbow Method** (WCSS/inertia), lalu model dilatih hingga konvergen. Evaluasi menggunakan _Silhouette Score_, _Davies-Bouldin Index_, dan _Calinski-Harabasz Index_. File `kmeans_trace.py` memperlihatkan perhitungan jarak dan pembaruan centroid tiap iterasi secara eksplisit.

### Apriori (`apriori/`)
Analisis asosiasi (_market basket analysis_) dengan prinsip _anti-monotone_. Program membentuk _frequent itemset_ bertahap (1-, 2-, 3-, 4-itemset) melalui _join_ dan _pruning_ berdasarkan _minimum support_, lalu membangkitkan aturan asosiasi dengan metrik _support_, _confidence_, dan _lift_. Evaluasi pada data uji menggunakan _Coverage_, _Hit Rate_, dan _Precision_.

### CART — Classification & Regression Tree (`cart/`)
Pohon keputusan berbasis _recursive binary partitioning_. **Classification Tree** menggunakan **Gini Impurity** untuk memisahkan kelas galaksi, dievaluasi dengan _Accuracy_, _Precision_, _Recall_, dan _F1-Score_. **Regression Tree** menggunakan **Sum of Squared Residuals (SSR)** untuk mengestimasi _redshift_, dievaluasi dengan MSE, RMSE, dan R². Tahap `preprocessing.ipynb` mencakup penanganan nilai _placeholder_ (-9999), eliminasi fitur non-fisis, seleksi fitur, serta _balanced/representative sampling_ dari 100.000 menjadi 10.000 baris.

## Ringkasan Hasil (Dataset Makro)

| Model | Metrik | Hasil |
| --- | --- | --- |
| K-Means | Silhouette / Davies-Bouldin | 0,583 / 0,611 (K=3) |
| Apriori | Rules / Coverage / Precision | 240 rules / 93% / 65,6% |
| Classification Tree | Accuracy / F1-Score | 80,55% / 80,49% |
| Regression Tree | R² / RMSE | 66,30% / 0,0557 |

## Dataset

- **Instagram Visits** dan **Grocery Transactions**: dataset yang digunakan untuk K-Means dan Apriori (disertakan dalam repositori).
- **SDSS Galaxy Classification DR18**: bersumber dari Kaggle (kompilasi oleh *bryancimo*). Data mentah `sdss_100k_galaxy_form_burst.csv` (±39 MB) disertakan untuk reproduktibilitas tahap _preprocessing_; dataset hasil olahan (`DataClass.csv`, `DataReg.csv`) juga sudah tersedia.

## Dependensi

`pandas`, `numpy`, `scikit-learn`, `matplotlib` (lihat `requirements.txt`). Algoritma inti ditulis manual; `scikit-learn` hanya dipakai untuk metrik evaluasi dan pembagian data.

# -*- coding: utf-8 -*-
"""
TRACE: K-Means Manual Step-by-Step
Menggunakan 12 baris pertama dari 'Instagram visits clustering.csv'
(dataset yang sama dengan kmeans.py) untuk menunjukkan setiap langkah:
  1. Preprocessing (fill NA, drop duplikat, normalisasi Min-Max)
  2. Inisialisasi centroid (sama persis seperti kmeans.py: OUT simetris)
  3. Iterasi assign + update centroid, hingga konvergen
  4. Hasil akhir cluster
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import math
import random

# ============================================================
# PARAMETER
# ============================================================
DATASET_FILE = 'Instagram visits clustering.csv'
N_TRACE      = 12   # ambil N baris pertama untuk trace
K            = 3
RANDOM_SEED  = 3773  # seed sama dengan kmeans.py agar deterministik

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ============================================================
# LANGKAH 1 — LOAD & PREPROCESSING
# ============================================================
print("=" * 68)
print("  TRACE K-MEANS — Dataset: Instagram visits clustering.csv")
print(f"  Menggunakan {N_TRACE} baris pertama  |  K = {K}")
print("=" * 68)

print(f"\n{'─'*68}")
print("  LANGKAH 1 — Load & Preprocessing")
print(f"{'─'*68}")

df_raw = pd.read_csv(DATASET_FILE).head(N_TRACE)
print(f"\n  [1a] Data mentah ({N_TRACE} baris):")
print(f"  {'UserID':>6}  {'Insta Score':>11}  {'Spending Rank':>13}")
print("  " + "─" * 36)
for _, row in df_raw.iterrows():
    print(f"  {int(row['User ID']):>6}  {row['Instagram visit score']:>11.2f}  {row['Spending_rank(0 to 100)']:>13.5f}")

# A. Fill NA
print(f"\n  [1b] Handling Missing Values (fill dengan mean):")
insta_mean = df_raw['Instagram visit score'].mean()
spend_mean = df_raw['Spending_rank(0 to 100)'].mean()
na_insta   = df_raw['Instagram visit score'].isna().sum()
na_spend   = df_raw['Spending_rank(0 to 100)'].isna().sum()
print(f"       Instagram visit score — NaN: {na_insta}  mean={insta_mean:.4f}")
print(f"       Spending_rank         — NaN: {na_spend}  mean={spend_mean:.4f}")
df_raw['Instagram visit score']     = df_raw['Instagram visit score'].fillna(insta_mean)
df_raw['Spending_rank(0 to 100)']   = df_raw['Spending_rank(0 to 100)'].fillna(spend_mean)
if na_insta == 0 and na_spend == 0:
    print(f"       → Tidak ada NaN, tidak ada perubahan.")
else:
    print(f"       → NaN diganti nilai mean.")

# B. Drop duplicates
before = len(df_raw)
df_raw = df_raw.drop_duplicates()
after  = len(df_raw)
print(f"\n  [1c] Drop Duplicates: {before} → {after} baris ({before-after} duplikat dihapus)")

# C. Normalisasi Min-Max
x_min, x_max = df_raw['Instagram visit score'].min(),    df_raw['Instagram visit score'].max()
y_min, y_max = df_raw['Spending_rank(0 to 100)'].min(),  df_raw['Spending_rank(0 to 100)'].max()

print(f"\n  [1d] Normalisasi Min-Max  →  X_scaled = (X - X_min)/(X_max - X_min)")
print(f"       Instagram score : min={x_min:.2f}  max={x_max:.2f}")
print(f"       Spending rank   : min={y_min:.5f}  max={y_max:.5f}")
print(f"\n  {'UserID':>6}  {'Insta(raw)':>10}  {'Spend(raw)':>10}  {'X_scaled':>9}  {'Y_scaled':>9}")
print("  " + "─" * 55)

points = []
for _, row in df_raw.iterrows():
    x_raw = row['Instagram visit score']
    y_raw = row['Spending_rank(0 to 100)']
    x_sc  = (x_raw - x_min) / (x_max - x_min)
    y_sc  = (y_raw - y_min) / (y_max - y_min)
    points.append((round(x_sc, 5), round(y_sc, 5)))
    print(f"  {int(row['User ID']):>6}  {x_raw:>10.2f}  {y_raw:>10.5f}  {x_sc:>9.5f}  {y_sc:>9.5f}")

N = len(points)

# ============================================================
# LANGKAH 2 — INISIALISASI CENTROID
# ============================================================
print(f"\n{'─'*68}")
print("  LANGKAH 2 — Inisialisasi Centroid (Random dari Data)")
print(f"{'─'*68}")
print(f"\n  Centroid dipilih secara acak (random.sample) dari {N} titik data.")
print(f"  Seed = {RANDOM_SEED} → deterministik, sama dengan kmeans.py")
print()

random.seed(RANDOM_SEED)
centroids = random.sample(points, K)
centroid_sources = []
for i, c in enumerate(centroids):
    # Cari index titik yang dipilih
    idx = points.index(c)
    row = df_raw.iloc[idx]
    centroid_sources.append(idx)
    print(f"  C{i} dipilih → titik ke-{idx+1}  UserID={int(row['User ID'])}  ({c[0]:.5f}, {c[1]:.5f})")
centroids = [list(c) for c in centroids]

# ============================================================
# LANGKAH 3 — ITERASI K-MEANS
# ============================================================
print(f"\n{'─'*68}")
print("  LANGKAH 3 — Iterasi K-Means hingga Konvergen")
print(f"{'─'*68}")

iteration = 0
while True:
    iteration += 1
    print(f"\n  {'='*64}")
    print(f"  ITERASI {iteration}")
    print(f"  {'='*64}")

    # --- STEP 1: Assign setiap titik ke centroid terdekat ---
    clusters = [[] for _ in range(K)]
    print(f"\n  [STEP 1] Hitung jarak Euclidean & assign titik ke centroid terdekat:")
    print(f"  {'Titik (X,Y)':<22}", end="")
    for i in range(K):
        print(f"  {'dist→C'+str(i):>10}", end="")
    print(f"  {'Masuk':>6}")
    print("  " + "─" * (22 + K*12 + 8))

    for p in points:
        distances = [dist(p, c) for c in centroids]
        assigned  = distances.index(min(distances))
        clusters[assigned].append(p)
        label = f"({p[0]:.3f}, {p[1]:.3f})"
        dist_str = "".join([f"  {d:>10.5f}" for d in distances])
        min_mark = f"  → C{assigned}"
        print(f"  {label:<22}{dist_str}{min_mark}")

    # --- STEP 2: Update centroid baru ---
    print(f"\n  [STEP 2] Update centroid baru (rata-rata koordinat tiap cluster):")
    new_centroids = []
    for i, cluster in enumerate(clusters):
        if not cluster:
            new_centroids.append(centroids[i])
            print(f"  C{i}: KOSONG → tetap ({centroids[i][0]:.5f}, {centroids[i][1]:.5f})")
        else:
            x_avg = sum(p[0] for p in cluster) / len(cluster)
            y_avg = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append([x_avg, y_avg])
            pts = ", ".join([f"({p[0]:.3f},{p[1]:.3f})" for p in cluster])
            print(f"  C{i}: [{pts}]")
            print(f"       → x_avg = {x_avg:.5f}   y_avg = {y_avg:.5f}")
            anggota = len(cluster)
            print(f"       → {anggota} titik | centroid lama=({centroids[i][0]:.5f},{centroids[i][1]:.5f})  baru=({x_avg:.5f},{y_avg:.5f})")

    # --- STEP 3: Cek konvergensi ---
    print(f"\n  [STEP 3] Cek konvergensi (centroid berubah?):")
    converged = True
    for i in range(K):
        old = centroids[i]
        new = new_centroids[i]
        dx  = abs(old[0] - new[0])
        dy  = abs(old[1] - new[1])
        berubah = (dx > 1e-9 or dy > 1e-9)
        tanda   = "🔄 BERUBAH" if berubah else "✅ TETAP"
        inside  = "✅ di dalam" if (0 <= new[0] <= 1 and 0 <= new[1] <= 1) else "❗ DI LUAR"
        print(f"  C{i}: ({old[0]:.5f},{old[1]:.5f}) → ({new[0]:.5f},{new[1]:.5f})  |Δ|=({dx:.6f},{dy:.6f})  {tanda}  [{inside}]")
        if berubah:
            converged = False

    centroids = new_centroids

    if converged:
        print(f"\n  ✅ SEMUA CENTROID TIDAK BERUBAH → KONVERGEN setelah {iteration} iterasi!")
        break

# ============================================================
# HASIL AKHIR
# ============================================================
print(f"\n{'='*68}")
print(f"  HASIL AKHIR setelah {iteration} iterasi")
print(f"{'='*68}")

for i, cluster in enumerate(clusters):
    pts_str = ", ".join([f"({p[0]:.3f},{p[1]:.3f})" for p in cluster])
    print(f"\n  Cluster {i}  ({len(cluster)} titik)")
    print(f"  Centroid  : ({centroids[i][0]:.5f}, {centroids[i][1]:.5f})")
    print(f"  Anggota   : [{pts_str}]")
    sse_c = sum((p[0]-centroids[i][0])**2 + (p[1]-centroids[i][1])**2 for p in cluster)
    print(f"  SSE cluster: {sse_c:.5f}")

total_sse = sum(
    (p[0]-centroids[i][0])**2 + (p[1]-centroids[i][1])**2
    for i, cl in enumerate(clusters) for p in cl
)
print(f"\n  Total SSE (Inertia) : {total_sse:.5f}")

print(f"\n{'='*68}")
print("  CATATAN:")
print(f"  - Centroid awal dipilih secara random dari data (seed={RANDOM_SEED})")
print("  - Setiap run dengan seed berbeda bisa menghasilkan cluster yang berbeda")
print("    (local minimum berbeda) — ini sifat normal K-Means random init.")
print(f"  - Seed dikunci sama dengan kmeans.py agar trace konsisten.")
print(f"{'='*68}")

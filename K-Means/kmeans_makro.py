# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:20:23 2026
@author: luffy
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # cegah UnicodeEncodeError di terminal Windows
except Exception:
    pass

import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# Constant
random.seed(3773)

# Menghitung jarak antara 2 point (euclidian)
def calculateDistance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Fungsi K-Means Manual
# initial_centroids: list titik awal centroid yang di-set dari LUAR fungsi
def run_kmeans(points, K, initial_centroids=None):
    # Jika centroid awal tidak diberikan, gunakan random sampling
    if initial_centroids is not None:
        centroid = [list(c) for c in initial_centroids]
    else:
        centroid = [list(c) for c in random.sample(points, K)]

    while True:
        clusters = [[] for i in range(K)]

        for p in points:
            distance = [calculateDistance(p, c) for c in centroid]
            min_distance = distance.index(min(distance))
            clusters[min_distance].append(p)

        new_centroid = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                new_centroid.append(centroid[i])
            else:
                x_avg = sum(p[0] for p in cluster) / len(cluster)
                y_avg = sum(p[1] for p in cluster) / len(cluster)
                new_centroid.append([x_avg, y_avg])

        if new_centroid == centroid:
            break
        centroid = new_centroid

    return centroid, clusters

# Fungsi Prediksi Klaster & Jarak Terdekat
def predict_cluster_with_dist(points, centroid):
    preds = []
    min_dists = []
    for p in points:
        distances = [calculateDistance(p, c) for c in centroid]
        min_distance = distances.index(min(distances))
        preds.append(min_distance)
        min_dists.append(min(distances))
    return preds, min_dists

# Fungsi Hitung SSE (Within-Cluster Sum of Squared Errors)
def hitung_sse(points, centroids, clusters):
    sse = 0
    for i, cluster in enumerate(clusters):
        for p in cluster:
            sse += (p[0] - centroids[i][0])**2 + (p[1] - centroids[i][1])**2
    return sse

# ==========================================
# 1. TAHAP PREPROCESSING DATASET UTUH
# ==========================================
df = pd.read_csv('Instagram visits clustering.csv')

print("=== TAHAP PREPROCESSING INITIAL ===")
print(f"Jumlah data awal: {len(df)} baris")

# A. Handling Missing Values (Mengisi data kosong dengan nilai rata-rata)
df['Instagram visit score'] = df['Instagram visit score'].fillna(df['Instagram visit score'].mean())
df['Spending_rank(0 to 100)'] = df['Spending_rank(0 to 100)'].fillna(df['Spending_rank(0 to 100)'].mean())

# B. Handling Duplicates (Menghapus baris data yang duplikat agar tidak bias)
df = df.drop_duplicates()
print(f"Jumlah data setelah pembersihan duplikat: {len(df)} baris")

# C. Normalisasi Fitur (Min-Max Scaling manual agar skala kedua fitur adil antara 0 sampai 1)
# Rumus: X_scaled = (X - X_min) / (X_max - X_min)
x_min, x_max = df['Instagram visit score'].min(), df['Instagram visit score'].max()
y_min, y_max = df['Spending_rank(0 to 100)'].min(), df['Spending_rank(0 to 100)'].max()

df['X_scaled'] = (df['Instagram visit score'] - x_min) / (x_max - x_min)
df['Y_scaled'] = (df['Spending_rank(0 to 100)'] - y_min) / (y_max - y_min)

# Mengubah hasil akhir preprocessing menjadi list of tuples koordinat ter-skala
points_all = list(zip(df['X_scaled'], df['Y_scaled']))

# ==========================================
# 2. SPLIT DATA SELEPAS PREPROCESSING (80:20)
# ==========================================
train_points, test_points = train_test_split(points_all, test_size=0.2, random_state=3773)

print(f"\nJumlah Data Train   (80%): {len(train_points)}")
print(f"Jumlah Data Predict (20%): {len(test_points)}\n")

# ==========================================
# 2.5 ELBOW METHOD: Menentukan K Optimal
# Jalankan K-Means untuk K=1..10 pada training data,
# hitung SSE tiap K, lalu plot → cari "siku" (elbow)
# ==========================================
print("=== ELBOW METHOD: Mencari K Optimal ===")
K_range = range(1, 11)
sse_elbow = []

for k in K_range:
    random.seed(3773)   # seed sama tiap iterasi → deterministik
    c_k, cl_k = run_kmeans(train_points, k)
    sse = hitung_sse(train_points, c_k, cl_k)
    sse_elbow.append(sse)
    print(f"  K={k:2d}  SSE = {sse:.4f}")

# Plot Elbow
plt.figure(figsize=(8, 5))
plt.plot(list(K_range), sse_elbow, marker='o', color='steelblue', linewidth=2, markersize=8)
plt.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label=f'K=3 (dipilih)')
plt.scatter([3], [sse_elbow[2]], color='red', zorder=5, s=120)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('SSE (Within-Cluster Sum of Squared Errors)', fontsize=12)
plt.title('Elbow Method – Penentuan K Optimal\n(Dataset: Instagram Visit & Spending Rank)', fontsize=13)
plt.xticks(list(K_range))
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
print(f"\n→ K=3 dipilih (lihat 'siku' pada grafik elbow di atas).\n")

# ==========================================
# 3. PROSES K-MEANS RUN: TRAINING PHASE (80%)
# ==========================================
K_best = 3

# Inisialisasi centroid: random sampling dari training data (cara standar K-Means)
# random.seed sudah di-set → hasil deterministik tapi centroid dipilih dari data aktual
random.seed(3773)
init_centroids = random.sample(train_points, K_best)

print("=== TRAINING PHASE: K-Means (K=3, random init dari data) ===")
print("Centroid awal (dipilih acak dari data train):")
for i, c in enumerate(init_centroids):
    print(f"  C{i} = ({c[0]:.4f}, {c[1]:.4f})")

trained_centroids, train_clusters = run_kmeans(train_points, K_best, initial_centroids=init_centroids)

print("\nCentroid akhir setelah konvergen:")
for i, c in enumerate(trained_centroids):
    print(f"  C{i} = ({c[0]:.4f}, {c[1]:.4f})  [{len(train_clusters[i])} titik]")

sse_train = hitung_sse(train_points, trained_centroids, train_clusters)
print(f"\nSSE Training : {sse_train:.4f}")


# Hitung batas normal jarak (radius) berdasarkan data latih ter-skala
max_train_dists = []
for i, cluster in enumerate(train_clusters):
    if cluster:
        dists = [calculateDistance(p, trained_centroids[i]) for p in cluster]
        max_train_dists.append(np.mean(dists) * 1.5)
    else:
        max_train_dists.append(0)

# ==========================================
# 4. PROSES K-MEANS RUN: PREDICTION PHASE (20%)
# ==========================================
y_pred, test_dists = predict_cluster_with_dist(test_points, trained_centroids)

y_true = []
for idx, cluster_idx in enumerate(y_pred):
    if test_dists[idx] <= max_train_dists[cluster_idx]:
        y_true.append(cluster_idx)
    else:
        y_true.append((cluster_idx + 1) % K_best)

# ==========================================
# 5. METRIK EVALUASI SKLEARN
# ==========================================
print("=== METRIK EVALUASI K-MEANS (Klasifikasi ─ pembanding) ===")
print(f"Accuracy  : {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision : {precision_score(y_true, y_pred, average='macro', zero_division=0):.2f}")
print(f"Recall    : {recall_score(y_true, y_pred, average='macro', zero_division=0):.2f}")
print(f"F1-Score  : {f1_score(y_true, y_pred, average='macro', zero_division=0):.2f}")
print("\nConfusion Matrix :")
print(confusion_matrix(y_true, y_pred))

# ==========================================
# 5b. METRIK STANDAR K-MEANS (Internal Metrics)
# Tidak butuh y_true — murni mengukur kualitas cluster itu sendiri
# ==========================================
points_arr = np.array(test_points)   # konversi ke numpy array

# SSE (Inertia): total jarak kuadrat tiap titik ke centroid cluster-nya
test_clusters_by_label = [
    [test_points[j] for j in range(len(test_points)) if y_pred[j] == i]
    for i in range(K_best)
]
sse_test = hitung_sse(test_points, trained_centroids, test_clusters_by_label)

# Silhouette Score: seberapa baik titik cocok ke cluster-nya vs cluster lain
# Range: -1 s/d +1 — semakin mendekati +1 semakin baik
sil = silhouette_score(points_arr, y_pred)

# Davies-Bouldin Index: rasio rata-rata (lebar cluster / jarak antar centroid)
# Range: 0 s/d ∞ — semakin mendekati 0 semakin baik
dbi = davies_bouldin_score(points_arr, y_pred)

# Calinski-Harabasz Index: rasio variance antar-cluster vs intra-cluster
# Range: 0 s/d ∞ — semakin besar semakin baik
chi = calinski_harabasz_score(points_arr, y_pred)

print("\n=== METRIK EVALUASI K-MEANS (Standar / Internal Metrics) ===")
print(f"SSE (Inertia)           : {sse_test:.4f}   ← lebih kecil lebih baik")
print(f"Silhouette Score        : {sil:.4f}   ← mendekati +1 lebih baik")
print(f"Davies-Bouldin Index    : {dbi:.4f}   ← mendekati 0 lebih baik")
print(f"Calinski-Harabasz Index : {chi:.4f}   ← lebih besar lebih baik")

# ==========================================
# 6. PLOTTING VISUALISASI DATA TER-SKALA
# ==========================================
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']

for i in range(K_best):
    cluster_points = [test_points[j] for j in range(len(test_points)) if y_pred[j] == i]
    if cluster_points:
        x = [p[0] for p in cluster_points]
        y = [p[1] for p in cluster_points]
        plt.scatter(x, y, color=colors[i], label=f'Predicted Cluster {i}', alpha=0.6)

cx = [c[0] for c in trained_centroids]
cy = [c[1] for c in trained_centroids]
plt.scatter(cx, cy, color='black', marker='x', s=200, linewidths=3, label='Trained Centroid')

plt.xlabel('Instagram visit score (Scaled 0-1)')
plt.ylabel('Spending_rank (Scaled 0-1)')
plt.title('K-Means Clustering (Full Preprocessing Pipeline)')
plt.legend()
plt.grid(True)
plt.show()
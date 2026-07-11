# -*- coding: utf-8 -*-
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')  # cegah UnicodeEncodeError di terminal Windows
except Exception:
    pass

import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split

# ==========================================
# 1. KONSTANTA
# ==========================================
DATASET_FILE = 'grocery_transactions.csv'   # CSV dengan header, kolom 'Items' berisi item pisah koma
MAX_ROWS     = 500      # batasi 500 transaksi agar ringan
MIN_SUPPORT  = 0.04     # minimal muncul di 4% transaksi (diturunkan agar 3-itemset bisa muncul)
MIN_CONF     = 0.40     # confidence minimal 40%
MIN_LIFT     = 1.0      # lift > 1 → rule lebih baik dari acak

# ==========================================
# 2. LOAD & PREPROCESSING
# Format CSV: punya header lengkap (No_Transaksi, Tanggal, Waktu,
#             Customer_ID, Items, Total_Item, Total_Harga, Metode_Pembayaran)
# Kolom 'Items' berisi daftar item pisah koma → dipakai apriori
# ==========================================
print("=== TAHAP PREPROCESSING ===")

# Baca semua kolom dengan pandas
df = pd.read_csv(DATASET_FILE, encoding='utf-8')
df = df.head(MAX_ROWS)   # batasi ke MAX_ROWS baris

print(f"Jumlah baris dimuat       : {len(df)}")
print(f"Fitur dataset             : {list(df.columns)}")
print()
print("--- Preview 3 baris pertama ---")
print(df.head(3).to_string(index=False))
print()

# Statistik fitur tambahan
print(f"Rentang tanggal           : {df['Tanggal'].min()} s/d {df['Tanggal'].max()}")
print(f"Jumlah customer unik      : {df['Customer_ID'].nunique()}")
print(f"Rata-rata total item/tx   : {df['Total_Item'].mean():.1f}")
print(f"Rata-rata harga/tx        : Rp {df['Total_Harga'].mean():,.0f}")
print()
print("Distribusi Metode Pembayaran:")
for metode, cnt in df['Metode_Pembayaran'].value_counts().items():
    print(f"  {metode:8s}: {cnt} transaksi ({cnt/len(df):.1%})")
print()

# Ekstrak kolom Items → list of list untuk apriori
all_transactions = []
for items_str in df['Items']:
    items = [item.strip() for item in str(items_str).split(',') if item.strip()]
    if len(items) >= 2:
        all_transactions.append(items)

total_items = set(item for t in all_transactions for item in t)
avg_per_tx  = sum(len(t) for t in all_transactions) / len(all_transactions)
print(f"Transaksi valid (>=2 item) : {len(all_transactions)}")
print(f"Jumlah item unik           : {len(total_items)}")
print(f"Rata-rata item/transaksi   : {avg_per_tx:.1f}\n")

# ==========================================
# 3. SPLIT DATA 80% TRAIN & 20% TEST
# ==========================================
train_data, test_data = train_test_split(all_transactions, test_size=0.2, random_state=42)

print(f"Jumlah Data Train   (80%) : {len(train_data)}")
print(f"Jumlah Data Test    (20%) : {len(test_data)}\n")


# ==========================================
# 4. FUNGSI APRIORI MANUAL
# ==========================================

def get_support(itemset, transactions):
    """Hitung support: proporsi transaksi yang mengandung itemset."""
    itemset_set = frozenset(itemset)
    count = sum(1 for t in transactions if itemset_set.issubset(set(t)))
    return count / len(transactions)


def get_frequent_itemsets(transactions, min_support):
    """
    Algoritma Apriori standar dengan join + pruning (anti-monotone property).
    Kembalikan dict {frozenset: support}.
    """
    all_items = set(item for t in transactions for item in t)

    # --- L1: Frequent 1-itemsets ---
    freq = {}
    for item in all_items:
        sup = get_support([item], transactions)
        if sup >= min_support:
            freq[frozenset([item])] = sup

    all_freq = dict(freq)
    k = 2

    while freq:
        prev_freq_list = list(freq.keys())
        candidates = set()

        # Join step: gabung 2 itemset (k-1) yang berbeda tepat 1 item
        for i in range(len(prev_freq_list)):
            for j in range(i + 1, len(prev_freq_list)):
                union = prev_freq_list[i] | prev_freq_list[j]
                if len(union) == k:
                    candidates.add(union)

        # Pruning step: hapus kandidat yang ada (k-1)-subsetnya tidak frequent
        freq = {}
        for candidate in candidates:
            all_subsets_ok = all(
                frozenset(sub) in all_freq
                for sub in combinations(candidate, k - 1)
            )
            if not all_subsets_ok:
                continue
            sup = get_support(candidate, transactions)
            if sup >= min_support:
                freq[candidate] = sup
                all_freq[candidate] = sup

        k += 1

    return all_freq


def generate_rules(all_freq, min_conf, min_lift):
    """
    Generate association rules dari frequent itemsets.
    Kembalikan list of (antecedent, consequent, support, confidence, lift).
    """
    rules = []
    for itemset, sup_itemset in all_freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for k in range(1, len(items)):
            for ant_tuple in combinations(items, k):
                antecedent = frozenset(ant_tuple)
                consequent = itemset - antecedent

                sup_A = all_freq.get(antecedent, 0)
                sup_C = all_freq.get(consequent, 0)
                if sup_A == 0 or sup_C == 0:
                    continue

                confidence = sup_itemset / sup_A
                lift = confidence / sup_C

                if confidence >= min_conf and lift >= min_lift:
                    rules.append((antecedent, consequent, sup_itemset, confidence, lift))

    rules.sort(key=lambda x: (x[4], x[3]), reverse=True)
    return rules


# ==========================================
# 5. TRAINING PHASE
# ==========================================
print("=== TRAINING PHASE: Mencari Frequent Itemset ===")

all_freq = get_frequent_itemsets(train_data, MIN_SUPPORT)

freq_by_size = {}
for itemset in all_freq:
    s = len(itemset)
    freq_by_size[s] = freq_by_size.get(s, 0) + 1

for size in sorted(freq_by_size):
    print(f"  Frequent {size}-itemset : {freq_by_size[size]}")

rules = generate_rules(all_freq, MIN_CONF, MIN_LIFT)
print(f"  Total rules       : {len(rules)}\n")

if not rules:
    print("[!] Tidak ada rules. Coba turunkan MIN_SUPPORT atau MIN_CONF.")
else:
    print("=== TOP 10 ASSOCIATION RULES (urut: Lift ↓, Confidence ↓) ===")
    for i, (A, B, sup, conf, lift) in enumerate(rules[:10], 1):
        print(f"  {i:2d}. {set(A)}")
        print(f"      --> {set(B)}")
        print(f"      Support={sup:.3f}  Confidence={conf:.3f}  Lift={lift:.3f}")
    print()


# ==========================================
# 6. EVALUASI PHASE (20% Test Data)
# Metrik evaluasi yang TEPAT untuk Apriori:
#   - Coverage  : % transaksi test yang bisa diberi rekomendasi
#   - Hit Rate  : % rekomendasi yang benar (dari semua test)
#   - Precision : akurasi rekomendasi yang diberikan
# ==========================================
print("=== EVALUASI PHASE: Uji Rules di Data Test (20%) ===")

if not rules:
    print("[!] Tidak ada rules → evaluasi tidak bisa dilakukan.")
else:
    # Lookup: antecedent → list (consequent, conf, lift), sudah urut terbaik
    rule_dict = {}
    for A, B, sup, conf, lift in rules:
        if A not in rule_dict:
            rule_dict[A] = []
        rule_dict[A].append((B, conf, lift))

    total_test = 0
    covered    = 0   # transaksi yang punya matching rule
    hit        = 0   # rekomendasi benar

    for t in test_data:
        t_set = set(t)
        if len(t) < 2:
            continue
        total_test += 1

        matched = False
        correct = False

        # Coba setiap subset item dalam transaksi sebagai antecedent
        for size in range(1, len(t)):
            for ant_tuple in combinations(t, size):
                ant = frozenset(ant_tuple)
                if ant in rule_dict:
                    matched = True
                    best_cons = rule_dict[ant][0][0]   # consequent terbaik (lift tertinggi)
                    if best_cons.issubset(t_set):
                        correct = True
                        break
            if correct:
                break

        if matched:
            covered += 1
        if correct:
            hit += 1

    coverage   = covered / total_test if total_test > 0 else 0
    hit_rate   = hit / total_test if total_test > 0 else 0
    precision  = hit / covered if covered > 0 else 0

    print(f"  Transaksi test              : {total_test}")
    print(f"  Terkover rules              : {covered}  → Coverage  = {coverage:.2%}")
    print(f"  Prediksi benar (Hit)        : {hit}  → Hit Rate  = {hit_rate:.2%}")
    print(f"  Precision (hit/covered)     : {precision:.2%}")
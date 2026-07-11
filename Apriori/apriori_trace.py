# -*- coding: utf-8 -*-
"""
TRACE: Apriori Manual Step-by-Step
Menggunakan 15 transaksi pertama dari grocery_transactions.csv
untuk menunjukkan proses pembentukan L1, L2, L3, dan rules secara eksplisit.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')   # fix encoding Windows terminal

import pandas as pd
from itertools import combinations

# ============================================================
# PARAMETER
# ============================================================
DATASET_FILE = 'grocery_transactions.csv'
N_TRACE      = 15       # gunakan N transaksi pertama untuk tracing
MIN_SUPPORT  = 0.20     # lebih tinggi agar L1 tidak terlalu banyak di trace (≥3/15)
MIN_CONF     = 0.50
MIN_LIFT     = 1.0

# ============================================================
# LOAD 15 TRANSAKSI PERTAMA
# ============================================================
df = pd.read_csv(DATASET_FILE, encoding='utf-8').head(N_TRACE)
transactions = []
for items_str in df['Items']:
    items = [item.strip() for item in str(items_str).split(',') if item.strip()]
    if len(items) >= 2:
        transactions.append(items)

N = len(transactions)   # jumlah transaksi aktual (mungkin < N_TRACE jika ada yg < 2 item)

print("=" * 68)
print("  TRACE APRIORI MANUAL – Dataset: grocery_transactions.csv")
print(f"  Menggunakan {N} transaksi pertama")
print(f"  MIN_SUPPORT={MIN_SUPPORT} (≥{MIN_SUPPORT*N:.0f}/{N} transaksi)")
print(f"  MIN_CONF={MIN_CONF}  MIN_LIFT={MIN_LIFT}")
print("=" * 68)

# ── tampilkan transaksi
print(f"\n📋 DATA TRANSAKSI ({N} baris pertama):")
print(f"{'No':>3}  {'No_Transaksi':<20}  Items")
print("─" * 68)
for idx, (_, row) in enumerate(df.iterrows(), 1):
    print(f"  {idx:>2}. [{row['No_Transaksi']}]  {row['Items']}")

# ============================================================
# HELPER
# ============================================================
def hitung_support(itemset, txns):
    fs = frozenset(itemset)
    count = sum(1 for t in txns if fs.issubset(set(t)))
    return count, count / len(txns)


# ============================================================
# LANGKAH 1 — Buat L1 (Frequent 1-itemset)
# ============================================================
print(f"\n{'='*68}")
print("  LANGKAH 1 — Buat L1 (Frequent 1-Itemset)")
print("  Hitung support setiap item, ambil yang ≥ MIN_SUPPORT")
print(f"{'='*68}")

all_items = sorted(set(item for t in transactions for item in t))
print(f"\n  Total item unik ditemukan: {len(all_items)}")
print(f"\n  {'Item':<30} {'Count':>5}  {'Support':>8}  {'Status'}")
print("  " + "─" * 55)

L1 = {}
for item in all_items:
    count, sup = hitung_support([item], transactions)
    status = "✅ FREQUENT" if sup >= MIN_SUPPORT else "❌ pruned"
    print(f"  {item:<30} {count:>5}  {sup:>8.3f}  {status}")
    if sup >= MIN_SUPPORT:
        L1[frozenset([item])] = sup

print(f"\n  → L1 menghasilkan {len(L1)} frequent 1-itemset:")
for fs, sup in sorted(L1.items(), key=lambda x: -x[1]):
    print(f"     {set(fs)}  (support={sup:.3f})")


# ============================================================
# LANGKAH 2 — Generate kandidat C2, buat L2
# ============================================================
print(f"\n{'='*68}")
print("  LANGKAH 2 — Generate Kandidat C2 & Buat L2 (Frequent 2-Itemset)")
print("  Gabungkan setiap pasang item dari L1, hitung support-nya")
print(f"{'='*68}")

L1_items = sorted([list(fs)[0] for fs in L1.keys()])
C2 = list(combinations(L1_items, 2))
print(f"\n  Jumlah kandidat C2 dari L1: {len(C2)}")
print(f"\n  {'Kandidat':<45} {'Count':>5}  {'Support':>8}  {'Status'}")
print("  " + "─" * 65)

L2 = {}
for pair in C2:
    candidate = list(pair)
    count, sup = hitung_support(candidate, transactions)
    status = "✅ FREQUENT" if sup >= MIN_SUPPORT else "❌ pruned"
    label = f"{{{pair[0]!r}, {pair[1]!r}}}"
    print(f"  {label:<45} {count:>5}  {sup:>8.3f}  {status}")
    if sup >= MIN_SUPPORT:
        L2[frozenset(candidate)] = sup

print(f"\n  → L2 menghasilkan {len(L2)} frequent 2-itemset:")
for fs, sup in sorted(L2.items(), key=lambda x: -x[1]):
    print(f"     {set(fs)}  (support={sup:.3f})")


# ============================================================
# LANGKAH 3 — Generate kandidat C3, buat L3
# ============================================================
print(f"\n{'='*68}")
print("  LANGKAH 3 — Generate Kandidat C3 & Buat L3 (Frequent 3-Itemset)")
print("  Join L2 + pruning: buang kandidat jika ada (k-1)-subsetnya tidak frequent")
print(f"{'='*68}")

L2_list = list(L2.keys())
C3_raw = set()
for i in range(len(L2_list)):
    for j in range(i + 1, len(L2_list)):
        union = L2_list[i] | L2_list[j]
        if len(union) == 3:
            C3_raw.add(union)

print(f"\n  Kandidat C3 dari join L2: {len(C3_raw)}")

L3 = {}
if C3_raw:
    print(f"\n  {'Kandidat':<55} {'Pruned?':>10}  {'Support':>8}  {'Status'}")
    print("  " + "─" * 80)
    for candidate in sorted(C3_raw, key=lambda x: sorted(x)):
        # Pruning: cek semua 2-subset ada di L2
        subs = list(combinations(candidate, 2))
        pruned_subs = [frozenset(s) for s in subs if frozenset(s) not in L2]
        if pruned_subs:
            prune_label = f"✂ pruned ({set(list(pruned_subs[0]))} ∉ L2)"
            label = "{" + ", ".join(f"'{x}'" for x in sorted(candidate)) + "}"
            print(f"  {label:<55} {prune_label}")
            continue

        count, sup = hitung_support(list(candidate), transactions)
        status = "✅ FREQUENT" if sup >= MIN_SUPPORT else "❌ low sup"
        label = "{" + ", ".join(f"'{x}'" for x in sorted(candidate)) + "}"
        print(f"  {label:<55} {'ok':>10}  {sup:>8.3f}  {status}")
        if sup >= MIN_SUPPORT:
            L3[candidate] = sup
else:
    print("\n  (Tidak ada kandidat C3 yang bisa dibentuk dari L2)")

if L3:
    print(f"\n  → L3 menghasilkan {len(L3)} frequent 3-itemset:")
    for fs, sup in sorted(L3.items(), key=lambda x: -x[1]):
        print(f"     {set(fs)}  (support={sup:.3f})")
else:
    print(f"\n  → L3: tidak ada 3-itemset yang memenuhi MIN_SUPPORT={MIN_SUPPORT}")


# ============================================================
# LANGKAH 4 — Generate Association Rules dari L2 + L3
# ============================================================
print(f"\n{'='*68}")
print("  LANGKAH 4 — Generate Association Rules")
print(f"  Dari setiap frequent itemset (size ≥ 2), buat semua kemungkinan A → B")
print(f"  Filter: Confidence ≥ {MIN_CONF}  dan  Lift ≥ {MIN_LIFT}")
print(f"{'='*68}")

all_freq = {**L1, **L2, **L3}
rules = []

print()
for itemset, sup_itemset in sorted(all_freq.items(), key=lambda x: len(x[0])):
    if len(itemset) < 2:
        continue
    items_list = list(itemset)
    print(f"  📦 Itemset: {set(itemset)}  (support={sup_itemset:.3f})")
    found_rule = False
    for k in range(1, len(items_list)):
        for ant_tuple in combinations(items_list, k):
            ant = frozenset(ant_tuple)
            con = itemset - ant
            sup_A = all_freq.get(ant, 0)
            sup_C = all_freq.get(con, 0)
            if sup_A == 0 or sup_C == 0:
                continue
            conf = sup_itemset / sup_A
            lift = conf / sup_C
            ok_conf = conf >= MIN_CONF
            ok_lift = lift >= MIN_LIFT
            passed  = ok_conf and ok_lift
            status  = "✅ RULE" if passed else f"❌ (conf={conf:.2f} {'ok' if ok_conf else 'RENDAH'}, lift={lift:.2f} {'ok' if ok_lift else 'RENDAH'})"
            print(f"     {set(ant)} → {set(con)}")
            print(f"       conf={conf:.3f}  lift={lift:.3f}  {status}")
            if passed:
                rules.append((ant, con, sup_itemset, conf, lift))
                found_rule = True
    if not found_rule:
        print("     (tidak ada rule yang lolos filter)")
    print()


# ============================================================
# HASIL AKHIR
# ============================================================
print("=" * 68)
print("  HASIL AKHIR — ASSOCIATION RULES YANG TERBENTUK")
print("=" * 68)

if not rules:
    print(f"\n  (Tidak ada rule yang memenuhi MIN_CONF={MIN_CONF} & MIN_LIFT={MIN_LIFT})")
    print(f"  → Coba turunkan MIN_SUPPORT atau MIN_CONF")
else:
    rules_sorted = sorted(rules, key=lambda x: (x[4], x[3]), reverse=True)
    print(f"\n  Total rules: {len(rules_sorted)}")
    print(f"\n  {'No':>3}  {'Antecedent → Consequent':<45}  {'Support':>7}  {'Conf':>5}  {'Lift':>6}")
    print("  " + "─" * 72)
    for i, (A, B, sup, conf, lift) in enumerate(rules_sorted, 1):
        rule_str = f"{set(A)} → {set(B)}"
        print(f"  {i:>3}. {rule_str:<45}  {sup:>7.3f}  {conf:>5.3f}  {lift:>6.3f}")

print(f"\n{'='*68}")
print("  SELESAI.")
print(f"{'='*68}")

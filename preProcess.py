# -*- coding: utf-8 -*-
"""
Created on Thu May 28 17:16:06 2026

@author: luffy
"""

import pandas as pd

# Load dataset
df = pd.read_csv("sdss_100k_galaxy_form_burst.csv")

# Drop kolom yang tidak dipakai
columns_to_drop = [
    "objid",
    "specobjid",
    "ra",
    "dec",
    "class",
    "redshift"
]

df = df.drop(columns=columns_to_drop)

# Drop row yang memiliki placeholder null -9999.00
df = df[df.ne(-9999.00).all(axis=1)]

# Optional: reset index
df = df.reset_index(drop=True)

# Save hasil preprocessing
df.to_csv("dataset_clean.csv", index=False)

print("Preprocessing selesai.")
print("Shape akhir:", df.shape)
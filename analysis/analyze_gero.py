#!/usr/bin/env python3
"""
Gero Identity Cues Dataset - Sperm Whale Acoustic Identification
Analysis script for ICI-based individual whale classification.

Dataset: Gero et al. coda ICI timing data
Features: Inter-Click Intervals (ICI1-ICI9), coda length, number of clicks
Target: Individual whale identification (WhaleID)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "/mnt/archive/datasets/whale_communication/gero_identity_cues/gero_coda_ici.xlsx"
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/gero_identity_cues/"

print("=" * 70)
print("GERO IDENTITY CUES DATASET - SPERM WHALE ACOUSTIC IDENTIFICATION")
print("=" * 70)

# ============================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================
df = pd.read_excel(DATA_PATH)
df["WhaleID"] = df["WhaleID"].astype(str)

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)
print(f"Total rows (codas): {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Total unique WhaleIDs: {df['WhaleID'].nunique()}")
print(f"Total social units: {df['Unit'].nunique()}")
print(f"Total coda types: {df['CodaName'].nunique()}")

# Breakdown
n_unidentified = (df["WhaleID"] == "0").sum()
n_noise = (df["CodaName"] == "NOISE").sum()
print(f"\nUnidentified whale codas (WhaleID=0): {n_unidentified}")
print(f"NOISE codas: {n_noise}")

print("\n--- Whales per Social Unit ---")
for unit in sorted(df["Unit"].unique()):
    mask = df["Unit"] == unit
    whales = sorted(df.loc[mask, "WhaleID"].unique())
    identified = [w for w in whales if w != "0"]
    print(f"  Unit {unit}: {len(identified)} identified whales, {mask.sum()} codas total")
    if identified:
        for w in identified:
            n = ((df["WhaleID"] == w) & mask).sum()
            print(f"    Whale {w}: {n} codas")

print("\n--- Coda Type Distribution ---")
coda_counts = df["CodaName"].value_counts().sort_values(ascending=False)
for name, count in coda_counts.items():
    print(f"  {name}: {count}")

print("\n--- Click Count Distribution ---")
click_counts = df["nClicks"].value_counts().sort_index()
for n, count in click_counts.items():
    print(f"  {n} clicks: {count}")

# ============================================================
# 2. PREPARE DATA FOR CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("2. DATA PREPARATION")
print("=" * 70)

# Filter: only identified whales (WhaleID != 0) and non-NOISE codas
df_clean = df[(df["WhaleID"] != "0") & (df["CodaName"] != "NOISE")].copy()
print(f"After removing unidentified (WhaleID=0) and NOISE: {len(df_clean)} codas")
print(f"Identified whales: {df_clean['WhaleID'].nunique()}")

# Check per-whale counts
whale_counts = df_clean["WhaleID"].value_counts()
print("\n--- Codas per Identified Whale ---")
for wid, count in whale_counts.items():
    unit = df_clean.loc[df_clean["WhaleID"] == wid, "Unit"].iloc[0]
    print(f"  Whale {wid} (Unit {unit}): {count} codas")

# Filter whales with enough samples (at least 10 codas for meaningful classification)
MIN_CODAS = 10
valid_whales = whale_counts[whale_counts >= MIN_CODAS].index.tolist()
df_ml = df_clean[df_clean["WhaleID"].isin(valid_whales)].copy()
print(f"\nWhales with >= {MIN_CODAS} codas: {len(valid_whales)}")
print(f"Codas for classification: {len(df_ml)}")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("3. FEATURE ENGINEERING")
print("=" * 70)

# ICI columns
ici_cols = ["ICI1", "ICI2", "ICI3", "ICI4", "ICI5", "ICI6", "ICI7", "ICI8", "ICI9"]

# Build features similar to the voiceprint pipeline but adapted for ICI data
features = pd.DataFrame(index=df_ml.index)

# Raw ICI values (primary timing features)
for col in ici_cols:
    features[col] = df_ml[col]

# Basic coda properties
features["nClicks"] = df_ml["nClicks"]
features["Length"] = df_ml["Length"]

# ICI statistics (computed across the active ICIs per coda)
ici_matrix = df_ml[ici_cols].values
# Only consider non-zero ICIs for statistics
active_icis = []
for i in range(len(ici_matrix)):
    row = ici_matrix[i]
    active = row[row > 0]
    active_icis.append(active)

features["ici_mean"] = [np.mean(a) if len(a) > 0 else 0 for a in active_icis]
features["ici_std"] = [np.std(a) if len(a) > 1 else 0 for a in active_icis]
features["ici_min"] = [np.min(a) if len(a) > 0 else 0 for a in active_icis]
features["ici_max"] = [np.max(a) if len(a) > 0 else 0 for a in active_icis]
features["ici_range"] = features["ici_max"] - features["ici_min"]
features["ici_cv"] = np.where(features["ici_mean"] > 0, features["ici_std"] / features["ici_mean"], 0)

# Tempo features
features["ici_median"] = [np.median(a) if len(a) > 0 else 0 for a in active_icis]
features["avg_click_rate"] = np.where(features["Length"] > 0, features["nClicks"] / features["Length"], 0)

# ICI ratios (relative timing - key identity cue per Gero et al.)
features["ici_ratio_1_2"] = np.where(df_ml["ICI2"] > 0, df_ml["ICI1"] / df_ml["ICI2"], 0)
features["ici_ratio_2_3"] = np.where(df_ml["ICI3"] > 0, df_ml["ICI2"] / df_ml["ICI3"], 0)
features["ici_ratio_3_4"] = np.where(df_ml["ICI4"] > 0, df_ml["ICI3"] / df_ml["ICI4"], 0)
features["ici_ratio_1_last"] = np.where(
    features["ici_max"] > 0, df_ml["ICI1"] / features["ici_max"], 0
)

# Rhythm features (differences between consecutive ICIs)
features["ici_diff_1_2"] = df_ml["ICI1"] - df_ml["ICI2"]
features["ici_diff_2_3"] = df_ml["ICI2"] - df_ml["ICI3"]
features["ici_diff_3_4"] = df_ml["ICI3"] - df_ml["ICI4"]

# Acceleration (second derivative of ICI timing)
features["ici_accel_1"] = features["ici_diff_1_2"] - features["ici_diff_2_3"]
features["ici_accel_2"] = features["ici_diff_2_3"] - features["ici_diff_3_4"]

# Normalized ICIs (proportion of total length)
for col in ici_cols:
    features[f"{col}_norm"] = np.where(features["Length"] > 0, df_ml[col] / features["Length"], 0)

feature_names = list(features.columns)
print(f"Total features engineered: {len(feature_names)}")
print("Feature list:")
for i, f in enumerate(feature_names):
    print(f"  {i+1}. {f}")

X = features.values
y = df_ml["WhaleID"].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
print(f"Classes (WhaleIDs): {list(class_names)}")
print(f"Class distribution:")
for cls in class_names:
    print(f"  Whale {cls}: {(y == cls).sum()} codas")

# ============================================================
# 4. ML CLASSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("4. CLASSIFICATION RESULTS")
print("=" * 70)

# Scale features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle any NaN/inf
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42
    ),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
}

results = {}

for name, clf in classifiers.items():
    print(f"\n--- {name} ---")

    if "SVM" in name:
        X_input = X_scaled
    else:
        X_input = X

    y_pred = cross_val_predict(clf, X_input, y_encoded, cv=cv)
    acc = accuracy_score(y_encoded, y_pred)
    results[name] = acc

    print(f"5-Fold CV Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\nClassification Report:")
    report = classification_report(y_encoded, y_pred, target_names=[f"Whale {c}" for c in class_names])
    print(report)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_encoded, y_pred)
    # Print header
    header = "         " + " ".join([f"{c:>6s}" for c in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {class_names[i]:>6s} " + " ".join([f"{v:6d}" for v in row])
        print(row_str)

# ============================================================
# 5. FEATURE IMPORTANCE (from Random Forest)
# ============================================================
print("\n" + "=" * 70)
print("5. FEATURE IMPORTANCE (Random Forest)")
print("=" * 70)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y_encoded)

importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print("\nTop 20 most discriminating features:")
for rank, idx in enumerate(sorted_idx[:20]):
    print(f"  {rank+1:2d}. {feature_names[idx]:20s} importance={importances[idx]:.4f}")

# ============================================================
# 6. ADDITIONAL ANALYSES
# ============================================================
print("\n" + "=" * 70)
print("6. SOCIAL UNIT CLASSIFICATION")
print("=" * 70)

# Can we classify social units? (should be easier - group-level identity)
# Use all non-noise codas (including unidentified whales) that have unit labels
df_unit = df[df["CodaName"] != "NOISE"].copy()

# Build same features for unit classification
ici_matrix_u = df_unit[ici_cols].values
active_icis_u = []
for i in range(len(ici_matrix_u)):
    row = ici_matrix_u[i]
    active = row[row > 0]
    active_icis_u.append(active)

features_u = pd.DataFrame(index=df_unit.index)
for col in ici_cols:
    features_u[col] = df_unit[col]
features_u["nClicks"] = df_unit["nClicks"]
features_u["Length"] = df_unit["Length"]
features_u["ici_mean"] = [np.mean(a) if len(a) > 0 else 0 for a in active_icis_u]
features_u["ici_std"] = [np.std(a) if len(a) > 1 else 0 for a in active_icis_u]
features_u["ici_range"] = [np.ptp(a) if len(a) > 0 else 0 for a in active_icis_u]
features_u["ici_cv"] = np.where(features_u["ici_mean"] > 0, features_u["ici_std"] / features_u["ici_mean"], 0)
features_u["avg_click_rate"] = np.where(features_u["Length"] > 0, features_u["nClicks"] / features_u["Length"], 0)
features_u["ici_ratio_1_2"] = np.where(df_unit["ICI2"] > 0, df_unit["ICI1"] / df_unit["ICI2"], 0)

X_unit = features_u.values
X_unit = np.nan_to_num(X_unit, nan=0.0, posinf=0.0, neginf=0.0)
y_unit = df_unit["Unit"].values

le_unit = LabelEncoder()
y_unit_enc = le_unit.fit_transform(y_unit)

cv_unit = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_unit = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
y_unit_pred = cross_val_predict(rf_unit, X_unit, y_unit_enc, cv=cv_unit)
unit_acc = accuracy_score(y_unit_enc, y_unit_pred)
print(f"Social Unit Classification (RF, 5-fold CV): {unit_acc:.4f} ({unit_acc*100:.1f}%)")
print(f"  {len(le_unit.classes_)} units, {len(X_unit)} codas")

# ============================================================
# 7. CODA-TYPE CONTROLLED ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("7. CODA-TYPE CONTROLLED ANALYSIS (within-type individual ID)")
print("=" * 70)

# The most important test: can we ID individuals WITHIN the same coda type?
# This isolates the "voice" from the "word"
top_coda_types = df_ml["CodaName"].value_counts()
print(f"\nCoda types available in identified-whale data:")
for ct, count in top_coda_types.items():
    n_whales = df_ml.loc[df_ml["CodaName"] == ct, "WhaleID"].nunique()
    print(f"  {ct}: {count} codas from {n_whales} whales")

for coda_type in top_coda_types.index:
    subset = df_ml[df_ml["CodaName"] == coda_type]
    whale_cts = subset["WhaleID"].value_counts()
    # Need at least 2 whales with >= 5 codas each
    valid = whale_cts[whale_cts >= 5]
    if len(valid) < 2:
        continue

    sub = subset[subset["WhaleID"].isin(valid.index)]
    X_sub = features.loc[sub.index].values
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    y_sub = sub["WhaleID"].values
    le_sub = LabelEncoder()
    y_sub_enc = le_sub.fit_transform(y_sub)

    n_splits = min(5, min(valid.values))
    if n_splits < 2:
        continue

    cv_sub = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_sub = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    try:
        y_sub_pred = cross_val_predict(rf_sub, X_sub, y_sub_enc, cv=cv_sub)
        sub_acc = accuracy_score(y_sub_enc, y_sub_pred)
        print(f"\n  Coda type '{coda_type}': {sub_acc:.4f} ({sub_acc*100:.1f}%) "
              f"- {len(le_sub.classes_)} whales, {len(X_sub)} codas")
        for wid in le_sub.classes_:
            print(f"    Whale {wid}: {(y_sub == wid).sum()} codas")
    except Exception as e:
        print(f"  Coda type '{coda_type}': skipped ({e})")

# ============================================================
# 8. COMPARISON WITH DSWP RESULTS
# ============================================================
print("\n" + "=" * 70)
print("8. COMPARISON WITH DSWP BASELINE")
print("=" * 70)

print(f"\nDSWP Baseline:")
print(f"  - 3 whales, Gradient Boosting")
print(f"  - Accuracy: 91.5%")
print(f"  - Features: 30 (audio-derived: ZCR, spectral, frequency bands, ICI stats)")

print(f"\nGero Dataset Results:")
print(f"  - {len(class_names)} whales from {df_ml['Unit'].nunique()} social units")
print(f"  - {len(df_ml)} total codas")
for name, acc in results.items():
    print(f"  - {name}: {acc*100:.1f}%")

best_name = max(results, key=results.get)
best_acc = results[best_name]
print(f"\n  Best classifier: {best_name} at {best_acc*100:.1f}%")

print(f"\nKey differences:")
print(f"  - Gero uses ICI timing only (no raw audio spectral features)")
print(f"  - Gero has {len(class_names)} whales vs 3 in DSWP (harder problem)")
print(f"  - Gero dataset is from published research (Gero et al.) with expert labels")
print(f"  - DSWP uses full audio features; Gero uses only temporal ICI patterns")

if best_acc > 0.5:
    print(f"\nCONCLUSION: ICI timing alone can discriminate individual whales")
    print(f"  at {best_acc*100:.1f}% accuracy across {len(class_names)} individuals.")
    print(f"  This supports the hypothesis that sperm whales have individual")
    print(f"  'voiceprints' encoded in their click timing patterns.")
else:
    print(f"\nCONCLUSION: ICI timing alone shows limited discriminability")
    print(f"  ({best_acc*100:.1f}%). Full audio features may be needed for")
    print(f"  reliable individual identification.")

# Chance level
chance = 1.0 / len(class_names)
print(f"\n  Chance level (random): {chance*100:.1f}%")
print(f"  Improvement over chance: {(best_acc - chance)*100:.1f} percentage points")
print(f"  Relative improvement: {best_acc/chance:.1f}x chance")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

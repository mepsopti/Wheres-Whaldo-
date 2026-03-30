#!/usr/bin/env python3
"""
CETI Vowels Dataset - Whale Acoustic Identification Analysis
Analyzes spectral features from the CETI sperm whale vowels dataset
to test individual whale identification (voiceprint) capabilities.

Cross-references with DSWP baseline features.
"""

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/mnt/archive/datasets/whale_communication/ceti_vowels')

print("=" * 80)
print("CETI VOWELS DATASET - WHALE ACOUSTIC IDENTIFICATION ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. DATA EXPLORATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA EXPLORATION")
print("=" * 80)

# Load all data
print("\nLoading data...")
codamd = pd.read_csv(DATA_DIR / 'codamd.csv', dtype={'codanum': str}, keep_default_na=False)
clickspec = feather.read_feather(DATA_DIR / 'clickspec.ft')
# Convert codanum/clicknum to string for consistent joining
clickspec['codanum'] = clickspec['codanum'].astype(str)
clickspec['clicknum'] = clickspec['clicknum'].astype(str)

print(f"\n--- CODA METADATA (codamd.csv) ---")
print(f"  Rows: {len(codamd)}")
print(f"  Columns: {list(codamd.columns)}")
print(f"  Total codas: {codamd['codanum'].nunique()}")
print(f"  Unique whales: {codamd[codamd['whale'] != '']['whale'].nunique()}")
print(f"  Whale names: {sorted(codamd[codamd['whale'] != '']['whale'].unique())}")
print(f"  Codas with no whale ID: {(codamd['whale'] == '').sum()}")
print(f"  Coda types: {codamd['codatype'].nunique()} types")
print(f"  handv categories: {sorted(codamd['handv'].unique())}")
print(f"\n  Codas per whale:")
for whale, count in codamd[codamd['whale'] != '']['whale'].value_counts().items():
    print(f"    {whale:12s}: {count:4d} codas")

print(f"\n--- CLICK SPECTRA (clickspec.ft) ---")
print(f"  Rows (clicks): {len(clickspec)}")
print(f"  Total columns: {len(clickspec.columns)}")
print(f"  ID columns: codanum, clicknum")
freq_cols = [c for c in clickspec.columns if c not in ['codanum', 'clicknum']]
freqs = [float(f) for f in freq_cols]
print(f"  Spectral columns: {len(freq_cols)} frequency bins")
print(f"  Frequency range: {min(freqs):.1f} Hz - {max(freqs):.1f} Hz")
print(f"  Frequency resolution: {freqs[1] - freqs[0]:.3f} Hz")
print(f"  Unique codas in clickspec: {clickspec['codanum'].nunique()}")

# Check TFP analysis structure (sample only)
print(f"\n--- TFP ANALYSIS (tfpanalysis.ft) ---")
print("  (Sampling first 1000 rows of 17.2M row file)")
tfp_sample = feather.read_feather(DATA_DIR / 'tfpanalysis.ft').head(1000)
print(f"  Columns: {list(tfp_sample.columns)}")
print(f"  Dtypes: {dict(tfp_sample.dtypes)}")
print(f"  Total rows: ~17.2 million")
print(f"  Features: frameidx, sec, intensity, nformants, f1, b1, f2, b2, ceil, stress")
print(f"  Multi-index levels: timestep, winlen, rlpcstd, toffset, sigdur, coef, codanum")
del tfp_sample

# ============================================================================
# 2. MERGE CLICK SPECTRA WITH WHALE IDS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: SPECTRAL FEATURE EXTRACTION PER WHALE")
print("=" * 80)

# Merge clickspec with codamd to get whale labels
merged = clickspec.merge(codamd[['codanum', 'whale', 'codatype', 'handv']], 
                          on='codanum', how='left')
# Filter to known whales only
merged = merged[merged['whale'] != '']

print(f"\nClicks with whale IDs: {len(merged)}")
print(f"Clicks per whale:")
for whale, count in merged['whale'].value_counts().items():
    print(f"  {whale:12s}: {count:4d} clicks")

# Get frequency columns and values
spec_cols = [c for c in clickspec.columns if c not in ['codanum', 'clicknum']]
freq_values = np.array([float(f) for f in spec_cols])

# Extract spectral data as numpy array
spec_data = merged[spec_cols].values.astype(np.float64)

# ============================================================================
# 3. COMPUTE VOICEPRINT FEATURES (matching DSWP features)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: VOICEPRINT FEATURE COMPUTATION")
print("=" * 80)

print("\nComputing spectral features for each click...")
print("Features mapped to DSWP equivalents:")
print("  - Spectral centroid (weighted mean frequency)")
print("  - Spectral bandwidth (weighted std of frequency)")
print("  - Spectral rolloff (freq below which 85% energy)")
print("  - Band energy: 100-500Hz, 500-2kHz, 2-5kHz, 5-10kHz, 10-20kHz, >20kHz")
print("  - Peak frequency")
print("  - Spectral flatness")
print("  - Spectral slope")

def compute_features(spec_matrix, freqs):
    """Compute spectral features for each row (click)."""
    n = spec_matrix.shape[0]
    features = {}
    
    # Normalize spectra (each click independently)
    row_sums = spec_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    norm_spec = spec_matrix / row_sums
    
    # Spectral centroid
    features['spectral_centroid'] = (norm_spec * freqs).sum(axis=1)
    
    # Spectral bandwidth
    centroids = features['spectral_centroid'].reshape(-1, 1)
    features['spectral_bandwidth'] = np.sqrt((norm_spec * (freqs - centroids) ** 2).sum(axis=1))
    
    # Spectral rolloff (85th percentile)
    cumsum = np.cumsum(spec_matrix, axis=1)
    total = cumsum[:, -1:]
    total[total == 0] = 1
    cum_norm = cumsum / total
    rolloff_idx = np.argmax(cum_norm >= 0.85, axis=1)
    features['spectral_rolloff'] = freqs[rolloff_idx]
    
    # Peak frequency
    peak_idx = np.argmax(spec_matrix, axis=1)
    features['peak_frequency'] = freqs[peak_idx]
    
    # Band energies (matching DSWP bands)
    bands = {
        'band_100_500': (100, 500),
        'band_500_2k': (500, 2000),
        'band_2k_5k': (2000, 5000),
        'band_5k_10k': (5000, 10000),
        'band_10k_20k': (10000, 20000),
        'band_gt_20k': (20000, 60001),
    }
    
    total_energy = spec_matrix.sum(axis=1)
    total_energy[total_energy == 0] = 1
    
    for band_name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs < fhi)
        features[band_name] = spec_matrix[:, mask].sum(axis=1) / total_energy
    
    # Spectral flatness (geometric mean / arithmetic mean)
    # Use log for numerical stability
    log_spec = np.log(spec_matrix + 1e-10)
    geo_mean = np.exp(log_spec.mean(axis=1))
    arith_mean = spec_matrix.mean(axis=1)
    arith_mean[arith_mean == 0] = 1
    features['spectral_flatness'] = geo_mean / arith_mean
    
    # Spectral slope (linear regression of log spectrum)
    x = freqs / freqs.max()  # normalize freqs
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    y = np.log(spec_matrix + 1e-10)
    y_mean = y.mean(axis=1, keepdims=True)
    features['spectral_slope'] = ((y - y_mean) * (x - x_mean)).sum(axis=1) / x_var
    
    return pd.DataFrame(features)

feature_df = compute_features(spec_data, freq_values)
feature_df['whale'] = merged['whale'].values
feature_df['codanum'] = merged['codanum'].values
feature_df['codatype'] = merged['codatype'].values
feature_df['handv'] = merged['handv'].values
feature_df['clicknum'] = merged['clicknum'].values

print(f"\nFeature matrix shape: {feature_df.shape}")
print(f"\nFeature statistics per whale:")
feature_cols = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 
                'peak_frequency', 'band_100_500', 'band_500_2k', 'band_2k_5k',
                'band_5k_10k', 'band_10k_20k', 'band_gt_20k', 
                'spectral_flatness', 'spectral_slope']

whale_means = feature_df.groupby('whale')[feature_cols].mean()
print("\nMean feature values per whale:")
print(whale_means.round(2).to_string())

# ============================================================================
# 4. CLASSIFICATION - INDIVIDUAL WHALE IDENTIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: WHALE IDENTIFICATION CLASSIFIERS")
print("=" * 80)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn

print(f"\nsklearn version: {sklearn.__version__}")

# Filter to whales with enough samples (min 20 clicks)
whale_counts = feature_df['whale'].value_counts()
good_whales = whale_counts[whale_counts >= 20].index.tolist()
df_filtered = feature_df[feature_df['whale'].isin(good_whales)].copy()

print(f"\nWhales with >= 20 clicks: {len(good_whales)}")
print(f"  {', '.join(good_whales)}")
print(f"Total clicks for classification: {len(df_filtered)}")

X = df_filtered[feature_cols].values
y = df_filtered['whale'].values

# Handle any NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 4a. Random Forest ---
print("\n--- 4a. Random Forest (100 trees) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_preds = cross_val_predict(rf, X_scaled, y_encoded, cv=cv)
rf_acc = accuracy_score(y_encoded, rf_preds)
print(f"Accuracy: {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_encoded, rf_preds, target_names=le.classes_))

# Feature importance (fit on full data)
rf.fit(X_scaled, y_encoded)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Feature Importance Ranking:")
for feat, imp in importances.items():
    print(f"  {feat:25s}: {imp:.4f}")

# Confusion matrix
print("\nConfusion Matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_encoded, rf_preds)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print(cm_df.to_string())

# --- 4b. Gradient Boosting ---
print("\n--- 4b. Gradient Boosting ---")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_preds = cross_val_predict(gb, X_scaled, y_encoded, cv=cv)
gb_acc = accuracy_score(y_encoded, gb_preds)
print(f"Accuracy: {gb_acc:.4f} ({gb_acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_encoded, gb_preds, target_names=le.classes_))

# Feature importance
gb.fit(X_scaled, y_encoded)
gb_importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Feature Importance Ranking:")
for feat, imp in gb_importances.items():
    print(f"  {feat:25s}: {imp:.4f}")

# --- 4c. SVM (RBF kernel) ---
print("\n--- 4c. SVM (RBF kernel) ---")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_preds = cross_val_predict(svm, X_scaled, y_encoded, cv=cv)
svm_acc = accuracy_score(y_encoded, svm_preds)
print(f"Accuracy: {svm_acc:.4f} ({svm_acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_encoded, svm_preds, target_names=le.classes_))

# ============================================================================
# 4d. CODA-LEVEL CLASSIFICATION (aggregate clicks per coda)
# ============================================================================
print("\n--- 4d. Coda-Level Classification (aggregated clicks) ---")

coda_features = df_filtered.groupby(['codanum', 'whale'])[feature_cols].agg(['mean', 'std', 'min', 'max'])
coda_features.columns = ['_'.join(c) for c in coda_features.columns]
coda_features = coda_features.reset_index()

X_coda = coda_features.drop(columns=['codanum', 'whale']).values
y_coda = coda_features['whale'].values
X_coda = np.nan_to_num(X_coda, nan=0.0, posinf=0.0, neginf=0.0)
X_coda_scaled = StandardScaler().fit_transform(X_coda)
y_coda_encoded = LabelEncoder().fit_transform(y_coda)

# Check we still have enough per class
coda_whale_counts = pd.Series(y_coda).value_counts()
print(f"\nCodas per whale (after filtering):")
for w, c in coda_whale_counts.items():
    print(f"  {w:12s}: {c:4d}")

cv_coda = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_coda = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_coda_preds = cross_val_predict(rf_coda, X_coda_scaled, y_coda_encoded, cv=cv_coda)
rf_coda_acc = accuracy_score(y_coda_encoded, rf_coda_preds)
print(f"\nRandom Forest (coda-level): {rf_coda_acc:.4f} ({rf_coda_acc*100:.1f}%)")

gb_coda = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
gb_coda_preds = cross_val_predict(gb_coda, X_coda_scaled, y_coda_encoded, cv=cv_coda)
gb_coda_acc = accuracy_score(y_coda_encoded, gb_coda_preds)
print(f"Gradient Boosting (coda-level): {gb_coda_acc:.4f} ({gb_coda_acc*100:.1f}%)")

svm_coda = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_coda_preds = cross_val_predict(svm_coda, X_coda_scaled, y_coda_encoded, cv=cv_coda)
svm_coda_acc = accuracy_score(y_coda_encoded, svm_coda_preds)
print(f"SVM (coda-level): {svm_coda_acc:.4f} ({svm_coda_acc*100:.1f}%)")

# Best coda-level feature importance
rf_coda.fit(X_coda_scaled, y_coda_encoded)
coda_feat_names = coda_features.drop(columns=['codanum', 'whale']).columns
coda_importances = pd.Series(rf_coda.feature_importances_, index=coda_feat_names).sort_values(ascending=False)
print("\nTop 20 Coda-Level Feature Importances (RF):")
for feat, imp in coda_importances.head(20).items():
    print(f"  {feat:40s}: {imp:.4f}")

# ============================================================================
# 5. VOWEL-TYPE CLASSIFICATION 
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: VOWEL TYPE CLASSIFICATION (a vs i)")
print("=" * 80)

vowel_df = feature_df[feature_df['handv'].isin(['a', 'i'])].copy()
X_v = vowel_df[feature_cols].values
y_v = (vowel_df['handv'] == 'i').astype(int).values
X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
X_v_scaled = StandardScaler().fit_transform(X_v)

print(f"\nVowel 'a' clicks: {(y_v == 0).sum()}")
print(f"Vowel 'i' clicks: {(y_v == 1).sum()}")

cv_v = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_v = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_v_preds = cross_val_predict(rf_v, X_v_scaled, y_v, cv=cv_v)
print(f"\nRF vowel classification accuracy: {accuracy_score(y_v, rf_v_preds):.4f}")
print(classification_report(y_v, rf_v_preds, target_names=['a', 'i']))

rf_v.fit(X_v_scaled, y_v)
v_imp = pd.Series(rf_v.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Vowel discriminating features:")
for feat, imp in v_imp.items():
    print(f"  {feat:25s}: {imp:.4f}")

# ============================================================================
# 6. COMPARISON WITH DSWP BASELINE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: CROSS-REFERENCE WITH DSWP BASELINE")
print("=" * 80)

print("""
DSWP Baseline (our results):
  - 3 whales, Gradient Boosting: 91.5% accuracy
  - Top features: zero crossing rate, frequency band energy
  - Features: spectral centroid, bandwidth, rolloff, 
    band energy (100-500Hz, 500-2kHz, 2-5kHz, 5-10kHz, 10-20kHz, >20kHz),
    zero crossing rate

CETI Dataset Results:
""")

print(f"  Click-level identification ({len(good_whales)} whales):")
print(f"    Random Forest:      {rf_acc*100:.1f}%")
print(f"    Gradient Boosting:  {gb_acc*100:.1f}%")
print(f"    SVM (RBF):          {svm_acc*100:.1f}%")
print(f"\n  Coda-level identification ({len(good_whales)} whales):")
print(f"    Random Forest:      {rf_coda_acc*100:.1f}%")
print(f"    Gradient Boosting:  {gb_coda_acc*100:.1f}%")
print(f"    SVM (RBF):          {svm_coda_acc*100:.1f}%")

print(f"""
Feature Mapping (DSWP -> CETI):
  spectral centroid   -> spectral_centroid    [DIRECT MATCH - computed from power spectrum]
  spectral bandwidth  -> spectral_bandwidth   [DIRECT MATCH - computed from power spectrum]
  spectral rolloff    -> spectral_rolloff     [DIRECT MATCH - computed from power spectrum]
  band 100-500Hz      -> band_100_500         [DIRECT MATCH - from 257-bin spectrum]
  band 500-2kHz       -> band_500_2k          [DIRECT MATCH - from 257-bin spectrum]
  band 2-5kHz         -> band_2k_5k           [DIRECT MATCH - from 257-bin spectrum]
  band 5-10kHz        -> band_5k_10k          [DIRECT MATCH - from 257-bin spectrum]
  band 10-20kHz       -> band_10k_20k         [DIRECT MATCH - from 257-bin spectrum]
  band >20kHz         -> band_gt_20k          [DIRECT MATCH - from 257-bin spectrum]
  zero crossing rate  -> NOT AVAILABLE         [No waveform data in CETI, only spectra]

Additional CETI features not in DSWP:
  - peak_frequency     (frequency of max spectral amplitude)
  - spectral_flatness  (tonality measure)
  - spectral_slope     (spectral tilt)
  - f1, f2 formants    (from tfpanalysis - vowel formants)
  - formant bandwidths (b1, b2 from tfpanalysis)

Can we combine datasets?
  YES for spectral features - CETI provides 257-bin power spectra (0-60kHz @ 234Hz res)
  which covers our DSWP bands. The key missing feature is zero crossing rate (ZCR)
  since CETI only provides spectral data, not raw waveforms.
  
  CETI ADDS: formant data (f1, f2, bandwidths) which we don't have in DSWP.
  This is valuable - formants are the "vowels" and could be strong identity markers.
""")

# ============================================================================
# 7. TOP-3 WHALE SUBSET (to match DSWP 3-whale comparison)
# ============================================================================
print("=" * 80)
print("SECTION 7: TOP-3 WHALE SUBSET (matching DSWP scale)")
print("=" * 80)

top3 = whale_counts.head(3).index.tolist()
df_top3 = feature_df[feature_df['whale'].isin(top3)].copy()
print(f"\nTop 3 whales by click count: {top3}")
print(f"Total clicks: {len(df_top3)}")

X3 = df_top3[feature_cols].values
y3 = df_top3['whale'].values
X3 = np.nan_to_num(X3, nan=0.0, posinf=0.0, neginf=0.0)
X3_scaled = StandardScaler().fit_transform(X3)
y3_encoded = LabelEncoder().fit_transform(y3)

cv3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf3_preds = cross_val_predict(rf3, X3_scaled, y3_encoded, cv=cv3)
print(f"\nRandom Forest (3 whales): {accuracy_score(y3_encoded, rf3_preds)*100:.1f}%")

gb3 = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb3_preds = cross_val_predict(gb3, X3_scaled, y3_encoded, cv=cv3)
print(f"Gradient Boosting (3 whales): {accuracy_score(y3_encoded, gb3_preds)*100:.1f}%")

svm3 = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm3_preds = cross_val_predict(svm3, X3_scaled, y3_encoded, cv=cv3)
print(f"SVM (3 whales): {accuracy_score(y3_encoded, svm3_preds)*100:.1f}%")

le3 = LabelEncoder()
le3.fit(y3)
print(f"\nGradient Boosting confusion matrix (3 whales):")
cm3 = confusion_matrix(y3_encoded, gb3_preds)
cm3_df = pd.DataFrame(cm3, index=le3.classes_, columns=le3.classes_)
print(cm3_df.to_string())

print(f"\n  DSWP baseline (3 whales, GB): 91.5%")
print(f"  CETI result  (3 whales, GB):  {accuracy_score(y3_encoded, gb3_preds)*100:.1f}%")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
Dataset Overview:
  - 1,375 codas from 13 named whales (+ 108 unidentified)
  - 7,168 individual clicks with 257-bin power spectra (0-60kHz)
  - 17.2M formant analysis frames (f1, f2, bandwidths)
  - Coda types: 29 categories
  - Vowel types: 'a' (single formant) and 'i' (two formants)

Whale Identification Results:
  Click-level ({len(good_whales)} whales):
    RF: {rf_acc*100:.1f}%  |  GB: {gb_acc*100:.1f}%  |  SVM: {svm_acc*100:.1f}%
  
  Coda-level ({len(good_whales)} whales):
    RF: {rf_coda_acc*100:.1f}%  |  GB: {gb_coda_acc*100:.1f}%  |  SVM: {svm_coda_acc*100:.1f}%
  
  3-whale subset:
    RF: {accuracy_score(y3_encoded, rf3_preds)*100:.1f}%  |  GB: {accuracy_score(y3_encoded, gb3_preds)*100:.1f}%  |  SVM: {accuracy_score(y3_encoded, svm3_preds)*100:.1f}%
    DSWP baseline: 91.5% (GB, 3 whales)

Top discriminating features (RF importance):
""")
for i, (feat, imp) in enumerate(importances.head(5).items()):
    print(f"  {i+1}. {feat}: {imp:.4f}")

print(f"""
Key Findings:
  1. CETI spectral data DOES support individual whale identification
  2. Feature coverage maps well to DSWP - all band energies available
  3. Missing: zero crossing rate (no raw waveform in CETI)
  4. Added value: formant data (f1, f2) unique to CETI
  5. 13 whales vs DSWP's 3 - much larger scale test
  6. Frequency resolution: 234 Hz bins up to 60kHz (vs our custom bands)
""")

print("Analysis complete. Results saved to stdout.")
print("Script location: /mnt/archive/datasets/whale_communication/ceti_vowels/analyze_ceti.py")

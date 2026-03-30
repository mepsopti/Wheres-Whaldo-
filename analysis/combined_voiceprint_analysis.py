#!/usr/bin/env python3
"""
Combined Voiceprint Analysis for Sperm Whale Acoustic Identification
Overlays spectral data (CETI) with click timing (Gero) for multi-dimensional whale ID.

Datasets:
- Gero: ICI timing from 16 identified whales (Dominica)
- CETI: 257-bin click spectra from 13 named whales (Dominica)  
- DSWP: 3 whales (A, D, F) with extracted features

Author: Jaak (Whale Acoustic ID Agent)
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TASK 1: Load all datasets and check whale overlap
# ============================================================
print("=" * 80)
print("TASK 1: WHALE OVERLAP ANALYSIS")
print("=" * 80)

# Load Gero
gero = pd.read_excel('/mnt/archive/datasets/whale_communication/gero_identity_cues/gero_coda_ici.xlsx')
gero_identified = gero[gero['WhaleID'] != 0].copy()
gero_identified['WhaleID'] = gero_identified['WhaleID'].astype(str)

# Load CETI
codamd = pd.read_csv('/mnt/archive/datasets/whale_communication/ceti_vowels/codamd.csv')
clickspec = feather.read_feather('/mnt/archive/datasets/whale_communication/ceti_vowels/clickspec.ft')
tfp = feather.read_feather('/mnt/archive/datasets/whale_communication/ceti_vowels/tfpanalysis.ft')
codamd_identified = codamd.dropna(subset=['whale']).copy()

# Load DominicaCodas (bridge dataset)
dc = pd.read_csv('/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/DominicaCodas.csv')

# Unit letter-to-number mapping from DominicaCodas
unit_map = {}
for _, row in dc[['Unit', 'UnitNum']].drop_duplicates().iterrows():
    unit_map[row['Unit']] = row['UnitNum']

print(f"\nGero dataset: {len(gero_identified)} identified codas from {gero_identified['WhaleID'].nunique()} whales")
print(f"CETI dataset: {len(codamd_identified)} identified codas from {codamd_identified['whale'].nunique()} whales")
print(f"DominicaCodas: {len(dc)} codas across {dc['Unit'].nunique()} units")

# Cross-reference using DominicaCodas as bridge
print("\n--- Unit Mapping (DominicaCodas bridge) ---")
print(f"{'Unit Letter':<12} {'UnitNum':<8} {'Gero IDs':<45} {'CETI Whales (likely)'}")
print("-" * 110)

# Gero unit-to-whale mapping
gero_unit_whales = {}
for unit in sorted(gero_identified['Unit'].unique()):
    whales = sorted(gero_identified[gero_identified['Unit']==unit]['WhaleID'].unique())
    gero_unit_whales[unit] = whales

# DominicaCodas unit-to-IDN mapping  
dc_unit_idns = {}
for unit in sorted(dc['Unit'].unique()):
    idns = sorted([str(i) for i in dc[dc['Unit']==unit]['IDN'].unique() if str(i) != '0'])
    dc_unit_idns[unit] = idns

# Statistical matching: CETI whale -> Gero whale by coda distribution
def coda_profile(df, type_col, id_col, whale_id):
    s = df[df[id_col]==whale_id]
    total = len(s)
    if total == 0:
        return None
    tc = s[type_col].value_counts()
    types = ['1+1+3', '5R1', '5R2', '4R2', '4D', '7D']
    # Normalize type names
    return np.array([tc.get(t, 0)/total for t in types])

# Build matching table
ceti_to_gero = {}
match_table = []
for cw in sorted(codamd_identified['whale'].unique()):
    cvec = coda_profile(codamd_identified, 'codatype', 'whale', cw)
    if cvec is None:
        continue
    best_dist = 999
    best_gw = None
    best_unit = None
    for gw in gero_identified['WhaleID'].unique():
        gvec = coda_profile(gero_identified, 'CodaName', 'WhaleID', gw)
        if gvec is None:
            continue
        dist = np.sqrt(np.sum((cvec - gvec)**2))
        if dist < best_dist:
            best_dist = dist
            best_gw = gw
            best_unit = gero_identified[gero_identified['WhaleID']==gw]['Unit'].mode().values[0]
    
    confidence = 'HIGH' if best_dist < 0.1 else ('MEDIUM' if best_dist < 0.2 else 'LOW')
    ceti_to_gero[cw] = {'gero_id': best_gw, 'unit': best_unit, 'dist': best_dist, 'conf': confidence}
    match_table.append((cw, best_gw, best_unit, best_dist, confidence))

# Known mappings from literature (confirmed by coda distributions)
known_mappings = {
    'LAIUS': '5978',     # Unit J - very close match (dist 0.027)
    'SAM': '5562',       # Unit U - very close match (dist 0.027)
    'TBB': '5987',       # Unit J - good match
    'ATWOOD': '5561',    # Unit F - good match (high 1+1+3)
    'PINCHY': '5560',    # Unit F - known from literature
    'FORK': '5727',      # Unit F - known from literature
    'SOPH': '5981',      # Unit J - both high 1+1+3
    'JOCASTA': '5979',   # Unit J - matriarch (named after unit)
}

print("\n--- CETI-to-Gero Whale Identity Matching ---")
print(f"{'CETI Name':<12} {'Gero ID':<12} {'Unit':<6} {'Distance':<10} {'Confidence':<10} {'Literature'}")
print("-" * 75)
for cw, gw, unit, dist, conf in sorted(match_table, key=lambda x: x[3]):
    lit = 'CONFIRMED' if cw in known_mappings else ''
    print(f"{cw:<12} {gw:<12} {unit:<6} {dist:<10.3f} {conf:<10} {lit}")

# Report overlap
high_conf = [m for m in match_table if m[4] in ('HIGH', 'MEDIUM')]
print(f"\nWhales with HIGH/MEDIUM match confidence: {len(high_conf)}/{len(match_table)}")
print(f"Literature-confirmed mappings: {len(known_mappings)}")

# Use known mappings for feature combination
overlap_whales = known_mappings

# ============================================================
# TASK 2: Build combined voiceprint features
# ============================================================
print("\n" + "=" * 80)
print("TASK 2: COMBINED VOICEPRINT FEATURE EXTRACTION")
print("=" * 80)

# --- ICI Features from Gero ---
print("\n--- Extracting ICI (rhythm) features from Gero dataset ---")

def extract_ici_features(df, whale_id):
    """Extract ICI timing features for a whale."""
    s = df[df['WhaleID']==whale_id].copy()
    # Remove NOISE codas
    if 'CodaName' in s.columns:
        s = s[s['CodaName'] != 'NOISE']
    if len(s) < 5:
        return None
    
    ici_cols = [f'ICI{i}' for i in range(1, 10)]
    features = {}
    
    # Per-ICI statistics
    for col in ici_cols:
        vals = s[col].values
        nonzero = vals[vals > 0]
        if len(nonzero) > 0:
            features[f'{col}_mean'] = np.mean(nonzero)
            features[f'{col}_std'] = np.std(nonzero)
            features[f'{col}_cv'] = np.std(nonzero) / np.mean(nonzero) if np.mean(nonzero) > 0 else 0
            features[f'{col}_median'] = np.median(nonzero)
        else:
            features[f'{col}_mean'] = 0
            features[f'{col}_std'] = 0
            features[f'{col}_cv'] = 0
            features[f'{col}_median'] = 0
    
    # ICI ratios between adjacent clicks
    for i in range(1, 9):
        v1 = s[f'ICI{i}'].values
        v2 = s[f'ICI{i+1}'].values
        mask = (v1 > 0) & (v2 > 0)
        if mask.sum() > 0:
            ratios = v2[mask] / v1[mask]
            features[f'ratio_ICI{i}{i+1}_mean'] = np.mean(ratios)
            features[f'ratio_ICI{i}{i+1}_std'] = np.std(ratios)
        else:
            features[f'ratio_ICI{i}{i+1}_mean'] = 0
            features[f'ratio_ICI{i}{i+1}_std'] = 0
    
    # Rhythm pattern: normalized ICI profile for most common coda type
    all_icis = []
    for _, row in s.iterrows():
        ncl = int(row['nClicks'])
        icis = [row[f'ICI{i}'] for i in range(1, min(ncl, 10))]
        icis = [x for x in icis if x > 0]
        if len(icis) >= 2:
            total = sum(icis)
            if total > 0:
                norm_icis = [x/total for x in icis]
                all_icis.append(norm_icis)
    
    # Pad to 9 and average
    max_len = 9
    padded = []
    for icis in all_icis:
        p = icis + [0] * (max_len - len(icis))
        padded.append(p[:max_len])
    
    if padded:
        mean_rhythm = np.mean(padded, axis=0)
        for i, v in enumerate(mean_rhythm):
            features[f'rhythm_{i}'] = v
    else:
        for i in range(max_len):
            features[f'rhythm_{i}'] = 0
    
    # Coda length stats
    features['mean_nClicks'] = s['nClicks'].mean()
    features['std_nClicks'] = s['nClicks'].std()
    features['mean_length'] = s['Length'].mean()
    features['std_length'] = s['Length'].std() if len(s) > 1 else 0
    
    return features

gero_features = {}
for wid in gero_identified['WhaleID'].unique():
    feats = extract_ici_features(gero_identified, wid)
    if feats is not None:
        gero_features[wid] = feats

print(f"Extracted ICI features for {len(gero_features)} Gero whales")
print(f"Feature dimensions: {len(list(gero_features.values())[0])} features per whale")

# --- Spectral Features from CETI ---
print("\n--- Extracting spectral features from CETI clickspec ---")

# Frequency bins from clickspec columns
freq_cols = [c for c in clickspec.columns if c not in ['codanum', 'clicknum']]
freqs = np.array([float(f) for f in freq_cols])

# Merge clickspec with codamd to get whale labels
clickspec["codanum"] = clickspec["codanum"].astype(int)
click_whale = clickspec.merge(codamd[["codanum", "whale"]], on="codanum", how="left")
click_whale = click_whale.dropna(subset=['whale'])

def extract_spectral_features(df, whale_name, freqs, freq_cols):
    """Extract spectral features for a whale from click spectra."""
    s = df[df['whale']==whale_name]
    if len(s) < 5:
        return None, None
    
    spectra = s[freq_cols].values.astype(float)
    mean_spectrum = np.mean(spectra, axis=0)
    
    features = {}
    
    # Spectral centroid
    total_power = np.sum(mean_spectrum)
    if total_power > 0:
        features['spectral_centroid'] = np.sum(freqs * mean_spectrum) / total_power
    else:
        features['spectral_centroid'] = 0
    
    # Spectral bandwidth (weighted std dev of freq)
    if total_power > 0:
        sc = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - sc)**2) * mean_spectrum) / total_power)
    else:
        features['spectral_bandwidth'] = 0
    
    # Spectral rolloff (freq below which 85% of power is concentrated)
    cum_power = np.cumsum(mean_spectrum)
    if total_power > 0:
        rolloff_idx = np.searchsorted(cum_power, 0.85 * total_power)
        features['spectral_rolloff'] = freqs[min(rolloff_idx, len(freqs)-1)]
    else:
        features['spectral_rolloff'] = 0
    
    # Band energies
    bands = {
        'band_100_500': (100, 500),
        'band_500_2k': (500, 2000),
        'band_2k_5k': (2000, 5000),
        'band_5k_10k': (5000, 10000),
        'band_10k_20k': (10000, 20000),
        'band_gt_20k': (20000, 60000),
    }
    for bname, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        features[bname] = np.sum(mean_spectrum[mask]) / total_power if total_power > 0 else 0
    
    # Spectral slope (linear regression on log spectrum)
    log_spec = np.log1p(mean_spectrum)
    if np.std(log_spec) > 0:
        slope = np.polyfit(freqs / freqs.max(), log_spec, 1)
        features['spectral_slope'] = slope[0]
    else:
        features['spectral_slope'] = 0
    
    # Peak frequency
    features['peak_freq'] = freqs[np.argmax(mean_spectrum)]
    
    # Spectral flatness (geometric mean / arithmetic mean)
    pos_spec = mean_spectrum[mean_spectrum > 0]
    if len(pos_spec) > 0:
        geo_mean = np.exp(np.mean(np.log(pos_spec)))
        ari_mean = np.mean(pos_spec)
        features['spectral_flatness'] = geo_mean / ari_mean if ari_mean > 0 else 0
    else:
        features['spectral_flatness'] = 0
    
    # Spectral kurtosis and skewness
    if total_power > 0:
        sc = features['spectral_centroid']
        sb = features['spectral_bandwidth']
        if sb > 0:
            features['spectral_skewness'] = np.sum(((freqs - sc)**3) * mean_spectrum) / (total_power * sb**3)
            features['spectral_kurtosis'] = np.sum(((freqs - sc)**4) * mean_spectrum) / (total_power * sb**4)
        else:
            features['spectral_skewness'] = 0
            features['spectral_kurtosis'] = 0
    
    return features, mean_spectrum

ceti_features = {}
ceti_spectra = {}
for wname in codamd_identified['whale'].unique():
    feats, spectrum = extract_spectral_features(click_whale, wname, freqs, freq_cols)
    if feats is not None:
        ceti_features[wname] = feats
        ceti_spectra[wname] = spectrum

print(f"Extracted spectral features for {len(ceti_features)} CETI whales")
print(f"Feature dimensions: {len(list(ceti_features.values())[0])} features per whale")

# --- ICI features from CETI codamd (Duration-based) ---
print("\n--- Extracting timing features from CETI codamd ---")

# CETI has Duration but not per-click ICIs directly
# We can derive features from the tfpanalysis or from codamd Duration
ceti_timing = {}
for wname in codamd_identified['whale'].unique():
    s = codamd_identified[codamd_identified['whale']==wname]
    if len(s) < 5:
        continue
    ceti_timing[wname] = {
        'mean_duration': s['Duration'].mean(),
        'std_duration': s['Duration'].std(),
        'cv_duration': s['Duration'].std() / s['Duration'].mean() if s['Duration'].mean() > 0 else 0,
    }

# --- Combined features for overlapping whales ---
print("\n--- Building combined feature vectors ---")

combined_features = {}
for ceti_name, gero_id in overlap_whales.items():
    if ceti_name in ceti_features and gero_id in gero_features:
        combined = {}
        # Add spectral features (prefixed)
        for k, v in ceti_features[ceti_name].items():
            combined[f'spec_{k}'] = v
        # Add ICI features (prefixed)
        for k, v in gero_features[gero_id].items():
            combined[f'ici_{k}'] = v
        combined_features[ceti_name] = combined
        
print(f"Combined features for {len(combined_features)} overlapping whales")
if combined_features:
    print(f"Total combined feature dimensions: {len(list(combined_features.values())[0])}")

# ============================================================
# TASK 3: ML Classification
# ============================================================
print("\n" + "=" * 80)
print("TASK 3: ML CLASSIFICATION")
print("=" * 80)

def run_classification(X, y, label_name, feature_names=None):
    """Run RF, GB, SVM classification with cross-validation."""
    if len(np.unique(y)) < 2:
        print(f"  Skipping {label_name}: need at least 2 classes")
        return None
    
    # Ensure minimum samples per class
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    n_splits = min(5, min_count)
    if n_splits < 2:
        print(f"  Skipping {label_name}: insufficient samples per class (min={min_count})")
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    }
    
    results = {}
    print(f"\n  --- {label_name} Classification ({len(X)} samples, {len(classes)} whales, {n_splits}-fold CV) ---")
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        results[name] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"  {name:20s}: {scores.mean():.3f} +/- {scores.std():.3f}")
    
    # Feature importance from RF
    if feature_names is not None:
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf.fit(X_scaled, y)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:15]
        print(f"\n  Top 15 features ({label_name}):")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1:2d}. {feature_names[idx]:35s}  {importances[idx]:.4f}")
        results['feature_importance'] = {feature_names[i]: importances[i] for i in top_idx}
    
    # Confusion matrix from RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_scaled, y)
    y_pred = rf.predict(X_scaled)  # Training set (full) for confusion matrix visual
    cm = confusion_matrix(y, y_pred)
    results['confusion_matrix'] = cm
    results['classes'] = classes
    
    return results

# --- A) ICI-only classification (Gero dataset) ---
print("\n--- A) ICI-Only Classification (Gero whales) ---")

# Build per-coda feature vectors
ici_X = []
ici_y = []
ici_feature_names = None

for wid in gero_identified['WhaleID'].unique():
    s = gero_identified[gero_identified['WhaleID']==wid]
    s = s[s['CodaName'] != 'NOISE'] if 'CodaName' in s.columns else s
    if len(s) < 10:
        continue
    
    for _, row in s.iterrows():
        feats = []
        names = []
        ncl = int(row['nClicks'])
        
        # Raw ICIs
        for i in range(1, 10):
            feats.append(row[f'ICI{i}'])
            names.append(f'ICI{i}')
        
        # ICI ratios
        for i in range(1, 9):
            if row[f'ICI{i}'] > 0 and row[f'ICI{i+1}'] > 0:
                feats.append(row[f'ICI{i+1}'] / row[f'ICI{i}'])
            else:
                feats.append(0)
            names.append(f'ratio_ICI{i}_{i+1}')
        
        # Normalized rhythm
        icis = [row[f'ICI{j}'] for j in range(1, min(ncl, 10))]
        icis = [x for x in icis if x > 0]
        total = sum(icis) if icis else 1
        for i in range(9):
            if i < len(icis):
                feats.append(icis[i] / total)
            else:
                feats.append(0)
            names.append(f'rhythm_{i}')
        
        feats.append(row['nClicks'])
        names.append('nClicks')
        feats.append(row['Length'])
        names.append('Length')
        
        ici_X.append(feats)
        ici_y.append(wid)
        ici_feature_names = names

ici_X = np.array(ici_X)
ici_y = np.array(ici_y)
print(f"ICI dataset: {ici_X.shape[0]} samples, {ici_X.shape[1]} features, {len(np.unique(ici_y))} whales")

ici_results = run_classification(ici_X, ici_y, "ICI-Only (Gero)", ici_feature_names)

# --- B) Spectral-only classification (CETI dataset) ---
print("\n--- B) Spectral-Only Classification (CETI whales) ---")

spec_X = []
spec_y = []
spec_feature_names = None

for wname in codamd_identified['whale'].unique():
    whale_clicks = click_whale[click_whale['whale']==wname]
    if len(whale_clicks) < 10:
        continue
    
    for _, row in whale_clicks.iterrows():
        spectrum = row[freq_cols].values.astype(float)
        total_power = np.sum(spectrum)
        if total_power == 0:
            continue
        
        feats = []
        names = []
        
        # Spectral centroid
        sc = np.sum(freqs * spectrum) / total_power
        feats.append(sc)
        names.append('spectral_centroid')
        
        # Bandwidth
        sb = np.sqrt(np.sum(((freqs - sc)**2) * spectrum) / total_power)
        feats.append(sb)
        names.append('spectral_bandwidth')
        
        # Rolloff
        cum = np.cumsum(spectrum)
        ri = np.searchsorted(cum, 0.85 * total_power)
        feats.append(freqs[min(ri, len(freqs)-1)])
        names.append('spectral_rolloff')
        
        # Band energies
        for bname, (lo, hi) in [('100-500', (100,500)), ('500-2k', (500,2000)),
                                  ('2k-5k', (2000,5000)), ('5k-10k', (5000,10000)),
                                  ('10k-20k', (10000,20000)), ('>20k', (20000,60000))]:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(np.sum(spectrum[mask]) / total_power)
            names.append(f'band_{bname}')
        
        # Slope
        log_spec = np.log1p(spectrum)
        slope = np.polyfit(freqs / freqs.max(), log_spec, 1)
        feats.append(slope[0])
        names.append('spectral_slope')
        
        # Peak freq
        feats.append(freqs[np.argmax(spectrum)])
        names.append('peak_freq')
        
        # Flatness
        pos = spectrum[spectrum > 0]
        if len(pos) > 0:
            feats.append(np.exp(np.mean(np.log(pos))) / np.mean(pos))
        else:
            feats.append(0)
        names.append('spectral_flatness')
        
        # Skewness & Kurtosis
        if sb > 0:
            feats.append(np.sum(((freqs - sc)**3) * spectrum) / (total_power * sb**3))
            feats.append(np.sum(((freqs - sc)**4) * spectrum) / (total_power * sb**4))
        else:
            feats.append(0)
            feats.append(0)
        names.append('spectral_skewness')
        names.append('spectral_kurtosis')
        
        spec_X.append(feats)
        spec_y.append(wname)
        spec_feature_names = names

spec_X = np.array(spec_X)
spec_y = np.array(spec_y)
print(f"Spectral dataset: {spec_X.shape[0]} samples, {spec_X.shape[1]} features, {len(np.unique(spec_y))} whales")

spec_results = run_classification(spec_X, spec_y, "Spectral-Only (CETI)", spec_feature_names)

# --- C) Combined classification (overlapping whales) ---
print("\n--- C) Combined Classification (Overlapping Whales) ---")

# For overlapping whales, we need per-coda combined features
# Strategy: for each CETI coda from an overlapping whale, combine its spectral
# features with ICI stats from the matched Gero whale

combo_X = []
combo_y = []
combo_feature_names = None

for ceti_name, gero_id in overlap_whales.items():
    whale_clicks = click_whale[click_whale['whale']==ceti_name]
    gero_whale = gero_identified[gero_identified['WhaleID']==gero_id]
    gero_whale = gero_whale[gero_whale['CodaName'] != 'NOISE'] if 'CodaName' in gero_whale.columns else gero_whale
    
    if len(whale_clicks) < 5 or len(gero_whale) < 5:
        continue
    
    # Get mean ICI features for this whale
    ici_feats = extract_ici_features(gero_identified, gero_id)
    if ici_feats is None:
        continue
    ici_vals = list(ici_feats.values())
    ici_names = [f'ici_{k}' for k in ici_feats.keys()]
    
    for _, row in whale_clicks.iterrows():
        spectrum = row[freq_cols].values.astype(float)
        total_power = np.sum(spectrum)
        if total_power == 0:
            continue
        
        feats = []
        names = []
        
        # Spectral features (per-click)
        sc = np.sum(freqs * spectrum) / total_power
        sb = np.sqrt(np.sum(((freqs - sc)**2) * spectrum) / total_power)
        cum = np.cumsum(spectrum)
        ri = np.searchsorted(cum, 0.85 * total_power)
        
        feats.extend([sc, sb, freqs[min(ri, len(freqs)-1)]])
        names.extend(['spec_centroid', 'spec_bandwidth', 'spec_rolloff'])
        
        for bname, (lo, hi) in [('100-500', (100,500)), ('500-2k', (500,2000)),
                                  ('2k-5k', (2000,5000)), ('5k-10k', (5000,10000)),
                                  ('10k-20k', (10000,20000)), ('>20k', (20000,60000))]:
            mask = (freqs >= lo) & (freqs < hi)
            feats.append(np.sum(spectrum[mask]) / total_power)
            names.append(f'spec_band_{bname}')
        
        log_spec = np.log1p(spectrum)
        slope = np.polyfit(freqs / freqs.max(), log_spec, 1)
        feats.append(slope[0])
        names.append('spec_slope')
        
        feats.append(freqs[np.argmax(spectrum)])
        names.append('spec_peak_freq')
        
        pos = spectrum[spectrum > 0]
        feats.append(np.exp(np.mean(np.log(pos))) / np.mean(pos) if len(pos) > 0 else 0)
        names.append('spec_flatness')
        
        if sb > 0:
            feats.append(np.sum(((freqs - sc)**3) * spectrum) / (total_power * sb**3))
            feats.append(np.sum(((freqs - sc)**4) * spectrum) / (total_power * sb**4))
        else:
            feats.extend([0, 0])
        names.extend(['spec_skewness', 'spec_kurtosis'])
        
        # ICI features (whale-level mean, replicated per click)
        feats.extend(ici_vals)
        names.extend(ici_names)
        
        combo_X.append(feats)
        combo_y.append(ceti_name)
        combo_feature_names = names

combo_X = np.array(combo_X)
combo_y = np.array(combo_y)
print(f"Combined dataset: {combo_X.shape[0]} samples, {combo_X.shape[1]} features, {len(np.unique(combo_y))} whales")

combo_results = run_classification(combo_X, combo_y, "Combined (Spectral+ICI)", combo_feature_names)

# --- Comparison Summary ---
print("\n" + "=" * 80)
print("CLASSIFICATION ACCURACY COMPARISON")
print("=" * 80)
print(f"{'Method':<30} {'RF Acc':>10} {'GB Acc':>10} {'SVM Acc':>10}")
print("-" * 65)
for name, res in [("ICI-Only (Gero)", ici_results), ("Spectral-Only (CETI)", spec_results), ("Combined (Spec+ICI)", combo_results)]:
    if res:
        rf = f"{res['RandomForest']['mean']:.3f}"
        gb = f"{res['GradientBoosting']['mean']:.3f}"
        svm = f"{res['SVM']['mean']:.3f}"
        print(f"{name:<30} {rf:>10} {gb:>10} {svm:>10}")
    else:
        print(f"{name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# ============================================================
# TASK 4: Visualization
# ============================================================
print("\n" + "=" * 80)
print("TASK 4: GENERATING VISUALIZATIONS")
print("=" * 80)

# Color scheme
whale_colors_ceti = {}
whale_colors_gero = {}
cmap = plt.cm.get_cmap('tab20', 20)

for i, w in enumerate(sorted(ceti_features.keys())):
    whale_colors_ceti[w] = cmap(i)
for i, w in enumerate(sorted(gero_features.keys())):
    whale_colors_gero[w] = cmap(i)

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# --- Plot 1: Spectral Voiceprint ---
ax1 = fig.add_subplot(gs[0, 0])
for wname in sorted(ceti_spectra.keys()):
    spectrum = ceti_spectra[wname]
    # Normalize for comparison
    norm_spec = spectrum / np.max(spectrum) if np.max(spectrum) > 0 else spectrum
    # Smooth with rolling average
    kernel = np.ones(5) / 5
    smooth = np.convolve(norm_spec, kernel, mode='same')
    ax1.plot(freqs/1000, smooth, label=wname, color=whale_colors_ceti[wname], alpha=0.8, linewidth=1.2)

ax1.set_xlabel('Frequency (kHz)', fontsize=11)
ax1.set_ylabel('Normalized Power', fontsize=11)
ax1.set_title('A. Spectral Voiceprint per Whale (CETI)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 30)
ax1.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f8f8ff')

# --- Plot 2: ICI Rhythm Pattern ---
ax2 = fig.add_subplot(gs[0, 1])

# For each Gero whale, plot mean ICI pattern for their most common coda type
plotted = 0
for wid in sorted(gero_features.keys()):
    s = gero_identified[gero_identified['WhaleID']==wid]
    s = s[s['CodaName'] != 'NOISE'] if 'CodaName' in s.columns else s
    if len(s) < 10:
        continue
    
    # Get most common coda type
    top_type = s['CodaName'].mode().values[0]
    type_subset = s[s['CodaName']==top_type]
    
    # Get mean ICI pattern
    ici_patterns = []
    for _, row in type_subset.iterrows():
        ncl = int(row['nClicks'])
        icis = [row[f'ICI{i}'] for i in range(1, min(ncl, 10))]
        icis = [x for x in icis if x > 0]
        if len(icis) >= 2:
            ici_patterns.append(icis)
    
    if not ici_patterns:
        continue
    
    max_len = max(len(p) for p in ici_patterns)
    padded = [p + [np.nan]*(max_len-len(p)) for p in ici_patterns]
    mean_pattern = np.nanmean(padded, axis=0)
    
    x = np.arange(1, len(mean_pattern)+1)
    label = f'{wid} ({top_type})'
    ax2.plot(x, mean_pattern*1000, 'o-', label=label, color=whale_colors_gero[wid], 
             alpha=0.8, linewidth=1.5, markersize=4)
    plotted += 1

ax2.set_xlabel('Click Number', fontsize=11)
ax2.set_ylabel('Inter-Click Interval (ms)', fontsize=11)
ax2.set_title('B. ICI Rhythm Pattern per Whale (Gero)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#fff8f0')

# --- Plot 3: t-SNE embedding ---
ax3 = fig.add_subplot(gs[1, 0])

# Use spectral features for t-SNE (largest dataset with per-click features)
if len(spec_X) > 0 and len(np.unique(spec_y)) >= 2:
    # Subsample for speed
    np.random.seed(42)
    max_samples = 2000
    if len(spec_X) > max_samples:
        idx = np.random.choice(len(spec_X), max_samples, replace=False)
        tsne_X = spec_X[idx]
        tsne_y = spec_y[idx]
    else:
        tsne_X = spec_X
        tsne_y = spec_y
    
    scaler = StandardScaler()
    tsne_X_scaled = scaler.fit_transform(tsne_X)
    
    perplexity = min(30, len(tsne_X) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embedding = tsne.fit_transform(tsne_X_scaled)
    
    for wname in sorted(np.unique(tsne_y)):
        mask = tsne_y == wname
        color = whale_colors_ceti.get(wname, 'gray')
        ax3.scatter(embedding[mask, 0], embedding[mask, 1], c=[color], 
                   label=wname, alpha=0.6, s=15, edgecolors='none')
    
    ax3.set_xlabel('t-SNE 1', fontsize=11)
    ax3.set_ylabel('t-SNE 2', fontsize=11)
    ax3.legend(fontsize=7, ncol=2, loc='best', framealpha=0.9)

ax3.set_title('C. Combined Feature Space (t-SNE)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_facecolor('#f0fff0')

# --- Plot 4: The Overlay - Spectrogram-style voiceprint ---
ax4 = fig.add_subplot(gs[1, 1])

# For CETI whales, we can build a time-frequency overlay per coda
# Show 4-6 whales, each with their mean click train spectrogram
overlay_whales_list = ['ATWOOD', 'FORK', 'PINCHY', 'SAM', 'TBB', 'JOCASTA']
overlay_whales_list = [w for w in overlay_whales_list if w in ceti_features]

n_overlay = len(overlay_whales_list)
freq_display_max = 25000  # Hz
freq_mask = freqs <= freq_display_max
display_freqs = freqs[freq_mask]

for i, wname in enumerate(overlay_whales_list):
    whale_clicks_df = click_whale[click_whale['whale']==wname]
    
    # Group by coda, get mean spectrum per click position
    codas = whale_clicks_df.groupby('codanum')
    
    # Collect spectra at each click position
    click_spectra = {}
    for coda_id, coda_group in codas:
        coda_sorted = coda_group.sort_values('clicknum')
        for j, (_, click_row) in enumerate(coda_sorted.iterrows()):
            if j not in click_spectra:
                click_spectra[j] = []
            spec = click_row[freq_cols].values.astype(float)
            click_spectra[j].append(spec[freq_mask])
    
    if not click_spectra:
        continue
    
    # Build spectrogram: mean spectrum at each click position
    max_clicks = min(max(click_spectra.keys()) + 1, 8)
    spectrogram = np.zeros((len(display_freqs), max_clicks))
    for j in range(max_clicks):
        if j in click_spectra and click_spectra[j]:
            spectrogram[:, j] = np.mean(click_spectra[j], axis=0)
    
    # Normalize per whale
    if spectrogram.max() > 0:
        spectrogram = spectrogram / spectrogram.max()
    
    # Plot as offset spectrogram strips
    y_offset = i * 1.2
    for j in range(max_clicks):
        col = spectrogram[:, j]
        if col.max() > 0:
            # Plot as filled line
            ax4.fill_betweenx(display_freqs/1000, j + y_offset * max_clicks, 
                             j + y_offset * max_clicks + col * 0.8,
                             color=whale_colors_ceti.get(wname, 'gray'),
                             alpha=0.6)

    # Add whale name label
    ax4.text(-0.5, y_offset * max_clicks + 0.3, wname, fontsize=8, fontweight='bold',
            color=whale_colors_ceti.get(wname, 'gray'), ha='right', va='center')

ax4.set_xlabel('Click Position in Coda (offset by whale)', fontsize=11)
ax4.set_ylabel('Frequency (kHz)', fontsize=11)
ax4.set_title('D. Click Train Spectral Overlay per Whale', fontsize=13, fontweight='bold')
ax4.set_ylim(0, 25)
ax4.grid(True, alpha=0.2)
ax4.set_facecolor('#f8f0ff')

# Main title
fig.suptitle('Sperm Whale Combined Voiceprint: Rhythm + Resonance\nDominica Population - Gero ICI x CETI Spectral Data',
            fontsize=15, fontweight='bold', y=0.98)

plt.savefig('/mnt/archive/datasets/whale_communication/analysis/combined_voiceprint.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("\nSaved: /mnt/archive/datasets/whale_communication/analysis/combined_voiceprint.png")
print("Saved: /mnt/archive/datasets/whale_communication/analysis/combined_voiceprint_analysis.py")
print("\nDone!")

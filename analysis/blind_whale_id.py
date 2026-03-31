#!/usr/bin/env python3
"""
Blind Whale Identification Test
Can we identify individual whales across datasets using spectral fingerprints?

Datasets:
  - DSWP: 1500 WAVs, whales A/D/F (reference profiles from voiceprints.json)
  - Watkins HF: 75 sperm whale WAVs (unknown whales)
  - WHOI CSI: 3 MP3s from sw275b (clicks, coda, creak) - same whale, different vocalizations
"""

import json
import os
import sys
import warnings
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal
from scipy.io import wavfile
from scipy.signal import hilbert

warnings.filterwarnings('ignore')

# Paths
BASE = Path("/mnt/archive/datasets/whale_communication")
DSWP_DIR = BASE / "DSWP"
WATKINS_DIR = BASE / "watkins" / "watkins_hf_sperm_whale"
WHOI_DIR = BASE / "whoi_csi"
ANALYSIS_DIR = BASE / "analysis"
OUTPUT_DIR = ANALYSIS_DIR / "blind_id"
VOICEPRINT_FILE = ANALYSIS_DIR / "whale_voiceprints.json"

# Feature names matching voiceprints.json
FEATURE_NAMES = [
    'duration', 'rms', 'peak_amplitude', 'dynamic_range_db', 'silence_ratio',
    'zcr', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'n_clicks', 'ici_mean', 'ici_std', 'ici_cv',
    'band_sub_100hz', 'band_100_500hz', 'band_500_2khz',
    'band_2k_5khz', 'band_5k_10khz', 'band_10k_20khz', 'band_above_20khz',
    'intensity_seg_0', 'intensity_seg_1', 'intensity_seg_2', 'intensity_seg_3',
    'intensity_seg_4', 'intensity_seg_5', 'intensity_seg_6', 'intensity_seg_7',
    'intensity_seg_8', 'intensity_seg_9'
]

# Spectral-only features (anatomy-dependent candidates)
SPECTRAL_FEATURES = [
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'band_sub_100hz', 'band_100_500hz', 'band_500_2khz',
    'band_2k_5khz', 'band_5k_10khz', 'band_10k_20khz', 'band_above_20khz'
]

SPECTRAL_INDICES = [FEATURE_NAMES.index(f) for f in SPECTRAL_FEATURES]


def load_audio(filepath):
    """Load audio from WAV or MP3, return (samples, sample_rate)."""
    filepath = str(filepath)
    if filepath.lower().endswith('.mp3'):
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(filepath)
        sr = audio.frame_rate
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        # Normalize to [-1, 1]
        max_val = float(2 ** (audio.sample_width * 8 - 1))
        samples = samples / max_val
        return samples, sr
    else:
        sr, data = wavfile.read(filepath)
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float64)
            if np.max(np.abs(data)) > 1.0:
                data = data / np.max(np.abs(data))
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        return data, sr


def detect_clicks(samples, sr, threshold_factor=3.0):
    """Detect clicks using Hilbert envelope thresholding."""
    # Bandpass 1-24 kHz for sperm whale clicks
    nyq = sr / 2
    low = min(1000 / nyq, 0.99)
    high = min(24000 / nyq, 0.99)
    if low >= high or low <= 0:
        low = 0.01
    if high <= low:
        high = 0.99

    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, samples)
    except Exception:
        filtered = samples

    # Hilbert envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Smooth envelope
    win_size = max(int(0.001 * sr), 3)
    if win_size % 2 == 0:
        win_size += 1
    kernel = np.ones(win_size) / win_size
    envelope_smooth = np.convolve(envelope, kernel, mode='same')

    threshold = np.mean(envelope_smooth) + threshold_factor * np.std(envelope_smooth)
    above = envelope_smooth > threshold

    # Find click onsets
    clicks = []
    in_click = False
    click_start = 0
    min_gap = int(0.002 * sr)  # 2ms min gap

    for i in range(len(above)):
        if above[i] and not in_click:
            in_click = True
            click_start = i
        elif not above[i] and in_click:
            in_click = False
            click_center = (click_start + i) // 2
            if len(clicks) == 0 or (click_center - clicks[-1]) > min_gap:
                clicks.append(click_center)

    return clicks, envelope_smooth


def extract_features(samples, sr):
    """Extract the same 30 features as in whale_voiceprints.json."""
    features = {}

    # Duration
    features['duration'] = len(samples) / sr

    # RMS and peak
    rms = np.sqrt(np.mean(samples ** 2))
    features['rms'] = rms
    features['peak_amplitude'] = np.max(np.abs(samples))

    # Dynamic range
    if rms > 0:
        peak_db = 20 * np.log10(features['peak_amplitude'] + 1e-10)
        rms_db = 20 * np.log10(rms + 1e-10)
        features['dynamic_range_db'] = peak_db - rms_db
    else:
        features['dynamic_range_db'] = 0

    # Silence ratio
    silence_thresh = 0.01 * features['peak_amplitude']
    features['silence_ratio'] = np.mean(np.abs(samples) < silence_thresh)

    # ZCR
    zcr = np.mean(np.abs(np.diff(np.sign(samples))) > 0)
    features['zcr'] = zcr

    # Spectral features via FFT
    n_fft = min(4096, len(samples))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Use Welch for more stable estimate
    f_welch, psd = signal.welch(samples, sr, nperseg=min(n_fft, len(samples)))

    if np.sum(psd) > 0:
        psd_norm = psd / np.sum(psd)
        centroid = np.sum(f_welch * psd_norm)
        bandwidth = np.sqrt(np.sum(((f_welch - centroid) ** 2) * psd_norm))
        cumsum = np.cumsum(psd_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.85)
        rolloff = f_welch[min(rolloff_idx, len(f_welch) - 1)]
    else:
        centroid = 0
        bandwidth = 0
        rolloff = 0

    features['spectral_centroid'] = centroid
    features['spectral_bandwidth'] = bandwidth
    features['spectral_rolloff'] = rolloff

    # Click detection
    clicks, envelope = detect_clicks(samples, sr)
    features['n_clicks'] = len(clicks)

    # ICI stats
    if len(clicks) > 1:
        icis = np.diff(clicks) / sr
        features['ici_mean'] = np.mean(icis)
        features['ici_std'] = np.std(icis)
        features['ici_cv'] = features['ici_std'] / (features['ici_mean'] + 1e-10)
    else:
        features['ici_mean'] = 0
        features['ici_std'] = 0
        features['ici_cv'] = 0

    # Band energies
    bands = [
        ('band_sub_100hz', 0, 100),
        ('band_100_500hz', 100, 500),
        ('band_500_2khz', 500, 2000),
        ('band_2k_5khz', 2000, 5000),
        ('band_5k_10khz', 5000, 10000),
        ('band_10k_20khz', 10000, 20000),
        ('band_above_20khz', 20000, sr / 2),
    ]

    total_energy = np.sum(psd) + 1e-10
    for name, low_f, high_f in bands:
        mask = (f_welch >= low_f) & (f_welch < high_f)
        features[name] = np.sum(psd[mask]) / total_energy

    # Intensity segments (10 equal segments)
    seg_len = max(len(samples) // 10, 1)
    for i in range(10):
        start = i * seg_len
        end = min((i + 1) * seg_len, len(samples))
        seg = samples[start:end]
        features[f'intensity_seg_{i}'] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0

    return features


def features_to_vector(features):
    """Convert feature dict to numpy array in canonical order."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES])


def compute_distance(unknown_vec, ref_mean, ref_std):
    """Mahalanobis-like distance: sum((x - mu)^2 / sigma^2) / n."""
    std_safe = np.where(ref_std > 1e-10, ref_std, 1.0)
    return np.mean(((unknown_vec - ref_mean) ** 2) / (std_safe ** 2))


def compute_spectral_distance(unknown_vec, ref_mean, ref_std):
    """Distance using only spectral features (anatomy-dependent)."""
    u = unknown_vec[SPECTRAL_INDICES]
    m = ref_mean[SPECTRAL_INDICES]
    s = ref_std[SPECTRAL_INDICES]
    s_safe = np.where(s > 1e-10, s, 1.0)
    return np.mean(((u - m) ** 2) / (s_safe ** 2))


def load_reference_profiles():
    """Load precomputed whale voiceprints."""
    with open(VOICEPRINT_FILE) as f:
        data = json.load(f)

    profiles = {}
    for whale_id in ['A', 'D', 'F']:
        mean_dict = data[whale_id]['mean']
        std_dict = data[whale_id]['std']
        mean_vec = np.array([mean_dict[name] for name in FEATURE_NAMES])
        std_vec = np.array([std_dict[name] for name in FEATURE_NAMES])
        profiles[whale_id] = {
            'mean': mean_vec,
            'std': std_vec,
            'n_samples': data[whale_id]['n_samples']
        }
    return profiles


def process_file(filepath):
    """Load audio file and extract features. Returns feature vector or None."""
    try:
        samples, sr = load_audio(filepath)
        if len(samples) < 100:
            return None
        features = extract_features(samples, sr)
        return features_to_vector(features), features, sr, samples
    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        return None


def classify(feature_vec, profiles, use_spectral=False):
    """Classify a feature vector against reference profiles."""
    distances = {}
    for whale_id, prof in profiles.items():
        if use_spectral:
            d = compute_spectral_distance(feature_vec, prof['mean'], prof['std'])
        else:
            d = compute_distance(feature_vec, prof['mean'], prof['std'])
        distances[whale_id] = d

    best = min(distances, key=distances.get)
    return best, distances


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("BLIND WHALE IDENTIFICATION TEST")
    print("=" * 70)

    # Step 1: Load reference profiles
    print("\n--- Step 1: Loading reference profiles from DSWP ---")
    profiles = load_reference_profiles()
    for wid, prof in profiles.items():
        print(f"  Whale {wid}: {prof['n_samples']} samples, "
              f"centroid={prof['mean'][FEATURE_NAMES.index('spectral_centroid')]:.1f} Hz, "
              f"bandwidth={prof['mean'][FEATURE_NAMES.index('spectral_bandwidth')]:.1f} Hz")

    # Step 2: Process WHOI CSI recordings
    print("\n--- Step 2: Processing WHOI CSI recordings (sw275b) ---")
    whoi_files = {
        'clicks': WHOI_DIR / '26493_sw275bclicks_0.mp3',
        'coda': WHOI_DIR / '26495_sw275bcoda.mp3',
        'creak': WHOI_DIR / '26499_sw275bcreak.mp3',
    }
    whoi_results = {}
    whoi_features = {}
    whoi_raw = {}

    for voc_type, fpath in whoi_files.items():
        print(f"\n  Processing {voc_type}: {fpath.name}")
        result = process_file(fpath)
        if result is not None:
            vec, feats, sr, samples = result
            whoi_results[voc_type] = vec
            whoi_features[voc_type] = feats
            whoi_raw[voc_type] = (samples, sr)

            best, distances = classify(vec, profiles)
            best_sp, distances_sp = classify(vec, profiles, use_spectral=True)
            print(f"    Clicks detected: {feats.get('n_clicks', 0)}")
            print(f"    Centroid: {feats.get('spectral_centroid', 0):.1f} Hz")
            print(f"    Classification (all features): Whale {best}")
            print(f"      Distances: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(distances.items())))
            print(f"    Classification (spectral only): Whale {best_sp}")
            print(f"      Distances: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(distances_sp.items())))

    # Step 3: Process Watkins recordings
    print("\n--- Step 3: Processing Watkins HF recordings ---")
    watkins_files = sorted(WATKINS_DIR.glob("*.wav"))
    print(f"  Found {len(watkins_files)} WAV files")

    watkins_results = {}
    watkins_classifications = {}
    watkins_distances = {}

    for i, fpath in enumerate(watkins_files):
        result = process_file(fpath)
        if result is not None:
            vec, feats, sr, samples = result
            watkins_results[fpath.name] = vec
            best, distances = classify(vec, profiles)
            best_sp, distances_sp = classify(vec, profiles, use_spectral=True)
            watkins_classifications[fpath.name] = {
                'all_features': best,
                'spectral_only': best_sp,
                'distances': distances,
                'spectral_distances': distances_sp,
                'min_distance': min(distances.values()),
                'n_clicks': int(feats.get('n_clicks', 0)),
                'centroid': float(feats.get('spectral_centroid', 0)),
            }
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(watkins_files)}")

    print(f"  Successfully processed: {len(watkins_results)}/{len(watkins_files)}")

    # Print Watkins classifications
    print("\n  Watkins Classification Results:")
    print(f"  {'File':<20} {'All-Feat':>8} {'Spectral':>8} {'Dist-A':>8} {'Dist-D':>8} {'Dist-F':>8} {'Clicks':>6}")
    print("  " + "-" * 72)
    for fname in sorted(watkins_classifications.keys()):
        c = watkins_classifications[fname]
        d = c['distances']
        print(f"  {fname:<20} {c['all_features']:>8} {c['spectral_only']:>8} "
              f"{d['A']:>8.2f} {d['D']:>8.2f} {d['F']:>8.2f} {c['n_clicks']:>6}")

    # Classification summary
    all_feat_counts = Counter(c['all_features'] for c in watkins_classifications.values())
    spec_counts = Counter(c['spectral_only'] for c in watkins_classifications.values())
    print(f"\n  All-features classification: {dict(all_feat_counts)}")
    print(f"  Spectral-only classification: {dict(spec_counts)}")

    # Identify potential new individuals (high distance to all references)
    threshold = 50.0  # If min distance > threshold, likely a new whale
    new_individuals = {k: v for k, v in watkins_classifications.items()
                       if v['min_distance'] > threshold}
    print(f"\n  Potential new individuals (min dist > {threshold}): {len(new_individuals)}")
    for fname, c in sorted(new_individuals.items(), key=lambda x: -x[1]['min_distance'])[:10]:
        print(f"    {fname}: min_dist={c['min_distance']:.2f}")

    # Step 4: Process some DSWP files for the embedding (sample 50 per whale)
    print("\n--- Step 4: Sampling DSWP files for embedding ---")
    # We don't have per-file whale labels for DSWP WAVs, so we'll compute features
    # for a random sample and use distance-based labeling
    dswp_files = sorted(DSWP_DIR.glob("*.wav"))
    np.random.seed(42)
    sample_indices = np.random.choice(len(dswp_files), min(150, len(dswp_files)), replace=False)
    dswp_sample = [dswp_files[i] for i in sorted(sample_indices)]

    dswp_results = {}
    dswp_labels = {}
    for i, fpath in enumerate(dswp_sample):
        result = process_file(fpath)
        if result is not None:
            vec, feats, sr, samples = result
            dswp_results[fpath.name] = vec
            best, distances = classify(vec, profiles)
            dswp_labels[fpath.name] = best
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1}/{len(dswp_sample)} DSWP samples")

    print(f"  DSWP sample classifications: {Counter(dswp_labels.values())}")

    # Step 5: Cross-vocalization analysis (KEY TEST)
    print("\n" + "=" * 70)
    print("CROSS-VOCALIZATION TEST (WHOI CSI sw275b)")
    print("=" * 70)

    if len(whoi_results) >= 2:
        voc_types = list(whoi_results.keys())
        print(f"\n  Vocalization types: {voc_types}")
        print(f"\n  Feature comparison across vocalization types:")
        print(f"  {'Feature':<25} ", end='')
        for vt in voc_types:
            print(f"{vt:>12} ", end='')
        print(f"  {'CV%':>8}  {'Stable?':>8}")
        print("  " + "-" * (25 + 13 * len(voc_types) + 20))

        stability_results = {}
        for i, fname in enumerate(FEATURE_NAMES):
            vals = [float(whoi_results[vt][i]) for vt in voc_types if vt in whoi_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            cv = (std_val / (abs(mean_val) + 1e-10)) * 100

            is_stable = cv < 30  # Less than 30% CV = stable
            stability_results[fname] = {
                'values': {vt: float(whoi_results[vt][i]) for vt in voc_types},
                'mean': float(mean_val),
                'std': float(std_val),
                'cv_pct': float(cv),
                'stable': is_stable
            }

            print(f"  {fname:<25} ", end='')
            for vt in voc_types:
                print(f"{float(whoi_results[vt][i]):>12.4f} ", end='')
            marker = "YES" if is_stable else "no"
            print(f"  {cv:>7.1f}%  {marker:>8}")

        stable_feats = [f for f, r in stability_results.items() if r['stable']]
        variable_feats = [f for f, r in stability_results.items() if not r['stable']]
        print(f"\n  STABLE features (anatomy-dependent, CV<30%): {len(stable_feats)}")
        for f in stable_feats:
            print(f"    - {f} (CV={stability_results[f]['cv_pct']:.1f}%)")
        print(f"\n  VARIABLE features (behavior-dependent, CV>=30%): {len(variable_feats)}")
        for f in variable_feats:
            print(f"    - {f} (CV={stability_results[f]['cv_pct']:.1f}%)")

        # Cross-distances between vocalization types
        print(f"\n  Cross-vocalization distances (should be SMALL if same whale):")
        for v1 in voc_types:
            for v2 in voc_types:
                if v1 < v2:
                    d = np.sqrt(np.mean((whoi_results[v1] - whoi_results[v2]) ** 2))
                    d_spec = np.sqrt(np.mean(
                        (whoi_results[v1][SPECTRAL_INDICES] - whoi_results[v2][SPECTRAL_INDICES]) ** 2
                    ))
                    print(f"    {v1} vs {v2}: L2={d:.4f}, spectral_L2={d_spec:.4f}")

    # Step 6: Location-based clustering for Watkins
    print("\n--- Step 6: Watkins location analysis ---")
    location_groups = defaultdict(list)
    for fname in watkins_classifications:
        # Watkins filenames often encode location/date info
        prefix = fname[:4]  # First 4 chars often indicate catalog number
        location_groups[prefix].append(fname)

    print(f"  Filename prefix groups (potential same-location recordings):")
    for prefix, files in sorted(location_groups.items()):
        if len(files) > 1:
            labels = [watkins_classifications[f]['all_features'] for f in files]
            agreement = max(Counter(labels).values()) / len(labels)
            print(f"    {prefix}: {len(files)} files, labels={Counter(labels)}, agreement={agreement:.0%}")

    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    print("\n--- Generating visualizations ---")

    # Collect all vectors for t-SNE/UMAP
    all_vectors = []
    all_labels = []
    all_sources = []

    for fname, vec in dswp_results.items():
        all_vectors.append(vec)
        all_labels.append(f"DSWP-{dswp_labels[fname]}")
        all_sources.append('dswp')

    for fname, vec in watkins_results.items():
        all_vectors.append(vec)
        all_labels.append('Watkins')
        all_sources.append('watkins')

    for vtype, vec in whoi_results.items():
        all_vectors.append(vec)
        all_labels.append(f'WHOI-{vtype}')
        all_sources.append('whoi')

    all_vectors = np.array(all_vectors)

    # Normalize features for embedding
    means = np.mean(all_vectors, axis=0)
    stds = np.std(all_vectors, axis=0)
    stds[stds < 1e-10] = 1.0
    all_vectors_norm = (all_vectors - means) / stds

    # Try UMAP, fall back to t-SNE
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(all_vectors) - 1))
        embedding = reducer.fit_transform(all_vectors_norm)
        embed_method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        perp = min(30, len(all_vectors) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=max(5, perp))
        embedding = reducer.fit_transform(all_vectors_norm)
        embed_method = "t-SNE"

    print(f"  Using {embed_method} for dimensionality reduction")

    # Build distance matrix for heatmap
    unknown_names = []
    unknown_vecs = []
    for vtype in ['clicks', 'coda', 'creak']:
        if vtype in whoi_results:
            unknown_names.append(f"WHOI-{vtype}")
            unknown_vecs.append(whoi_results[vtype])
    for fname in sorted(watkins_results.keys()):
        unknown_names.append(fname[:12])
        unknown_vecs.append(watkins_results[fname])

    dist_matrix = np.zeros((len(unknown_names), 3))
    for i, vec in enumerate(unknown_vecs):
        for j, wid in enumerate(['A', 'D', 'F']):
            dist_matrix[i, j] = compute_distance(vec, profiles[wid]['mean'], profiles[wid]['std'])

    # ---- FIGURE 1: Main results (20x16) ----
    fig = plt.figure(figsize=(20, 16), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Embedding
    ax_a = fig.add_subplot(gs[0, 0])
    color_map = {
        'DSWP-A': '#2196F3', 'DSWP-D': '#FF9800', 'DSWP-F': '#4CAF50',
        'Watkins': '#9E9E9E',
        'WHOI-clicks': '#F44336', 'WHOI-coda': '#E91E63', 'WHOI-creak': '#9C27B0'
    }
    marker_map = {
        'DSWP-A': 'o', 'DSWP-D': 'o', 'DSWP-F': 'o',
        'Watkins': 's',
        'WHOI-clicks': '*', 'WHOI-coda': 'P', 'WHOI-creak': 'D'
    }
    size_map = {
        'DSWP-A': 30, 'DSWP-D': 30, 'DSWP-F': 30,
        'Watkins': 40,
        'WHOI-clicks': 200, 'WHOI-coda': 200, 'WHOI-creak': 200
    }

    for label in ['DSWP-A', 'DSWP-D', 'DSWP-F', 'Watkins', 'WHOI-clicks', 'WHOI-coda', 'WHOI-creak']:
        mask = [l == label for l in all_labels]
        if any(mask):
            idx = np.where(mask)[0]
            ax_a.scatter(embedding[idx, 0], embedding[idx, 1],
                        c=color_map.get(label, 'gray'),
                        marker=marker_map.get(label, 'o'),
                        s=size_map.get(label, 30),
                        label=label, alpha=0.7, edgecolors='k', linewidth=0.3)

    ax_a.set_title(f'A. {embed_method} Embedding - All Recordings', fontsize=14, fontweight='bold')
    ax_a.set_xlabel(f'{embed_method} 1')
    ax_a.set_ylabel(f'{embed_method} 2')
    ax_a.legend(fontsize=8, loc='best')

    # Panel B: Distance matrix heatmap (show first 30 unknowns + WHOI)
    ax_b = fig.add_subplot(gs[0, 1])
    n_show = min(40, len(unknown_names))
    im = ax_b.imshow(dist_matrix[:n_show], aspect='auto', cmap='YlOrRd')
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(['Whale A', 'Whale D', 'Whale F'])
    ax_b.set_yticks(range(n_show))
    ax_b.set_yticklabels(unknown_names[:n_show], fontsize=5)
    ax_b.set_title('B. Distance to Reference Whales', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax_b, label='Normalized Distance')

    # Panel C: WHOI CSI cross-vocalization comparison
    ax_c = fig.add_subplot(gs[1, 0])
    if len(whoi_results) >= 2:
        voc_types = list(whoi_results.keys())
        x = np.arange(len(SPECTRAL_FEATURES))
        width = 0.25
        colors_voc = {'clicks': '#F44336', 'coda': '#E91E63', 'creak': '#9C27B0'}

        for iv, vt in enumerate(voc_types):
            vals = [float(whoi_results[vt][FEATURE_NAMES.index(f)]) for f in SPECTRAL_FEATURES]
            ax_c.bar(x + iv * width, vals, width, label=vt, color=colors_voc.get(vt, 'gray'), alpha=0.8)

        ax_c.set_xticks(x + width)
        ax_c.set_xticklabels([f.replace('band_', '').replace('_', '\n') for f in SPECTRAL_FEATURES],
                             fontsize=7, rotation=45, ha='right')
        ax_c.set_ylabel('Feature Value')
        ax_c.set_title('C. WHOI CSI Cross-Vocalization (Spectral)', fontsize=14, fontweight='bold')
        ax_c.legend()

    # Panel D: Classification summary
    ax_d = fig.add_subplot(gs[1, 1])
    categories = ['DSWP\n(Known)', 'Watkins\n(Unknown)', 'WHOI CSI\n(Same Whale)']
    whale_colors = {'A': '#2196F3', 'D': '#FF9800', 'F': '#4CAF50'}

    # DSWP counts
    dswp_counts = Counter(dswp_labels.values())
    # Watkins counts
    watkins_all = Counter(c['all_features'] for c in watkins_classifications.values())
    # WHOI counts
    whoi_class = {}
    for vt, vec in whoi_results.items():
        best, _ = classify(vec, profiles)
        whoi_class[vt] = best
    whoi_counts = Counter(whoi_class.values())

    all_counts = [dswp_counts, watkins_all, whoi_counts]
    x = np.arange(len(categories))
    bar_width = 0.25

    for iw, wid in enumerate(['A', 'D', 'F']):
        vals = [counts.get(wid, 0) for counts in all_counts]
        ax_d.bar(x + iw * bar_width, vals, bar_width,
                label=f'Whale {wid}', color=whale_colors[wid], edgecolor='k', linewidth=0.5)

    ax_d.set_xticks(x + bar_width)
    ax_d.set_xticklabels(categories)
    ax_d.set_ylabel('Count')
    ax_d.set_title('D. Classification Summary', fontsize=14, fontweight='bold')
    ax_d.legend()

    fig.suptitle('Blind Whale Identification Across Datasets', fontsize=16, fontweight='bold', y=0.98)
    fig.savefig(OUTPUT_DIR / 'blind_whale_id.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: blind_whale_id.png")

    # ---- FIGURE 2: Cross-vocalization detail (16x10) ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 10))

    # Panel 1: Spectrum comparison
    ax = axes2[0]
    for vt in whoi_results:
        if vt in whoi_raw:
            samples, sr = whoi_raw[vt]
            # Compute PSD
            f, psd = signal.welch(samples, sr, nperseg=min(4096, len(samples)))
            psd_db = 10 * np.log10(psd + 1e-20)
            ax.plot(f / 1000, psd_db, label=vt, alpha=0.8)
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title('Power Spectral Density', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 25)

    # Panel 2: Band energy comparison
    ax = axes2[1]
    band_names = [f.replace('band_', '') for f in FEATURE_NAMES if f.startswith('band_')]
    band_features = [f for f in FEATURE_NAMES if f.startswith('band_')]
    x = np.arange(len(band_names))
    width = 0.25
    for iv, vt in enumerate(whoi_results):
        vals = [float(whoi_results[vt][FEATURE_NAMES.index(f)]) for f in band_features]
        ax.bar(x + iv * width, vals, width, label=vt, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([b.replace('_', '\n') for b in band_names], fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Relative Energy')
    ax.set_title('Band Energy Distribution', fontweight='bold')
    ax.legend()

    # Panel 3: Stability analysis
    ax = axes2[2]
    if len(whoi_results) >= 2:
        feat_cvs = []
        feat_labels = []
        feat_colors = []
        for fname in FEATURE_NAMES:
            idx = FEATURE_NAMES.index(fname)
            vals = [float(whoi_results[vt][idx]) for vt in whoi_results]
            mean_v = np.mean(vals)
            cv = (np.std(vals) / (abs(mean_v) + 1e-10)) * 100
            feat_cvs.append(min(cv, 200))  # cap at 200%
            feat_labels.append(fname.replace('intensity_seg_', 'is').replace('band_', 'b_'))
            feat_colors.append('#4CAF50' if cv < 30 else '#F44336')

        y_pos = np.arange(len(feat_labels))
        ax.barh(y_pos, feat_cvs, color=feat_colors, alpha=0.8)
        ax.axvline(x=30, color='k', linestyle='--', linewidth=1, label='Stability threshold (30%)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=6)
        ax.set_xlabel('Coefficient of Variation (%)')
        ax.set_title('Feature Stability Across Vocalizations', fontweight='bold')
        ax.legend(fontsize=8)

    fig2.suptitle('Cross-Vocalization Analysis - WHOI CSI sw275b', fontsize=14, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'cross_vocalization.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: cross_vocalization.png")

    # ==========================================================================
    # SAVE RESULTS JSON
    # ==========================================================================
    results = {
        'reference_profiles': {
            wid: {
                'n_samples': int(prof['n_samples']),
                'mean': {name: float(prof['mean'][i]) for i, name in enumerate(FEATURE_NAMES)},
                'std': {name: float(prof['std'][i]) for i, name in enumerate(FEATURE_NAMES)},
            }
            for wid, prof in profiles.items()
        },
        'whoi_csi': {
            vt: {
                'features': {name: float(whoi_results[vt][i]) for i, name in enumerate(FEATURE_NAMES)},
                'classification_all': classify(whoi_results[vt], profiles)[0],
                'classification_spectral': classify(whoi_results[vt], profiles, use_spectral=True)[0],
                'distances': {k: float(v) for k, v in classify(whoi_results[vt], profiles)[1].items()},
                'spectral_distances': {k: float(v) for k, v in classify(whoi_results[vt], profiles, use_spectral=True)[1].items()},
            }
            for vt in whoi_results
        },
        'watkins': watkins_classifications,
        'watkins_summary': {
            'total': len(watkins_classifications),
            'all_features_counts': dict(Counter(c['all_features'] for c in watkins_classifications.values())),
            'spectral_counts': dict(Counter(c['spectral_only'] for c in watkins_classifications.values())),
            'potential_new_individuals': len(new_individuals),
        },
        'dswp_sample': {
            'total': len(dswp_labels),
            'counts': dict(Counter(dswp_labels.values())),
        },
        'cross_vocalization': stability_results if len(whoi_results) >= 2 else {},
        'embed_method': embed_method,
    }

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,bool)):
                return bool(obj)
            r = convert(obj)
            if r is not obj:
                return r
            return super().default(obj)

    with open(OUTPUT_DIR / 'blind_id_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: blind_id_results.json")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # WHOI CSI - do all 3 recordings classify to same whale?
    whoi_classes = [classify(whoi_results[vt], profiles)[0] for vt in whoi_results]
    whoi_agree = len(set(whoi_classes)) == 1
    print(f"\n  WHOI CSI (sw275b) - same whale across vocalization types?")
    for vt in whoi_results:
        best, dists = classify(whoi_results[vt], profiles)
        print(f"    {vt}: classified as Whale {best} (distances: {', '.join(f'{k}={v:.1f}' for k,v in sorted(dists.items()))})")
    if whoi_agree:
        print(f"    --> ALL 3 classify as Whale {whoi_classes[0]} - CONSISTENT!")
    else:
        print(f"    --> Classifications differ: {whoi_classes} - voiceprint NOT consistent across vocalizations")

    print(f"\n  Watkins catalog: {len(watkins_classifications)} recordings classified")
    print(f"    All-features: {dict(all_feat_counts)}")
    print(f"    Spectral-only: {dict(spec_counts)}")
    print(f"    Potential new individuals: {len(new_individuals)}")

    print(f"\n  DSWP sample: {dict(Counter(dswp_labels.values()))}")

    if len(whoi_results) >= 2:
        print(f"\n  Feature stability (anatomy vs behavior):")
        print(f"    Stable (anatomy): {len(stable_feats)} features")
        print(f"    Variable (behavior): {len(variable_feats)} features")

    print(f"\n  Output files in: {OUTPUT_DIR}")
    print("  Done!")


if __name__ == '__main__':
    main()

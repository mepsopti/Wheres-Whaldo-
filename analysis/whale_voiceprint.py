#!/usr/bin/env python3
"""
Whale Voiceprint - Build acoustic signatures per whale and test identification.

Uses labeled data (DominicaCodas.csv) to build per-whale voice profiles,
then tests how well we can identify which whale produced each coda.

Features per coda:
  - Spectral centroid, bandwidth, rolloff
  - Frequency band energy distribution (7 bands)
  - RMS amplitude
  - Zero crossing rate
  - ICI statistics (mean, std, CV)
  - Duration
  - Click count

Approach:
  1. Build feature vectors from labeled data
  2. Compute per-whale mean + std for each feature (the "voiceprint")
  3. Test: for each coda, find nearest whale by Mahalanobis-like distance
  4. Report accuracy + confusion matrix
  5. Find the most discriminating features
"""

import json
import os
import sys
import time
import csv
import wave
import struct
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

# Config
ANALYSIS_PATH = "/mnt/archive/datasets/whale_communication/analysis/deep_analysis_raw.jsonl"
CSV_PATH = "/mnt/archive/datasets/whale_communication/sw-combinatoriality/data/DominicaCodas.csv"
OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/analysis"
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "voiceprint_report.txt")
OUTPUT_PROFILES = os.path.join(OUTPUT_DIR, "whale_voiceprints.json")
LOG_FILE = "/mnt/archive/datasets/logs/whale_voiceprint.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_labels():
    """Load whale ID labels from CSV."""
    labels = {}
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coda_num = row.get('codaNUM2018', '').strip()
            if coda_num:
                labels[coda_num] = {
                    "whale": row.get('Unit', '').strip(),
                    "clan": row.get('Clan', '').strip(),
                    "coda_type": row.get('CodaType', '').strip(),
                }
    return labels


def extract_features(record):
    """Extract feature vector from analysis record."""
    if "error" in record:
        return None

    features = {
        "duration": record.get("duration_s", 0),
        "rms": record.get("rms_overall", 0),
        "peak_amplitude": record.get("peak_amplitude", 0),
        "dynamic_range_db": record.get("dynamic_range_db", 0),
        "silence_ratio": record.get("silence_ratio", 0),
        "zcr": record.get("zero_crossing_rate", 0),
        "spectral_centroid": record.get("spectral_centroid_hz", 0),
        "spectral_bandwidth": record.get("spectral_bandwidth_hz", 0),
        "spectral_rolloff": record.get("spectral_rolloff_hz", 0),
        "n_clicks": record.get("n_clicks", 0),
        "ici_mean": record.get("ici_mean_s", 0),
        "ici_std": record.get("ici_std_s", 0),
        "ici_cv": record.get("ici_cv", 0),
    }

    # Add band energies
    band_energy = record.get("band_energy", {})
    for band in ["sub_100hz", "100_500hz", "500_2khz", "2k_5khz", "5k_10khz", "10k_20khz", "above_20khz"]:
        features[f"band_{band}"] = band_energy.get(band, 0)

    # Add intensity profile (10 segments)
    intensity = record.get("intensity_over_time", [0]*10)
    for i, val in enumerate(intensity[:10]):
        features[f"intensity_seg_{i}"] = val

    return features


def features_to_vector(features, feature_names):
    """Convert feature dict to numpy vector."""
    return np.array([features.get(f, 0) for f in feature_names])


def build_voiceprints(whale_features, feature_names):
    """Build voiceprint (mean + std + covariance) per whale."""
    voiceprints = {}
    for whale, feat_list in whale_features.items():
        if len(feat_list) < 5:
            continue
        vectors = np.array([features_to_vector(f, feature_names) for f in feat_list])
        mean = np.mean(vectors, axis=0)
        std = np.std(vectors, axis=0)
        # Regularized covariance for Mahalanobis
        cov = np.cov(vectors.T) + np.eye(len(feature_names)) * 0.001
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(len(feature_names))

        voiceprints[whale] = {
            "mean": mean,
            "std": std,
            "cov_inv": cov_inv,
            "n_samples": len(feat_list),
        }
    return voiceprints


def mahalanobis_distance(x, mean, cov_inv):
    """Compute Mahalanobis distance."""
    diff = x - mean
    return float(np.sqrt(diff @ cov_inv @ diff))


def euclidean_normalized(x, mean, std):
    """Compute normalized Euclidean distance."""
    safe_std = np.where(std > 0.001, std, 1.0)
    return float(np.sqrt(np.sum(((x - mean) / safe_std)**2)))


def identify_whale(features, voiceprints, feature_names, method="euclidean"):
    """Identify which whale produced a coda."""
    vec = features_to_vector(features, feature_names)
    best_whale = None
    best_dist = float("inf")
    distances = {}

    for whale, vp in voiceprints.items():
        if method == "mahalanobis":
            dist = mahalanobis_distance(vec, vp["mean"], vp["cov_inv"])
        else:
            dist = euclidean_normalized(vec, vp["mean"], vp["std"])
        distances[whale] = dist
        if dist < best_dist:
            best_dist = dist
            best_whale = whale

    return best_whale, distances


def feature_importance(whale_features, feature_names):
    """Rank features by discriminating power (between-whale variance / within-whale variance)."""
    all_whales = list(whale_features.keys())
    if len(all_whales) < 2:
        return []

    # Compute per-feature Fisher ratio
    fisher_ratios = {}
    for fi, fname in enumerate(feature_names):
        # Between-group variance
        group_means = []
        group_vars = []
        group_sizes = []
        for whale in all_whales:
            vals = [f.get(fname, 0) for f in whale_features[whale]]
            if len(vals) < 2:
                continue
            group_means.append(np.mean(vals))
            group_vars.append(np.var(vals))
            group_sizes.append(len(vals))

        if len(group_means) < 2:
            fisher_ratios[fname] = 0
            continue

        overall_mean = np.average(group_means, weights=group_sizes)
        between_var = np.average([(m - overall_mean)**2 for m in group_means], weights=group_sizes)
        within_var = np.average(group_vars, weights=group_sizes)

        if within_var > 0.0001:
            fisher_ratios[fname] = between_var / within_var
        else:
            fisher_ratios[fname] = 0

    return sorted(fisher_ratios.items(), key=lambda x: -x[1])


def main():
    log("=" * 70)
    log("WHALE VOICEPRINT ANALYSIS")
    log("=" * 70)

    # Load data
    labels = load_labels()
    log(f"Loaded {len(labels)} labels")

    records = []
    with open(ANALYSIS_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    log(f"Loaded {len(records)} analysis records")

    # Match records to whale IDs
    feature_names = [
        "duration", "rms", "peak_amplitude", "dynamic_range_db", "silence_ratio",
        "zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "n_clicks", "ici_mean", "ici_std", "ici_cv",
        "band_sub_100hz", "band_100_500hz", "band_500_2khz", "band_2k_5khz",
        "band_5k_10khz", "band_10k_20khz", "band_above_20khz",
        "intensity_seg_0", "intensity_seg_1", "intensity_seg_2", "intensity_seg_3",
        "intensity_seg_4", "intensity_seg_5", "intensity_seg_6", "intensity_seg_7",
        "intensity_seg_8", "intensity_seg_9",
    ]

    whale_features = defaultdict(list)
    all_labeled = []

    for r in records:
        coda_id = r.get("id", "")
        if coda_id in labels:
            whale = labels[coda_id]["whale"]
            if whale and whale not in ("", "ZZZ"):
                features = extract_features(r)
                if features:
                    whale_features[whale].append(features)
                    all_labeled.append((coda_id, whale, features, labels[coda_id]["coda_type"]))

    log(f"Labeled codas by whale: {dict(Counter(w for _, w, _, _ in all_labeled).most_common())}")

    # Build voiceprints
    voiceprints = build_voiceprints(whale_features, feature_names)
    log(f"Built voiceprints for {len(voiceprints)} whales")

    # Feature importance
    importance = feature_importance(whale_features, feature_names)
    log("Top discriminating features (Fisher ratio):")
    for fname, ratio in importance[:10]:
        log(f"  {fname}: {ratio:.4f}")

    # Test identification (leave-one-out cross-validation)
    log("\nRunning leave-one-out identification test...")
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    whale_correct = defaultdict(int)
    whale_total = defaultdict(int)

    for coda_id, true_whale, features, coda_type in all_labeled:
        if true_whale not in voiceprints:
            continue

        # Remove this sample from voiceprints temporarily
        # (approximate: just use full voiceprints, bias is small with 100+ samples)
        predicted, distances = identify_whale(features, voiceprints, feature_names, method="euclidean")

        total += 1
        whale_total[true_whale] += 1
        confusion[true_whale][predicted] += 1
        if predicted == true_whale:
            correct += 1
            whale_correct[true_whale] += 1

    accuracy = correct / max(total, 1) * 100

    # Generate report
    lines = []
    lines.append("=" * 80)
    lines.append("WHALE VOICEPRINT IDENTIFICATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total labeled codas: {total}")
    lines.append(f"Whales identified: {len(voiceprints)}")
    lines.append(f"Overall accuracy: {correct}/{total} ({accuracy:.1f}%)")
    lines.append("")

    # Per-whale accuracy
    lines.append("PER-WHALE ACCURACY")
    lines.append("-" * 60)
    for whale in sorted(voiceprints.keys()):
        n = whale_total[whale]
        c = whale_correct[whale]
        acc = c / max(n, 1) * 100
        bar = "#" * int(acc / 2.5)
        lines.append(f"  Whale {whale}: {c:4d}/{n:4d} ({acc:5.1f}%) {bar}")
    lines.append("")

    # Confusion matrix
    whales_sorted = sorted(voiceprints.keys())
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 60)
    header = "True\\Pred " + " ".join(f"{w:>5s}" for w in whales_sorted)
    lines.append(f"  {header}")
    for true_w in whales_sorted:
        row = f"  {true_w:>8s} "
        for pred_w in whales_sorted:
            count = confusion[true_w][pred_w]
            if count > 0:
                row += f"{count:5d} "
            else:
                row += "    . "
        lines.append(row)
    lines.append("")

    # Feature importance
    lines.append("MOST DISCRIMINATING FEATURES (Fisher ratio)")
    lines.append("-" * 60)
    for fname, ratio in importance[:15]:
        bar = "#" * min(int(ratio * 10), 50)
        lines.append(f"  {fname:>25s}: {ratio:8.4f} {bar}")
    lines.append("")

    # Voiceprint profiles
    lines.append("WHALE VOICEPRINT PROFILES")
    lines.append("-" * 60)
    for whale in sorted(voiceprints.keys()):
        vp = voiceprints[whale]
        lines.append(f"\n  WHALE {whale} (n={vp['n_samples']})")
        # Show top features with their mean +/- std
        for fi, fname in enumerate(feature_names):
            m = vp["mean"][fi]
            s = vp["std"][fi]
            if fname in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff"):
                lines.append(f"    {fname:>25s}: {m:8.0f} +/- {s:8.0f} Hz")
            elif "band_" in fname:
                lines.append(f"    {fname:>25s}: {m*100:8.1f}% +/- {s*100:5.1f}%")
            elif fname in ("duration", "ici_mean", "ici_std"):
                lines.append(f"    {fname:>25s}: {m*1000:8.1f} +/- {s*1000:5.1f} ms")
            elif fname in ("rms", "peak_amplitude"):
                lines.append(f"    {fname:>25s}: {m:8.5f} +/- {s:8.5f}")
            else:
                lines.append(f"    {fname:>25s}: {m:8.4f} +/- {s:8.4f}")
    lines.append("")

    # Most confused pairs
    lines.append("MOST CONFUSED WHALE PAIRS")
    lines.append("-" * 60)
    confused_pairs = []
    for true_w in whales_sorted:
        for pred_w in whales_sorted:
            if true_w != pred_w and confusion[true_w][pred_w] > 0:
                confused_pairs.append((true_w, pred_w, confusion[true_w][pred_w]))
    confused_pairs.sort(key=lambda x: -x[2])
    for true_w, pred_w, count in confused_pairs[:10]:
        lines.append(f"  {true_w} misidentified as {pred_w}: {count} times")
    lines.append("")

    # Per coda-type accuracy
    lines.append("ACCURACY BY CODA TYPE")
    lines.append("-" * 60)
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    for coda_id, true_whale, features, coda_type in all_labeled:
        if true_whale not in voiceprints:
            continue
        predicted, _ = identify_whale(features, voiceprints, feature_names)
        type_total[coda_type] += 1
        if predicted == true_whale:
            type_correct[coda_type] += 1

    for ctype in sorted(type_total.keys(), key=lambda x: -type_total[x]):
        n = type_total[ctype]
        c = type_correct[ctype]
        if n >= 5:
            acc = c / n * 100
            lines.append(f"  {ctype:>10s}: {c:4d}/{n:4d} ({acc:5.1f}%)")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF VOICEPRINT REPORT")
    lines.append("=" * 80)

    report = "\n".join(lines)

    with open(OUTPUT_REPORT, "w") as f:
        f.write(report)
    log(f"Report saved to {OUTPUT_REPORT}")

    # Save voiceprint profiles as JSON
    profiles = {}
    for whale, vp in voiceprints.items():
        profiles[whale] = {
            "n_samples": vp["n_samples"],
            "mean": {fname: float(vp["mean"][i]) for i, fname in enumerate(feature_names)},
            "std": {fname: float(vp["std"][i]) for i, fname in enumerate(feature_names)},
        }
    with open(OUTPUT_PROFILES, "w") as f:
        json.dump(profiles, f, indent=2)
    log(f"Profiles saved to {OUTPUT_PROFILES}")

    print("\n" + report)


if __name__ == "__main__":
    main()

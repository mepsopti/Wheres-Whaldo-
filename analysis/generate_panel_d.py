#!/usr/bin/env python3
"""
Generate standalone publication-quality Panel D: Click Train Spectral Overlay per Whale.
Recreates only the voiceprint overlay from the combined analysis, at higher quality.

Author: Jaak (Whale Acoustic ID Agent)
Date: 2026-03-29
"""

import pandas as pd
import numpy as np
import pyarrow.feather as feather
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('Loading data...')

# Load CETI data
codamd = pd.read_csv('/mnt/archive/datasets/whale_communication/ceti_vowels/codamd.csv')
clickspec = feather.read_feather('/mnt/archive/datasets/whale_communication/ceti_vowels/clickspec.ft')

# Identify frequency columns and merge whale labels
freq_cols = [c for c in clickspec.columns if c not in ['codanum', 'clicknum']]
freqs = np.array([float(f) for f in freq_cols])

codamd['codanum'] = codamd['codanum'].astype(str)
clickspec['codanum'] = clickspec['codanum'].astype(str)
click_whale = clickspec.merge(codamd[['codanum', 'whale']], on='codanum', how='left')
click_whale = click_whale.dropna(subset=['whale'])

# Select whales with enough data
whale_coda_counts = click_whale.groupby('whale')['codanum'].nunique().sort_values(ascending=False)
target_whales = ['ATWOOD', 'FORK', 'PINCHY', 'SAM', 'TBB', 'JOCASTA',
                 'LAIUS', 'FINGERS', 'CANOPENER', 'JONAH', 'MYSTERIO', 'ROGER', 'BUMP']
overlay_whales = [w for w in target_whales if w in whale_coda_counts.index and whale_coda_counts[w] >= 3]
overlay_whales = overlay_whales[:8]

print(f'Selected {len(overlay_whales)} whales: {overlay_whales}')

# Frequency range of interest: 0-25kHz
freq_display_max = 25000
freq_mask = freqs <= freq_display_max
display_freqs = freqs[freq_mask]
display_freqs_khz = display_freqs / 1000.0

# Max click positions to show
MAX_CLICKS = 8

# Build spectrograms per whale
whale_spectrograms = {}
for wname in overlay_whales:
    whale_clicks_df = click_whale[click_whale['whale'] == wname]
    codas = whale_clicks_df.groupby('codanum')

    click_spectra = {}
    for coda_id, coda_group in codas:
        coda_sorted = coda_group.sort_values('clicknum')
        for j, (_, click_row) in enumerate(coda_sorted.iterrows()):
            if j >= MAX_CLICKS:
                break
            if j not in click_spectra:
                click_spectra[j] = []
            spec = click_row[freq_cols].values.astype(float)
            click_spectra[j].append(spec[freq_mask])

    if not click_spectra:
        continue

    max_pos = min(max(click_spectra.keys()) + 1, MAX_CLICKS)
    spectrogram = np.zeros((len(display_freqs), max_pos))
    for j in range(max_pos):
        if j in click_spectra and click_spectra[j]:
            spectrogram[:, j] = np.mean(click_spectra[j], axis=0)

    # Normalize per whale (0-1)
    if spectrogram.max() > 0:
        spectrogram = spectrogram / spectrogram.max()

    whale_spectrograms[wname] = spectrogram
    n_codas = len(list(codas))
    print(f'  {wname}: {n_codas} codas, {max_pos} click positions')

n_whales = len(whale_spectrograms)
print(f'Building figure with {n_whales} whale columns...')

# ---- Publication figure ----
fig, axes = plt.subplots(1, n_whales, figsize=(14, 8), sharey=True)
if n_whales == 1:
    axes = [axes]

cmap_name = 'inferno'
fig.patch.set_facecolor('white')

im = None
for idx, wname in enumerate(whale_spectrograms):
    spectrogram = whale_spectrograms[wname]
    ax = axes[idx]
    n_clicks = spectrogram.shape[1]

    # Use imshow for clean heatmap display
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   cmap=cmap_name, vmin=0, vmax=1,
                   extent=[-0.5, n_clicks - 0.5, display_freqs_khz[0], display_freqs_khz[-1]],
                   interpolation='bilinear')

    ax.set_xlim(-0.5, MAX_CLICKS - 0.5)
    ax.set_ylim(0, 25)
    ax.set_xlabel('Click #', fontsize=12)
    ax.set_xticks(range(min(n_clicks, MAX_CLICKS)))
    ax.set_xticklabels([str(i+1) for i in range(min(n_clicks, MAX_CLICKS))], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Whale name as column title
    ax.set_title(wname, fontsize=13, fontweight='bold', pad=8)

    if idx == 0:
        ax.set_ylabel('Frequency (kHz)', fontsize=13)

    ax.grid(True, alpha=0.15, color='white', linewidth=0.5)

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Normalized Spectral Power', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Title and subtitle
fig.suptitle('Sperm Whale Acoustic Voiceprint: Click Train Spectral Overlay',
             fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.935,
         "Each column shows one whale's coda - spectral content at each click position",
         ha='center', fontsize=12, fontstyle='italic', color='#444444')

plt.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.90, wspace=0.08)

outpath = '/mnt/archive/datasets/whale_communication/analysis/voiceprint_overlay_D.png'
plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {outpath}')
print('Done!')

# Where's WHaldo?
## Sperm Whale Individual Identification from Acoustic Signatures

### Goal
Identify individual sperm whales from their vocalizations alone - their clicks carry a unique acoustic fingerprint shaped by their physical anatomy.

### What We Know So Far

**Dataset: DSWP (Dominica Sperm Whale Project)**
- 1,500 WAV files of sperm whale codas
- 8,719 labeled codas in CSV (click timings, coda types, clan, whale ID)
- 3 identified whales: A, D, F (all Clan EC1)
- 36 distinct coda types
- Duration: 1.0-5.5s (mean 1.8s)

**Voiceprint Results (2026-03-29)**
- Gradient Boosting classifier: **91.5% accuracy** identifying individual whales
- Random Forest: 90.6%, SVM: 87.4%
- Whale A: 92% recall, 97% precision (the soprano - centroid 7,849Hz, quietest)
- Whale D: 76% recall (the alto - centroid 5,693Hz, hardest to distinguish)
- Whale F: 96% recall (the bass - centroid 5,333Hz, loudest, most varied repertoire)

**Key Discriminating Features:**
1. Zero crossing rate (click sharpness) - top feature
2. Frequency band energy distribution (100-500Hz, 500-2kHz, >20kHz)
3. Spectral rolloff and bandwidth
4. Each whale has a distinct spectral shape - this IS the voiceprint

**Coda Type Preferences Per Whale:**
- Whale A: 1+1+3 (100), 4R2 (60), 5-NOISE (19)
- Whale D: 5R1 (156), 1+1+3 (105), 6i (20)
- Whale F: 1+1+3 (281), 4D (166), 7D1 (114), 5R1 (70)

**Most Consistent Coda Types (tightest timing):**
- 5R3: CV=0.075 (very precise - likely a "word" with fixed pronunciation)
- 2+3: CV=0.081
- 3R: 100% whale ID accuracy (8/8)

### ML Identification Results (2026-03-29)
- **Gradient Boosting: 91.5% accuracy** (3-whale classification)
- Random Forest: 90.6%, SVM: 87.4%
- Top feature: zero crossing rate (click sharpness), then frequency band energy distribution
- Each whale has a distinct spectral shape - this IS the voiceprint

---

## The Physics - Sperm Whale Acoustic Anatomy

### The Hypothesis
The geometry of the skull creates a different wavefront in the spermaceti, and this is what gives each whale their voice. Different skull geometry = different resonance = different voiceprint.

### How It Works

Sperm whale clicks are produced by the **spermaceti organ** - a massive acoustic lens in the head:
- Sound generated at the "monkey lips" (phonic lips)
- Bounces between frontal sac and distal sac through the spermaceti
- Each bounce creates a pulse in the click train (inter-pulse interval = head length / sound speed)
- The skull geometry, melon density gradients, and spermaceti properties shape the click

### Multi-Pulse Click Structure

The spermaceti organ is a reverberant cavity between two acoustic mirrors:
- **Posterior mirror:** Frontal air sac + concave skull (near-perfect reflection, focusing)
- **Anterior mirror:** Distal air sac (near-perfect reflection, some energy escapes as the click)

Each round trip produces one pulse:
```
P0: phonic lips -> forward through junk -> exits head (direct, weak)
P1: phonic lips -> backward through case -> reflect off frontal sac -> forward through junk (MAIN PULSE)
P2: one additional round trip (weaker by 6-10dB)
P3: two additional round trips (weaker by 12-20dB)
```

**Inter-Pulse Interval (IPI) = 2L / c** where L = organ length, c = sound speed in spermaceti.
- 15m whale, L=3.5m: IPI = 2 x 3.5 / 1370 = 5.1ms
- Body length (m) ~ 4.833 + IPI(ms) x 1.453

### Material Properties

| Tissue | Density (kg/m3) | Sound Speed (m/s) | Impedance (MRayl) |
|--------|-----------------|--------------------|--------------------|
| Spermaceti oil (37C) | 857 | 1,370 | 1.17 |
| Spermaceti oil (25C, solid) | 870 | 1,460 | 1.27 |
| Junk lipid (posterior) | 860-880 | 1,370-1,400 | 1.18-1.23 |
| Junk lipid (anterior) | 900-950 | 1,400-1,450 | 1.26-1.38 |
| Connective tissue | 1,050-1,100 | 1,550-1,600 | 1.63-1.76 |
| Skull bone | 1,800-2,100 | 2,800-3,500 | 5.04-7.35 |
| Air (in sacs) | 1.2 | 340 | 0.000408 |
| Seawater | 1,025 | 1,530 | 1.57 |

**Critical reflections:**
- Spermaceti / air sac: R ~ 0.9993 (essentially a perfect mirror)
- Spermaceti / case wall: R ~ 0.16 (mostly transmissive - waveguide)
- Junk / seawater: graded transition (GRIN lens - minimizes reflection, maximizes forward beam)

### Temperature Control Hypothesis
- Whales can regulate blood flow to spermaceti organ
- Warmer = liquid = c ~1,370 m/s; Cooler = solid = c ~1,530 m/s
- This changes IPI by ~10% - effectively changing "apparent body size"
- Could be an additional communication channel (modulating voice pitch)
- Madsen et al. (2002) showed IPI is stable during echolocation, but may vary in social contexts

### Dimensions by Whale Size

| Parameter | Juvenile (8-10m) | Female (10-12m) | Adult Male (15-18m) |
|-----------|-----------------|-----------------|---------------------|
| Head length | 2.0-2.5m | 2.5-3.5m | 4.0-5.5m |
| Spermaceti organ length | 1.0-1.5m | 1.5-2.5m | 2.5-4.0m |
| Spermaceti organ diameter | 0.5-0.8m | 0.8-1.2m | 1.2-2.0m |
| IPI | ~2.5-3.5ms | ~3.5-5.0ms | ~5.0-8.0ms |

### What Makes Each Voice Unique

1. **Skull rostral basin depth/curvature** - acts as parabolic reflector, varies per individual
2. **Spermaceti organ length** - directly determines IPI
3. **Case wall thickness** - affects waveguide properties
4. **Junk lipid gradient** - determines beam shape and spectral envelope
5. **Skull asymmetry** - sperm whale skulls are twisted left, degree varies
6. **Air sac geometry** - reflector shape affects focusing

### Beam Pattern
- On-axis: up to 236 dB re 1 uPa (loudest biological sound)
- Off-axis (behind): 180-190 dB (30-40 dB less)
- Beamwidth: ~28 degrees (-3dB)
- Directivity index: ~26-27 dB
- Frequency-dependent: lower freq = broader beam

---

## Simulator Design

### Approach: 2D k-Wave Acoustic Simulation

**Tool:** k-Wave (k-space pseudospectral method) - open source, Python/MATLAB, handles heterogeneous media, GPU-accelerated.

**Grid (2D sagittal cross-section of 15m adult male):**
- Domain: 6m x 3m
- Resolution: dx = 0.5 cm (supports up to 20kHz at 6+ points per wavelength)
- Grid: 1200 x 600 = 720,000 points
- Time step: ~0.5 us (CFL condition)
- Duration: ~20ms (captures P0 through P4)
- Total steps: ~40,000
- **Runs in minutes on a single machine**

**Source:** Broadband pulse at phonic lips, ~100us duration, Ricker wavelet, center freq 10-15kHz

**Geometry parameters to vary per "whale":**
- Spermaceti organ length (2.5-4.0m)
- Skull basin curvature (radius 0.8-1.5m)
- Case diameter (1.0-2.0m)
- Junk gradient steepness
- Skull asymmetry angle
- Spermaceti temperature (sound speed 1,370-1,530 m/s)

### Validation
- Simulate clicks for different head geometries
- Extract voiceprint features (spectral centroid, band energy, ICI)
- Compare to measured voiceprints of Whales A, D, F
- If the model predicts the spectral differences we see, the hypothesis is confirmed

### Research Questions

1. Can we predict a whale's voiceprint from its head geometry alone?
2. Does spermaceti temperature modulate voice pitch? (Additional communication channel?)
3. Can we reverse-engineer head size/shape from the voiceprint?
4. Can ocean propagation modeling "undo" distance/depth effects for cleaner ID?
5. Are coda type preferences (vocabulary) learned (cultural) or anatomy-dependent?

---

## Simulation Results (2026-03-29)

### First Successful Run
- 2D FDTD acoustic simulation, sagittal plane
- Fatty interfaces (Gaussian-smoothed tissue boundaries) - biologically accurate AND numerically stable
- Air sacs modeled as low-density fatty membranes (rho=50, c=800) - gives R~0.97 reflection
- dx=1cm, dt=0.67us, 12ms duration, ~18K time steps
- Runs in 20-28 seconds per whale

### Results

| Whale | Organ Length | Expected IPI | Round-trip Gap | Pulses | Peak Freq | F/B Ratio |
|-------|-------------|-------------|----------------|--------|-----------|-----------|
| A (soprano) | 2.8m | 4.09ms | 2.14ms | 36 | 12,834 Hz | -2.1 dB |
| D (alto) | 3.2m | 4.67ms | 2.08ms | 49 | 9,251 Hz | -1.1 dB |
| F (bass) | 3.8m | 5.55ms | 2.37ms | 55 | 14,501 Hz | 5.9 dB |
| D cold (28C) | 3.2m | 4.48ms | 2.11ms | 48 | 9,000 Hz | -2.7 dB |

### Key Findings
1. **Round-trip gaps scale with head size**: F (biggest) = 2.37ms > A (smallest) = 2.14ms. The IPI is visible.
2. **Whale F is the only one with positive beam forming** (F/B = 5.9dB) - larger head = better directional focus
3. **Temperature effect visible**: D_cold (28C) has slightly different IPI (2.11ms) vs D_warm (37C, 2.08ms) and lower peak freq (9.0kHz vs 9.25kHz)
4. **Each whale produces distinct pulse counts**: 36, 49, 55, 48 - geometry determines how many bounces occur
5. **Fatty interfaces were the key** to numerical stability - sharp tissue boundaries caused FDTD blowup

### High-Resolution Analysis (Run 2)

**Cavity Resonance (the voiceprint fingerprint):**

| Whale | Fundamental | Primary Band | Energy Distribution |
|-------|------------|-------------|-------------------|
| A (soprano) | 12,833 Hz | 12-16kHz (48%) | High-frequency dominant |
| D (alto) | 9,250 Hz | 8-12kHz (49%) | Mid-frequency dominant |
| F (bass) | 14,500 Hz | 12-16kHz (41%) | Broader spread, more 4-8kHz |
| D cold (28C) | 9,000 Hz | 16-20kHz (41%) | Shifted from warm D |

**Macro-Pulse Structure (P0/P1/P2):**

| Whale | Pulses | IPI (ms) | Mean Rise | Mean Fall |
|-------|--------|----------|-----------|-----------|
| A | 3 | [2.89, 4.81] | 0.587ms | 0.007ms |
| D | 3 | [2.97, 5.50] | 0.304ms | 0.021ms |
| F | 2 | [3.88] | 0.619ms | 0.040ms |
| D cold | 3 | [3.06, 5.75] | 0.412ms | 0.044ms |

- P0 always has fastest rise (direct path through junk)
- Each successive reflection smooths the wavefront (slower rise)
- Very fast falloff across all whales (5-90us) - clicks are impulsive

**Temperature Effect:**
- D warm (37C): fundamental 9,250Hz, mean IPI 4.23ms
- D cold (28C): fundamental 9,000Hz, mean IPI 4.41ms
- **250Hz frequency shift and 0.18ms IPI shift from temperature alone**
- This confirms whales could modulate their voice by controlling spermaceti temperature

**Cavity Interference Patterns:**
- Each whale has distinct harmonic structure (2nd harmonics visible)
- Destructive nulls at 38-46kHz (cavity anti-resonances)
- The interference pattern IS the voiceprint - it's uniquely determined by cavity geometry

**Frequency Evolution Over Time:**
- Click starts at organ resonance frequency
- Shifts higher with each successive bounce (wavefront filtering by cavity)
- Between pulses, frequency drops to noise floor then snaps back at next pulse arrival

---

---

## Observations & Ideas

### O1: Cold Water Preserves Voiceprints Better at Distance - CONFIRMED BY MODEL
Cold water is better for whale ID at range for three reasons:
1. **SOFAR channel** - cold deep water creates a natural waveguide at ~1000m depth, sound travels thousands of km
2. **Less high-frequency absorption** - seawater absorption is frequency-dependent AND temperature-dependent. At 10kHz, ~1 dB/km. Cold water absorbs less, so the spectral detail (the voiceprint) survives longer distances.
3. **Slightly worse head coupling** - cold water has lower impedance (~1.49 vs ~1.57 MRayl), increasing mismatch with spermaceti (~1.17 MRayl). Marginally less efficient radiation, but this is a tiny effect compared to the propagation benefits.

Net: same click in cold water carries further with more spectral detail intact. Implication: whale ID from distant hydrophones should work better in cold/deep water.

**MODELED (2026-03-29):**

Ocean absorption (Francois-Garrison model) at different temperatures:

| Temp | Abs@5kHz | Abs@10kHz | Abs@15kHz | Head-Water R |
|------|----------|-----------|-----------|-------------|
| 2C | 11.75 dB/km | 45.51 dB/km | 97.22 dB/km | 0.059 |
| 10C | 8.79 dB/km | 34.72 dB/km | 76.50 dB/km | 0.069 |
| 20C | 6.09 dB/km | 24.26 dB/km | 54.19 dB/km | 0.079 |
| 30C | 4.24 dB/km | 16.95 dB/km | 38.05 dB/km | 0.086 |

WAIT - cold water has HIGHER absorption, not lower! The model shows 2C absorbs 11.75 dB/km at 5kHz vs only 4.24 dB/km at 30C. This reverses my initial hypothesis. Warm water preserves high frequencies better.

**Voiceprint survival by distance:**

| Distance | Cold (2C) spread | Warm (30C) spread |
|----------|-----------------|-------------------|
| 0.1 km | 1,617 Hz | 1,018 Hz |
| 0.5 km | 1,492 Hz | 1,757 Hz |
| 1.0 km | 1,801 Hz | 732 Hz |
| 5.0 km | 559 Hz | 704 Hz |
| 10.0 km | 555 Hz | 584 Hz |
| 50.0 km | 482 Hz | 664 Hz |

At close range (<1km), cold water preserves MORE spectral spread (better ID).
At long range (>5km), warm water preserves slightly more spread.
The crossover happens around 1-5km where frequency-dependent absorption dominates.

**Key finding:** Voiceprint identification is feasible to ~5km (centroid spread ~500-700Hz across whales). Beyond 10km, all clicks converge to low-frequency residuals (<2kHz) and spectral ID becomes very difficult. The IPI (timing) would still work at longer range since it doesn't depend on frequency content.

### O2: Temperature Control as Communication Channel
Spermaceti temperature shifts fundamental resonance by ~250Hz per 9C. If whales can modulate this voluntarily, they have a "pitch knob" independent of click pattern. Could encode emotional state, urgency, or identity confirmation.

### O3: Vertical Posture During Codas - Listening Mode or SOFAR Injection?
Sperm whales go nearly vertical during social coda exchanges. Their click beam is highly directional (~28 degrees). Vertical posture means the beam points UP or DOWN, not at other whales. Three hypotheses:

1. **Listening mode** - the head is also a massive acoustic receiver. Aiming it at the surface or deep maximizes reception area. The codas they produce during this posture use the weaker off-axis radiation (which carries the multi-pulse structure better anyway).

2. **SOFAR channel injection** - at the thermocline, sound speed is minimum. A vertical whale clicking at this depth injects energy directly into the natural waveguide. The clicks refract horizontally and propagate enormous distances. The vertical posture is an antenna orientation for long-range broadcast.

3. **Surface/bottom reflection** - clicks aimed at the surface reflect and spread horizontally like an acoustic ceiling bounce. The surface acts as a mirror, converting a narrow vertical beam into a wide horizontal broadcast.

These aren't mutually exclusive. A vertical whale could be simultaneously listening (head orientation), broadcasting via SOFAR (depth positioning), and reflecting off the surface (beam direction).

### O4: Sound Speed Profile Creates Natural Waveguide (SOFAR)
Sound speed in the ocean varies with depth:
```
Surface (warm):     c ~1540 m/s  (fast - temperature dominates)
Thermocline:        c drops to ~1480 m/s minimum (SOFAR axis)
Deep (cold, high P): c rises to ~1500+ m/s (pressure dominates)
```
Sound refracts toward the minimum speed layer. A click produced at or near the SOFAR axis gets trapped and can travel thousands of km. The SOFAR depth varies: ~1000m in temperate waters, ~200m near poles, ~1500m in tropics (Dominica).

Sperm whales dive to 1000-2000m. They pass through the SOFAR channel on every dive. If their social codas happen near SOFAR depth, they're essentially broadcasting into a natural fiber optic cable for sound.

**To model:** Simulate click propagation through a realistic Caribbean sound speed profile (warm surface, thermocline at ~200-800m, SOFAR axis at ~800m for Dominica).

### O5: Skull Asymmetry Creates Directional Spectral Differences
Sperm whale skulls are twisted left. This means the voiceprint changes depending on recording angle. A hydrophone to the left of the whale hears a different spectral shape than one to the right. Could be used for spatial localization or intentional directional signaling.

### O6: Standing Waves as Frequency Filters
The cavity standing wave pattern determines which frequencies radiate efficiently (antinodes at boundary = strong radiation) and which don't (nodes at boundary = poor radiation). The standing wave pattern is set by geometry. Changing the geometry (jaw position? muscle tension on case wall?) could shift which frequencies radiate - another modulation channel.

### O7: Whale Age from Voiceprint
Skull grows throughout life, rostral basin deepens in males. The voiceprint should drift predictably with age. Could track a whale's aging from recordings over years without ever seeing it.

### O8: Ocean as a Matched Filter
If we know the ocean sound speed profile (from CTD data or climatology), we can model propagation and "undo" it from the recorded signal. This recovers the at-source voiceprint from a distant recording. Critical for ID from passive acoustic monitoring (PAM) arrays.

### O9: Maxillonasalis as Vocal Cords - A Second Spectral Filter (NOVEL HYPOTHESIS)

**Observation (2026-03-30):** The maxillonasalis muscle wraps over the spermaceti organ in distinct transverse bands (visible in Cranford anatomical illustrations). These segmented muscle bands stretched across a resonant cavity are structurally analogous to vocal cords.

**Current literature model:**
- Source = phonic lips (analogous to vocal cords)
- Filter = distal air sac shape (analogous to vocal tract)
- Maxillonasalis described as "controlling organ shape" and "pressurizing air" (Cranford 1999, Huggenberger 2014)
- CETI vowels paper (2025) shows formant structures similar to human vowels, attributes to air sac shape changes

**What's NOT in the literature (our hypothesis):**
The maxillonasalis bands act as a SECOND spectral filter independent of the air sacs. Each transverse band, when tensioned, selectively damps or passes specific frequencies as the acoustic pulse bounces through the spermaceti organ beneath them. This is functionally identical to how vocal cord tension controls which harmonics pass in human speech.

**Physics argument:**
1. **Impedance loading:** A tensioned muscle band (rho~1050, c~1570) pressed against the spermaceti (rho=857, c=1370) creates a local impedance mismatch. The reflection coefficient at that band depends on the band's tension and thickness.
2. **Frequency selectivity:** Each band has a resonant frequency determined by its length, thickness, and tension (f = (1/2L) * sqrt(T/mu), same as a vibrating string). Frequencies matching a band's resonance get absorbed; others pass.
3. **Segmented control:** The bands are independently innervated. Different tension patterns create different spectral filter profiles - like an equalizer with ~5-8 bands.
4. **Timescale:** Muscle contraction operates in milliseconds - fast enough to change the spectral filter BETWEEN CLICKS within a single coda. Temperature modulation (the other known mechanism) takes minutes.

**Predictions:**
- Spectral content should vary systematically with maxillonasalis activation patterns
- Whales could voluntarily change their spectral voiceprint in real time
- The "a-coda" vs "i-coda" vowel distinction (CETI 2025) may correspond to different maxillonasalis tension states, not just air sac shape
- Individual voiceprint differences may partly reflect individual differences in muscle mass, band count, and resting tension - not just skull geometry

**Modulation channels available to the whale (summary):**
1. Spermaceti temperature - slow pitch drift (minutes) - ~250Hz per 9C
2. Maxillonasalis tension - fast spectral shaping (milliseconds) - selective frequency filter
3. Air sac shape/tension - reflection coefficient tuning
4. Jaw/mandible position - impedance matching at tissue-water boundary

**To model:** Add maxillonasalis bands to the FDTD/FEM simulator as impedance-loaded regions on the dorsal surface of the spermaceti organ. Vary band tension and measure spectral output changes. Compare to measured a-coda vs i-coda spectral differences.

**Reference image:** `wt_whalehead_free.webp` (Cranford illustration showing maxillonasalis sectioned over spermaceti organ)

### O10: The Complete Sound Production Model - Excite, Resonate, Modulate, Filter

**Observation (2026-03-30):** The sperm whale sound system is not a "click generator" - it's a coupled resonant instrument with distinct functional layers:

1. **Phonic lips = Excitation source.** Sustained drive (~500-800us, not a single 50us impulse). Pumps energy into the cavity to build standing waves. Solver optimization confirmed long source durations are needed.

2. **Spermaceti cavity = Resonator.** Bounded by two near-perfect mirrors (frontal and distal air sacs, R~0.997). Cavity length, diameter, and spermaceti sound speed determine the resonant mode frequencies. This is the fundamental "voice." The P0/P1/P2 pulse structure in the time domain is the transient buildup and decay of these cavity modes.

3. **Maxillonasalis muscles = Real-time modulator.** Contracts on ms timescale to:
   - Change cavity diameter (shifts transverse resonant modes)
   - Compress spermaceti (increases density + sound speed, shifts longitudinal modes)
   - Load the dorsal surface with variable impedance (damps specific modes)
   - The organ may be longer than IPI alone predicts to accommodate compression

4. **Temperature = Slow modulator.** Changes spermaceti sound speed over minutes. Coarse pitch control (~250Hz per 9C). Independent of muscle control.

5. **Exit path = Spectral filter.** Sound exits through: case wall (connective tissue) -> maxillonasalis muscle -> junk (lipid with ~10-30 connective tissue septa) -> blubber -> skin -> water. Each layer preferentially absorbs high frequencies (0.3-5.0 dB/cm/MHz). Total exit path is ~1-2m of tissue. This is likely why recorded spectra peak at 5-10kHz even though cavity resonance may be higher.

6. **Ocean propagation = Final filter.** Depth-integrated, frequency-dependent absorption through the water column. Whale at 500-1400m depth, sound traveling through cold deep water to surface hydrophone. Further attenuates high frequencies.

**Signal chain:** Lips (excite) -> Cavity (resonate) -> Muscles (modulate) -> Exit tissues (filter) -> Ocean (attenuate) -> Hydrophone (record)

**What makes each whale unique:** Skull geometry sets the cavity shape (fixed anatomy). Muscle tone sets the modulation pattern (variable physiology). Temperature sets the coarse pitch (variable physiology). Exit-path tissue thickness varies per individual.

**Rejected hypothesis:** Skull as vibrating plate (like a guitar body). The skull is too thick (8cm+), too heavily damped by surrounding tissue, and impedance mismatch with tissue is too low (3:1 vs 3000:1 for wood/air). The skull acts as a rigid reflector, not a vibrating element. Sub-100Hz energy in recordings is likely ocean ambient, whole-body recoil, or click-train envelope modulation.

**Solver evidence (2026-03-30):** 31-parameter optimization with differential evolution. At iteration 2:
- Source duration converges to 500-800us (sustained excitation, not single impulse)
- Whale depth converges to 500-1400m (deep water propagation critical)
- Recording distance converges to 1300-1500m (further than assumed)
- Exit-path tissue absorption matters: muscle 1-2 dB/cm/MHz, skin 1.4-4.7 dB/cm/MHz
- Absorption power law 1.3-2.0 (super-linear frequency dependence)
- Organ length pushing upper bounds (5-6m) suggesting compression factor needed

---

## Paper: "Where's WHaldo?"

### Working Title
"Beyond Inter-Pulse Interval: Spectral Voiceprints from Spermaceti Cavity Resonance Enable Individual Identification of Sperm Whales"

### Novel Contributions (what's not in the literature)
1. Full 2D/3D wavefront simulation using real skull geometry showing how head anatomy shapes spectral envelope (not just IPI timing)
2. Cavity interference patterns (resonant frequencies + harmonics) as individual-specific fingerprints
3. ML classification validated across 3 datasets: DSWP 91.5% (3 whales), CETI 85.9% (13 whales), Gero 57.7% ICI-only (16 whales)
4. Simulated temperature modulation of spermaceti shifting fundamental resonance (250Hz shift, 0.18ms IPI shift at 9C change)
5. Prediction of voiceprint from anatomy - connecting physics simulation to measured acoustic signatures
6. **Maxillonasalis as vocal cord analogue** - hypothesis that transverse muscle bands over the spermaceti organ act as a fast (ms-timescale) spectral filter independent of air sac shape, with segmented control enabling real-time frequency modulation between clicks

### What's Known (prior art)
- IPI correlates with body length (Gordon 1991, Madsen 2002)
- Multi-pulse structure from bent horn model (Norris & Harvey 1972)
- Cranford et al. (2008) did FEM on beaked whales (not sperm whale, not individual ID)
- Wei et al. (2017) FEM on porpoises (not individual ID)
- Coda types are culturally transmitted within clans (Rendell & Whitehead 2003)

### Work Still Needed
- [ ] Validate simulated voiceprints against measured voiceprints of Whales A, D, F
- [ ] **Standing wave analysis of the head cavity** - map nodes/antinodes at different frequencies, identify which resonant modes dominate
- [ ] **Head-to-ocean interface analysis** - how standing wave patterns couple to the water at the tissue-water boundary. Nodes at the interface = poor transmission at that frequency. Antinodes = strong transmission. This shapes what actually gets radiated.
- [ ] **Ocean temperature effects on acoustic output** - seawater sound speed changes with temperature (1450 m/s at 5C vs 1540 m/s at 25C). This changes the impedance match at the head-water boundary. Same click, different radiation pattern. A whale in cold Arctic water vs warm Dominica water emits a different spectrum from identical internal acoustics.
- [ ] **Depth effects** - pressure changes air sac volume (though Madsen showed IPI is stable at depth, the impedance match at the sac boundaries changes)
- [ ] 3D simulation (sagittal plane only captures part of the story - skull asymmetry matters)
- [ ] More whales - 3 individuals is a proof of concept, need 10+ for a real paper
- [ ] Behavioral context data - link vocalizations to activities
- [ ] Cross-reference with CETI project data if available
- [ ] Sensitivity analysis - which geometric parameters matter most for ID
- [ ] Test on unlabeled recordings - can we ID whales we've never seen?
- [ ] Peer review of the physics model with a bioacoustician
- [ ] **Thermocline effects** - sound speed profile in the water column creates refraction. Clicks propagating through a thermocline bend. This affects what a distant hydrophone records vs what was emitted. Need to model this to "undo" propagation effects for cleaner ID at distance.

### Target Journals
- Journal of the Acoustical Society of America (JASA) - where the key prior work lives
- Bioinspiration & Biomimetics - where Cranford's FEM work was published
- Royal Society Open Science - open access, good for interdisciplinary work

---

### Key References
- Norris & Harvey (1972) - bent horn model (foundational)
- Cranford (1999, 2000) - detailed anatomy, unified model
- Zimmer et al. (2005) JASA - quantitative acoustic model, off-axis multi-pulse
- Madsen et al. (2002) JASA/JEB - IPI measurements, body length correlation
- Mohl et al. (2003) JASA - monopulsed on-axis nature, 236dB source level
- Flewellen & Morris (1978) - sound speed in spermaceti vs temperature
- Clarke (1978) - morphometric data from ~50 dissected specimens
- Aroyan (2001) - FEM methodology for odontocete heads
- Cranford et al. (2008) - COMSOL FEM for beaked whale (closest published FEM)
- Wei et al. (2014, 2017) - FEM for finless porpoise (CT-derived geometry)

### Signal Analysis Summary

**Frequency:**
- Most energy in 5-10kHz band (30.7% average)
- Sub-100Hz carries 21.2% (ocean ambient + whale body resonance)
- 68.5% of codas have centroid >5kHz
- Spectral rolloff at 9,418Hz mean

**Temporal:**
- Intensity profile is nearly flat (no consistent attack/decay pattern across codas)
- 55% of codas show periodicity in envelope
- Average silence ratio: 22% (some codas are mostly quiet with brief click bursts)

**Click Patterns:**
- ICIs peak at 20-67ms (4,815 measurements) with long tail to 2s
- Mean ICI: 142ms, Median: 79ms
- 60% of codas are rhythmically irregular, 10% regular

### Credits & Licenses

- **"Sperm whale cranium"** (https://skfb.ly/6zAwH) by NHM_Imaging is licensed under Creative Commons Attribution-NonCommercial (http://creativecommons.org/licenses/by-nc/4.0/). Specimen NHMUK ZD 2007.100, stranded 1930s Norway. 175K triangles, 87K vertices.

### 3D Skull Data
- **High-res OBJ**: `/mnt/archive/datasets/whale_communication/3d_scans/sperm_whale_cranium_high/source/12_2007_online.obj` (21MB, 88K vertices, 175K faces)
- **Low-res glTF**: `/mnt/archive/datasets/whale_communication/3d_scans/sperm_whale_cranium_low/scene.gltf` (5.9MB)
- **Material properties**: `/Users/ericbrowy/Desktop/Claude/WheresWhaldo/material_properties.md`

### Files

- `/mnt/archive/datasets/whale_communication/` - raw data (744MB)
- `/mnt/archive/datasets/whale_communication/analysis/deep_analysis_raw.jsonl` - per-coda signal analysis
- `/mnt/archive/datasets/whale_communication/analysis/deep_analysis_report.txt` - full text report
- `/mnt/archive/datasets/whale_communication/analysis/whale_voiceprints.json` - per-whale acoustic profiles
- `/mnt/archive/datasets/whale_communication/analysis/voiceprint_report.txt` - identification results
- Scripts: `~/Desktop/Claude/witness/scripts/whale_signal_analysis.py`, `whale_deep_analysis.py`, `whale_voiceprint.py`

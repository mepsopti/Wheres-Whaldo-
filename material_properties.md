# Sperm Whale Head - Material Properties Reference

All values from published literature. Sources cited per measurement.

## Spermaceti Oil (Case)

The main acoustic waveguide. Fills the barrel-shaped case dorsal to the junk.

| Property | Value | Condition | Source |
|----------|-------|-----------|--------|
| Density | 857 kg/m3 | 37C, liquid | Clarke 1978 |
| Density | 870 kg/m3 | 25C, partial solid | Clarke 1978 |
| Sound speed | 1,373 m/s | 35C | Flewellen & Morris 1978 |
| Sound speed | 1,393 m/s | 30C | Flewellen & Morris 1978 |
| Sound speed | 1,460 m/s | 25C (phase transition) | Flewellen & Morris 1978 |
| Sound speed | 1,530 m/s | 20C, solid | Flewellen & Morris 1978 |
| Temperature coefficient | -3 to -4 m/s per C | Liquid phase | Flewellen & Morris 1978 |
| Impedance | 1.17 MRayl | 37C | Calculated (rho x c) |
| Impedance | 1.27 MRayl | 25C | Calculated |
| Solidification point | 29-31 C | - | Clarke 1978 |
| Composition | Wax esters + triglycerides | Ratio varies along organ length | Clarke 1978 |

### Temperature-Speed Curve (Flewellen & Morris 1978)
```
Temp (C)  |  Speed (m/s)  |  Phase
----------|---------------|--------
  37      |   1,370       |  Liquid
  35      |   1,373       |  Liquid
  33      |   1,380       |  Liquid
  31      |   1,395       |  Transition begins
  29      |   1,420       |  Partially solid
  27      |   1,440       |  Partially solid
  25      |   1,460       |  Mostly solid
  20      |   1,530       |  Solid
```

## Junk (Melon Analogue)

The acoustic lens. Graded lipid composition creates GRIN (gradient-index) lens effect.

| Property | Value | Position | Source |
|----------|-------|----------|--------|
| Density (posterior) | 860-880 kg/m3 | Near skull | Cranford 1996 |
| Density (anterior) | 900-950 kg/m3 | Near tip | Cranford 1996 |
| Sound speed (posterior) | 1,370-1,400 m/s | High wax ester | Goold 1996 |
| Sound speed (anterior) | 1,400-1,450 m/s | More triglyceride | Goold 1996 |
| Impedance (posterior) | 1.18-1.23 MRayl | - | Calculated |
| Impedance (anterior) | 1.26-1.38 MRayl | - | Calculated |
| Septa (connective tissue) | 1,550-1,600 m/s | Between lipid layers | Cranford 1996 |
| Septa density | 1,050-1,100 kg/m3 | - | Cranford 1996 |
| Layer thickness (lipid) | 2-15 mm | Individual compartments | Clarke 1978 |
| Layer thickness (septa) | 1-3 mm | Connective tissue | Clarke 1978 |

### GRIN Lens Gradient
The junk has nested elliptical lipid compartments. Sound speed increases from center to periphery:
- Core: ~1,370 m/s (similar to spermaceti)
- Periphery: ~1,450 m/s (more triglyceride, more water)
- This focuses the sound beam forward without a curved surface (same principle as a fiber optic GRIN lens)

### Comparison to Dolphin Melon
| Property | Sperm Whale Junk | Dolphin Melon |
|----------|-----------------|---------------|
| Center speed | 1,370 m/s | 1,350-1,370 m/s |
| Periphery speed | 1,430-1,450 m/s | 1,400-1,450 m/s |
| Gradient | ~80 m/s over ~1m | ~80 m/s over ~10cm |
| Structure | Nested elliptical layers | Continuous gradient |

## Connective Tissue (Case Wall, Septa)

| Property | Value | Source |
|----------|-------|--------|
| Density | 1,050-1,100 kg/m3 | Cranford 1996 |
| Sound speed | 1,550-1,600 m/s | Cranford 1996 |
| Impedance | 1.63-1.76 MRayl | Calculated |
| Case wall thickness | 3-8 mm | Clarke 1978 |

## Skull Bone

| Property | Value | Source |
|----------|-------|--------|
| Density | 1,800-2,100 kg/m3 | General cetacean bone |
| Sound speed | 2,800-3,500 m/s | General cetacean bone |
| Impedance | 5.04-7.35 MRayl | Calculated |
| Rostral basin depth | 5-15 cm (varies with age/sex) | Clarke 1978 |
| Rostral basin radius | 0.8-1.5 m | Estimated from dissections |

## Air Sacs (Frontal and Distal)

| Property | Value | Notes | Source |
|----------|-------|-------|--------|
| Air density | 1.2 kg/m3 | At surface pressure | - |
| Air sound speed | 340 m/s | At surface | - |
| Air impedance | 0.000408 MRayl | - | Calculated |
| Reflection coeff (tissue/air) | 0.9993 | Near-perfect mirror | Calculated |
| Frontal sac dimensions | ~0.8m x 0.8m x 0.02m | Large male | Cranford 1999 |
| Distal sac dimensions | ~0.3m diameter x 0.02m | - | Cranford 1999 |

### Depth Effects on Air Sacs
At depth, air compresses:
- 100m: pressure = 11 atm, air volume = 1/11 of surface
- 1000m: pressure = 101 atm, air volume = 1/101 of surface
- Despite this, Madsen et al. (2002) showed IPI is stable at depth
- The sacs may collapse to thin films that still maintain impedance mismatch
- Or the whales may actively maintain sac pressure using their right nasal passage

## Blubber

| Property | Value | Source |
|----------|-------|--------|
| Density | 900-950 kg/m3 | General cetacean |
| Sound speed | 1,400-1,450 m/s | General cetacean |
| Impedance | 1.26-1.38 MRayl | Calculated |
| Thickness (head) | 10-20 cm | Variable |

## Muscle

| Property | Value | Source |
|----------|-------|--------|
| Density | 1,040-1,060 kg/m3 | General |
| Sound speed | 1,540-1,600 m/s | General |
| Impedance | 1.60-1.70 MRayl | Calculated |

## Seawater (for reference)

| Property | Value | Condition | Source |
|----------|-------|-----------|--------|
| Density | 1,025 kg/m3 | 15C, 35ppt | - |
| Sound speed | 1,530 m/s | 15C, 35ppt, surface | Mackenzie 1981 |
| Impedance | 1.57 MRayl | - | Calculated |

## Key Impedance Mismatches (Reflection Coefficients)

| Interface | Z1 (MRayl) | Z2 (MRayl) | R | % Energy Reflected |
|-----------|-----------|-----------|---|-------------------|
| Spermaceti / air sac | 1.17 | 0.0004 | 0.9993 | 99.86% |
| Spermaceti / case wall | 1.17 | 1.70 | 0.185 | 3.4% |
| Junk (ant) / seawater | 1.33 | 1.57 | 0.083 | 0.7% |
| Bone / air sac | 6.00 | 0.0004 | 0.9999 | 99.99% |
| Bone / seawater | 6.00 | 1.57 | 0.585 | 34.2% |
| Blubber / seawater | 1.33 | 1.57 | 0.083 | 0.7% |

## Sources

- Clarke, M.R. (1978). Physical properties of spermaceti oil in the sperm whale. J Mar Biol Assoc UK, 58, 19-26.
- Cranford, T.W., Amundin, M., & Norris, K.S. (1996). Functional morphology and homology in the odontocete nasal complex. J Morphology, 228(3), 223-285.
- Cranford, T.W. (1999). The sperm whale nose: sexual selection on a grand scale? Marine Mammal Science, 15(4), 1133-1157.
- Flewellen, C.G. & Morris, R.J. (1978). Sound velocity measurements on samples from the spermaceti organ of the sperm whale. Deep-Sea Research, 25, 269-277.
- Goold, J.C., Bennell, J.D., & Jones, S.E. (1996). Sound velocity measurements in spermaceti oil under the combined influences of temperature and pressure. Deep-Sea Research I, 43(6), 961-969.
- Madsen, P.T., Wahlberg, M., & Mohl, B. (2002). Male sperm whale acoustics in a high-latitude habitat. JASA, 111(3), 1346-1352.
- Mackenzie, K.V. (1981). Nine-term equation for sound speed in the oceans. JASA, 70(3), 807-812.

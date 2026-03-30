#!/usr/bin/env python3
"""
Depth-Dependent Sound Propagation Model

Simulates how sperm whale clicks propagate through a realistic
Caribbean (Dominica) sound speed profile. Models:
  - Temperature/pressure/salinity vs depth -> sound speed profile
  - Ray tracing through the profile (Snell's law)
  - SOFAR channel trapping
  - Click injection at different depths (surface, thermocline, deep)
  - Vertical vs horizontal whale orientation
"""

import json
import os
import numpy as np
import time

OUTPUT_DIR = "/mnt/archive/datasets/whale_communication/simulation"
REPORT_PATH = os.path.join(OUTPUT_DIR, "depth_propagation_report.txt")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# DOMINICA SOUND SPEED PROFILE
# ============================================================

def dominica_temperature(depth_m):
    """Approximate temperature profile for Dominica (Caribbean).
    Based on typical tropical Atlantic profiles."""
    if depth_m < 50:
        return 27.0 - depth_m * 0.02  # warm mixed layer
    elif depth_m < 200:
        return 26.0 - (depth_m - 50) * 0.06  # upper thermocline
    elif depth_m < 800:
        return 17.0 - (depth_m - 200) * 0.015  # main thermocline
    elif depth_m < 1500:
        return 8.0 - (depth_m - 800) * 0.004  # deep thermocline
    else:
        return 2.5 - (depth_m - 1500) * 0.0005  # deep water


def dominica_salinity(depth_m):
    """Approximate salinity profile (ppt)."""
    if depth_m < 100:
        return 35.5
    elif depth_m < 500:
        return 35.5 - (depth_m - 100) * 0.001
    else:
        return 35.1


def sound_speed_profile(max_depth=2500, step=10):
    """Compute full sound speed profile for Dominica waters."""
    depths = np.arange(0, max_depth + step, step)
    profile = []
    for d in depths:
        T = dominica_temperature(d)
        S = dominica_salinity(d)
        # Mackenzie equation
        c = (1448.96 + 4.591 * T - 0.05304 * T**2 + 0.0002374 * T**3
             + 1.340 * (S - 35) + 0.0163 * d + 1.675e-7 * d**2
             - 0.01025 * T * (S - 35) - 7.139e-13 * T * d**3)
        profile.append({"depth_m": float(d), "temp_c": round(T, 2),
                        "salinity_ppt": round(S, 2), "sound_speed_ms": round(c, 1)})
    return profile


# ============================================================
# RAY TRACING
# ============================================================

def trace_ray(profile, source_depth_m, initial_angle_deg, max_range_km=50, dt_s=0.01):
    """Trace an acoustic ray through the sound speed profile.

    Args:
        profile: list of {depth_m, sound_speed_ms}
        source_depth_m: starting depth (m)
        initial_angle_deg: angle from horizontal (0=horizontal, +90=down, -90=up)
        max_range_km: maximum horizontal range
        dt_s: time step

    Returns:
        list of (range_m, depth_m, time_s) points along the ray
    """
    # Build interpolation for c(depth)
    depths = np.array([p["depth_m"] for p in profile])
    speeds = np.array([p["sound_speed_ms"] for p in profile])

    def c_at_depth(z):
        z = max(0, min(z, depths[-1]))
        return float(np.interp(z, depths, speeds))

    def dc_dz(z):
        """Gradient of sound speed with depth."""
        dz = 1.0  # 1m for numerical gradient
        z1 = max(0, z - dz)
        z2 = min(depths[-1], z + dz)
        return (c_at_depth(z2) - c_at_depth(z1)) / (z2 - z1 + 1e-10)

    # Initial conditions
    angle = np.radians(initial_angle_deg)
    x = 0.0  # horizontal range (m)
    z = float(source_depth_m)  # depth (m)
    t = 0.0

    c0 = c_at_depth(z)
    # Snell's law constant: cos(angle) / c = constant along ray
    snell_const = np.cos(angle) / c0

    ray_points = [(x, z, t)]
    max_range_m = max_range_km * 1000

    for _ in range(int(max_range_m / (c0 * dt_s * 2))):
        c_local = c_at_depth(z)

        # Current angle from Snell's law
        cos_angle = snell_const * c_local
        cos_angle = max(-0.9999, min(0.9999, cos_angle))
        angle = np.arccos(cos_angle)

        # Determine if ray is going up or down
        grad = dc_dz(z)
        # Ray curves toward lower sound speed
        if z > source_depth_m and grad > 0:
            angle = -angle  # curving upward

        # Step
        dx = c_local * np.cos(angle) * dt_s
        dz = c_local * np.sin(angle) * dt_s

        x += abs(dx)
        z += dz
        t += dt_s

        # Boundary reflections
        if z < 0:
            z = -z  # reflect off surface
            angle = -angle
        if z > depths[-1]:
            z = 2 * depths[-1] - z  # reflect off bottom (approximate)
            angle = -angle

        ray_points.append((x, z, t))

        if x >= max_range_m:
            break

    return ray_points


def analyze_sofar(profile):
    """Find the SOFAR channel axis and properties."""
    depths = [p["depth_m"] for p in profile]
    speeds = [p["sound_speed_ms"] for p in profile]

    min_idx = np.argmin(speeds)
    sofar_depth = depths[min_idx]
    sofar_speed = speeds[min_idx]

    # Channel width (depths where c < surface c)
    surface_c = speeds[0]
    channel_top = sofar_depth
    channel_bottom = sofar_depth
    for i in range(min_idx, -1, -1):
        if speeds[i] >= surface_c:
            channel_top = depths[i]
            break
    for i in range(min_idx, len(speeds)):
        if speeds[i] >= surface_c:
            channel_bottom = depths[i]
            break

    return {
        "sofar_depth_m": sofar_depth,
        "sofar_speed_ms": sofar_speed,
        "channel_top_m": channel_top,
        "channel_bottom_m": channel_bottom,
        "channel_width_m": channel_bottom - channel_top,
    }


def main():
    log("Depth-Dependent Sound Propagation Model")
    log("Location: Dominica, Caribbean")

    # Build profile
    profile = sound_speed_profile()

    # SOFAR analysis
    sofar = analyze_sofar(profile)

    lines = []
    lines.append("=" * 80)
    lines.append("DEPTH-DEPENDENT PROPAGATION - DOMINICA CARIBBEAN")
    lines.append("=" * 80)

    # Sound speed profile
    lines.append("\nSOUND SPEED PROFILE:")
    lines.append(f"{'Depth':>8s} {'Temp':>7s} {'Salinity':>9s} {'Speed':>8s}")
    lines.append("-" * 40)
    for p in profile[::10]:  # every 100m
        lines.append(f"{p['depth_m']:>6.0f}m {p['temp_c']:>5.1f}C {p['salinity_ppt']:>7.1f}ppt {p['sound_speed_ms']:>6.1f}m/s")

    # SOFAR
    lines.append(f"\nSOFAR CHANNEL:")
    lines.append(f"  Axis depth: {sofar['sofar_depth_m']:.0f}m")
    lines.append(f"  Speed at axis: {sofar['sofar_speed_ms']:.1f} m/s")
    lines.append(f"  Channel: {sofar['channel_top_m']:.0f}m - {sofar['channel_bottom_m']:.0f}m ({sofar['channel_width_m']:.0f}m wide)")

    # Ray tracing from different depths and angles
    source_configs = [
        ("Surface whale (10m), horizontal", 10, 0),
        ("Surface whale (10m), 10deg down", 10, 10),
        ("Surface whale (10m), 30deg down", 10, 30),
        ("Thermocline whale (500m), horizontal", 500, 0),
        ("Thermocline whale (500m), 10deg down", 500, 10),
        ("Thermocline whale (500m), 10deg up", 500, -10),
        ("SOFAR axis whale (800m), horizontal", 800, 0),
        ("SOFAR axis whale (800m), 5deg down", 800, 5),
        ("SOFAR axis whale (800m), 5deg up", 800, -5),
        ("Deep whale (1200m), horizontal", 1200, 0),
        ("Deep whale (1200m), 45deg up (vertical posture)", 1200, -45),
        ("Deep whale (1200m), 90deg up (fully vertical)", 1200, -89),
        ("Deep whale (1200m), 90deg down", 1200, 89),
    ]

    lines.append(f"\n{'='*80}")
    lines.append("RAY TRACING RESULTS")
    lines.append(f"{'='*80}")

    for desc, depth, angle in source_configs:
        ray = trace_ray(profile, depth, angle, max_range_km=20, dt_s=0.005)

        # Analyze ray path
        ranges = [p[0] for p in ray]
        depths_ray = [p[1] for p in ray]
        max_range = max(ranges) / 1000  # km
        min_depth = min(depths_ray)
        max_depth = max(depths_ray)

        # Count surface bounces
        surface_bounces = sum(1 for i in range(1, len(depths_ray))
                            if depths_ray[i-1] > 5 and depths_ray[i] < 5)

        # Check if trapped in SOFAR
        trapped = all(sofar["channel_top_m"] - 50 < d < sofar["channel_bottom_m"] + 50
                     for d in depths_ray[len(depths_ray)//4:])  # check last 75%

        lines.append(f"\n  {desc}")
        lines.append(f"    Source: {depth}m, angle: {angle}deg")
        lines.append(f"    Range achieved: {max_range:.1f} km")
        lines.append(f"    Depth range: {min_depth:.0f}m - {max_depth:.0f}m")
        lines.append(f"    Surface bounces: {surface_bounces}")
        lines.append(f"    SOFAR trapped: {'YES' if trapped else 'NO'}")

        # Depth at key ranges
        for target_km in [1, 5, 10, 20]:
            target_m = target_km * 1000
            closest = min(ray, key=lambda p: abs(p[0] - target_m))
            if abs(closest[0] - target_m) < 500:
                lines.append(f"    At {target_km}km: depth={closest[1]:.0f}m")

    # Vertical whale analysis
    lines.append(f"\n{'='*80}")
    lines.append("VERTICAL WHALE HYPOTHESIS")
    lines.append(f"{'='*80}")
    lines.append("""
A vertical sperm whale at 800-1200m depth with head pointing up:
- Main beam (P1) fires UPWARD through water column
- Hits surface at 0m, reflects back down
- Surface reflection spreads the beam horizontally (rough surface = diffuse)
- Off-axis radiation (weaker, but carries multi-pulse coda structure) goes sideways

A vertical whale at SOFAR depth (800m) with head horizontal:
- Even slight downward angle injects into SOFAR channel
- Sound gets trapped and propagates hundreds of km

IMPLICATION: The vertical posture may serve DUAL purpose:
1. HEAD UP = Surface bounce for local broadcast (coda exchange with nearby whales)
2. HEAD ANGLED = SOFAR injection for long-range broadcast
3. The highly directional on-axis beam is for echolocation (horizontal swimming)
4. Social codas use the off-axis radiation pattern (wider, multi-pulse)
""")

    # Sperm whale dive profile context
    lines.append("TYPICAL SPERM WHALE DIVE PROFILE (Dominica):")
    lines.append("  Surface: 0-15m, breathing, socializing (codas here)")
    lines.append("  Descent: 0-800m, ~1.5 m/s, mostly silent")
    lines.append("  Foraging: 800-1200m, echolocation clicks (regular)")
    lines.append("  Ascent: 1200-0m, ~1.2 m/s")
    lines.append("  Social time: 10-15 min at surface between dives")
    lines.append(f"  SOFAR axis: {sofar['sofar_depth_m']:.0f}m - whales pass through it on every dive")
    lines.append("")

    # If codas happen at surface (10-15m), they DON'T benefit from SOFAR
    # But surface reflections still help spread the signal
    lines.append("CRITICAL: Most coda exchanges happen at the SURFACE (10-15m),")
    lines.append("NOT at depth. This means SOFAR injection is NOT the primary")
    lines.append("mechanism for coda propagation. Surface duct and direct path")
    lines.append("are more relevant for social communication.")
    lines.append("")
    lines.append("However, 'slow clicks' (a different vocalization type) are")
    lines.append("produced during ascent through SOFAR depth - these COULD be")
    lines.append("long-range identity broadcasts.")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    log(f"Report: {REPORT_PATH}")
    print("\n" + report)


if __name__ == "__main__":
    main()

"""
Test script to verify planogram parsing for the new Ejemplo 2.csv data.
Generates planogram images to test_output/ for visual verification.

Usage:
    python3 test_new_data.py
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exploracion import load_planograms

OUTPUT_DIR = "test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEST 1: Verify original data still loads correctly
# ============================================================

print("\n" + "=" * 70)
print("TEST 1: Loading original data (datos/ejemplo_planograma.csv)")
print("=" * 70)

orig_planograms = load_planograms("datos/ejemplo_planograma.csv")
print("  -> Loaded {} planograms from original data".format(len(orig_planograms)))

# Plot first 2 from original for comparison
for i, p in enumerate(orig_planograms[:2]):
    n_prods = sum(len(s.products) for s in p.shelves.values())
    fname = os.path.join(
        OUTPUT_DIR,
        "original_{}_{}_{}_{}_{}.png".format(
            p.segmento_id, p.mueble_id, p.tamano, p.direccion, p.conjunto_id
        ),
    )
    print("  Plotting: {} ({} products)".format(p.title, n_prods))
    p.plot_planogram(save_path=fname)
    plt.close("all")

print("  Saved {} original planogram images.".format(min(2, len(orig_planograms))))


# ============================================================
# TEST 2: Load new data (subset for speed)
# ============================================================

print("\n" + "=" * 70)
print("TEST 2: Loading new data subset (datos/Ejemplo 2.csv)")
print("=" * 70)

# Pre-filter to diverse groups for manageable test size
df_full = pd.read_csv("datos/Ejemplo 2.csv", encoding="latin-1")
print("  Full CSV: {} rows".format(len(df_full)))

test_configs = [
    # (SEGMENTO, MUEBLE, TAMANO, DIR) — diverse selection
    ("BCO", "CF",  4.0, "DI"),   # 4-door CF (same config as original)
    ("BCO", "CFC", 4.0, "DI"),   # 4-door CFC (different furniture type)
    ("BCO", "CF",  1.0, "DI"),   # 1-door (smallest)
    ("HRN", "CF",  3.0, "ID"),   # 3-door, left-to-right direction
    ("PTC", "CF",  5.0, "DI"),   # 5-door (large)
    ("OFC", "CF",  2.0, "ID"),   # 2-door
    ("CLA", "CF",  3.5, "DI"),   # 3.5-door (half-door variant)
    ("RET", "CF",  4.5, "ID"),   # 4.5-door (half-door, larger)
]

mask = pd.Series(False, index=df_full.index)
for seg, mueble, tam, direc in test_configs:
    group_mask = (
        (df_full["SEGMENTO_ID"] == seg)
        & (df_full["MUEBLE_ID"] == mueble)
        & (df_full["TAMANO_DESC"] == tam)
        & (df_full["DIRECCION_LEGO_ID"] == direc)
    )
    mask |= group_mask

subset = df_full[mask]
print("  Selected {} rows from {} groups".format(len(subset), len(test_configs)))

# Save filtered CSV for the loader
temp_csv = os.path.join("datos", "_test_subset.csv")
subset.to_csv(temp_csv, index=False, encoding="latin-1")

# Load with updated parser
new_planograms = load_planograms(temp_csv)
print("  -> Loaded {} planograms from new data".format(len(new_planograms)))


# ============================================================
# TEST 3: Generate planogram images (first 2 per config)
# ============================================================

print("\n" + "=" * 70)
print("TEST 3: Generating planogram images")
print("=" * 70)

from collections import defaultdict

config_seen = defaultdict(int)
plotted = 0

for p in new_planograms:
    config = (p.segmento_id, p.mueble_id, p.tamano, p.direccion)
    if config_seen[config] >= 2:
        continue
    config_seen[config] += 1

    n_prods = sum(len(s.products) for s in p.shelves.values())
    fname = os.path.join(
        OUTPUT_DIR,
        "new_{}_{}_{}_{}_{}.png".format(
            p.segmento_id, p.mueble_id, p.tamano, p.direccion, p.conjunto_id
        ),
    )
    print("  Plotting: {} ({} products, {} charolas)".format(
        p.title, n_prods, len(p.shelves)
    ))
    p.plot_planogram(save_path=fname)
    plt.close("all")
    plotted += 1

# Clean up temp file
os.remove(temp_csv)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("  Original planograms loaded: {}".format(len(orig_planograms)))
print("  New planograms loaded (from subset): {}".format(len(new_planograms)))
print("  Images generated: {}".format(plotted + min(2, len(orig_planograms))))
print("  Output directory: {}/".format(OUTPUT_DIR))
print()
print("  Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print("    {}".format(f))

print()
print("WHAT TO CHECK:")
print("  1. Products are placed on shelves (not empty)")
print("  2. Door count matches TAMANO (e.g., 4.0 = 4 doors)")
print("  3. Product labels are readable")
print("  4. CFC planograms show different layout than CF")
print("  5. Half-door sizes (3.5, 4.5) show correct structure")
print("  6. Compare original_*.png vs new_*.png for same config")

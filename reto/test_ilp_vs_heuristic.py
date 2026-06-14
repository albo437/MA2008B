"""
Test: compare ILP (exact) vs Heuristic (your planogram_model.py) on a
small problem to verify they solve the same formulation.

We take the smallest historic planogram (BCO/CF/3.0, 92 UPCs), pick
25 products as the target set, and run both approaches.
"""
import os, sys, time, random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exploracion import load_planograms, Planogram, Shelf, ProductPlacement
from planogram_model import (
    build_product_catalog, find_best_match, adapt_planogram,
    mine_placement_rules, evaluate_rule_adherence,
)
import pulp

OUTPUT_DIR = "test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================================
# 1. Load data & mine rules
# =================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

csv_path = "datos/ejemplo_planograma.csv"
planograms = load_planograms(csv_path)
catalog = build_product_catalog(csv_path)
rules = mine_placement_rules(planograms, catalog)

print(f"  {len(planograms)} planograms loaded")
print(f"  {len(catalog)} products in catalog")
print(f"  {len(rules['level_probs'])} products with level data")
print(f"  {len(rules['adjacencies'])} adjacency pairs")


# =================================================================
# 2. Build REALISTIC target problem (20% swap, full product set)
# =================================================================
print("\n" + "=" * 70)
print("BUILDING REALISTIC TARGET PROBLEM (20% product swap)")
print("=" * 70)

# Use the smallest planogram as the template
smallest = min(planograms,
               key=lambda p: len({pr.upc for s in p.shelves.values()
                                  for pr in s.products}))
orig_upcs = {pr.upc for s in smallest.shelves.values() for pr in s.products}

# Build set of all known products (universe)
all_known_upcs = set()
for p in planograms:
    for shelf in p.shelves.values():
        for pr in shelf.products:
            all_known_upcs.add(pr.upc)

# 20% swap: remove 20% of products, add different ones
random.seed(42)
n_swap = int(len(orig_upcs) * 0.20)
to_remove = set(random.sample(sorted(orig_upcs), n_swap))
available_replacements = sorted(all_known_upcs - orig_upcs)
replacements = set(random.sample(available_replacements, min(n_swap, len(available_replacements))))
target_upcs = (orig_upcs - to_remove) | replacements

print(f"  Template: {smallest.title}")
print(f"  Original products: {len(orig_upcs)}")
print(f"  Removed: {len(to_remove)}, Added: {len(replacements)}")
print(f"  Target products (P_t): {len(target_upcs)}")
print(f"  Shelves: {len(smallest.shelves)}")
print(f"  Template tamano: {smallest.tamano}")


# =================================================================
# 3. Run the HEURISTIC (planogram_model.py)
# =================================================================
print("\n" + "=" * 70)
print("RUNNING HEURISTIC (planogram_model.py)")
print("=" * 70)

t0 = time.perf_counter()
match, overlap, total = find_best_match(
    target_upcs, smallest.tamano, planograms)
heuristic_result = adapt_planogram(match, target_upcs, catalog)
heuristic_time = time.perf_counter() - t0

heuristic_metrics = evaluate_rule_adherence(heuristic_result, rules, catalog)

placed_h = {pr.upc for s in heuristic_result.shelves.values()
             for pr in s.products}
print(f"  Template matched: {match.title}")
print(f"  Overlap: {overlap}/{total}")
print(f"  Products placed: {len(placed_h)}/{len(target_upcs)}")
print(f"  Time: {heuristic_time*1000:.1f} ms")


# =================================================================
# 4. Run the ILP (from the notebook)
# =================================================================
print("\n" + "=" * 70)
print("RUNNING ILP EXACT SOLVER (PuLP/CBC)")
print("=" * 70)

# Use same template as the heuristic
template = match
lp = rules['level_probs']
ac = rules['adjacencies']
N_H = rules['n_planograms']

# Facings from template
tfac = {p.upc: p.facings
        for s in template.shelves.values()
        for p in s.products}

S_items = list(template.shelves.items())
S_idx = list(range(len(S_items)))
S_arr = [s for _, s in S_items]

# Compute EFFECTIVE shelf heights from Y-coordinate gaps
# (the raw HEIGHT=2.5 is just shelf thickness, not clearance)
ys = sorted({s.y for s in S_arr})
eff_h = {}
for i, yv in enumerate(ys):
    if i < len(ys) - 1:
        eff_h[yv] = ys[i + 1] - yv
    else:
        eff_h[yv] = (ys[i] - ys[i - 1]) if i > 0 else 40.0

for si, shelf in enumerate(S_arr):
    shelf.shelf_height = eff_h[shelf.y]

print(f"  Effective shelf heights (Y-gaps): "
      f"{sorted(set(eff_h.values()))} cm")

# Pre-filter: only products that have catalog info
P_set = sorted(target_upcs & set(catalog.keys()))

# Further filter: only products that fit in at least one shelf
P_set = [p for p in P_set
         if any(catalog[p].height <= s.shelf_height * 1.05
                for s in S_arr)]

n_P = len(P_set)
print(f"  Feasible products for ILP: {n_P}")
print(f"  Shelves: {len(S_idx)}")

# Pairs for co-occurrence
pairs = []
for i in range(n_P):
    for k in range(i + 1, n_P):
        pairs.append((P_set[i], P_set[k]))

lambda1, lambda2 = 1.0, 0.0  # lambda2=0: skip co-occurrence for full-size ILP

prob = pulp.LpProblem("Planogram_BILP", pulp.LpMaximize)

# Variables x_ps — product p assigned to shelf s
x = {}
for pi, p in enumerate(P_set):
    for si in S_idx:
        x[p, si] = pulp.LpVariable(f"x_{pi}_{si}", cat="Binary")

# Variables y_pqs — co-location of p and q on shelf s
y = {}
if lambda2 > 0 and pairs:
    for pair_i, (p, q) in enumerate(pairs):
        for si in S_idx:
            y[p, q, si] = pulp.LpVariable(f"y_{pair_i}_{si}", cat="Binary")

# Objective: lambda1 * Phi_nivel + lambda2 * Phi_adj
phi_nivel = (1.0 / max(n_P, 1)) * pulp.lpSum(
    lp.get(p, {}).get(S_arr[si].level, 0.0) * x[p, si]
    for p in P_set for si in S_idx)

if lambda2 > 0 and pairs:
    n_pairs = len(pairs)
    phi_adj = (1.0 / max(n_pairs, 1)) * pulp.lpSum(
        (ac.get(tuple(sorted([p, q])), 0) / N_H) * y[p, q, si]
        for p, q in pairs for si in S_idx)
else:
    phi_adj = 0
    n_pairs = len(pairs)

prob += lambda1 * phi_nivel + lambda2 * phi_adj, "FO"

# Constraint (6): coverage — each product in exactly one shelf
for p in P_set:
    prob += pulp.lpSum(x[p, si] for si in S_idx) == 1, f"cov_{p}"

# Constraint (7): width capacity
for si, shelf in zip(S_idx, S_arr):
    prob += pulp.lpSum(
        catalog[p].width * tfac.get(p, 1) * x[p, si]
        for p in P_set) <= shelf.shelf_width, f"w_{si}"

# Constraint (8): height compatibility
for p in P_set:
    h_p = catalog[p].height
    for si, shelf in zip(S_idx, S_arr):
        if h_p > shelf.shelf_height * 1.05:
            prob += x[p, si] == 0, f"ht_{p}_{si}"

# Constraints (9)-(11): McCormick linearization
if lambda2 > 0 and pairs:
    for pair_i, (p, q) in enumerate(pairs):
        for si in S_idx:
            prob += y[p, q, si] <= x[p, si], f"mc1_{pair_i}_{si}"
            prob += y[p, q, si] <= x[q, si], f"mc2_{pair_i}_{si}"
            prob += y[p, q, si] >= x[p, si] + x[q, si] - 1, f"mc3_{pair_i}_{si}"

# Solve
solver = pulp.PULP_CBC_CMD(timeLimit=120, msg=0, gapRel=0.01)
t0 = time.perf_counter()
status = prob.solve(solver)
ilp_time = time.perf_counter() - t0

print(f"  Status: {pulp.LpStatus[status]}")
obj_val = pulp.value(prob.objective)
print(f"  Objective: {obj_val:.6f}" if obj_val is not None else "  Objective: None")
print(f"  Variables x_ps: {len(x)}")
print(f"  Variables y_pqs: {len(y)}")
print(f"  Constraints: {len(prob.constraints)}")
print(f"  Time: {ilp_time:.3f} s")

# Extract ILP assignment
ilp_assignment = {}  # {upc: shelf_index}
if prob.status == 1:
    for p in P_set:
        for si in S_idx:
            v = pulp.value(x[p, si])
            if v is not None and v > 0.5:
                ilp_assignment[p] = si
                break

# Build a Planogram from ILP solution for visualization
ilp_planogram = Planogram(
    segmento_id=template.segmento_id,
    mueble_id=template.mueble_id,
    planogrupo=template.planogrupo,
    tamano=template.tamano,
    direccion=template.direccion,
    conjunto_id="ILP_OPTIMAL",
)

for ch_num, shelf in template.shelves.items():
    ilp_planogram.shelves[ch_num] = Shelf(
        charola=shelf.charola, door=shelf.door, level=shelf.level,
        x=shelf.x, y=shelf.y,
        shelf_width=shelf.shelf_width, shelf_height=shelf.shelf_height,
    )

for upc, si in ilp_assignment.items():
    ch_num, shelf = S_items[si]
    info = catalog[upc]
    pos = max((p.position for p in ilp_planogram.shelves[ch_num].products),
              default=0) + 1
    ilp_planogram.shelves[ch_num].products.append(ProductPlacement(
        upc=upc, description=info.description,
        shelf=ch_num, position=pos,
        facings=tfac.get(upc, 1),
        width=info.width, height=info.height,
    ))

ilp_metrics = evaluate_rule_adherence(ilp_planogram, rules, catalog)
placed_ilp = {pr.upc for s in ilp_planogram.shelves.values()
               for pr in s.products}


# =================================================================
# 5. COMPARE RESULTS
# =================================================================
print("\n" + "=" * 70)
print("COMPARISON: HEURISTIC vs ILP")
print("=" * 70)

print(f"\n{'Metric':<28} {'Heuristic':>12} {'ILP Exact':>12} {'Delta':>10}")
print("-" * 65)

for k in ['level_rule_score', 'level_mode_accuracy',
          'adjacency_hit_rate', 'adjacency_weighted',
          'width_feasibility', 'height_level_corr',
          'n_products_placed']:
    hv = heuristic_metrics[k]
    iv = ilp_metrics[k]
    delta = iv - hv
    print(f"  {k:<26} {hv:>10.3f} {iv:>10.3f} {delta:>+10.3f}")

print(f"\n  {'Time':<26} {heuristic_time*1000:>8.1f} ms {ilp_time:>8.3f} s")

# Shelf assignment comparison
print("\n" + "-" * 65)
print("PRODUCT ASSIGNMENT COMPARISON")
print("-" * 65)

common = placed_h & placed_ilp
same_shelf = 0
for upc in sorted(common):
    # Find shelf in heuristic
    h_shelf = None
    for s in heuristic_result.shelves.values():
        if any(p.upc == upc for p in s.products):
            h_shelf = s.level
            break
    # Find shelf in ILP
    i_shelf = None
    for s in ilp_planogram.shelves.values():
        if any(p.upc == upc for p in s.products):
            i_shelf = s.level
            break
    match_str = "✓" if h_shelf == i_shelf else "✗"
    if h_shelf == i_shelf:
        same_shelf += 1
    desc = catalog[upc].description[:30] if upc in catalog else upc
    print(f"  {match_str} {desc:<32} Heuristic: level {h_shelf}  ILP: level {i_shelf}")

print(f"\n  Same level assignment: {same_shelf}/{len(common)} "
      f"({same_shelf/max(len(common),1)*100:.0f}%)")


# =================================================================
# 6. Generate comparison images
# =================================================================
print("\n" + "=" * 70)
print("GENERATING COMPARISON IMAGES")
print("=" * 70)

heuristic_result.plot_planogram(
    save_path=os.path.join(OUTPUT_DIR, "compare_heuristic.png"))
plt.close("all")

ilp_planogram.plot_planogram(
    save_path=os.path.join(OUTPUT_DIR, "compare_ilp_optimal.png"))
plt.close("all")

print("\nDone! Compare:")
print(f"  {OUTPUT_DIR}/compare_heuristic.png")
print(f"  {OUTPUT_DIR}/compare_ilp_optimal.png")

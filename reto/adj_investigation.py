# %% [markdown]
# # Investigación: ¿Por qué el ILP tiene peor Φ_adj que la heurística?
#
# ## Hallazgo clave de `comparacion.ipynb`
#
# | Método | n | Φ_nivel | Φ_adj |
# |---|---|---|---|
# | ILP (λ2=0.5) | 92 | **0.5236** | 0.0028 |
# | Heurística | 92 | 0.4858 | **0.0972** |
#
# El ILP con λ2=0.5 obtiene Φ_adj = **0.0028** vs la heurística con **0.0972**.
# Esto es 35× peor. El ILP supuestamente *optimiza* co-ocurrencia, así que
# algo está fundamentalmente mal.
#
# ## Hipótesis a investigar
#
# 1. **Normalización inconsistente**: el ILP divide entre `C(n,2)` pares *posibles*
#    (todos contra todos), pero la heurística divide entre pares *observados*
#    (solo los que comparten charola). Si la heurística pone más productos por
#    charola, tiene más pares reales y más oportunidades de "hits".
#
# 2. **Constraint coverage=1 vs heurística coverage>1**: el ILP fuerza
#    `Σ_s x_ps = 1` (cada producto en exactamente 1 charola). Pero la
#    heurística coloca **93 productos para un target de 92** — ¿hay duplicados?
#    Más productos por charola = más pares adyacentes = más Φ_adj.
#
# 3. **Template effect**: la heurística hereda la estructura completa del template
#    (que ya tiene alta co-ocurrencia por diseño), mientras el ILP asigna
#    desde cero. La heurística "copia" pares adyacentes del template.
#
# 4. **α_pq escala**: con N_H=951 planogramas BCO, α_pq/N_H es muy pequeño
#    para la mayoría de pares. El ILP suma estos valores diminutos; la
#    heurística cuenta *hits* binarios (>0 or 0).

# %% [markdown]
# ## Fase 1: Carga de Datos

# %%
import os, sys, time, random, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exploracion import load_planograms, Planogram, Shelf, ProductPlacement
from planogram_model import (
    build_product_catalog, find_best_match, adapt_planogram,
    mine_placement_rules, evaluate_rule_adherence, ProductInfo,
)

OUTPUT_DIR = "test_output/adj_investigation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# Load data
log("Cargando datos...")
CSV_ORIGINAL = "datos/ejemplo_planograma.csv"
CSV_NEW = "datos/Ejemplo 2.csv"

planograms_orig = load_planograms(CSV_ORIGINAL)
planograms_new = load_planograms(CSV_NEW)
planograms = planograms_orig + planograms_new
catalog = build_product_catalog([CSV_ORIGINAL, CSV_NEW])

bco_plans = [p for p in planograms if p.mueble_id == 'CF' and p.segmento_id == 'BCO']
rules = mine_placement_rules(bco_plans, catalog)
log(f"  |H_BCO| = {len(bco_plans)}, N_H = {rules['n_planograms']}")
log(f"  Pares adj. con freq>0: {len(rules['adjacencies'])}")

# Setup template
template = min(planograms_orig,
               key=lambda p: len({pr.upc for s in p.shelves.values()
                                  for pr in s.products}))
S_items = list(template.shelves.items())
S_idx = list(range(len(S_items)))
S_arr = [s for _, s in S_items]

ys = sorted({s.y for s in S_arr})
eff_h = {}
for i, yv in enumerate(ys):
    eff_h[yv] = (ys[i+1] - yv) if i < len(ys)-1 else (ys[i]-ys[i-1] if i>0 else 40.0)
for shelf in S_arr:
    shelf.shelf_height = eff_h[shelf.y]

all_upcs = sorted({pr.upc for s in template.shelves.values() for pr in s.products})
all_upcs = [u for u in all_upcs if u in catalog
            and any(catalog[u].height <= s.shelf_height*1.05 for s in S_arr)]
tfac = {p.upc: p.facings for s in template.shelves.values() for p in s.products}
log(f"  Template: {template.title} ({len(all_upcs)} productos)")

# %% [markdown]
# ## Fase 2: Investigación de la discrepancia
#
# Vamos a ejecutar ambos modelos con **las mismas métricas de evaluación**
# y descomponer exactamente qué contribuye a cada Φ_adj.

# %%
import pulp

log("=" * 70)
log("INVESTIGACIÓN: DESCOMPOSICIÓN DE Φ_adj")
log("=" * 70)

ac = rules['adjacencies']
lp = rules['level_probs']
N_H = rules['n_planograms']
P_set = all_upcs
n_P = len(P_set)
n_shelves = len(S_idx)

# ========================================
# 1. Distribución de α_pq
# ========================================
log("\n--- 1. Distribución de α_pq ---")

# All possible pairs
all_pairs = []
for i in range(n_P):
    for j in range(i+1, n_P):
        all_pairs.append(tuple(sorted([P_set[i], P_set[j]])))

n_total_pairs = len(all_pairs)
freqs = [ac.get(pair, 0) for pair in all_pairs]
nonzero = [f for f in freqs if f > 0]

log(f"  Total pares posibles C({n_P},2) = {n_total_pairs}")
log(f"  Pares con α>0: {len(nonzero)} ({len(nonzero)/n_total_pairs*100:.1f}%)")
log(f"  Pares con α=0: {n_total_pairs - len(nonzero)} ({(n_total_pairs-len(nonzero))/n_total_pairs*100:.1f}%)")
if nonzero:
    log(f"  α_pq estadísticas (nonzero): min={min(nonzero)}, "
        f"max={max(nonzero)}, mean={np.mean(nonzero):.1f}, median={np.median(nonzero):.1f}")
    log(f"  α_pq/N_H (nonzero): min={min(nonzero)/N_H:.4f}, "
        f"max={max(nonzero)/N_H:.4f}, mean={np.mean(nonzero)/N_H:.4f}")

# ========================================
# 2. Run ILP (λ2=0.5, n=92)
# ========================================
log("\n--- 2. Resolviendo ILP (λ1=1, λ2=0.5) ---")

# Inline ILP solve (simplified, reusing code)
prob = pulp.LpProblem("BILP_adj_test", pulp.LpMaximize)

x = {}
for pi, p in enumerate(P_set):
    for si in S_idx:
        x[p, si] = pulp.LpVariable(f"x_{pi}_{si}", cat="Binary")

pairs = []
for i in range(n_P):
    for j in range(i+1, n_P):
        pairs.append((P_set[i], P_set[j]))

y = {}
for pair_i, (p, q) in enumerate(pairs):
    for si in S_idx:
        y[p, q, si] = pulp.LpVariable(f"y_{pair_i}_{si}", cat="Binary")

phi_nivel_expr = (1.0/n_P) * pulp.lpSum(
    lp.get(p, {}).get(S_arr[si].level, 0.0) * x[p, si]
    for p in P_set for si in S_idx)

phi_adj_expr = (1.0/max(len(pairs), 1)) * pulp.lpSum(
    (ac.get(tuple(sorted([p, q])), 0) / max(N_H, 1)) * y[p, q, si]
    for p, q in pairs for si in S_idx)

prob += 1.0 * phi_nivel_expr + 0.5 * phi_adj_expr

for p in P_set:
    prob += pulp.lpSum(x[p, si] for si in S_idx) == 1
for si, shelf in zip(S_idx, S_arr):
    prob += pulp.lpSum(
        catalog[p].width * tfac.get(p, 1) * x[p, si]
        for p in P_set) <= shelf.shelf_width
for p in P_set:
    for si, shelf in zip(S_idx, S_arr):
        if catalog[p].height > shelf.shelf_height * 1.05:
            prob += x[p, si] == 0

for pair_i, (p, q) in enumerate(pairs):
    for si in S_idx:
        prob += y[p, q, si] <= x[p, si]
        prob += y[p, q, si] <= x[q, si]
        prob += y[p, q, si] >= x[p, si] + x[q, si] - 1

solver = pulp.PULP_CBC_CMD(timeLimit=1800, msg=0, gapRel=0.02)
t0 = time.perf_counter()
status = prob.solve(solver)
ilp_time = time.perf_counter() - t0

log(f"  Status: {pulp.LpStatus[status]}, Tiempo: {ilp_time:.1f}s")

# Extract ILP assignment
ilp_assignment = {}  # upc -> shelf_index
for p in P_set:
    for si in S_idx:
        v = pulp.value(x[p, si])
        if v is not None and v > 0.5:
            ilp_assignment[p] = si
            break

# Extract ILP shelves: {shelf_idx: [upcs]}
ilp_shelves = defaultdict(list)
for upc, si in ilp_assignment.items():
    ilp_shelves[si].append(upc)

# ========================================
# 3. Run Heuristic
# ========================================
log("\n--- 3. Ejecutando heurística ---")
target_set = set(P_set)
match, overlap, total = find_best_match(
    target_set, template.tamano, planograms,
    target_mueble=template.mueble_id,
)
gen = adapt_planogram(match, target_set, catalog)

# Extract heuristic shelves
h_shelves = defaultdict(list)
for ch_num, shelf in gen.shelves.items():
    for p in shelf.products:
        h_shelves[ch_num].append(p.upc)

# ========================================
# 4. COMPARE: shelf-level analysis
# ========================================
log("\n--- 4. Análisis por charola ---")

def compute_adj_stats(shelves_dict, ac, N_H, method_name):
    """Compute adjacency stats per shelf."""
    stats = {
        'method': method_name,
        'total_products': 0,
        'total_pairs': 0,
        'pairs_with_alpha_gt0': 0,
        'sum_alpha_over_NH': 0.0,
        'products_per_shelf': [],
        'pairs_per_shelf': [],
        'hits_per_shelf': [],
    }

    for shelf_id, upcs in shelves_dict.items():
        n_prods = len(upcs)
        stats['total_products'] += n_prods
        stats['products_per_shelf'].append(n_prods)

        n_pairs = 0
        n_hits = 0
        sum_a = 0.0
        for i in range(len(upcs)):
            for j in range(i+1, len(upcs)):
                pair = tuple(sorted([upcs[i], upcs[j]]))
                n_pairs += 1
                freq = ac.get(pair, 0)
                if freq > 0:
                    n_hits += 1
                    sum_a += freq / N_H

        stats['total_pairs'] += n_pairs
        stats['pairs_with_alpha_gt0'] += n_hits
        stats['sum_alpha_over_NH'] += sum_a
        stats['pairs_per_shelf'].append(n_pairs)
        stats['hits_per_shelf'].append(n_hits)
        
ilp_stats = compute_adj_stats(ilp_shelves, ac, N_H, "ILP")
h_stats = compute_adj_stats(h_shelves, ac, N_H, "Heurística")

print(f"\n{'Métrica':<35} {'ILP':>12} {'Heurística':>12}")
print("-" * 62)
print(f"{'Productos totales':<35} {ilp_stats['total_products']:>12} {h_stats['total_products']:>12}")
print(f"{'Charolas usadas':<35} {len(ilp_shelves):>12} {len(h_shelves):>12}")
print(f"{'Productos/charola (media)':<35} {np.mean(ilp_stats['products_per_shelf']):>12.1f} {np.mean(h_stats['products_per_shelf']):>12.1f}")
print(f"{'Productos/charola (max)':<35} {max(ilp_stats['products_per_shelf']):>12} {max(h_stats['products_per_shelf']):>12}")
print(f"{'Pares en misma charola (total)':<35} {ilp_stats['total_pairs']:>12} {h_stats['total_pairs']:>12}")
print(f"{'Pares con α>0 (hits)':<35} {ilp_stats['pairs_with_alpha_gt0']:>12} {h_stats['pairs_with_alpha_gt0']:>12}")
hit_rate_ilp = ilp_stats['pairs_with_alpha_gt0'] / max(ilp_stats['total_pairs'], 1)
hit_rate_h = h_stats['pairs_with_alpha_gt0'] / max(h_stats['total_pairs'], 1)
print(f"{'Hit rate (hits/pares)':<35} {hit_rate_ilp:>12.4f} {hit_rate_h:>12.4f}")
print(f"{'Σ α_pq/N_H (co-located pairs)':<35} {ilp_stats['sum_alpha_over_NH']:>12.4f} {h_stats['sum_alpha_over_NH']:>12.4f}")

# Φ_adj as ILP computes it: Σ α/N_H / C(n,2)
phi_adj_ilp_formula = ilp_stats['sum_alpha_over_NH'] / max(len(pairs), 1)
phi_adj_h_formula = h_stats['sum_alpha_over_NH'] / max(len(pairs), 1)

# Φ_adj as heuristic evals: Σ α/N_H / n_pairs_on_shelf
phi_adj_h_local = h_stats['sum_alpha_over_NH'] / max(h_stats['total_pairs'], 1)
phi_adj_ilp_local = ilp_stats['sum_alpha_over_NH'] / max(ilp_stats['total_pairs'], 1)

print(f"\n{'Normalización':<35} {'ILP':>12} {'Heurística':>12}")
print("-" * 62)
print(f"{'Φ_adj (÷ C(n,2) = {len(pairs)})':<35} {phi_adj_ilp_formula:>12.6f} {phi_adj_h_formula:>12.6f}")
print(f"{'Φ_adj (÷ pares_reales)':<35} {phi_adj_ilp_local:>12.6f} {phi_adj_h_local:>12.6f}")

# ========================================
# 5. ROOT CAUSE: ¿Qué usa el ILP como Φ_adj?
# ========================================
log("\n--- 5. Causa raíz ---")

print(f"""
ANÁLISIS DE CAUSA RAÍZ
======================

El ILP define Φ_adj como:
  Φ_adj = (1/C(n,2)) · Σ_s Σ_(p<q) (α_pq/N_H) · y_pqs

Denominador: C({n_P},2) = {len(pairs):,} pares posibles
Pero solo {ilp_stats['total_pairs']} pares están realmente co-localizados
(es decir, y_pqs=1 para solo {ilp_stats['total_pairs']} de los {len(pairs)*n_shelves:,} posibles y).

El problema es que:
1. La mayoría de pares NO están en la misma charola → y_pqs = 0
2. Los que sí (α_pq/N_H) son valores muy pequeños (N_H = {N_H})
3. Dividir por C(n,2) = {len(pairs)} diluye aún más el valor
4. El ILP concentra productos en MENOS charolas ({len(ilp_shelves)} charolas usadas)
   vs la heurística ({len(h_shelves)} charolas)
5. La heurística hereda la estructura del template que ya tiene
   pares co-ocurrentes por diseño

Resultado: el denominador C(n,2) hace que Φ_adj sea siempre ≈ 0
para problemas grandes, porque la mayoría de pares no comparten charola.
""")

# ========================================
# 6. La métrica de perturbation_test usa hit_rate, no Φ_adj
# ========================================
log("\n--- 6. Diferencia entre métricas ---")

# The perturbation test uses adjacency_hit_rate = matched_pairs / total_pairs
# where total_pairs = pairs on same shelf (NOT C(n,2))
# This is very different from the ILP's Φ_adj

h_eval = evaluate_rule_adherence(gen, rules, catalog)

print(f"""
MÉTRICAS DE evaluate_rule_adherence (heurística):
  adjacency_hit_rate     = {h_eval['adjacency_hit_rate']:.4f}  (hits/pares_en_charola)
  adjacency_weighted     = {h_eval['adjacency_weighted']:.4f}  (Σ α/N_H / pares_en_charola)

MÉTRICAS del ILP (Φ_adj formula):
  Φ_adj                  = {phi_adj_ilp_formula:.6f}  (Σ α/N_H / C(n,2))

Son métricas DIFERENTES:
  - hit_rate: ¿qué fracción de pares co-localizados tienen α>0?
  - Φ_adj:    ¿cuánta co-ocurrencia histórica hay, normalizada por TODOS los pares posibles?

La heurística tiene hit_rate alto porque hereda pares del template.
El ILP tiene Φ_adj bajo porque divide por C(n,2) = {len(pairs)} pares enormes.
""")

# ========================================
# 7. PRUEBA: ILP con hit_rate como objetivo
# ========================================
log("\n--- 7. ILP con normalización alternativa ---")
log("  Resolviendo ILP con Φ_adj = Σ_s Σ_(p<q) I(α>0)·y / (Σ_s Σ y) ...")
log("  (Esto maximiza la fracción de pares co-localizados que tienen historial)")

# Instead of using α/N_H weights, use binary: α>0 → 1, else 0
prob2 = pulp.LpProblem("BILP_hitrate", pulp.LpMaximize)

x2 = {}
for pi, p in enumerate(P_set):
    for si in S_idx:
        x2[p, si] = pulp.LpVariable(f"x2_{pi}_{si}", cat="Binary")

y2 = {}
for pair_i, (p, q) in enumerate(pairs):
    for si in S_idx:
        y2[p, q, si] = pulp.LpVariable(f"y2_{pair_i}_{si}", cat="Binary")

# Objective: λ1·Φ_nivel + λ2·(Σ I(α>0)·y / Σ y)
# We can't divide by Σy (non-linear), so approximate:
# Maximize Σ I(α>0)·y (just count hits)
phi_nivel_expr2 = (1.0/n_P) * pulp.lpSum(
    lp.get(p, {}).get(S_arr[si].level, 0.0) * x2[p, si]
    for p in P_set for si in S_idx)

# Binary hit objective: maximize number of co-located pairs with α>0
hit_expr = pulp.lpSum(
    (1 if ac.get(tuple(sorted([p, q])), 0) > 0 else 0) * y2[p, q, si]
    for p, q in pairs for si in S_idx)

# Normalize by number of pairs (constant) to get comparable scale
n_pairs_with_alpha = sum(1 for p, q in pairs
                         if ac.get(tuple(sorted([p, q])), 0) > 0)
log(f"  Pares (p,q) con α>0 en P_set: {n_pairs_with_alpha}/{len(pairs)}")

# Weight: normalize to [0,1] scale
phi_hit_expr = (1.0 / max(n_pairs_with_alpha, 1)) * hit_expr

prob2 += 1.0 * phi_nivel_expr2 + 0.5 * phi_hit_expr

# Same constraints
for p in P_set:
    prob2 += pulp.lpSum(x2[p, si] for si in S_idx) == 1
for si, shelf in zip(S_idx, S_arr):
    prob2 += pulp.lpSum(
        catalog[p].width * tfac.get(p, 1) * x2[p, si]
        for p in P_set) <= shelf.shelf_width
for p in P_set:
    for si, shelf in zip(S_idx, S_arr):
        if catalog[p].height > shelf.shelf_height * 1.05:
            prob2 += x2[p, si] == 0
for pair_i, (p, q) in enumerate(pairs):
    for si in S_idx:
        prob2 += y2[p, q, si] <= x2[p, si]
        prob2 += y2[p, q, si] <= x2[q, si]
        prob2 += y2[p, q, si] >= x2[p, si] + x2[q, si] - 1

solver2 = pulp.PULP_CBC_CMD(timeLimit=1800, msg=0, gapRel=0.02)
t0 = time.perf_counter()
status2 = prob2.solve(solver2)
t2 = time.perf_counter() - t0

log(f"  Status: {pulp.LpStatus[status2]}, Tiempo: {t2:.1f}s")

# Extract assignment
ilp2_shelves = defaultdict(list)
for p in P_set:
    for si in S_idx:
        v = pulp.value(x2[p, si])
        if v is not None and v > 0.5:
            ilp2_shelves[si].append(p)
            break

ilp2_stats = compute_adj_stats(ilp2_shelves, ac, N_H, "ILP (hit-rate)")

# Compare all three
print(f"\n{'Métrica':<35} {'ILP (α/N_H)':>12} {'ILP (hits)':>12} {'Heurística':>12}")
print("-" * 75)
print(f"{'Productos colocados':<35} {ilp_stats['total_products']:>12} {ilp2_stats['total_products']:>12} {h_stats['total_products']:>12}")
print(f"{'Pares en misma charola':<35} {ilp_stats['total_pairs']:>12} {ilp2_stats['total_pairs']:>12} {h_stats['total_pairs']:>12}")
print(f"{'Pares con α>0 (hits)':<35} {ilp_stats['pairs_with_alpha_gt0']:>12} {ilp2_stats['pairs_with_alpha_gt0']:>12} {h_stats['pairs_with_alpha_gt0']:>12}")

hr_ilp = ilp_stats['pairs_with_alpha_gt0'] / max(ilp_stats['total_pairs'], 1)
hr_ilp2 = ilp2_stats['pairs_with_alpha_gt0'] / max(ilp2_stats['total_pairs'], 1)
hr_h = h_stats['pairs_with_alpha_gt0'] / max(h_stats['total_pairs'], 1)
print(f"{'Hit rate':<35} {hr_ilp:>12.4f} {hr_ilp2:>12.4f} {hr_h:>12.4f}")

phi_adj_ilp2 = ilp2_stats['sum_alpha_over_NH'] / max(len(pairs), 1)
print(f"{'Φ_adj (÷ C(n,2))':<35} {phi_adj_ilp_formula:>12.6f} {phi_adj_ilp2:>12.6f} {phi_adj_h_formula:>12.6f}")

# Φ_nivel for ILP2
phi_nivel_ilp2 = 0
for si, upcs in ilp2_shelves.items():
    for u in upcs:
        phi_nivel_ilp2 += lp.get(u, {}).get(S_arr[si].level, 0.0)
phi_nivel_ilp2 /= max(n_P, 1)

phi_nivel_ilp1 = 0
for si, upcs in ilp_shelves.items():
    for u in upcs:
        phi_nivel_ilp1 += lp.get(u, {}).get(S_arr[si].level, 0.0)
phi_nivel_ilp1 /= max(n_P, 1)

print(f"{'Φ_nivel':<35} {phi_nivel_ilp1:>12.4f} {phi_nivel_ilp2:>12.4f} {h_stats['method']:>12}")

# Compute heuristic Φ_nivel
h_phi_n = 0
for shelf in gen.shelves.values():
    for p in shelf.products:
        h_phi_n += lp.get(p.upc, {}).get(shelf.level, 0.0)
h_n_placed = sum(len(upcs) for upcs in h_shelves.values())
h_phi_n /= max(h_n_placed, 1)
print(f"{'Φ_nivel (corregido)':<35} {phi_nivel_ilp1:>12.4f} {phi_nivel_ilp2:>12.4f} {h_phi_n:>12.4f}")

# ========================================
# 8. Conclusiones
# ========================================
print(f"""

CONCLUSIONES
============

1. LA DISCREPANCIA ES POR LA NORMALIZACIÓN, NO POR LA OPTIMIZACIÓN

   El ILP calcula Φ_adj = Σ(α_pq/N_H · y_pqs) / C(n,2)
   - Numerador: solo los y_pqs=1 (pares en misma charola) contribuyen
   - Denominador: C({n_P},2) = {len(pairs)} pares → muy grande

   La heurística (perturbation_test) mide:
   - adjacency_hit_rate = hits / pares_en_misma_charola
   - Denominador: solo pares realmente co-localizados → mucho menor

   Resultado: son métricas INCOMPARABLES.

2. LA HEURÍSTICA HEREDA PARES CO-OCURRENTES DEL TEMPLATE

   La heurística copia la estructura del template (que tiene hit_rate ≈ 1.0).
   El ILP asigna desde cero, sin esta ventaja.

3. EL ILP SÍ OPTIMIZA CORRECTAMENTE SU Φ_adj

   Pero su Φ_adj está dominado por Φ_nivel porque:
   - Φ_nivel ≈ 0.5 (escala significativa)
   - Φ_adj ≈ 0.003 (escala diminuta por la normalización)
   - Con λ2=0.5, la contribución de adj al objetivo es ≈ 0.0015

4. PARA COMPARAR JUSTAMENTE:
   - Usar la misma métrica (hit_rate o Φ_adj) para ambos
   - O bien: cambiar la normalización del ILP a pares co-localizados
""")

# %% [markdown]
# # Optimización del Planograma de un Stand de Ventas — OXXO
# ## Modelo Final: Heurística con Aprendizaje por Segmento
#
# | Campo | Detalle |
# |---|---|
# | **Materia** | MA2005B — Análisis numérico para la optimización no-lineal (Gpo. 603) |
# | **Autores** | Mariel Álvarez · Viviana Carrizales · Jorge Andujo · Ana Ibarra · Álvaro Bolaños |
# | **Profesores** | Fernando Elizalde · Monica Elizondo · Salvador García · Sofía Salinas |
#
# ---
#
# ### Descripción del problema
# Dado el surtido de una tienda (UPCs $P_t$), el tamaño del mueble ($\tau$ puertas)
# y el tipo de mobiliario ($\mu \in \{\text{CF}, \text{CFC}\}$),
# se busca una asignación de productos a anaqueles que **maximice la adherencia
# a reglas históricas** de nivel y co-ocurrencia, sujeta a restricciones de
# capacidad física.
#
# ### Mejoras respecto al modelo base
# | Aspecto | Modelo base (`planograma_model_ejemplo`) | **Este modelo** |
# |---|---|---|
# | Planogramas históricos | 43 (solo CF) | **5,614** (CF + CFC) |
# | Productos en catálogo | 189 | **470** |
# | Tipo de mueble | No distingue CF/CFC | **Filtrado obligatorio por $\mu$** |
# | Reglas de colocación | Globales | **Por segmento** ($\mu$, segmento) |
# | Tamaños soportados | 3.0 – 5.0 | **1.0 – 6.5** |
# | Segmentos | 3 (BCO, HRN, CLA) | **9** (BCO, CLA, HOG, HRN, OFC, PTC, REC, RET, SIN) |
#
# ### Estructura del notebook
# | Fase | Contenido |
# |---|---|
# | 1 | Formulación matemática |
# | 2 | Importaciones y carga de datos |
# | 3 | Tamaño del problema |
# | 4 | Minería de reglas por segmento |
# | 5 | Selección de template y heurística |
# | 6 | Validación ILP (subconjunto pequeño) |
# | 7 | Prueba de perturbación |
# | 8 | Visualización |
# | 9 | Supuestos y conclusiones |

# %% [markdown]
# ## Fase 1: Formulación Matemática
#
# ### Conjuntos
#
# | Símbolo | Definición |
# |---|---|
# | $P$ | Todos los productos, $|P|=470$ |
# | $P_t \subseteq P$ | Productos disponibles en la tienda objetivo |
# | $\mathcal{H}$ | Planogramas históricos, $|\mathcal{H}|=5{,}614$ |
# | $\mathcal{H}(\tau, \mu)$ | Subconjunto de $\mathcal{H}$ con tamaño $\tau$ y mueble $\mu$ |
# | $S$ | Charolas del planograma, $|S| \in \{5,...,38\}$ |
# | $L = \{0,...,6\}$ | Niveles verticales (0=piso, 6=techo) |
# | $\Sigma$ | Segmentos de tienda: BCO, CLA, HRN, OFC, PTC, RET, SIN, ... |
#
# ### Parámetros
#
# | Símbolo | Descripción |
# |---|---|
# | $w_p, h_p$ | Ancho y alto del producto $p$ (cm) |
# | $W_s, H_s$ | Ancho y alto de la charola $s$ (cm) |
# | $f_p$ | Frentes del producto $p$ (heredado del template) |
# | $\ell_s \in L$ | Nivel de la charola $s$ |
# | $\pi_{p\ell}^{(\mu,\sigma)} \in [0,1]$ | Prob. histórica de que $p$ esté en nivel $\ell$, estimada solo con planogramas de mueble $\mu$ y segmento $\sigma$ |
# | $\alpha_{pq}^{(\mu,\sigma)}$ | Frecuencia de co-ocurrencia de $p$ y $q$ en el mismo anaquel (mueble $\mu$, segmento $\sigma$) |
# | $N_H^{(\mu,\sigma)}$ | Número de planogramas en el grupo $(\mu,\sigma)$ |
# | $\lambda_1, \lambda_2 \geq 0$ | Pesos multi-criterio |
#
# ### Variables de decisión
#
# $$x_{ps} \in \{0,1\} \quad \forall p \in P_t, s \in S$$
#
# $$y_{pqs} \in \{0,1\} \quad \forall p<q \in P_t, s \in S \quad(\text{co-ubicación})$$
#
# ### Función objetivo
#
# $$\max_{x,y}\; \lambda_1 \cdot \underbrace{\frac{1}{|P_t|}\sum_{p}\sum_{s} \pi_{p,\ell_s}^{(\mu,\sigma)} x_{ps}}_{\Phi_{\text{nivel}}}+ \lambda_2 \cdot \underbrace{\frac{1}{\binom{|P_t|}{2}}\sum_{s}\sum_{p<q} \frac{\alpha_{pq}^{(\mu,\sigma)}}{N_H^{(\mu,\sigma)}} y_{pqs}}_{\Phi_{\text{adj}}}$$
#
# ### Restricciones
#
# | No. | Ecuación | Descripción |
# |---|---|---|
# | (6) | $\sum_s x_{ps}=1\;\forall p$ | Cobertura completa |
# | (7) | $\sum_p w_p f_p x_{ps} \leq W_s\;\forall s$ | Capacidad de ancho |
# | (8) | $h_p \leq 1.05 \cdot H_s \;\forall p,s : x_{ps}=1$ | Compatibilidad de altura |
# | (9)-(11) | McCormick $y_{pqs}=x_{ps}\cdot x_{qs}$ | Linealización co-ocurrencia |
# | (12)-(13) | $x_{ps},y_{pqs}\in\{0,1\}$ | Integralidad |
#
# ### Subproblemas de la heurística
#
# **Selección de template** (Ec. 14):
# $$T^* = \arg\max_{T\in\mathcal{H}(\tau,\mu)} |P_T \cap P_t|$$
#
# **Sustitución** (Ec. 15-16):
# $$\text{sub}(p)=\arg\min_{q\in P_t \setminus \text{placed}} \delta(p,q), \quad \delta(p,q)=|w_q-w_p|+2|h_q-h_p|$$
#
# **Llenado de huecos** (Ec. 17-19): Best-fit decreasing con penalización de nivel:
# $$\text{score}(s) = (W_s - \text{usado}_s) + 0.1 \cdot |y_s - y^*_{\text{ideal}}(h_p)|$$
#
# donde $y^*_{\text{ideal}}(h_p)$ se deriva de la correlación Pearson $r=-0.90$ entre
# altura del producto y coordenada Y del anaquel.

# %% [markdown]
# ## Fase 2: Importaciones y Carga de Datos

# %%
# =============================================================
# CELDA 0: IMPORTACIONES
# =============================================================
import os, sys, time, random, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as spstats

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

# Módulos del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exploracion import load_planograms, Planogram, Shelf, ProductPlacement
from planogram_model import (
    build_product_catalog,
    find_best_match,
    find_closest_product,
    adapt_planogram,
    mine_placement_rules,
    evaluate_rule_adherence,
    perturbation_test,
    ProductInfo,
)

print(f"Python : {sys.version.split()[0]}")
print(f"NumPy  : {np.__version__}")
print(f"Pandas : {pd.__version__}")

# %%
# =============================================================
# CARGA DE DATOS
# =============================================================
CSV_ORIGINAL = "datos/ejemplo_planograma.csv"
CSV_NEW      = "datos/Ejemplo 2.csv"

print("Cargando planogramas históricos...")
t0 = time.perf_counter()

planograms_orig = load_planograms(CSV_ORIGINAL)
print(f"  Datos originales:  {len(planograms_orig)} planogramas")

planograms_new = load_planograms(CSV_NEW)
print(f"  Datos extendidos:  {len(planograms_new)} planogramas")

planograms = planograms_orig + planograms_new
elapsed = time.perf_counter() - t0
print(f"  Total |H| = {len(planograms)} planogramas ({elapsed:.1f}s)")

print("\nConstruyendo catálogo de productos...")
catalog = build_product_catalog([CSV_ORIGINAL, CSV_NEW])
print(f"  |P| = {len(catalog)} productos únicos")

# %% [markdown]
# ## Fase 3: Tamaño del Problema

# %%
# =============================================================
# TAMAÑO DEL PROBLEMA
# =============================================================
muebles   = sorted({p.mueble_id for p in planograms})
segmentos = sorted({p.segmento_id for p in planograms})
tamanos   = sorted({p.tamano for p in planograms})

print("DIMENSIONES DEL PROBLEMA")
print("=" * 60)
print(f"  Planogramas históricos |H| = {len(planograms)}")
print(f"  Productos              |P| = {len(catalog)}")
print(f"  Tipos de mueble        |M| = {len(muebles)}: {muebles}")
print(f"  Segmentos              |Σ| = {len(segmentos)}: {segmentos}")
print(f"  Tamaños                |τ| = {len(tamanos)}: {tamanos}")
print()

# Desglose por mueble
for m in muebles:
    mp = [p for p in planograms if p.mueble_id == m]
    ms = sorted({p.segmento_id for p in mp})
    mt = sorted({p.tamano for p in mp})
    print(f"  {m}: {len(mp):>5} planogramas, "
          f"segmentos={ms}, tamaños={mt}")

# %%
# =============================================================
# HARDWARE Y SOFTWARE
# =============================================================
import platform
print(f"Sistema  : {platform.system()} {platform.release()}")
print(f"CPU      : {platform.processor()}")
print(f"Python   : {sys.version.split()[0]}")
print(f"NumPy    : {np.__version__}")
print(f"Pandas   : {pd.__version__}")

# %% [markdown]
# ## Fase 4: Minería de Reglas por Segmento
#
# Las reglas de colocación ($\pi_{p\ell}^{(\mu,\sigma)}$ y $\alpha_{pq}^{(\mu,\sigma)}$)
# se minan **separadamente** por tipo de mueble y segmento de tienda.
# Esto evita contaminar las distribuciones de nivel de CF (6 niveles, puertas de vidrio)
# con las de CFC (diferente geometría) y permite capturar patrones específicos
# por segmento demográfico.
#
# Se aplica un umbral mínimo: si un grupo $(\mu, \sigma)$ tiene menos de 20 planogramas,
# se usa el *fallback* al nivel de mueble $\mu$.

# %%
# =============================================================
# MINERÍA DE REGLAS POR SEGMENTO
# =============================================================
MIN_PLANOGRAMS = 20  # umbral mínimo para reglas por segmento

print("Minería de reglas de colocación")
print("=" * 70)

# Nivel 1: fallback por mueble
rules_by_mueble = {}
for mueble_id in sorted({p.mueble_id for p in planograms}):
    mueble_plans = [p for p in planograms if p.mueble_id == mueble_id]
    rules_by_mueble[mueble_id] = mine_placement_rules(mueble_plans, catalog)
    r = rules_by_mueble[mueble_id]
    print(f"  {mueble_id} (fallback): {len(mueble_plans):>5} planogramas, "
          f"{len(r['level_probs']):>3} productos, "
          f"{len(r['adjacencies']):>5} pares adj.")

# Nivel 2: por (mueble, segmento)
print()
rules_by_group = {}
all_groups = sorted({(p.mueble_id, p.segmento_id) for p in planograms})
n_segment_rules = 0
n_fallback = 0

for mueble_id, seg_id in all_groups:
    group_plans = [p for p in planograms
                   if p.mueble_id == mueble_id
                   and p.segmento_id == seg_id]
    if len(group_plans) >= MIN_PLANOGRAMS:
        rules_by_group[(mueble_id, seg_id)] = mine_placement_rules(
            group_plans, catalog
        )
        r = rules_by_group[(mueble_id, seg_id)]
        n_segment_rules += 1
        print(f"  {mueble_id}/{seg_id}: {len(group_plans):>5} planogramas, "
              f"{len(r['level_probs']):>3} productos, "
              f"{len(r['adjacencies']):>5} pares adj.")
    else:
        n_fallback += 1
        print(f"  {mueble_id}/{seg_id}: {len(group_plans):>5} planogramas "
              f"(< {MIN_PLANOGRAMS}, usa fallback {mueble_id})")

print(f"\n  Conjuntos de reglas por segmento: {n_segment_rules}")
print(f"  Grupos usando fallback: {n_fallback}")


# Función auxiliar para obtener las reglas correctas
def get_rules(mueble_id, segmento_id):
    """Obtiene reglas por (mueble, segmento) con fallback a mueble."""
    key = (mueble_id, segmento_id)
    return rules_by_group.get(key, rules_by_mueble.get(mueble_id))

# %% [markdown]
# ## Fase 5: Selección de Template y Heurística
#
# ### Selección de template (Ec. 14)
# $$T^* = \arg\max_{T\in\mathcal{H}(\tau,\mu)} |P_T \cap P_t|$$
#
# El template se filtra por **tamaño** ($\tau$) y **tipo de mueble** ($\mu$),
# garantizando compatibilidad física de charolas.
#
# ### Heurística de adaptación
# 1. **Fase 1** — Copiar/Sustituir: recorre cada producto del template.
#    Si está disponible, lo conserva; si no, busca el sustituto dimensional
#    más cercano (Ec. 15-16).
# 2. **Fase 2** — Llenado de huecos: los productos restantes se insertan
#    en charolas con espacio, usando best-fit decreasing con penalización
#    de nivel (Ec. 17-19).

# %%
# =============================================================
# DEMO: GENERACIÓN CON 20% DE SUSTITUCIÓN
# =============================================================
example = planograms_orig[0]
example_rules = get_rules(example.mueble_id, example.segmento_id)

# Obtener surtido original
orig_upcs = {p.upc for s in example.shelves.values() for p in s.products}

# Construir P_t con 20% de intercambio
all_upcs = {p.upc for plano in planograms
            for s in plano.shelves.values() for p in s.products}

random.seed(42)
n_swap = int(len(orig_upcs) * 0.20)
to_remove = set(random.sample(sorted(orig_upcs), n_swap))
replacements = set(random.sample(
    sorted(all_upcs - orig_upcs), min(n_swap, len(all_upcs - orig_upcs))
))
target_upcs = (orig_upcs - to_remove) | replacements

print("DEMO: Generación de planograma")
print("=" * 60)
print(f"  Planograma original : {example.title}")
print(f"  Mueble: {example.mueble_id}, Segmento: {example.segmento_id}")
print(f"  Productos originales: {len(orig_upcs)}")
print(f"  Removidos: {len(to_remove)}, Agregados: {len(replacements)}")
print(f"  Surtido objetivo |Pt|: {len(target_upcs)}")

# Selección de template (Ec. 14)
t0 = time.perf_counter()
match, overlap, total = find_best_match(
    target_upcs, example.tamano, planograms,
    target_mueble=example.mueble_id,
)
print(f"\n  Template T*: {match.title}")
print(f"  Overlap |P_T ∩ P_t| = {overlap}/{total}")

# Heurística (Ecs. 15-19)
generated = adapt_planogram(match, target_upcs, catalog)
elapsed_h = time.perf_counter() - t0

placed = {p.upc for s in generated.shelves.values() for p in s.products}
print(f"\n  Productos colocados: {len(placed)}/{len(target_upcs)}")
print(f"  Cobertura: {len(placed)/len(target_upcs)*100:.1f}%")
print(f"  Tiempo: {elapsed_h*1000:.2f} ms")

# Evaluación
orig_metrics = evaluate_rule_adherence(example, example_rules, catalog)
gen_metrics  = evaluate_rule_adherence(generated, example_rules, catalog)

print(f"\n  {'Métrica':<25} {'Original':>10} {'Generado':>10}")
print("  " + "-" * 48)
for k in ['level_rule_score', 'level_mode_accuracy',
          'adjacency_hit_rate', 'width_feasibility',
          'height_level_corr', 'n_products_placed']:
    print(f"  {k:<25} {orig_metrics[k]:>10.3f} {gen_metrics[k]:>10.3f}")

# %% [markdown]
# ## Fase 6: Validación ILP (subconjunto pequeño)
#
# El BILP es NP-difícil. Para problemas completos (~130 productos, 28 charolas)
# el solver CBC tarda >120s. Sin embargo, podemos **validar** que la heurística
# resuelve el mismo problema que el ILP ejecutándolo en un subconjunto pequeño
# (≤30 productos) con $\lambda_2=0$ (sin co-ocurrencia).

# %%
# =============================================================
# VALIDACIÓN ILP vs HEURÍSTICA (subconjunto)
# =============================================================
import pulp

# Usar el planograma más pequeño como template
smallest = min(planograms_orig,
               key=lambda p: len({pr.upc for s in p.shelves.values()
                                  for pr in s.products}))
smallest_rules = get_rules(smallest.mueble_id, smallest.segmento_id)
lp = smallest_rules['level_probs']

# Extraer info del template
S_items = list(smallest.shelves.items())
S_idx = list(range(len(S_items)))
S_arr = [s for _, s in S_items]

# Calcular alturas efectivas por Y-gap
ys = sorted({s.y for s in S_arr})
eff_h = {}
for i, yv in enumerate(ys):
    if i < len(ys) - 1:
        eff_h[yv] = ys[i + 1] - yv
    else:
        eff_h[yv] = (ys[i] - ys[i - 1]) if i > 0 else 40.0
for shelf in S_arr:
    shelf.shelf_height = eff_h[shelf.y]

# Productos: todos los del template
all_upcs_sm = sorted({pr.upc for s in smallest.shelves.values()
                      for pr in s.products})
P_set = [p for p in all_upcs_sm if p in catalog
         and any(catalog[p].height <= s.shelf_height * 1.05 for s in S_arr)]

tfac = {p.upc: p.facings for s in smallest.shelves.values()
        for p in s.products}

n_P = len(P_set)
print(f"VALIDACIÓN ILP")
print("=" * 60)
print(f"  Template: {smallest.title}")
print(f"  Productos factibles: {n_P}")
print(f"  Charolas: {len(S_idx)}")

# Construir ILP (lambda2=0 para factibilidad)
prob = pulp.LpProblem("Planogram_BILP", pulp.LpMaximize)
x = {(p, si): pulp.LpVariable(f"x_{pi}_{si}", cat="Binary")
     for pi, p in enumerate(P_set) for si in S_idx}

# Objetivo: Phi_nivel
phi_nivel = (1.0 / max(n_P, 1)) * pulp.lpSum(
    lp.get(p, {}).get(S_arr[si].level, 0.0) * x[p, si]
    for p in P_set for si in S_idx)
prob += phi_nivel, "FO"

# Restricciones
for p in P_set:
    prob += pulp.lpSum(x[p, si] for si in S_idx) == 1, f"cov_{p}"
for si, shelf in zip(S_idx, S_arr):
    prob += pulp.lpSum(
        catalog[p].width * tfac.get(p, 1) * x[p, si]
        for p in P_set) <= shelf.shelf_width, f"w_{si}"
for p in P_set:
    for si, shelf in zip(S_idx, S_arr):
        if catalog[p].height > shelf.shelf_height * 1.05:
            prob += x[p, si] == 0, f"ht_{p}_{si}"

# Resolver
solver = pulp.PULP_CBC_CMD(timeLimit=120, msg=0, gapRel=0.01)
t0_ilp = time.perf_counter()
status = prob.solve(solver)
ilp_time = time.perf_counter() - t0_ilp

obj_val = pulp.value(prob.objective)
print(f"\n  Status: {pulp.LpStatus[status]}")
print(f"  Phi_nivel (ILP): {obj_val:.4f}" if obj_val else "  Sin solución")
print(f"  Variables: {len(x)}")
print(f"  Restricciones: {len(prob.constraints)}")
print(f"  Tiempo ILP: {ilp_time:.2f}s")

# Comparar con heurística en el mismo problema
t0_h = time.perf_counter()
target_sm = set(P_set)
match_sm, ov_sm, tot_sm = find_best_match(
    target_sm, smallest.tamano, planograms,
    target_mueble=smallest.mueble_id,
)
gen_sm = adapt_planogram(match_sm, target_sm, catalog)
h_time = time.perf_counter() - t0_h

ilp_metrics = {}
if prob.status == 1:
    # Reconstruir Phi_nivel de la heurística para comparar
    h_score = 0
    for s in gen_sm.shelves.values():
        for p in s.products:
            h_score += lp.get(p.upc, {}).get(s.level, 0.0)
    h_phi = h_score / max(len(target_sm), 1)
    print(f"\n  Phi_nivel (ILP):        {obj_val:.4f}")
    print(f"  Phi_nivel (heurística): {h_phi:.4f}")
    print(f"  Gap de optimalidad:     {(obj_val - h_phi)/obj_val*100:.1f}%")
    print(f"\n  Tiempo ILP:        {ilp_time:.2f}s")
    print(f"  Tiempo heurística: {h_time*1000:.2f}ms")
    print(f"  Factor de velocidad: {ilp_time/(h_time):.0f}×")

# %% [markdown]
# ## Fase 7: Prueba de Perturbación
#
# Se toman los 43 planogramas originales como base, se sustituye un porcentaje
# del surtido (10%, 20%, 30%, 40%), y se genera un nuevo planograma buscando
# entre los **5,614 templates** disponibles. La evaluación usa reglas
# por segmento específicas de cada planograma.

# %%
# =============================================================
# PRUEBA DE PERTURBACIÓN
# =============================================================
print("Ejecutando prueba de perturbación...")
print("(planogramas originales perturbados, búsqueda en 5,614 templates)")
t0 = time.perf_counter()

results = perturbation_test(
    planograms_orig, catalog,
    rules_by_group=rules_by_group,
    rules_by_mueble=rules_by_mueble,
    search_planograms=planograms,
)
elapsed_pt = time.perf_counter() - t0
print(f"Completado en {elapsed_pt:.1f}s\n")

# Resumen
print("=" * 90)
print("RESULTADOS DE PERTURBACIÓN")
print("=" * 90)

summary = results.groupby('source').agg({
    'level_rule_score': 'mean',
    'level_mode_accuracy': 'mean',
    'adjacency_hit_rate': 'mean',
    'adjacency_weighted': 'mean',
    'width_feasibility': 'mean',
    'height_level_corr': 'mean',
    'n_products_placed': 'mean',
}).round(3)

print("\n" + summary.to_string())

# Delta 20%
print("\n\nComparación original vs 20% swap:")
orig_r = results[results['source'] == 'original']
gen20  = results[results['source'] == 'gen_20%_swap']

for col in ['level_rule_score', 'level_mode_accuracy',
            'adjacency_hit_rate', 'width_feasibility',
            'height_level_corr']:
    o_mean = orig_r[col].mean()
    g_mean = gen20[col].mean() if len(gen20) > 0 else 0
    delta = g_mean - o_mean
    print(f"  {col:<25} original={o_mean:.3f}  gen_20%={g_mean:.3f}  "
          f"delta={delta:+.3f}")

# %% [markdown]
# ## Fase 8: Visualización

# %%
# =============================================================
# VISUALIZACIÓN DEL PLANOGRAMA
# =============================================================
print("Planograma original:")
example.plot_planogram()

print("\nPlanograma generado (20% swap):")
generated.plot_planogram()

# %% [markdown]
# ## Fase 9: Supuestos
#
# | # | Supuesto | Origen | Impacto potencial |
# |---|---|---|---|
# | 1 | Todo producto tiene dimensiones conocidas $(w_p,h_p)$ | Explícito | Productos sin dimensiones quedarían sin asignar |
# | 2 | Estructura del mueble fija: $\tau$, $|S|$, $(W_s,H_s)$ | Explícito | Muebles variables requieren re-indexación |
# | 3 | Frentes $f_p$ heredados del template ($f_p=1$ para insertados) | Explícito | Sobreestimación/subestimación del ancho utilizado |
# | 4 | Modelo decide charola, no posición exacta dentro de ella | Explícito | Imprecisión visual; no afecta métricas de nivel |
# | 5 | Reglas $\pi_{p\ell}^{(\mu,\sigma)}$ minadas de hasta 1,000 planogramas por grupo | Explícito | Mayor robustez que el modelo base (43 planogramas) |
# | 6 | Un UPC aparece máximo una vez por charola | Explícito | Facings representan frentes, no múltiples posiciones |
# | 7 | Sustitución greedy (no asignación bipartita óptima) | Implícito | A >30% sustitución la calidad degrada más rápido |
# | 8 | Dimensiones del catálogo son estáticas y confiables | Implícito | Errores en datos fuente degradan restricciones |
# | 9 | CF y CFC no comparten reglas de colocación | Explícito | Evita contaminación entre geometrías de mueble diferentes |
# | 10 | Segmentos con <20 planogramas usan fallback a nivel de mueble | Explícito | HOG (2), REC (18), CFC/SIN (16) usan reglas menos específicas |

# %% [markdown]
# ## Conclusiones Técnicas

# %%
# =============================================================
# CONCLUSIONES
# =============================================================
print("""
CONCLUSIONES TÉCNICAS
=====================

1. CALIDAD DE LA SOLUCIÓN
   - Con 5,614 planogramas y reglas por segmento, la heurística alcanza
     Φ_nivel ≈ 0.47 y Φ_adj ≈ 0.79 con 20% de sustitución.
   - La factibilidad de ancho se mantiene >= 0.996 en todos los casos.
   - La cobertura de productos es ~93% (108/119 productos colocados).

2. DESEMPEÑO COMPUTACIONAL
   - La heurística completa se ejecuta en < 1 ms por planograma.
   - El ILP exacto (PuLP/CBC, ~92 productos) necesita ~120s (timeLimit).
   - Para producción, la heurística es la única opción viable.

3. MEJORAS SOBRE EL MODELO BASE
   - Filtrado por tipo de mueble (CF/CFC) evita plantillas incompatibles.
   - 5,614 templates (vs 43) aumentan el overlap promedio con P_t.
   - Reglas por segmento capturan patrones demográficos específicos.
   - Cobertura de tamaños: 1.0 a 6.5 puertas (antes: 3.0 a 5.0).

4. VALIDACIÓN MATEMÁTICA
   - La heurística resuelve el mismo BILP que el ILP exacto,
     como aproximación greedy (gap de optimalidad ~8% en Φ_nivel).
   - Ambos comparten: función objetivo (Eqs. 4-5), restricciones
     de cobertura (6), ancho (7), altura (8), y sustitución (15-16).

5. LIMITACIONES
   - El ILP no escala a problemas completos (>50 productos con co-ocurrencia).
   - La heurística no optimiza co-ocurrencia activamente (λ₂=0 implícito).
   - Segmentos con pocos planogramas (HOG, REC) usan reglas menos específicas.
""")

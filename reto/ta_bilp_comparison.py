# %% [markdown]
# # Modelo Matemático Completo: TA-BILP (Template-Anchored BILP)
#
# ## Objetivo
# Un modelo matemático **autocontenido** que, dado únicamente:
# - **Base de datos**: planogramas históricos $\mathcal{H}$, catálogo de productos
# - **Entrada**: surtido objetivo $P_t$, tipo de mueble $\mu$, tamaño $\tau$, segmento $\sigma$
#
# produce un planograma óptimo SIN depender de la heurística.
#
# ## Pipeline matemático completo
#
# ```
# ENTRADAS                          MODELO (auto-contenido)                    SALIDA
# ─────────                         ──────────────────────────                 ──────
# H (planogramas históricos)  ──►  Paso 1: Minería de reglas π, α         ──►  Planograma
# Catálogo (w_p, h_p)        ──►  Paso 2: Selección de template (Ec. 14) ──►  generado
# P_t (surtido objetivo)     ──►  Paso 3: Partición P_keep / P_new       ──►
# τ (tamaño mueble)          ──►  Paso 4: BILP → asignación óptima      ──►
# μ (tipo de mueble)         ──►
# σ (segmento de tienda)     ──►
# ```
#
# ---
#
# ## Formulación Matemática
#
# ### Paso 1: Minería de reglas (parámetros del modelo)
#
# A partir de $\mathcal{H}(\mu, \sigma)$ se calculan:
#
# $$\pi_{p\ell}^{(\mu,\sigma)} = \frac{\text{# veces } p \text{ en nivel } \ell}{\text{# apariciones de } p}$$
#
# Estos son **parámetros**, no variables. Se minan una vez.
#
# ### Paso 2: Selección de template (Ec. 14)
#
# $$T^* = \arg\max_{T \in \mathcal{H}(\tau, \mu)} |P_T \cap P_t|$$
#
# Enumeración finita sobre $\mathcal{H}$. Complejidad: $O(|\mathcal{H}| \cdot |P_t|)$.
#
# ### Paso 3: Partición del surtido
#
# $$P_{\text{keep}} = P_t \cap P_{T^*}, \quad P_{\text{gone}} = P_{T^*} \setminus P_t, \quad P_{\text{new}} = P_t \setminus P_{T^*}$$
#
# Los productos $P_{\text{keep}}$ quedan **fijos** en su charola del template $s_p^{T^*}$.
#
# ### Paso 4: BILP — Asignación óptima de $P_{\text{new}}$
#
# **Parámetros pre-calculados:**
#
# | Símbolo | Definición |
# |---|---|
# | $W_s^{\text{rem}} = W_s - \sum_{p \in P_{\text{keep}}, s_p^T=s} w_p f_p$ | Ancho residual en charola $s$ |
# | $H_s$ | Altura efectiva de la charola $s$ |
# | $\ell_s$ | Nivel de la charola $s$ |
# | $\delta(p,q) = |w_p-w_q| + 2|h_p-h_q|$ | Distancia dimensional |
# | $\gamma(p,s) = \max\left(0,\; 1 - \frac{\min_{q \in P_{\text{gone}}(s)} \delta(p,q)}{\delta_{\max}}\right)$ | Afinidad de sustitución |
# | $\phi(p,s) = 1 - \frac{|y_s - y^*_{\text{ideal}}(h_p)|}{y_{\max} - y_{\min}}$ | Afinidad de nivel (gap-fill) |
#
# **Variables:**
# $$x_{ps} \in \{0,1\} \quad \forall p \in P_{\text{new}}, s \in S$$
#
# **Función objetivo:**
# $$\max \; \underbrace{\sum_p \sum_s x_{ps}}_{\Phi_{\text{cob}}}
# + \lambda_1 \cdot \underbrace{\frac{1}{|P_{\text{new}}|}\sum_p \sum_s \pi_{p,\ell_s} \cdot x_{ps}}_{\Phi_{\text{nivel}}}
# + \lambda_2 \cdot \underbrace{\frac{1}{|P_{\text{new}}|}\sum_p \sum_s \gamma(p,s) \cdot x_{ps}}_{\Phi_{\text{afinidad}}}$$
#
# **Restricciones:**
#
# | No. | Ecuación | Descripción |
# |---|---|---|
# | (1) | $\sum_s x_{ps} \leq 1\;\forall p$ | Cada producto en máximo un anaquel |
# | (2) | $\sum_p w_p \cdot x_{ps} \leq W_s^{\text{rem}}\;\forall s$ | Capacidad de ancho residual |
# | (3) | $x_{ps} = 0 \;\forall (p,s): h_p > 1.05 \cdot H_s$ | Compatibilidad de altura |

# %% [markdown]
# ## Implementación

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
    build_product_catalog, mine_placement_rules, evaluate_rule_adherence,
    # Heuristic functions — imported ONLY for comparison, not used in the model
    find_best_match as heuristic_find_best_match,
    adapt_planogram as heuristic_adapt_planogram,
    ProductInfo,
)
import pulp

OUTPUT_DIR = "test_output/ta_bilp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# %%
# =============================================================
# MODELO COMPLETO AUTO-CONTENIDO
# =============================================================

def ideal_y(height: float) -> float:
    """
    Coordenada Y ideal según la altura del producto (Ec. 19).
    Derivado de la correlación Pearson r=-0.90 (alto vs Y).
    """
    if   height >= 30: return  20.0
    elif height >= 25: return  90.0
    elif height >= 20: return 130.0
    elif height >= 15: return 166.0
    else:              return 168.0


def generate_planogram_bilp(
    target_upcs: set[str],
    target_tamano: float,
    target_mueble: str,
    target_segmento: str,
    historic_planograms: list[Planogram],
    catalog: dict[str, ProductInfo],
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    min_planograms_for_rules: int = 20,
    time_limit: int = 300,
    gap_rel: float = 0.01,
    verbose: bool = False,
) -> dict:
    """
    Modelo matemático auto-contenido para generación de planogramas.

    Recibe SOLO parámetros de base de datos y entrada. Internamente
    ejecuta los 4 pasos del pipeline:
      1. Minería de reglas π_{pℓ} desde H(μ, σ)
      2. Selección de template T* (Ec. 14)
      3. Partición P_keep / P_gone / P_new
      4. BILP: asignación óptima de P_new

    Parámetros
    ----------
    target_upcs : set[str]
        Surtido de la tienda objetivo P_t (conjunto de UPCs).
    target_tamano : float
        Tamaño del mueble (puertas): 1.0, 1.5, ..., 6.5.
    target_mueble : str
        Tipo de mueble: 'CF' o 'CFC'.
    target_segmento : str
        Segmento de tienda: 'BCO', 'CLA', 'HRN', etc.
    historic_planograms : list[Planogram]
        Base de datos completa de planogramas históricos H.
    catalog : dict[str, ProductInfo]
        Catálogo de productos {UPC: ProductInfo(width, height, ...)}.
    lambda1 : float
        Peso de la adherencia de nivel Φ_nivel en la FO.
    lambda2 : float
        Peso de la afinidad de sustitución Φ_afinidad en la FO.
    min_planograms_for_rules : int
        Mínimo de planogramas para minar reglas por segmento.
    time_limit : int
        Tiempo máximo del solver (segundos).
    gap_rel : float
        Gap de optimalidad relativo aceptable.

    Retorna
    -------
    dict con:
        'planogram': Planogram generado
        'status': estado del solver
        'time_total': tiempo total (reglas + template + BILP)
        'time_bilp': tiempo del solver
        'n_placed', 'n_keep', 'n_new_placed', 'n_new', 'n_gone'
        'phi_nivel', 'phi_affinity'
        'template_title': título del template seleccionado
        'template_overlap': overlap |P_T ∩ P_t|
        'rules_source': origen de las reglas
        'n_vars', 'n_constraints'
    """
    t0_total = time.perf_counter()

    # ==========================================================
    # PASO 1: MINERÍA DE REGLAS π_{pℓ}^{(μ,σ)}
    # ==========================================================
    # Filtrar H por mueble y segmento
    seg_plans = [p for p in historic_planograms
                 if p.mueble_id == target_mueble
                 and p.segmento_id == target_segmento]

    if len(seg_plans) >= min_planograms_for_rules:
        rules = mine_placement_rules(seg_plans, catalog)
        rules_source = f"{target_mueble}/{target_segmento} ({len(seg_plans)} planogramas)"
    else:
        # Fallback: usar todos los planogramas del mismo mueble
        mueble_plans = [p for p in historic_planograms
                        if p.mueble_id == target_mueble]
        rules = mine_placement_rules(mueble_plans, catalog)
        rules_source = f"{target_mueble} (fallback, {len(mueble_plans)} planogramas)"

    level_probs = rules['level_probs']

    # ==========================================================
    # PASO 2: SELECCIÓN DE TEMPLATE T* (Ec. 14)
    #   T* = argmax_{T ∈ H(τ,μ)} |P_T ∩ P_t|
    # ==========================================================
    best_template = None
    best_overlap = -1
    best_total = 0

    for plano in historic_planograms:
        # Filtro: mismo tamaño y tipo de mueble
        if plano.tamano != target_tamano:
            continue
        if plano.mueble_id != target_mueble:
            continue

        # Calcular overlap
        template_upcs = {p.upc for s in plano.shelves.values()
                         for p in s.products}
        overlap = len(template_upcs & target_upcs)

        if overlap > best_overlap:
            best_overlap = overlap
            best_template = plano
            best_total = len(template_upcs)

    if best_template is None:
        return {
            'planogram': None,
            'status': 'Infeasible (no template)',
            'time_total': time.perf_counter() - t0_total,
            'time_bilp': 0,
            'n_placed': 0, 'n_keep': 0, 'n_new_placed': 0,
            'n_new': len(target_upcs), 'n_gone': 0,
            'phi_nivel': None, 'phi_affinity': None,
            'template_title': None, 'template_overlap': 0,
            'rules_source': rules_source,
            'n_vars': 0, 'n_constraints': 0,
        }

    template = best_template

    # ==========================================================
    # PASO 3: PARTICIÓN DEL SURTIDO
    # ==========================================================
    # Extraer estructura del template
    shelves = {}          # ch_num -> Shelf
    template_assign = {}  # upc -> ch_num
    template_facings = {} # upc -> facings

    for ch_num, shelf in template.shelves.items():
        shelves[ch_num] = shelf
        for p in shelf.products:
            template_assign[p.upc] = ch_num
            template_facings[p.upc] = p.facings

    P_T = set(template_assign.keys())
    P_keep = target_upcs & P_T       # productos a conservar
    P_gone = P_T - target_upcs        # productos removidos (slots libres)
    P_new  = target_upcs - P_T       # productos nuevos a colocar

    # Filtrar por catálogo
    P_new_sorted  = sorted([u for u in P_new if u in catalog])
    P_gone_sorted = sorted([u for u in P_gone if u in catalog])

    # Calcular ancho residual por charola
    S_list = sorted(shelves.keys())
    remaining_width = {}
    for ch_num in S_list:
        shelf = shelves[ch_num]
        kept_w = sum(
            catalog[p.upc].width * p.facings
            for p in shelf.products
            if p.upc in P_keep and p.upc in catalog
        )
        remaining_width[ch_num] = shelf.shelf_width - kept_w

    # Alturas efectivas de charolas (gap entre Y consecutivas)
    ys = sorted({s.y for s in shelves.values()})
    eff_h = {}
    for i, yv in enumerate(ys):
        if i < len(ys) - 1:
            eff_h[yv] = ys[i+1] - yv
        else:
            eff_h[yv] = (ys[i] - ys[i-1]) if i > 0 else 40.0

    # ==========================================================
    # PASO 4: BILP — ASIGNACIÓN ÓPTIMA DE P_new
    # ==========================================================
    n_new = len(P_new_sorted)

    # Caso trivial: no hay productos nuevos
    if n_new == 0:
        # Construir planograma solo con P_keep
        result = _build_planogram(template, shelves, P_keep, {},
                                  catalog, template_facings)
        return {
            'planogram': result,
            'status': 'Optimal (trivial)',
            'time_total': time.perf_counter() - t0_total,
            'time_bilp': 0,
            'n_placed': len(P_keep), 'n_keep': len(P_keep),
            'n_new_placed': 0, 'n_new': 0, 'n_gone': len(P_gone),
            'phi_nivel': 0, 'phi_affinity': 0,
            'template_title': template.title,
            'template_overlap': best_overlap,
            'rules_source': rules_source,
            'n_vars': 0, 'n_constraints': 0,
        }

    # --- Calcular parámetros de afinidad γ(p, s) ---

    # P_gone en cada charola
    gone_on_shelf = defaultdict(list)
    for upc in P_gone_sorted:
        if upc in template_assign:
            gone_on_shelf[template_assign[upc]].append(upc)

    # δ(p,q) = |w_p - w_q| + 2·|h_p - h_q|
    def delta(p_upc, q_upc):
        pi = catalog[p_upc]
        qi = catalog[q_upc]
        return abs(pi.width - qi.width) + 2.0 * abs(pi.height - qi.height)

    # δ_max para normalización
    all_deltas = [delta(p, q) for p in P_new_sorted for q in P_gone_sorted]
    delta_max = max(all_deltas) if all_deltas else 1.0

    # y ranges para normalización del gap-fill
    y_range = max(ys) - min(ys) if len(ys) > 1 else 1.0

    # γ(p, s): afinidad de sustitución
    gamma = {}
    for p in P_new_sorted:
        for ch_num in S_list:
            gone = gone_on_shelf.get(ch_num, [])
            if gone:
                # Charola con slots liberados → usar afinidad de sustitución
                best_d = min(delta(p, q) for q in gone)
                gamma[p, ch_num] = max(0.0, 1.0 - best_d / delta_max)
            else:
                # Charola sin slots liberados → usar afinidad de nivel (gap-fill)
                shelf = shelves[ch_num]
                iy = ideal_y(catalog[p].height)
                gamma[p, ch_num] = max(0.0, 1.0 - abs(shelf.y - iy) / y_range)

    # --- Construir BILP ---
    prob = pulp.LpProblem("TA_BILP", pulp.LpMaximize)

    # Variables x_{ps} ∈ {0,1}
    x = {}
    for p in P_new_sorted:
        for ch_num in S_list:
            x[p, ch_num] = pulp.LpVariable(f"x_{p}_{ch_num}", cat="Binary")

    # --- Función objetivo ---
    # Φ_cob: cobertura (maximizar productos colocados)
    phi_cob = pulp.lpSum(x[p, ch] for p in P_new_sorted for ch in S_list)

    # Φ_nivel: adherencia de nivel
    phi_niv = (1.0 / n_new) * pulp.lpSum(
        level_probs.get(p, {}).get(shelves[ch].level, 0.0) * x[p, ch]
        for p in P_new_sorted for ch in S_list
    )

    # Φ_afinidad: afinidad de sustitución
    phi_aff = (1.0 / n_new) * pulp.lpSum(
        gamma.get((p, ch), 0.0) * x[p, ch]
        for p in P_new_sorted for ch in S_list
    )

    prob += phi_cob + lambda1 * phi_niv + lambda2 * phi_aff, "FO"

    # --- Restricciones ---
    # (1) Cada producto en máximo un anaquel
    for p in P_new_sorted:
        prob += pulp.lpSum(x[p, ch] for ch in S_list) <= 1, f"cov_{p}"

    # (2) Capacidad de ancho residual
    for ch_num in S_list:
        prob += pulp.lpSum(
            catalog[p].width * x[p, ch_num] for p in P_new_sorted
        ) <= remaining_width[ch_num], f"w_{ch_num}"

    # (3) Compatibilidad de altura
    for p in P_new_sorted:
        h_p = catalog[p].height
        for ch_num in S_list:
            shelf_h = eff_h.get(shelves[ch_num].y, shelves[ch_num].shelf_height)
            if h_p > shelf_h * 1.05:
                prob += x[p, ch_num] == 0, f"ht_{p}_{ch_num}"

    n_vars = len(x)
    n_cons = len(prob.constraints)

    # --- Resolver ---
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=int(verbose),
                                gapRel=gap_rel)
    t0_bilp = time.perf_counter()
    status = prob.solve(solver)
    time_bilp = time.perf_counter() - t0_bilp

    # --- Extraer asignación ---
    ilp_assignment = {}  # upc -> ch_num
    for p in P_new_sorted:
        for ch_num in S_list:
            v = pulp.value(x[p, ch_num])
            if v is not None and v > 0.5:
                ilp_assignment[p] = ch_num
                break

    # --- Construir planograma final ---
    result = _build_planogram(template, shelves, P_keep, ilp_assignment,
                              catalog, template_facings)

    return {
        'planogram': result,
        'status': pulp.LpStatus[status],
        'time_total': time.perf_counter() - t0_total,
        'time_bilp': time_bilp,
        'n_placed': len(P_keep) + len(ilp_assignment),
        'n_keep': len(P_keep),
        'n_new_placed': len(ilp_assignment),
        'n_new': n_new,
        'n_gone': len(P_gone),
        'phi_nivel': pulp.value(phi_niv) if prob.status == 1 else None,
        'phi_affinity': pulp.value(phi_aff) if prob.status == 1 else None,
        'template_title': template.title,
        'template_overlap': best_overlap,
        'rules_source': rules_source,
        'rules': rules,
        'n_vars': n_vars,
        'n_constraints': n_cons,
    }


def _build_planogram(
    template: Planogram,
    shelves: dict,
    P_keep: set,
    ilp_assignment: dict,
    catalog: dict,
    template_facings: dict,
) -> Planogram:
    """Construye el planograma final a partir de P_keep + asignación ILP."""
    result = Planogram(
        segmento_id=template.segmento_id,
        mueble_id=template.mueble_id,
        planogrupo=template.planogrupo,
        tamano=template.tamano,
        direccion=template.direccion,
        conjunto_id="TA-BILP",
    )

    for ch_num, shelf in shelves.items():
        ns = Shelf(shelf.charola, shelf.door, shelf.level,
                   shelf.x, shelf.y, shelf.shelf_width, shelf.shelf_height)

        # P_keep: conservar en su posición original
        for p in shelf.products:
            if p.upc in P_keep:
                ns.products.append(ProductPlacement(
                    p.upc, p.description, p.shelf, p.position,
                    p.facings, p.width, p.height))

        # P_new: productos asignados por el BILP
        for upc, assigned_ch in ilp_assignment.items():
            if assigned_ch == ch_num and upc in catalog:
                ci = catalog[upc]
                pos = max((p.position for p in ns.products), default=0) + 1
                ns.products.append(ProductPlacement(
                    upc, ci.description, ch_num, pos,
                    1, ci.width, ci.height))

        result.shelves[ch_num] = ns

    return result

# %% [markdown]
# ## Carga de Datos (para el test de comparación)

# %%
log("Cargando datos...")
CSV_ORIGINAL = "datos/ejemplo_planograma.csv"
CSV_NEW = "datos/Ejemplo 2.csv"

planograms_orig = load_planograms(CSV_ORIGINAL)
planograms_new = load_planograms(CSV_NEW)
planograms = planograms_orig + planograms_new
catalog = build_product_catalog([CSV_ORIGINAL, CSV_NEW])
log(f"  |H| = {len(planograms)}, |catálogo| = {len(catalog)}")

# %% [markdown]
# ## Comparación: TA-BILP (auto-contenido) vs Heurística
#
# Ambos reciben los mismos inputs. El TA-BILP usa `generate_planogram_bilp()`;
# la heurística usa `find_best_match()` + `adapt_planogram()`.
# Se evalúan con las **mismas métricas** via `evaluate_rule_adherence()`.

# %%
log("=" * 70)
log("COMPARACIÓN: TA-BILP (auto-contenido) vs HEURÍSTICA")
log("=" * 70)

test_planograms = planograms_orig[:10]
swap_fractions = [0.10, 0.20, 0.30, 0.40]
all_results = []

for plano_idx, original in enumerate(test_planograms):
    orig_upcs = {p.upc for s in original.shelves.values() for p in s.products}
    all_upcs_pool = {p.upc for pl in planograms
                     for s in pl.shelves.values() for p in s.products}

    for swap_frac in swap_fractions:
        random.seed(42 + plano_idx * 100 + int(swap_frac * 100))

        # Construir P_t perturbado
        orig_list = sorted(orig_upcs)
        n_swap = max(1, int(len(orig_list) * swap_frac))
        to_remove = set(random.sample(orig_list, n_swap))
        available = sorted(all_upcs_pool - orig_upcs)
        replacements = set(random.sample(available, min(n_swap, len(available))))
        target_upcs = (orig_upcs - to_remove) | replacements

        # ==============================================
        # HEURÍSTICA (usa funciones importadas)
        # ==============================================
        t0_h = time.perf_counter()
        h_match, h_ov, h_tot = heuristic_find_best_match(
            target_upcs, original.tamano, planograms,
            target_mueble=original.mueble_id,
        )
        gen_h = heuristic_adapt_planogram(h_match, target_upcs, catalog)
        h_time = time.perf_counter() - t0_h

        # ==============================================
        # TA-BILP (auto-contenido)
        # ==============================================
        ta = generate_planogram_bilp(
            target_upcs=target_upcs,
            target_tamano=original.tamano,
            target_mueble=original.mueble_id,
            target_segmento=original.segmento_id,
            historic_planograms=planograms,
            catalog=catalog,
            lambda1=1.0,
            lambda2=1.0,
            time_limit=60,
            gap_rel=0.02,
        )

        # Evaluar con MISMAS reglas
        # (usar las reglas que el TA-BILP minó internamente para evaluar ambos)
        eval_rules = ta.get('rules')
        if eval_rules is None:
            # Fallback
            seg_plans = [p for p in planograms
                         if p.mueble_id == original.mueble_id
                         and p.segmento_id == original.segmento_id]
            eval_rules = mine_placement_rules(seg_plans, catalog)

        h_metrics = evaluate_rule_adherence(gen_h, eval_rules, catalog)
        ta_metrics = evaluate_rule_adherence(ta['planogram'], eval_rules, catalog) \
            if ta['planogram'] else {k: 0 for k in h_metrics}

        log(f"  [{plano_idx}] {original.segmento_id}/{original.mueble_id} "
            f"swap={swap_frac:.0%}: "
            f"H=[niv={h_metrics['level_rule_score']:.3f} "
            f"adj={h_metrics['adjacency_hit_rate']:.3f} "
            f"placed={h_metrics['n_products_placed']}] "
            f"TA=[niv={ta_metrics['level_rule_score']:.3f} "
            f"adj={ta_metrics['adjacency_hit_rate']:.3f} "
            f"placed={ta_metrics['n_products_placed']} "
            f"t={ta['time_bilp']:.1f}s {ta['status']}]")

        all_results.append({
            'plano_idx': plano_idx,
            'swap_frac': swap_frac,
            'segmento': original.segmento_id,
            'mueble': original.mueble_id,
            'n_target': len(target_upcs),
            'n_keep': ta['n_keep'], 'n_new': ta['n_new'], 'n_gone': ta['n_gone'],
            'template_overlap': ta['template_overlap'],
            # Heuristic
            'h_level': h_metrics['level_rule_score'],
            'h_mode': h_metrics['level_mode_accuracy'],
            'h_adj': h_metrics['adjacency_hit_rate'],
            'h_adj_w': h_metrics['adjacency_weighted'],
            'h_wfeas': h_metrics['width_feasibility'],
            'h_placed': h_metrics['n_products_placed'],
            'h_time': h_time,
            # TA-BILP
            'ta_level': ta_metrics['level_rule_score'],
            'ta_mode': ta_metrics['level_mode_accuracy'],
            'ta_adj': ta_metrics['adjacency_hit_rate'],
            'ta_adj_w': ta_metrics['adjacency_weighted'],
            'ta_wfeas': ta_metrics['width_feasibility'],
            'ta_placed': ta_metrics['n_products_placed'],
            'ta_time': ta['time_bilp'],
            'ta_status': ta['status'],
        })

# %% [markdown]
# ## Resultados

# %%
df = pd.DataFrame(all_results)

log("\n" + "=" * 90)
log("RESULTADOS PROMEDIO POR SWAP %")
log("=" * 90)

print(f"\n{'Swap':>5} | {'--- Heurística ---':^32} | {'--- TA-BILP ---':^32}")
print(f"{'%':>5} | {'Φ_niv':>7} {'Mode%':>7} {'AdjHR':>7} {'Placed':>7} | "
      f"{'Φ_niv':>7} {'Mode%':>7} {'AdjHR':>7} {'Placed':>7} {'Time':>6}")
print("-" * 85)

for swap in sorted(df['swap_frac'].unique()):
    sub = df[df['swap_frac'] == swap]
    print(f"{swap:>5.0%} | "
          f"{sub['h_level'].mean():>7.3f} {sub['h_mode'].mean():>7.1%} "
          f"{sub['h_adj'].mean():>7.3f} {sub['h_placed'].mean():>7.0f} | "
          f"{sub['ta_level'].mean():>7.3f} {sub['ta_mode'].mean():>7.1%} "
          f"{sub['ta_adj'].mean():>7.3f} {sub['ta_placed'].mean():>7.0f} "
          f"{sub['ta_time'].mean():>5.1f}s")

# Deltas
print(f"\n{'Swap':>5} | {'Δ Φ_niv':>9} {'Δ Mode':>9} {'Δ AdjHR':>9} {'Δ Placed':>9}")
print("-" * 50)
for swap in sorted(df['swap_frac'].unique()):
    sub = df[df['swap_frac'] == swap]
    print(f"{swap:>5.0%} | "
          f"{sub['ta_level'].mean()-sub['h_level'].mean():>+9.4f} "
          f"{sub['ta_mode'].mean()-sub['h_mode'].mean():>+9.1%} "
          f"{sub['ta_adj'].mean()-sub['h_adj'].mean():>+9.4f} "
          f"{sub['ta_placed'].mean()-sub['h_placed'].mean():>+9.0f}")

df.to_csv(os.path.join(OUTPUT_DIR, "ta_bilp_results.csv"), index=False)
log(f"Resultados: {OUTPUT_DIR}/ta_bilp_results.csv")

# %%
# =============================================================
# EJEMPLO: INVOCACIÓN DIRECTA DEL MODELO
# =============================================================
log("\n" + "=" * 70)
log("EJEMPLO: INVOCACIÓN DIRECTA DEL MODELO")
log("=" * 70)

# El modelo recibe SOLO parámetros de BD + entrada
example = planograms_orig[0]
orig_upcs = {p.upc for s in example.shelves.values() for p in s.products}

random.seed(42)
n_swap = int(len(orig_upcs) * 0.20)
to_remove = set(random.sample(sorted(orig_upcs), n_swap))
all_pool = {p.upc for pl in planograms for s in pl.shelves.values()
            for p in s.products}
replacements = set(random.sample(sorted(all_pool - orig_upcs), n_swap))
target_upcs = (orig_upcs - to_remove) | replacements

print(f"\nINPUT:")
print(f"  P_t          = {len(target_upcs)} UPCs")
print(f"  τ (tamaño)   = {example.tamano}")
print(f"  μ (mueble)   = {example.mueble_id}")
print(f"  σ (segmento) = {example.segmento_id}")
print(f"  |H|          = {len(planograms)}")
print(f"  |catálogo|   = {len(catalog)}")

result = generate_planogram_bilp(
    target_upcs=target_upcs,
    target_tamano=example.tamano,
    target_mueble=example.mueble_id,
    target_segmento=example.segmento_id,
    historic_planograms=planograms,
    catalog=catalog,
    lambda1=1.0,
    lambda2=1.0,
    time_limit=120,
)

print(f"\nOUTPUT:")
print(f"  Status:           {result['status']}")
print(f"  Template:         {result['template_title']}")
print(f"  Overlap P_T∩P_t:  {result['template_overlap']}")
print(f"  Reglas:           {result['rules_source']}")
print(f"  P_keep:           {result['n_keep']}")
print(f"  P_gone:           {result['n_gone']}")
print(f"  P_new:            {result['n_new']}")
print(f"  Colocados:        {result['n_new_placed']}/{result['n_new']} nuevos")
print(f"  Total colocados:  {result['n_placed']}")
print(f"  Φ_nivel:          {result['phi_nivel']:.4f}" if result['phi_nivel'] else "  Φ_nivel: N/A")
print(f"  Φ_afinidad:       {result['phi_affinity']:.4f}" if result['phi_affinity'] else "  Φ_afinidad: N/A")
print(f"  Variables:        {result['n_vars']}")
print(f"  Restricciones:    {result['n_constraints']}")
print(f"  Tiempo BILP:      {result['time_bilp']:.2f}s")
print(f"  Tiempo total:     {result['time_total']:.2f}s")

# Evaluar
if result['planogram']:
    eval_rules = result['rules']
    metrics = evaluate_rule_adherence(result['planogram'], eval_rules, catalog)
    print(f"\nMÉTRICAS DE EVALUACIÓN:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:.4f}")

    result['planogram'].plot_planogram()

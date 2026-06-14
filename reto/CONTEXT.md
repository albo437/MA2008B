# Planogram Generation Project — Full Context

> **Purpose**: This document captures the complete history and context of the planogram project
> so that any new AI conversation can continue work seamlessly. Paste the contents of this file
> (or reference it) at the start of a new conversation.

---

## Project Overview

**Goal**: Build an automated planogram generator for refrigerated beverage displays (Coca-Cola ecosystem).
Given a store's product assortment and shelf layout, the system generates a complete planogram
that places products on specific shelves and positions, learning implicit business rules from
historic data.

**Course**: MA2008B (Modelación de Sistemas Multiagentes con Gráficas Computacionales)
**Location**: `/Users/alvarobolanos/Desktop/MA2008B/reto/`

---

## Data

- **Source file**: `datos/ejemplo_planograma.csv` (latin-1 encoding)
- **43 planograms**, **189 unique products (UPCs)**, **5,201 product placements**
- Products per planogram: 93–151 (mean 121)
- Products per shelf: 1–9 (mean 5.2)
- Shelves per planogram: 18–30
- Most products appear in ~27/43 planograms (high overlap)

### Key Column Descriptions
| Column | Description |
|---|---|
| `SEGMENTO_ID` | Store segment (e.g., BCO) |
| `MUEBLE_ID` | Furniture type (e.g., CF = cooler/fridge) |
| `PLANOGRUPO` | Planogram group name |
| `TAMANO_POST` | Number of doors (3, 3.5, 4, 4.5, 5) — 3.5 means some doors have fewer shelves |
| `DIRECCION_LEGO_ID` | Reading direction: `DI` = right-to-left, `ID` = left-to-right |
| `CONJUNTO_ID` | Unique planogram variant identifier |
| `UPC_CVE` | Product barcode/identifier |
| `ITEM_DESC` | Product description |
| `CHAROLA` | Shelf (tray) number — sequential across doors |
| `UBICACION_BANDEJA` | Position index on the shelf (reading order) |
| `NUM_FRENTES` | Number of facings |
| `ALTO` | Product height (cm) |
| `ANCHO` | Product width (cm) |
| `WIDTH` | Shelf width (cm) — always 55cm |
| `HEIGHT` | Shelf height (cm) |
| `X` | Shelf horizontal position (0, 55, 110, 165, 220) — defines doors |
| `Y` | Shelf vertical position (0, 42, 84, 115.5, 147, 175) — defines levels |

### Spatial Structure Discovery
- **X coordinates** (0, 55, 110, 165) define **doors/columns** — each door is 55cm wide
- **Y coordinates** (0, 42, 84, 115.5, 147, 175) define **vertical shelf levels** — 6 levels per door
- `CHAROLA` numbering goes sequentially: door 1 bottom→top, then door 2 bottom→top, etc.
- `TAMANO_POST=4` → 4 doors × 6 levels = 24 charolas
- `TAMANO_POST=3.5` → some doors have fewer levels (e.g., door 0 only has 2 shelves)
- **Strong negative correlation (r ≈ -0.90)** between product height (ALTO) and shelf Y:
  - ALTO 30-40cm → Y ≈ 20 (bottom shelves)
  - ALTO < 15cm → Y ≈ 168 (top shelves)

---

## Project Architecture

### File: `exploracion.py` (606 lines)
**Data loader and visualization module.**

Key classes:
- **`ProductPlacement`** (dataclass): upc, description, shelf, position, facings, width, height
- **`Shelf`** (dataclass): charola, door, level, x, y, shelf_width, shelf_height, products list
  - `sorted_products(reverse)`: returns products sorted by UBICACION_BANDEJA
- **`Planogram`** (dataclass): segmento_id, mueble_id, planogrupo, tamano, direccion, conjunto_id, shelves dict
  - `key`: tuple identifier
  - `title`: human-readable string
  - `is_right_to_left`: True if direction is DI
  - `get_door_layout()`: groups shelves by door, sorted by level
  - `get_n_doors()`, `get_n_levels()`
  - `print_planogram()`: text display (top→bottom, door by door)
  - `plot_planogram(figsize, save_path)`: matplotlib visualization with colored product rectangles

Key functions:
- **`normalize_column(col)`**: Handles BOM, accents, encoding issues in column names
- **`load_planograms(csv_path)`**: Main loader — reads CSV, builds Planogram objects with spatial metadata
- **`_product_color(name, alpha)`**: Deterministic pastel color from product name hash

### File: `planogram_model.py` (738 lines)
**Retrieval + Gap-Fill planogram generator with rule-adherence evaluation.**

Key classes:
- **`ProductInfo`** (dataclass): upc, description, width, height

Key functions:
- **`build_product_catalog(csv_path)`**: Builds {upc: ProductInfo} from CSV
- **`find_best_match(target_upcs, target_tamano, historic_planograms, exclude_keys)`**: Finds historic planogram with most product overlap (filters by same tamaño)
- **`find_closest_product(original, available_upcs, catalog, already_used)`**: Substitution by physical similarity — `distance = Δwidth + 2×Δheight` (height weighted more because it determines shelf level)
- **`adapt_planogram(template, target_upcs, catalog)`**: Two-phase adaptation:
  - Phase 1: Keep available products, swap unavailable with closest substitute
  - Phase 2: Insert unplaced products into shelf gaps (best-fit decreasing, height-aware)
- **`_ideal_y_for_height(product_height)`**: Maps product height → ideal Y coordinate
- **`mine_placement_rules(planograms, catalog)`**: Extracts level probabilities and adjacency frequencies from all planograms
- **`evaluate_rule_adherence(planogram, rules, catalog)`**: Evaluates rule compliance (level score, adjacency hit rate, width feasibility, height-level correlation)
- **`perturbation_test(...)`**: Simulates stores with different assortments by swapping products
- **`plot_comparison(original, generated)`**: Side-by-side visualization

### Other Files
- `planograma_model.ipynb`: Notebook version (may be out of sync)
- `exploracion.ipynb`: Original exploration notebook
- `Figure_1.png`, `doc_original.png`, `doc_generated.png`: Generated visualizations
- `datos/ejemplo_planograma.csv`: Main dataset

---

## Model Results (Completed Work)

### Approach Implemented: Retrieval + Gap-Fill (Phase 1)

**Algorithm**:
1. Given target product assortment + shelf layout size
2. Find the closest historic planogram by product overlap (filtered by same tamaño)
3. Keep available products in place
4. Swap unavailable products with physically similar substitutes
5. Insert remaining products into shelf gaps

### Leave-One-Out Results (v1 — original evaluation)
| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| product_jaccard | 0.997 | 0.011 | 0.944 | 1.000 |
| level_accuracy | 0.968 | 0.040 | 0.794 | 1.000 |
| door_accuracy | 0.156 | 0.189 | 0.000 | 0.931 |
| charola_accuracy | 0.107 | 0.182 | 0.000 | 0.876 |
| width_feasibility | 1.000 | 0.000 | 1.000 | 1.000 |

**Key findings**:
- ✅ Level accuracy very high (96.8%) — products go to correct height
- ✅ Width feasibility perfect — all shelves fit within constraints
- ⚠️ Door accuracy low (15.6%) — expected, since door ordering differs across planograms
- ⚠️ Charola accuracy low (10.7%) — combination of door + level mismatch

### Evaluation v2: Rule-Adherence + Perturbation Test
The original leave-one-out evaluation was replaced with a more meaningful rule-adherence evaluation:
- Mine placement rules from ALL 43 planograms (level assignments, adjacency patterns)
- Perturb real planograms by swapping 10–40% of products
- Generate new planograms and score against mined rules
- Compare generated planograms vs real ones as baseline

---

## Research: Future Approaches Considered

5 approaches were researched and documented (from most to least feasible for 43 planograms):

1. **Retrieval + Gap-Fill** ← IMPLEMENTED ✅ (Low complexity, works with 43)
2. **Constraint Learning + CP-SAT** ← RECOMMENDED FOR PHASE 2 (Medium complexity, excellent for 43)
3. **Shelf-Level Sequence Model (Transformer)** — borderline feasible (~1,000 shelf sequences)
4. **Discrete Diffusion Model (LayoutDM-style)** — risky with only 43 examples
5. **Full-Planogram Seq2Seq** — not feasible with current data

**Recommended next step**: Implement Approach 4 (Constraint Learning + CP-SAT) — mine rules from data, use Google OR-Tools CP-SAT solver to generate planograms that satisfy learned constraints.

---

## Usage Example

```python
from planogram_model import (
    find_best_match, adapt_planogram,
    build_product_catalog, evaluate_rule_adherence,
    mine_placement_rules, perturbation_test
)
from exploracion import load_planograms

planograms = load_planograms("datos/ejemplo_planograma.csv")
catalog = build_product_catalog("datos/ejemplo_planograma.csv")

# Find and adapt
target_upcs = {"7501055305247", "7501055349555", ...}  # available UPCs
target_tamano = 4.0
match, overlap, total = find_best_match(target_upcs, target_tamano, planograms)
generated = adapt_planogram(match, target_upcs, catalog)
generated.plot_planogram()

# Evaluate
rules = mine_placement_rules(planograms, catalog)
metrics = evaluate_rule_adherence(generated, rules, catalog)
results_df = perturbation_test(planograms, catalog, rules)
```

---

## Open Questions (from previous conversation)

1. Will more planogram data become available? (500+ would unlock transformer/diffusion approaches)
2. Is this for a single category (Refrescos/CF) or expanding to other categories?
3. Should the model decide `NUM_FRENTES` (facings) or is that given?
4. What's the deployment target — academic prototype or production system?

---

## Conversation History Reference

The original conversation ID was `40b4e03c-4d29-464b-bc64-bda691a5f1e4`.
Transcript available at: `~/.gemini/antigravity-ide/brain/40b4e03c-4d29-464b-bc64-bda691a5f1e4/.system_generated/logs/transcript.jsonl`

Related conversations:
- `a14c318d-...`: Fixing Dataset 2 Notebook (different project — tda_project)
- `0d0860dd-...`: Follow-up dataset fix (tda_project)

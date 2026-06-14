
# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# %%
df = pd.read_csv('datos/ejemplo_planograma.csv', encoding='latin-1')

# %%
df.head()

# %%
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import unicodedata
import hashlib
import math


# =========================================================
# FIX COLUMN NAMES
# =========================================================

COLUMN_FIXES = {

    # Corrupted names (latin-1 encoding mangles ñ → ao)
    "ISEGMENTO_ID": "SEGMENTO_ID",
    "TAMAAO_POST": "TAMANO_POST",

    # Alternate names
    "ITEM": "UPC_CVE",

    # Sometimes found
    "TAMANO": "TAMANO_POST",
    "TAMAÑO": "TAMANO_POST",

    # New file column names (Ejemplo 2 format)
    "PLANOGRUPO_DESC": "PLANOGRUPO",
    "TAMANO_DESC": "TAMANO_POST",

}


def normalize_column(col):

    # Convert to string
    col = str(col)

    # Remove BOM characters
    col = col.replace("ï»¿", "")
    col = col.replace("\ufeff", "")

    # Strip whitespace
    col = col.strip()

    # Remove accents
    col = unicodedata.normalize('NFKD', col)
    col = col.encode('ascii', 'ignore').decode('utf-8')

    # Uppercase
    col = col.upper()

    # Replace spaces
    col = col.replace(" ", "_")

    # Apply manual fixes
    if col in COLUMN_FIXES:
        col = COLUMN_FIXES[col]

    return col


# =========================================================
# COLOR PALETTE
# =========================================================

def _product_color(name: str, alpha: float = 0.75):
    """Generate a deterministic pastel color from a product name."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    # Use golden-ratio hue spacing for variety
    hue = (h % 360) / 360.0
    sat = 0.35 + (h % 30) / 100.0  # 0.35–0.65
    val = 0.85 + (h % 15) / 100.0  # 0.85–1.0
    # HSV → RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (r, g, b, alpha)


# =========================================================
# PRODUCT
# =========================================================

@dataclass
class ProductPlacement:
    upc: str
    description: str
    shelf: int           # charola number
    position: int        # UBICACION_BANDEJA
    facings: int         # NUM_FRENTES
    width: float         # ANCHO (product width in cm)
    height: float        # ALTO (product height in cm)


# =========================================================
# SHELF (with spatial metadata)
# =========================================================

@dataclass
class Shelf:
    charola: int              # original charola number
    door: int                 # which door/column (0-indexed, derived from X)
    level: int                # vertical level within door (0 = bottom, derived from Y)
    x: float                  # X coordinate of shelf bottom-left (cm)
    y: float                  # Y coordinate of shelf bottom-left (cm)
    shelf_width: float        # Width of the shelf (cm)
    shelf_height: float       # Height of the shelf (cm)
    products: list[ProductPlacement] = field(default_factory=list)

    def sorted_products(self, reverse: bool = False):
        """Products sorted by UBICACION_BANDEJA, optionally reversed for DI."""
        prods = sorted(self.products, key=lambda p: p.position)
        if reverse:
            prods = list(reversed(prods))
        return prods


# =========================================================
# PLANOGRAM
# =========================================================

@dataclass
class Planogram:

    segmento_id: str
    mueble_id: str
    planogrupo: str
    tamano: float
    direccion: str
    conjunto_id: str

    shelves: dict[int, Shelf] = field(default_factory=dict)

    @property
    def key(self):
        return (self.segmento_id, self.mueble_id, self.planogrupo,
                self.tamano, self.direccion, self.conjunto_id)

    @property
    def title(self):
        return (f"{self.segmento_id} | {self.mueble_id} | "
                f"{self.planogrupo} | Tamaño={self.tamano} | "
                f"Dir={self.direccion} | Conjunto={self.conjunto_id}")

    @property
    def is_right_to_left(self):
        return str(self.direccion).upper() in ["DI", "RIGHT_LEFT", "RL"]

    # ---------------------------------------------------------
    # Door / level layout
    # ---------------------------------------------------------

    def get_door_layout(self) -> dict[int, list[Shelf]]:
        """
        Group shelves by door index, sorted by level (bottom → top).
        Returns {door_idx: [shelf_bottom, ..., shelf_top]}.
        """
        doors: dict[int, list[Shelf]] = defaultdict(list)
        for shelf in self.shelves.values():
            doors[shelf.door].append(shelf)

        # Sort each door's shelves by level (ascending = bottom to top)
        for door in doors:
            doors[door] = sorted(doors[door], key=lambda s: s.level)

        return dict(sorted(doors.items()))

    def get_n_doors(self) -> int:
        if not self.shelves:
            return 0
        return max(s.door for s in self.shelves.values()) + 1

    def get_n_levels(self) -> int:
        if not self.shelves:
            return 0
        return max(s.level for s in self.shelves.values()) + 1

    # ---------------------------------------------------------
    # Text display
    # ---------------------------------------------------------

    def print_planogram(self):

        print("\n" + "=" * 80)
        print(f"PLANOGRAM: {self.title}")
        print("=" * 80)

        door_layout = self.get_door_layout()
        n_doors = self.get_n_doors()
        n_levels = self.get_n_levels()

        print(f"Doors: {n_doors}  |  Levels: {n_levels}  |  "
              f"Total charolas: {len(self.shelves)}")

        # Print top to bottom (highest level first)
        for level in range(n_levels - 1, -1, -1):
            print(f"\n--- Level {level} " + "-" * 60)
            for door in range(n_doors):
                # Find the shelf at this door/level
                shelf = None
                for s in door_layout.get(door, []):
                    if s.level == level:
                        shelf = s
                        break
                if shelf is None:
                    print(f"  Door {door}: (empty)")
                    continue

                prods = shelf.sorted_products(reverse=self.is_right_to_left)
                items = []
                for p in prods:
                    label = p.description[:20]
                    if p.facings > 1:
                        label += f" x{p.facings}"
                    items.append(f"[{label}]")

                print(f"  Door {door} (charola {shelf.charola:>2}): "
                      f"{''.join(items)}")

    # ---------------------------------------------------------
    # Matplotlib visualization
    # ---------------------------------------------------------

    def plot_planogram(self, figsize=None, save_path=None):
        """
        Draws the planogram as a 2D grid (doors × levels) with
        products shown as proportionally-sized colored rectangles.
        """
        door_layout = self.get_door_layout()
        n_doors = self.get_n_doors()
        n_levels = self.get_n_levels()

        if n_doors == 0 or n_levels == 0:
            print("No shelves to plot.")
            return

        # ---- Figure sizing ----
        if figsize is None:
            fig_w = max(5 * n_doors, 12)
            fig_h = max(3 * n_levels, 8)
            figsize = (fig_w, fig_h)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=18)

        # ---- Layout constants ----
        DOOR_WIDTH = 1.2       # normalized width per door
        LEVEL_HEIGHT = 1.4     # normalized height per level (taller for vertical text)
        DOOR_GAP = 0.10        # gap between doors
        SHELF_PAD = 0.03       # padding inside shelf cell

        total_w = n_doors * DOOR_WIDTH + (n_doors - 1) * DOOR_GAP
        total_h = n_levels * LEVEL_HEIGHT

        # ---- Draw each shelf and its products ----
        for door_idx, shelf_list in door_layout.items():
            for shelf in shelf_list:

                # Cell position (bottom-left)
                cell_x = door_idx * (DOOR_WIDTH + DOOR_GAP)
                cell_y = shelf.level * LEVEL_HEIGHT

                # Draw shelf background
                shelf_rect = mpatches.FancyBboxPatch(
                    (cell_x, cell_y),
                    DOOR_WIDTH, LEVEL_HEIGHT,
                    boxstyle="round,pad=0.01",
                    facecolor='#f0f0f0',
                    edgecolor='#888888',
                    linewidth=1.0,
                    zorder=1
                )
                ax.add_patch(shelf_rect)

                # Charola label (small, top-right of cell)
                ax.text(
                    cell_x + DOOR_WIDTH - 0.02,
                    cell_y + LEVEL_HEIGHT - 0.04,
                    f"C{shelf.charola}",
                    fontsize=5.5, color='#aaaaaa',
                    va='top', ha='right', zorder=5
                )

                # Products
                prods = shelf.sorted_products(
                    reverse=self.is_right_to_left
                )
                if not prods:
                    continue

                # Available area for products inside cell
                prod_area_x = cell_x + SHELF_PAD
                prod_area_w = DOOR_WIDTH - 2 * SHELF_PAD
                prod_area_y = cell_y + SHELF_PAD
                prod_area_h = LEVEL_HEIGHT - 2 * SHELF_PAD

                # Calculate total product width (ANCHO * facings)
                total_product_w = sum(
                    p.width * p.facings for p in prods
                )

                if total_product_w <= 0:
                    continue

                # Scale factor: map cm to normalized units
                scale = prod_area_w / total_product_w

                # Draw each product
                cursor_x = prod_area_x
                for p in prods:
                    pw = p.width * p.facings * scale
                    color = _product_color(p.description)

                    # Product rectangle
                    prod_rect = mpatches.FancyBboxPatch(
                        (cursor_x, prod_area_y),
                        pw, prod_area_h,
                        boxstyle="round,pad=0.005",
                        facecolor=color,
                        edgecolor='#555555',
                        linewidth=0.5,
                        zorder=2
                    )
                    ax.add_patch(prod_rect)

                    # --- Product label (always vertical) ---
                    # Adaptive font size: scale with product width
                    base_font = min(6.5, max(4.0, pw * 55))

                    # Truncate label to fit available height
                    # More chars for taller cells, fewer for narrow products
                    max_chars = int(prod_area_h * 18)
                    label = p.description[:max_chars]
                    if p.facings > 1:
                        label = f"{p.facings}× {label}"

                    ax.text(
                        cursor_x + pw / 2,
                        prod_area_y + prod_area_h / 2,
                        label,
                        fontsize=base_font,
                        ha='center', va='center',
                        rotation=90,
                        color='#222222',
                        clip_on=True,
                        zorder=3
                    )

                    cursor_x += pw

        # ---- Door labels (top) ----
        for d in range(n_doors):
            cx = d * (DOOR_WIDTH + DOOR_GAP) + DOOR_WIDTH / 2
            ax.text(
                cx, total_h + 0.18,
                f"Puerta {d + 1}",
                fontsize=11, ha='center', va='bottom',
                fontweight='bold', color='#333333'
            )

        # ---- Level labels (left) ----
        for lv in range(n_levels):
            cy = lv * LEVEL_HEIGHT + LEVEL_HEIGHT / 2
            ax.text(
                -0.18, cy,
                f"Nivel {lv}",
                fontsize=9, ha='right', va='center',
                color='#666666'
            )

        # ---- Axis formatting ----
        ax.set_xlim(-0.35, total_w + 0.25)
        ax.set_ylim(-0.25, total_h + 0.55)
        ax.set_aspect('equal')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()


# =========================================================
# LOADER
# =========================================================

def load_planograms(csv_path, min_coverage=0.95):

    df = pd.read_csv(csv_path, encoding='latin-1')

    df.columns = [normalize_column(c) for c in df.columns]

    print("\nNORMALIZED COLUMNS:")
    for c in df.columns:
        print(f"  {c}")

    # -----------------------------------------------------
    # COLUMN MAPPING
    # -----------------------------------------------------

    SEGMENTO_COL = "SEGMENTO_ID"
    MUEBLE_COL = "MUEBLE_ID"
    PLANOGRUPO_COL = "PLANOGRUPO"

    # Sometimes these vary
    TAMANO_COL = "TAMANO_POST"
    DIRECCION_COL = "DIRECCION_LEGO_ID"
    CONJUNTO_COL = "CONJUNTO_ID"

    UPC_COL = "UPC_CVE"

    # Try multiple possibilities for description
    DESC_COL = (
        "ITEM_DESC"
        if "ITEM_DESC" in df.columns
        else "DESC1"
        if "DESC1" in df.columns
        else None
    )

    CHAROLA_COL = "CHAROLA"
    POS_COL = "UBICACION_BANDEJA"

    FRENTES_COL = "NUM_FRENTES"
    ANCHO_COL = "ANCHO"
    ALTO_COL = "ALTO"

    # Shelf spatial columns
    WIDTH_COL = "WIDTH"
    HEIGHT_COL = "HEIGHT"
    X_COL = "X"
    Y_COL = "Y"

    # -----------------------------------------------------
    # DETECT OPTIONAL COLUMNS
    # -----------------------------------------------------

    has_spatial = all(c in df.columns for c in [WIDTH_COL, HEIGHT_COL, X_COL, Y_COL])
    has_conjunto = CONJUNTO_COL in df.columns

    if not has_spatial:
        print("\nSpatial columns (X, Y, WIDTH, HEIGHT) not found "
              "— deriving layout from CHAROLA + TAMANO")

    # -----------------------------------------------------
    # VERIFY REQUIRED COLUMNS
    # -----------------------------------------------------

    required = [
        SEGMENTO_COL,
        MUEBLE_COL,
        PLANOGRUPO_COL,
        TAMANO_COL,
        DIRECCION_COL,
        UPC_COL,
        CHAROLA_COL,
        POS_COL,
        FRENTES_COL,
        ANCHO_COL,
        ALTO_COL,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise Exception(
            f"\nMissing columns:\n{missing}\n\n"
            f"Detected columns:\n{df.columns.tolist()}"
        )

    # -----------------------------------------------------
    # RECONSTRUCT VARIANTS (if CONJUNTO_ID missing)
    # -----------------------------------------------------

    if not has_conjunto:
        group_cols = [SEGMENTO_COL, MUEBLE_COL, PLANOGRUPO_COL,
                      TAMANO_COL, DIRECCION_COL]

        # Assign variant IDs by row occurrence order within positions
        df[CONJUNTO_COL] = df.groupby(
            group_cols + [CHAROLA_COL, POS_COL]
        ).cumcount().astype(str)

        # Compute coverage: fraction of charolas filled per variant
        variant_key = group_cols + [CONJUNTO_COL]
        config_charolas = df.groupby(group_cols)[CHAROLA_COL].transform('nunique')
        variant_charolas = df.groupby(variant_key)[CHAROLA_COL].transform('nunique')
        coverage = variant_charolas / config_charolas

        n_total = df.groupby(variant_key).ngroups
        df = df[coverage >= min_coverage].copy()
        n_kept = df.groupby(variant_key).ngroups if len(df) > 0 else 0

        print(f"\nVariant reconstruction: {n_total} total → "
              f"{n_kept} kept (>= {min_coverage:.0%} coverage)")

    # -----------------------------------------------------
    # GROUP ROWS BY PLANOGRAM KEY
    # -----------------------------------------------------

    grouped = defaultdict(list)

    for _, row in df.iterrows():

        key = (
            row[SEGMENTO_COL],
            row[MUEBLE_COL],
            row[PLANOGRUPO_COL],
            row[TAMANO_COL],
            row[DIRECCION_COL],
            row[CONJUNTO_COL]
        )

        grouped[key].append(row)

    # -----------------------------------------------------
    # BUILD PLANOGRAM OBJECTS
    # -----------------------------------------------------

    planograms = []

    for key, rows in grouped.items():

        planogram = Planogram(
            segmento_id=key[0],
            mueble_id=key[1],
            planogrupo=key[2],
            tamano=key[3],
            direccion=key[4],
            conjunto_id=key[5]
        )

        # --- Compute door/level indices for each charola ---
        if has_spatial:
            # Use X, Y, Width, Height columns directly
            charola_info = {}
            for row in rows:
                ch = int(row[CHAROLA_COL])
                if ch not in charola_info:
                    charola_info[ch] = {
                        'x': float(row[X_COL]),
                        'y': float(row[Y_COL]),
                        'width': float(row[WIDTH_COL]),
                        'height': float(row[HEIGHT_COL]),
                    }

            unique_xs = sorted(set(info['x'] for info in charola_info.values()))
            x_to_door = {x: i for i, x in enumerate(unique_xs)}
            unique_ys = sorted(set(info['y'] for info in charola_info.values()))
            y_to_level = {y: i for i, y in enumerate(unique_ys)}

            for ch, info in charola_info.items():
                shelf = Shelf(
                    charola=ch,
                    door=x_to_door[info['x']],
                    level=y_to_level[info['y']],
                    x=info['x'],
                    y=info['y'],
                    shelf_width=info['width'],
                    shelf_height=info['height'],
                )
                planogram.shelves[ch] = shelf
        else:
            # Derive spatial layout from CHAROLA + TAMANO
            charolas = sorted(set(int(row[CHAROLA_COL]) for row in rows))
            n_charolas = len(charolas)
            tamano_val = float(key[3])
            n_doors_total = max(1, math.ceil(tamano_val))
            n_levels = max(1, math.ceil(n_charolas / n_doors_total))

            DOOR_W = 55.0
            SHELF_H = 2.5
            Y_STD = [0, 42, 84, 115.5, 147, 175, 203, 231]

            for i, ch in enumerate(charolas):
                door = i // n_levels
                level = i % n_levels
                x = door * DOOR_W
                y = Y_STD[level] if level < len(Y_STD) else level * 28.0

                shelf = Shelf(
                    charola=ch,
                    door=door,
                    level=level,
                    x=x,
                    y=y,
                    shelf_width=DOOR_W,
                    shelf_height=SHELF_H,
                )
                planogram.shelves[ch] = shelf

        # --- Build ProductPlacement objects ---
        for row in rows:

            description = (
                str(row[DESC_COL])
                if DESC_COL and DESC_COL in row and pd.notna(row[DESC_COL])
                else str(row[UPC_COL])
            )

            product = ProductPlacement(
                upc=str(row[UPC_COL]),
                description=description,
                shelf=int(row[CHAROLA_COL]),
                position=int(row[POS_COL]),
                facings=int(row[FRENTES_COL]),
                width=float(row[ANCHO_COL]),
                height=float(row[ALTO_COL]),
            )

            # Place product into its Shelf object
            charola_num = product.shelf
            if charola_num in planogram.shelves:
                planogram.shelves[charola_num].products.append(product)

        planograms.append(planogram)

    print(f"\nLoaded {len(planograms)} planograms.")
    return planograms


# %%
# =========================================================
# USAGE
# =========================================================

if __name__ == "__main__":
    csv_path = "datos/ejemplo_planograma.csv"

    planograms = load_planograms(csv_path)

    # Print summary of all planograms
    print(f"\n{'='*100}")
    print(f"{'Segmento':<10} {'Mueble':<8} {'Grupo':<12} "
          f"{'Tamaño':<8} {'Dir':<6} {'Conjunto':<10} {'Puertas':<8} "
          f"{'Niveles':<8} {'Charolas':<10} {'Productos':<10}")
    print(f"{'='*100}")

    for p in planograms:
        n_prods = sum(len(s.products) for s in p.shelves.values())
        print(f"{p.segmento_id:<10} {p.mueble_id:<8} {p.planogrupo:<12} "
              f"{p.tamano:<8} {p.direccion:<6} {p.conjunto_id:<10} {p.get_n_doors():<8} "
              f"{p.get_n_levels():<8} {len(p.shelves):<10} {n_prods:<10}")

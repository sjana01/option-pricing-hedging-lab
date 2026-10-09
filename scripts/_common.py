"""Shared paths, plotting style and result-recording helpers for the pipeline scripts."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = ROOT / "output" / "figures"
TABLE_DIR = ROOT / "output" / "tables"
DATA_DIR = ROOT / "data"
RESULTS_JSON = TABLE_DIR / "key_results.json"
for d in (FIG_DIR, TABLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette, plus neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100,
})


def save_fig(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved figures/{name}")


def record(section, **values):
    """Merge key results into output/tables/key_results.json (read by the notebook and docs)."""
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    data.setdefault(section, {}).update(values)
    RESULTS_JSON.write_text(json.dumps(data, indent=2, default=float))

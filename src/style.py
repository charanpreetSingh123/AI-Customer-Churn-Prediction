import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

BG = "#0F172A"
PANEL = "#1E293B"
GRID = "#334155"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"

TEAL = "#2DD4BF"
SKY = "#38BDF8"
ORANGE = "#F97316"
VIOLET = "#A78BFA"
ROSE = "#FB7185"
AMBER = "#FBBF24"

RETAINED = SKY
CHURNED = ORANGE
STATUS_PALETTE = {"Retained": RETAINED, "Churned": CHURNED}

MODEL_COLORS = {
    "Logistic Regression": MUTED,
    "Decision Tree": VIOLET,
    "Random Forest": SKY,
    "XGBoost (tuned)": TEAL,
}
METRIC_COLORS = [SKY, ORANGE, VIOLET, TEAL]

SEQ_CMAP = LinearSegmentedColormap.from_list("seq", [PANEL, "#0E7490", TEAL])
DIV_CMAP = LinearSegmentedColormap.from_list("div", [SKY, PANEL, ORANGE])
SHAP_CMAP = LinearSegmentedColormap.from_list("shap", [SKY, VIOLET, ORANGE])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=[TEAL, ORANGE, SKY, VIOLET, ROSE, AMBER]),
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "grid.linewidth": 0.6,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT,
        "font.size": 10,
    })


def recolor_text(fig):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG)
    for t in fig.findobj(Text):
        t.set_color(TEXT)

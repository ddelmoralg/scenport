"""Figure style.

Okabe-Ito palette in a fixed order, assigned by series name rather than by plotting
order, plus a distinct line style and marker per series so nothing depends on colour
alone in greyscale print.
"""

from __future__ import annotations

from pathlib import Path

# Okabe-Ito, reordered so that no two adjacent entries are the deutan-confusable
# green/purple pair.
PALETTE = (
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#56B4E9",   # sky blue
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
)
REFERENCE = "#4D4D4D"      # benchmarks and reference lines, deliberately achromatic
GRID = "#D9D9D9"
INK = "#1A1A1A"
MUTED = "#666666"

LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "P")

#: Fixed assignment so a strategy keeps its identity across every figure.
FORECASTER_ORDER = (
    "rw", "rw_nodrift", "arima", "ets", "garch",
    "prophet", "prophet_rmse", "lgbm", "lstm", "deepar", "transformer",
)
ALLOCATOR_ORDER = (
    "equal", "mcap", "random", "invvol", "single",
    "minvar", "maxsharpe", "riskparity", "hrp", "cvar_lp",
    "ga", "de", "pso", "cmaes", "milp",
)

PRETTY = {
    "rw": "Historical sim.",
    "rw_nodrift": "Driftless RW",
    "arima": "ARIMA",
    "ets": "ETS",
    "garch": "AR-GARCH",
    "prophet": "Prophet (CRPS)",
    "prophet_rmse": "Prophet (RMSE)",
    "lgbm": "LightGBM",
    "lstm": "LSTM",
    "deepar": "DeepAR",
    "transformer": "Transformer",
    "equal": "1/N",
    "mcap": "Cap-weighted",
    "random": "Random",
    "invvol": "Inverse vol.",
    "single": "BTC only",
    "minvar": "Min. variance",
    "maxsharpe": "Max. Sharpe",
    "riskparity": "Risk parity",
    "hrp": "HRP",
    "cvar_lp": "CVaR LP",
    "ga": "GA",
    "de": "DE",
    "pso": "PSO",
    "cmaes": "CMA-ES",
    "milp": "MILP",
    "mase": "MASE",
    "crps": "CRPS",
    "pinball": "Pinball loss",
    "coverage_error": "Coverage error",
    "energy_score": "Energy score",
    "variogram_score": "Variogram score",
    "pit_ks": "PIT KS distance",
}


def label(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def style_for(name: str, order: tuple[str, ...]) -> dict:
    """Colour, line style and marker for one named series.

    Benchmarks that are not really competitors -- 1/N, the cap-weighted index --
    come back grey and dashed so they read as the backdrop rather than as another
    contender.
    """
    if name in ("equal", "mcap", "single"):
        return {"color": REFERENCE, "linestyle": "--", "marker": "", "linewidth": 1.4}
    idx = order.index(name) if name in order else len(order)
    return {
        "color": PALETTE[idx % len(PALETTE)],
        "linestyle": LINESTYLES[idx % len(LINESTYLES)],
        "marker": MARKERS[idx % len(MARKERS)],
        "linewidth": 1.6,
        "markersize": 4,
        "markevery": 0.12,
    }


def use_paper_style() -> None:
    """Set the rcParams. Call once, before building any figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 3.6),          # fits a single column with room to spare
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "text.color": INK,
        }
    )


def save(fig, directory: Path, name: str) -> Path:
    """Write a vector PDF for the manuscript and a PNG to eyeball."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.pdf"
    fig.savefig(target)
    fig.savefig(directory / f"{name}.png")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return target


def annotate_last(ax, x, y, text: str, colour: str) -> None:
    """Direct label at the right end of a line, so the legend is a backup."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=7.5,
        color=colour,
        clip_on=False,
    )

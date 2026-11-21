"""
app.py - Projet 19 : Smart datacenter cooling (Streamlit interface).

Design : strictly black & white, Inter Bold, no colors. Information levels are encoded with
         white opacity, line styles and marker shapes instead of hues.

Run locally :   streamlit run app.py
Deploy      :   GitHub repo + https://share.streamlit.io

Inputs (folder ./outputs, or the HVAC_DATA environment variable):
    hvac_simulation.csv  rf_metrics.json  rf_hvac.joblib     <- rf_hvac.py
    lstm_predictions.csv                                      <- LSTM notebook (optional)
Optional: ./fonts/Inter-Regular.otf and Inter-Bold.otf (used by the charts).
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager
from matplotlib.figure import Figure

# =============================================================================
# Constants
# =============================================================================
DATA_DIR = Path(os.environ.get("HVAC_DATA", "outputs"))
FONT_DIR = Path(__file__).resolve().parent / "fonts"

INK = "#000000"
LEVEL_OPACITY = (0.20, 0.55, 1.00)           # ECO / NORMAL / BOOST -> ink intensity replaces color
SHARE_COLORS = ("#9ec5ff", "#2563eb", INK)    # donut only: ECO / NORMAL / BOOST, darker = more cooling
PROBA_COLS = ["p_eco", "p_normal", "p_boost"]
BOOST = 2


def ink(alpha: float) -> tuple[float, float, float, float]:
    """Black with a given opacity (the only 'color' used by the charts, besides the donut blues)."""
    return (0.0, 0.0, 0.0, alpha)


# =============================================================================
# Style (CSS + matplotlib)
# =============================================================================
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;800&display=swap');

:root { --bg:#fff; --ink:#000; --surface:#fff; --line:rgba(0,0,0,.14); --muted:rgba(0,0,0,.58); --radius:16px; --box-h:190px; }

/* ---- typography : Inter Bold everywhere (icons are left untouched) ---- */
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp button, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp [data-testid="stWidgetLabel"] *, .stApp [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stCaptionContainer"] *, .stApp [data-testid^="stSlider"] *, .stApp [data-testid="stSelectSlider"] *,
.stApp [data-testid="stSidebar"] * :not([data-testid="stIconMaterial"]) {
    font-family: "Inter", system-ui, sans-serif !important; font-weight: 700 !important;
    -webkit-font-smoothing: antialiased; }

/* ---- layout ---- */
.stApp { background: var(--bg); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem; }
section[data-testid="stSidebar"] { background: var(--bg); border-right: 1px solid var(--line); }

/* ---- components ---- */
.label { font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); }
.hero h1 { font-size:3.2rem; font-weight:800 !important; letter-spacing:-.045em; line-height:1.05; margin:0; color:var(--ink); }
.hero p  { color:var(--muted); font-size:1.15rem; margin:.6rem 0 1.2rem 0; }
.flow span { display:inline-block; padding:.35rem .9rem; border:1px solid var(--line); border-radius:999px; font-size:.82rem; color:var(--ink); }
.flow i { font-style:normal; color:var(--muted); margin:0 .5rem; }

/* every box (KPI or card) has exactly the same height */
.kpi, .card { box-sizing:border-box; height:var(--box-h); padding:18px 20px; overflow:hidden; background:var(--surface);
              border:1px solid var(--line); border-radius:var(--radius);
              display:flex; flex-direction:column; align-items:flex-start; justify-content:space-between; }
.card.auto { height:auto; min-height:var(--box-h); justify-content:flex-start; gap:.6rem; }
.kpi .value, .card .big { font-size:2.2rem; font-weight:800 !important; letter-spacing:-.04em; line-height:1.1; color:var(--ink); margin:0; }
.kpi .unit, .card .big small { font-size:.95rem; color:var(--muted); margin-left:.3rem; letter-spacing:0; }
.kpi .note, .card .note { font-size:.82rem; color:var(--muted); line-height:1.3; }
.kpi .note { max-width:100%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

.bar { align-self:stretch; margin:0; }
.bar .row { display:flex; justify-content:space-between; font-size:.82rem; color:var(--ink); margin-bottom:.3rem; }
.bar .track { height:6px; border-radius:99px; background:rgba(0,0,0,.10); overflow:hidden; }
.bar .fill { height:100%; background:var(--ink); border-radius:99px; }

.badge { display:inline-block; padding:.4rem 1.3rem; border-radius:12px; border:1.5px solid var(--ink); font-size:1.8rem;
         font-weight:800 !important; letter-spacing:.02em; }
.l0 { background:transparent; color:var(--ink); }
.l1 { background:rgba(0,0,0,.14); color:var(--ink); }
.l2 { background:var(--ink); color:#fff; }
.swatch { display:inline-block; width:14px; height:14px; border-radius:4px; border:1px solid var(--ink); margin:0 .4rem -2px 1rem; }
.legend { font-size:.82rem; color:var(--muted); margin-top:.4rem; }
.stamp { display:inline-block; padding:.35rem 1rem; border:1px solid var(--line); border-radius:10px; font-size:.9rem; margin-bottom:1rem; }

/* ---- tabs : text only, spaced out, active tab = black pill, no underline (covers several Streamlit versions) ---- */
[role="tablist"] { gap:14px !important; background:transparent !important; padding:0 !important; border:0 !important; box-shadow:none !important; }
[role="tablist"] > :not([role="tab"]):not(:has([role="tab"])), [role="tablist"] [aria-hidden="true"],
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {
        display:none !important; height:0 !important; border:0 !important; background:transparent !important; box-shadow:none !important; }
[role="tablist"]::before, [role="tablist"]::after, [role="tab"]::before, [role="tab"]::after { display:none !important; }
[role="tab"], button[data-baseweb="tab"], [data-testid="stTab"] { height:42px !important; padding:0 24px !important; margin:0 !important;
        border-radius:12px !important; border:1px solid transparent !important; background:transparent !important;
        box-shadow:none !important; color:var(--muted) !important; transition:all .15s ease; }
[role="tab"] p, [data-testid="stTab"] p { color:inherit !important; font-size:.95rem !important; }
[role="tab"]:hover, [data-testid="stTab"]:hover { color:var(--ink) !important; border-color:var(--line) !important; }
[role="tab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"], [role="tab"][data-state="active"] {
        background:var(--ink) !important; color:#fff !important; border-color:var(--ink) !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"], .stTabs [data-baseweb="tab-panel"] { padding-top:1.6rem; }

/* ---- widgets ---- */
.stButton > button { background:transparent; color:var(--ink); border:1px solid var(--line); border-radius:12px; height:42px; }
.stButton > button:hover { background:var(--ink); color:#fff; border-color:var(--ink); }
.stButton > button p { color:inherit !important; }
[data-baseweb="input"], [data-baseweb="select"] > div { background:var(--surface) !important; border-radius:10px !important; }
[data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
div[data-testid="stImage"] { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); padding:14px 16px; }
div[data-testid="stCaptionContainer"] { margin-top:.8rem; }
</style>
"""


def configure_matplotlib() -> None:
    """Transparent monochrome charts using the bundled Inter font when available."""
    for path in FONT_DIR.glob("*.otf"):
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": ["Inter", "DejaVu Sans"], "font.weight": "bold", "font.size": 10,
        "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.titlecolor": INK,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": ink(.7), "ytick.color": ink(.7),
        "axes.edgecolor": ink(.25), "axes.grid": True, "grid.color": INK, "grid.alpha": .10,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


# =============================================================================
# Data & domain logic
# =============================================================================
@dataclass(frozen=True)
class Settings:
    """Values chosen in the sidebar."""
    test_only: bool
    eps: float
    p_rated: float
    emission_factor: float
    price: float
    power_frac: tuple[float, float, float]


class Kpis(NamedTuple):
    energy_kwh: float
    saved_kwh: float
    saved_pct: float
    co2_kg: float
    cost: float
    undercool_pct: float
    reactive_kwh: float


@dataclass
class Context:
    """Everything the tabs need."""
    sim: pd.DataFrame
    view: pd.DataFrame
    meta: dict
    lstm: pd.DataFrame | None
    cfg: Settings

    @property
    def levels(self) -> list[str]:
        return self.meta["levels"]

    @property
    def t_warm(self) -> float:
        return self.meta["thresholds_C"]["warm"]

    @property
    def t_hot(self) -> float:
        return self.meta["thresholds_C"]["hotspot"]

    @property
    def horizon(self) -> int:
        return self.meta["horizon_min"]


@st.cache_data
def load_data(folder: str) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    path = Path(folder)
    sim = pd.read_csv(path / "hvac_simulation.csv", index_col=0, parse_dates=True)
    meta = json.loads((path / "rf_metrics.json").read_text())
    lstm = None
    for candidate in (path / "lstm_predictions.csv", Path("lstm_predictions.csv")):
        if candidate.exists():
            lstm = pd.read_csv(candidate, index_col=0, parse_dates=True)
            break
    return sim, meta, lstm


@st.cache_resource
def load_model(folder: str):
    """Trained Random Forests, or an error message (e.g. scikit-learn version mismatch)."""
    try:
        import joblib
        return joblib.load(Path(folder) / "rf_hvac.joblib")
    except Exception as error:
        return str(error)


def decide(proba: np.ndarray, eps: float) -> np.ndarray:
    """Cheapest cooling level whose probability of being insufficient is <= eps."""
    tail = 1.0 - np.cumsum(proba, axis=1)
    return (tail <= eps + 1e-9).argmax(axis=1)


def energy_kwh(levels, cfg: Settings) -> float:
    """Energy for one-minute samples at the given cooling levels."""
    power_kw = cfg.p_rated * np.asarray(cfg.power_frac)[np.asarray(levels)]
    return float(power_kw.sum() / 60)


def build_view(sim: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    view = (sim[sim.split == "test"] if cfg.test_only else sim).copy()
    view["action"] = decide(view[PROBA_COLS].to_numpy(), cfg.eps)
    return view


def compute_kpis(ctx: Context) -> Kpis:
    view, cfg = ctx.view, ctx.cfg
    energy = energy_kwh(view.action, cfg)
    baseline = energy_kwh(np.full(len(view), BOOST), cfg)
    reactive = energy_kwh(np.digitize(view.hot_max, [ctx.t_warm, ctx.t_hot]), cfg)

    labelled = view.required.notna().to_numpy()
    missing = view.required.to_numpy()[labelled] - view.action.to_numpy()[labelled] > 0
    undercool = 100 * float(missing.mean()) if labelled.any() else float("nan")

    saved = baseline - energy
    return Kpis(energy, saved, 100 * saved / baseline if baseline else 0.0,
                saved * cfg.emission_factor, saved * cfg.price, undercool, reactive)


# =============================================================================
# HTML components
# =============================================================================
def kpi_html(label: str, value: str, unit: str = "", note: str = "") -> str:
    unit_html = f'<span class="unit">{unit}</span>' if unit else ""
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}{unit_html}</div><div class="note">{note}</div></div>')


def card_html(title: str, body: str, auto: bool = False) -> str:
    """Info box. All boxes share one fixed height; text-heavy boxes (auto=True) may grow."""
    css = "card auto" if auto else "card"
    return f'<div class="{css}"><div class="label">{title}</div>{body}</div>'


def bar_html(label: str, probability: float) -> str:
    p = float(np.clip(probability, 0, 1))
    return (f'<div class="bar"><div class="row"><span>{label}</span><span>{p:.0%}</span></div>'
            f'<div class="track"><div class="fill" style="width:{p * 100:.0f}%"></div></div></div>')


def badge_html(level: int, name: str) -> str:
    return f'<div class="badge l{level}">{name}</div>'


def big_html(value: str, unit: str) -> str:
    return f'<div class="big">{value}<small>{unit}</small></div>'


def render_html(html: str, container=st) -> None:
    container.markdown(html, unsafe_allow_html=True)


def show(fig: Figure, dpi: int = 170) -> None:
    """Render a figure as a transparent PNG at its native proportions."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    st.image(buffer.getvalue())


# =============================================================================
# Charts (black & white only)
# =============================================================================
def _time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def chart_overview(ctx: Context) -> Figure:
    view = ctx.view
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True,
                                    gridspec_kw={"height_ratios": [7, 0.8], "hspace": 0.08})
    top = float(view.hot_max.max()) + 1
    ax.axhspan(ctx.t_warm, ctx.t_hot, color=ink(.06), lw=0)
    ax.axhspan(ctx.t_hot, top, color=ink(.12), lw=0)
    ax.plot(view.index, view.hot_max, color=INK, lw=2, label="Hot side (max)")
    ax.plot(view.index, view.cold_mean, color=ink(.55), lw=1.6, label="Cold side (mean)")
    ax.axhline(ctx.t_warm, color=ink(.55), ls=":", lw=1.2, label=f"Warm {ctx.t_warm:.1f} °C")
    ax.axhline(ctx.t_hot, color=INK, ls="--", lw=1.2, label=f"Hotspot {ctx.t_hot:.1f} °C")
    ax.set_ylim(float(view.cold_mean.min()) - 1.5, top)
    ax.set_ylabel("Temperature (°C)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.14), fontsize=9)

    strip.vlines(view.index, 0, 1, colors=[ink(LEVEL_OPACITY[a]) for a in view.action], lw=1.6)
    strip.set_yticks([])
    strip.set_ylabel("HVAC", rotation=0, labelpad=26, va="center")
    strip.grid(False)
    _time_axis(strip)
    return fig


def chart_energy(ctx: Context) -> Figure:
    view, cfg = ctx.view, ctx.cfg
    boost = np.cumsum(np.full(len(view), cfg.p_rated * cfg.power_frac[BOOST] / 60))
    rf = np.cumsum(cfg.p_rated * np.asarray(cfg.power_frac)[view.action.to_numpy()] / 60)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(view.index, boost, color=ink(.55), ls="--", lw=1.8, label="Always BOOST")
    ax.plot(view.index, rf, color=INK, lw=2.2, label="RF controller")
    ax.fill_between(view.index, rf, boost, color=ink(.12), label="Energy saved")
    ax.set_title("Cumulative energy (kWh)", loc="left")
    ax.legend(loc="upper left")
    _time_axis(ax)
    return fig


def chart_share(ctx: Context) -> Figure:
    share = ctx.view.action.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.grid(False)
    wedges, _ = ax.pie(share + 1e-9, colors=list(SHARE_COLORS), startangle=90, counterclock=False,
                       wedgeprops=dict(width=.38, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{100 * (1 - share[BOOST]):.0f}%\nbelow BOOST", ha="center", va="center", fontsize=12)
    ax.set_title("Time in each level", loc="center")
    ax.legend(wedges, [f"{name} {100 * s:.0f}%" for name, s in zip(ctx.levels, share)], loc="lower center",
              bbox_to_anchor=(.5, -.16), ncol=3, fontsize=8, handlelength=1)
    return fig


def chart_window(ctx: Context, index: int) -> Figure:
    window = ctx.view.iloc[max(0, index - 60): index + 1]
    level = int(ctx.view.action.iloc[index])
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axhspan(ctx.t_warm, ctx.t_hot, color=ink(.06), lw=0)
    ax.axhline(ctx.t_hot, color=INK, ls="--", lw=1)
    ax.plot(window.index, window.hot_max, color=INK, lw=2)
    ax.plot(window.index, window.cold_mean, color=ink(.55), lw=1.6)
    ax.scatter([window.index[-1]], [window.hot_max.iloc[-1]], s=80, color=ink(max(LEVEL_OPACITY[level], .35)),
               edgecolor=INK, linewidth=1.5, zorder=5)
    ax.set_title("Last 60 minutes", loc="left")
    ax.set_ylabel("°C")
    _time_axis(ax)
    return fig


def chart_importance(meta: dict) -> Figure:
    importance = pd.Series(meta["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.barh(importance.index, importance.values, color=INK, height=.6)
    ax.grid(axis="y", alpha=0)
    ax.set_title("Feature importance", loc="left")
    return fig


def chart_tradeoff(meta: dict) -> Figure:
    table = pd.DataFrame(meta["test_table"]).T
    rf = table[table.index.str.startswith("RF controller")]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(rf.undercool_pct, rf.saving_vs_boost_pct, "-o", color=INK, lw=2, ms=6, label="RF controller (ε sweep)")
    for name, marker in [("Always NORMAL", "s"), ("Reactive rule (no ML)", "^")]:
        ax.scatter(table.loc[name, "undercool_pct"], table.loc[name, "saving_vs_boost_pct"], marker=marker, s=90,
                   facecolors="none", edgecolors=INK, linewidths=1.8, label=name, zorder=3)
    ax.set_xlabel("Minutes with too little cooling (%)")
    ax.set_ylabel("Saving vs BOOST (%)")
    ax.set_title("Energy saving vs thermal risk", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def chart_forecast(ctx: Context) -> Figure:
    test = ctx.sim[ctx.sim.split == "test"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(test.index, test.hot_max, color=INK, lw=2, label="Actual")
    ax.plot(test.index, test.pred_hot_max_future.shift(ctx.horizon), color=ink(.6), ls="--", lw=1.6,
            label=f"RF forecast (made {ctx.horizon} min earlier)")
    ax.set_title("Hot-side temperature forecast (test)", loc="left")
    ax.set_ylabel("°C")
    ax.legend(loc="upper right")
    _time_axis(ax)
    return fig


def chart_lstm(lstm: pd.DataFrame) -> Figure:
    window = lstm.iloc[:576]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(window.index, window.actual_load, color=ink(.55), lw=1.5, label="Actual")
    ax.plot(window.index, window.predicted_load, color=INK, lw=1.6, label="LSTM forecast")
    ax.set_title("LSTM · CPU load forecast (2 days)", loc="left")
    ax.set_ylabel("CPU load (%)")
    ax.legend(loc="upper right")
    return fig


# =============================================================================
# Page sections
# =============================================================================
def render_sidebar(meta: dict) -> Settings:
    levels = meta["levels"]
    render_html('<div style="font-size:1.5rem;font-weight:800;letter-spacing:-.03em">Smart Cooling</div>'
                '<div class="label" style="margin:.2rem 0 1.4rem 0">Projet 19 · Datacenter energy efficiency</div>',
                st.sidebar)
    period = st.sidebar.radio("Period", ["Test (unseen data)", "Whole recording"])
    eps = st.sidebar.select_slider("Safety level ε", options=meta["eps_grid"], value=meta["eps"],
                                   help="ε = accepted risk of too little cooling. Small ε is safer but uses more energy.")
    with st.sidebar.expander("Energy model (assumptions)"):
        p_rated = st.number_input("Rated cooling power (kW)", 1.0, 500.0, float(meta["p_rated_kw"]))
        factor = st.number_input("Emission factor (kgCO₂/kWh)", 0.0, 2.0, float(meta["ef_kg_per_kwh"]), 0.01)
        price = st.number_input("Electricity price (per kWh)", 0.0, 10.0, float(meta["price_per_kwh"]), 0.01)
        frac = tuple(st.slider(f"{levels[i]} power (% of rated)", 10, 100, int(100 * meta["power_frac"][i]), 5) / 100
                     for i in range(3))
    return Settings(period.startswith("Test"), eps, p_rated, factor, price, frac)


def render_header() -> None:
    render_html("""
<div class="hero">
  <h1>Smart Datacenter Cooling</h1>
  <p>Predict the heat. Cool only what is needed.</p>
  <div class="flow"><span>CNN · Thermal state</span><i>→</i><span>LSTM · CPU load</span><i>→</i><span>Random Forest · HVAC decision</span></div>
</div>""")
    st.write("")


def render_kpis(ctx: Context) -> None:
    k, cfg = compute_kpis(ctx), ctx.cfg
    direction = "below" if k.saved_pct >= 0 else "above"
    columns = st.columns(5)
    cards = [
        kpi_html("Energy used", f"{k.energy_kwh:,.1f}", "kWh", f"{abs(k.saved_pct):.1f}% {direction} BOOST"),
        kpi_html("Energy saved", f"{k.saved_kwh:,.1f}", "kWh", f"reactive rule {k.reactive_kwh:,.1f} kWh"),
        kpi_html("CO₂ avoided", f"{k.co2_kg:,.1f}", "kg", f"at {cfg.emission_factor:.2f} kg/kWh"),
        kpi_html("Cost saved", f"{k.cost:,.2f}", "", f"price {cfg.price:.2f} / kWh"),
        kpi_html("Too little cooling", f"{k.undercool_pct:.1f}", "%", "of minutes · lower is safer"),
    ]
    for column, html in zip(columns, cards):
        render_html(html, column)
    st.caption(f"{len(ctx.view)} minutes replayed · energy figures use the assumptions in the sidebar")


def render_dashboard(ctx: Context) -> None:
    show(chart_overview(ctx))
    swatches = "".join(f'<span class="swatch l{i}"></span>{name}' for i, name in enumerate(ctx.levels))
    render_html(f'<div class="legend">Cooling level chosen every minute:{swatches}</div>')
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        show(chart_energy(ctx))
    with right:
        show(chart_share(ctx))


def _move_step(delta: int, last: int) -> None:
    st.session_state.step = int(np.clip(st.session_state.step + delta, 0, last))


def render_replay(ctx: Context) -> None:
    view, n = ctx.view, len(ctx.view)
    st.write("Move through time and see what each part of the project contributes at that moment.")
    st.session_state.setdefault("step", 0)
    st.session_state.step = min(st.session_state.step, n - 1)

    buttons = st.columns([1, 1, 1, 1, 6])
    for column, (label, delta) in zip(buttons, [("−10", -10), ("−1", -1), ("+1", 1), ("+10", 10)]):
        column.button(label, on_click=_move_step, args=(delta, n - 1))
    step = st.slider("Time step (minutes)", 0, n - 1, key="step")

    row, level = view.iloc[step], int(view.action.iloc[step])
    render_html(f'<div class="stamp">{view.index[step]:%d %b %Y · %H:%M}</div>')

    cnn, lstm, rf, hvac = st.columns(4)
    render_html(card_html("CNN · Thermal state",
                          big_html(f"{row.hot_max:.1f}", " °C hot side")
                          + bar_html("P(hotspot)", row.p_hotspot) + bar_html("P(warm)", row.p_warm)), cnn)
    render_html(card_html("LSTM · CPU load",
                          big_html(f"{row.load:.2f}", " %")
                          + '<div class="note">forecast for this time of day</div>'
                          + f'<div class="note" style="margin-top:.7rem;color:var(--ink)">{row.load_trend:+.2f} pts in {ctx.horizon} min</div>'), lstm)
    actual = "" if np.isnan(row.hot_max_future) else f'<div class="note">actual later: {row.hot_max_future:.1f} °C</div>'
    render_html(card_html(f"Random Forest · in {ctx.horizon} min",
                          big_html(f"{row.pred_hot_max_future:.1f}", " °C predicted") + actual
                          + bar_html("P(BOOST needed)", row.p_boost)), rf)

    verdict = ""
    if not np.isnan(row.required):
        ok = level >= int(row.required)
        verdict = (f'<div class="note" style="color:var(--ink);margin-top:.5rem">{"Cooling sufficient" if ok else "Under-cooling · needed " + ctx.levels[int(row.required)]}</div>')
    kw, kw_boost = ctx.cfg.p_rated * ctx.cfg.power_frac[level], ctx.cfg.p_rated * ctx.cfg.power_frac[BOOST]
    render_html(card_html("HVAC decision",
                          badge_html(level, ctx.levels[level])
                          + f'<div class="note" style="margin-top:.8rem">{kw:.1f} kW instead of {kw_boost:.1f} kW</div>' + verdict), hvac)
    st.write("")
    show(chart_window(ctx, step))


def _what_if_inputs(ctx: Context, base: pd.Series) -> dict:
    sim, horizon = ctx.sim, ctx.horizon
    left, middle, right = st.columns(3)
    hot = left.slider("Hot-side max (°C)", float(sim.hot_max.min()) - 2, float(sim.hot_max.max()) + 3, float(base.hot_max), .1)
    cold = left.slider("Cold-side mean (°C)", float(sim.cold_mean.min()) - 2, float(sim.cold_mean.max()) + 3, float(base.cold_mean), .1)
    trend = middle.slider("Temperature change, last 5 min (°C)", -2.0, 2.0, float(base.d5), .05)
    load = middle.slider("Forecast CPU load (%)", 0.0, float(sim.load.max()) * 1.5, float(base.load), .1)
    load_trend = middle.slider(f"Load change expected in {horizon} min", -2.0, 2.0, float(base.load_trend), .05)
    p_hot = right.slider("CNN P(hotspot)", 0.0, 1.0, float(base.p_hotspot), .01)
    p_warm = right.slider("CNN P(warm)", 0.0, 1.0 - p_hot, min(float(base.p_warm), 1.0 - p_hot), .01)
    return dict(hot_max=hot, cold_mean=cold, d5=trend, load=load, load_trend=load_trend, p_hotspot=p_hot, p_warm=p_warm)


def render_what_if(ctx: Context) -> None:
    st.write("Describe a situation and ask the trained Random Forest what it would do.")
    model = load_model(str(DATA_DIR))
    if isinstance(model, str):
        st.warning(f"Model file could not be loaded ({model}). Re-run rf_hvac.py with the same scikit-learn "
                   f"version as the app (trained with {ctx.meta['sklearn_version']}).")
        return

    base = ctx.sim[model["features"]].median()
    values = _what_if_inputs(ctx, base)
    row = base.copy()
    for key, value in values.items():
        row[key] = value
    row["hot_mean"] = base.hot_mean + (values["hot_max"] - base.hot_max) * 0.8
    row["delta_T"] = row.hot_mean - values["cold_mean"]
    row["d1"] = values["d5"] / 5
    features = pd.DataFrame([row[model["features"]]])

    proba = np.zeros((1, 3))
    proba[:, model["clf"].classes_.astype(int)] = model["clf"].predict_proba(features)
    level = int(decide(proba, ctx.cfg.eps)[0])
    forecast = float(values["hot_max"] + model["reg"].predict(features)[0])

    st.write("")
    left, right = st.columns(2)
    render_html(card_html("Decision", badge_html(level, ctx.levels[level])
                          + f'<div class="note" style="margin-top:.9rem">Predicted hot-side max in {ctx.horizon} min</div>'
                          + big_html(f"{forecast:.1f}", " °C")), left)
    render_html(card_html("Probability of the level required",
                          "".join(bar_html(name, proba[0, i]) for i, name in enumerate(ctx.levels))), right)
    st.caption("Features not shown here (humidity, short-term std) are kept at their median value.")


def render_models(ctx: Context) -> None:
    meta, forecast = ctx.meta, ctx.meta["forecast_metrics"]
    columns = st.columns(3)
    render_html(kpi_html("Forecast error (RF)", f"{forecast['rf_mae_C']:.2f}", "°C", f"+{ctx.horizon} min · MAE"), columns[0])
    render_html(kpi_html("'Stays the same' error", f"{forecast['persistence_mae_C']:.2f}", "°C", "baseline MAE"), columns[1])
    render_html(kpi_html("R² of RF forecast", f"{forecast['rf_r2']:.2f}", "", f"baseline {forecast['persistence_r2']:.2f}"), columns[2])

    st.write("")
    st.markdown("##### Energy and thermal risk on unseen test data")
    table = pd.DataFrame(meta["test_table"]).T.rename(columns={
        "energy_kWh": "Energy (kWh)", "co2_kg": "CO₂ (kg)", "cost": "Cost", "undercool_pct": "Under-cooling (%)",
        "severe_undercool_pct": "Severe (%)", "saving_vs_boost_pct": "Saving vs BOOST (%)"})
    st.dataframe(table.round(2))

    left, right = st.columns(2)
    with left:
        show(chart_tradeoff(meta))
    with right:
        show(chart_importance(meta))
    st.markdown("##### What do CNN and LSTM bring? (ablation)")
    st.dataframe(pd.DataFrame(meta["ablation"]).T.round(3))
    show(chart_forecast(ctx))
    if ctx.lstm is not None:
        show(chart_lstm(ctx.lstm))


def render_about(ctx: Context) -> None:
    left, right = st.columns(2)
    render_html(card_html("How it works", f"""
<div class="note" style="font-size:.95rem;line-height:1.75;color:var(--ink)">
CNN classifies the thermal map of the racks (Normal / Warm / Hotspot).<br>
LSTM forecasts the total CPU load of the servers.<br>
Random Forest predicts the cooling level needed in {ctx.horizon} minutes and picks the cheapest level that stays safe (risk ≤ ε).</div>""", auto=True), left)
    render_html(card_html("Honest limits", """
<div class="note" style="font-size:.95rem;line-height:1.75;color:var(--ink)">
Workload (2013) and sensors (2018) are linked by time of day only.<br>
No HVAC actions in the data, so the replay is open-loop.<br>
Cooling power per level, CO₂ factor and price are assumptions.<br>
The recording is short (about 30 h): confirm on longer data.</div>""", auto=True), right)
    st.write("")
    st.json({key: ctx.meta[key] for key in ["horizon_min", "eps", "thresholds_C", "sklearn_version", "test_period"]},
            expanded=False)


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Smart Datacenter Cooling", layout="wide")
    render_html(STYLE)
    configure_matplotlib()

    try:
        sim, meta, lstm = load_data(str(DATA_DIR))
    except Exception as error:
        st.error(f"Could not load the result files from '{DATA_DIR}': {error}\n\n"
                 "Run rf_hvac.py first and copy its outputs into the 'outputs' folder.")
        st.stop()

    cfg = render_sidebar(meta)
    ctx = Context(sim, build_view(sim, cfg), meta, lstm, cfg)

    render_header()
    render_kpis(ctx)
    st.write("")

    tabs = st.tabs(["Dashboard", "Replay", "What-if", "Models", "About"])
    for tab, render in zip(tabs, [render_dashboard, render_replay, render_what_if, render_models, render_about]):
        with tab:
            render(ctx)


main()

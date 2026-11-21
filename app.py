"""
app.py - Projet 19 : interface (Streamlit) for the smart datacenter cooling demo.

Run locally :   streamlit run app.py
Deploy      :   GitHub repo + https://share.streamlit.io  (free, public link)

Reads the files produced by the 3 parts of the project (put them in ./outputs or set HVAC_DATA):
    hvac_simulation.csv   rf_metrics.json   rf_hvac.joblib      <- rf_hvac.py
    lstm_predictions.csv                                         <- LSTM notebook   (optional)
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Datacenter HVAC - Projet 19", page_icon="❄️", layout="wide")
DATA = Path(os.environ.get("HVAC_DATA", "outputs"))
COLORS = ["#2a7fba", "#e8a317", "#d9412b"]            # ECO / NORMAL / BOOST


# ----------------------------------------------------------------- helpers
@st.cache_data
def load_data(folder: str):
    d = Path(folder)
    sim = pd.read_csv(d / "hvac_simulation.csv", index_col=0, parse_dates=True)
    meta = json.load(open(d / "rf_metrics.json"))
    lstm = None
    for cand in [d / "lstm_predictions.csv", Path("lstm_predictions.csv")]:
        if cand.exists():
            lstm = pd.read_csv(cand, index_col=0, parse_dates=True)
            break
    return sim, meta, lstm

@st.cache_resource
def load_model(folder: str):
    try:
        import joblib
        return joblib.load(Path(folder) / "rf_hvac.joblib")
    except Exception as e:                              # missing file / sklearn version mismatch
        return str(e)

def decide(P, eps):
    """Cheapest cooling level whose probability of being insufficient is <= eps."""
    tail = 1.0 - np.cumsum(P, axis=1)
    return (tail <= eps + 1e-9).argmax(axis=1)

def energy(act, frac, p_rated):
    return float((p_rated * np.asarray(frac)[np.asarray(act)]).sum() / 60)   # 1 sample = 1 minute -> kWh


# ----------------------------------------------------------------- data
try:
    sim, meta, lstm = load_data(str(DATA))
except Exception as e:
    st.error(f"Could not load the result files from '{DATA}': {e}\n\n"
             "Run rf_hvac.py first and copy its outputs into the 'outputs' folder.")
    st.stop()

LEVELS = meta["levels"]
T_WARM, T_HOT = meta["thresholds_C"]["warm"], meta["thresholds_C"]["hotspot"]
H = meta["horizon_min"]

# ----------------------------------------------------------------- sidebar
st.sidebar.title("❄️ Smart cooling")
st.sidebar.caption("CNN → thermal state · LSTM → CPU load · Random Forest → HVAC decision")
period = st.sidebar.radio("Period", ["Test (unseen data)", "Whole recording"], index=0)
eps = st.sidebar.select_slider("Safety level (ε = accepted risk of too little cooling)",
                               options=meta["eps_grid"], value=meta["eps"],
                               help="Small ε = very safe but more energy. Large ε = more savings but more risk.")
st.sidebar.markdown("**Energy model (assumptions)**")
p_rated = st.sidebar.number_input("Rated cooling power (kW)", 1.0, 500.0, float(meta["p_rated_kw"]))
ef = st.sidebar.number_input("Emission factor (kgCO₂/kWh)", 0.0, 2.0, float(meta["ef_kg_per_kwh"]), 0.01)
price = st.sidebar.number_input("Electricity price (per kWh)", 0.0, 10.0, float(meta["price_per_kwh"]), 0.01)
frac = [st.sidebar.slider(f"{LEVELS[i]} power (% of rated)", 10, 100, int(100 * meta["power_frac"][i]), 5) / 100
        for i in range(3)]

view = sim[sim.split == "test"] if period.startswith("Test") else sim
view = view.copy()
P = view[["p_eco", "p_normal", "p_boost"]].to_numpy()
view["action"] = decide(P, eps)
req = view["required"].to_numpy()
lab = ~np.isnan(req)

# ----------------------------------------------------------------- KPIs
n = len(view)
e_rf, e_boost = energy(view.action, frac, p_rated), energy(np.full(n, 2), frac, p_rated)
e_react = energy(np.digitize(view.hot_max, [T_WARM, T_HOT]), frac, p_rated)
short = (req[lab] - view.action.to_numpy()[lab]) > 0
under = 100 * float(short.mean()) if lab.any() else float("nan")
saved = e_boost - e_rf

st.title("Datacenter energy efficiency – smart HVAC control")
c = st.columns(5)
c[0].metric("Energy (RF controller)", f"{e_rf:,.1f} kWh", f"{-100 * saved / e_boost:.1f}% vs always BOOST", delta_color="inverse")
c[1].metric("Energy saved", f"{saved:,.1f} kWh")
c[2].metric("CO₂ avoided", f"{saved * ef:,.1f} kg")
c[3].metric("Cost saved", f"{saved * price:,.2f}")
c[4].metric("Minutes with too little cooling", f"{under:.1f}%")
st.caption(f"{n} minutes replayed · energy figures use the assumptions in the sidebar · "
           f"reactive rule without ML would use {e_react:,.1f} kWh")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "▶ Replay", "🧪 What-if", "📈 Models", "ℹ️ About"])

# ----------------------------------------------------------------- TAB 1 : dashboard
with tab1:
    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1.3]})
    axs[0].plot(view.index, view.hot_max, lw=1, label="Hottest hot-side sensor")
    axs[0].plot(view.index, view.cold_mean, lw=1, label="Cold-side mean")
    axs[0].axhline(T_WARM, color="orange", ls="--", lw=.8, label="Warm threshold")
    axs[0].axhline(T_HOT, color="red", ls="--", lw=.8, label="Hotspot threshold")
    axs[0].set_ylabel("°C"); axs[0].legend(ncol=4, fontsize=8, loc="upper right")
    axs[1].scatter(view.index, view.action, c=[COLORS[a] for a in view.action], s=6)
    axs[1].set_yticks([0, 1, 2]); axs[1].set_yticklabels(LEVELS); axs[1].set_ylabel("RF decision")
    kw_rf = p_rated * np.asarray(frac)[view.action.to_numpy()]
    axs[2].plot(view.index, np.cumsum(np.full(n, p_rated * frac[2]) / 60), color="gray", label="Always BOOST")
    axs[2].plot(view.index, np.cumsum(kw_rf / 60), color="#2a7", label="RF controller")
    axs[2].set_ylabel("Cumulative kWh"); axs[2].legend()
    for a in axs: a.grid(alpha=.3)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    share = view.action.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0) * 100
    st.write("Time spent in each cooling level (%)")
    st.bar_chart(pd.Series(share.values, index=LEVELS))

# ----------------------------------------------------------------- TAB 2 : replay
with tab2:
    st.write("Move through time and watch what each part of the project contributes at that moment.")
    if "i" not in st.session_state:
        st.session_state.i = 0
    st.session_state.i = min(st.session_state.i, n - 1)
    b = st.columns([1, 1, 1, 1, 6])
    if b[0].button("⏮ -10"): st.session_state.i = max(0, st.session_state.i - 10)
    if b[1].button("◀ -1"): st.session_state.i = max(0, st.session_state.i - 1)
    if b[2].button("+1 ▶"): st.session_state.i = min(n - 1, st.session_state.i + 1)
    if b[3].button("+10 ⏭"): st.session_state.i = min(n - 1, st.session_state.i + 10)
    i = st.slider("Time step (minutes)", 0, n - 1, st.session_state.i, key="slider_i")
    st.session_state.i = i
    r = view.iloc[i]
    st.subheader(str(view.index[i]))
    k = st.columns(4)
    k[0].markdown("**🧠 CNN – thermal state**")
    k[0].progress(float(np.clip(r.p_hotspot, 0, 1)), text=f"P(hotspot) = {r.p_hotspot:.2f}")
    k[0].progress(float(np.clip(r.p_warm, 0, 1)), text=f"P(warm) = {r.p_warm:.2f}")
    k[0].metric("Hot-side max", f"{r.hot_max:.1f} °C")
    k[1].markdown("**📈 LSTM – CPU load**")
    k[1].metric("Forecast load (time-of-day)", f"{r.load:.2f} %", f"{r.load_trend:+.2f} in {H} min")
    k[2].markdown(f"**🌲 Random Forest – in {H} min**")
    k[2].metric("Predicted hot-side max", f"{r.pred_hot_max_future:.1f} °C",
                None if np.isnan(r.hot_max_future) else f"actual {r.hot_max_future:.1f} °C", delta_color="off")
    k[2].progress(float(np.clip(r.p_boost, 0, 1)), text=f"P(BOOST needed) = {r.p_boost:.2f}")
    k[3].markdown("**❄️ HVAC decision**")
    lvl = int(r.action)
    k[3].markdown(f"<h1 style='color:{COLORS[lvl]};margin:0'>{LEVELS[lvl]}</h1>", unsafe_allow_html=True)
    k[3].caption(f"{p_rated * frac[lvl]:.1f} kW instead of {p_rated * frac[2]:.1f} kW")
    if not np.isnan(r.required):
        ok_ = lvl >= int(r.required)
        k[3].success("Cooling sufficient") if ok_ else k[3].error(f"Under-cooling (needed {LEVELS[int(r.required)]})")
    w = view.iloc[max(0, i - 60): i + 1]
    st.line_chart(w[["hot_max", "cold_mean"]])

# ----------------------------------------------------------------- TAB 3 : what-if
with tab3:
    st.write("Try your own situation and ask the trained Random Forest what it would do.")
    mdl = load_model(str(DATA))
    if isinstance(mdl, str):
        st.warning(f"Model file could not be loaded ({mdl}). Re-run rf_hvac.py with the same scikit-learn version "
                   f"as the app (trained with {meta['sklearn_version']}).")
    else:
        base = sim[mdl["features"]].median()
        s1, s2, s3 = st.columns(3)
        hot_max = s1.slider("Hot-side max (°C)", float(sim.hot_max.min()) - 2, float(sim.hot_max.max()) + 3, float(base.hot_max), .1)
        cold = s1.slider("Cold-side mean (°C)", float(sim.cold_mean.min()) - 2, float(sim.cold_mean.max()) + 3, float(base.cold_mean), .1)
        trend = s2.slider(f"Temperature change last 5 min (°C)", -2.0, 2.0, float(base.d5), .05)
        loadv = s2.slider("Forecast CPU load (%)", 0.0, float(sim.load.max()) * 1.5, float(base.load), .1)
        ltr = s2.slider(f"Load change expected in {H} min", -2.0, 2.0, float(base.load_trend), .05)
        phot = s3.slider("CNN P(hotspot)", 0.0, 1.0, float(base.p_hotspot), .01)
        pwarm = s3.slider("CNN P(warm)", 0.0, 1.0 - phot, min(float(base.p_warm), 1.0 - phot), .01)
        x = base.copy()
        x["hot_max"], x["cold_mean"], x["d5"], x["load"], x["load_trend"] = hot_max, cold, trend, loadv, ltr
        x["p_hotspot"], x["p_warm"] = phot, pwarm
        x["hot_mean"] = base.hot_mean + (hot_max - base.hot_max) * .8
        x["delta_T"] = x.hot_mean - cold
        x["d1"] = trend / 5
        X = pd.DataFrame([x[mdl["features"]]])
        Pw = np.zeros((1, 3)); Pw[:, mdl["clf"].classes_.astype(int)] = mdl["clf"].predict_proba(X)
        a = int(decide(Pw, eps)[0])
        f = float(hot_max + mdl["reg"].predict(X)[0])
        o1, o2 = st.columns(2)
        o1.markdown(f"### Decision: <span style='color:{COLORS[a]}'>{LEVELS[a]}</span>", unsafe_allow_html=True)
        o1.metric(f"Predicted hot-side max in {H} min", f"{f:.1f} °C")
        o2.bar_chart(pd.Series(Pw[0], index=LEVELS), height=200)
        st.caption("Features not shown here (humidity, short-term std) are kept at their median value.")

# ----------------------------------------------------------------- TAB 4 : models
with tab4:
    st.subheader("Random Forest – results on unseen test data")
    tt = pd.DataFrame(meta["test_table"]).T
    st.dataframe(tt.round(2), use_container_width=True)
    m = meta["forecast_metrics"]
    st.write(f"**Temperature forecast (+{H} min)** – RF MAE {m['rf_mae_C']:.3f} °C vs "
             f"{m['persistence_mae_C']:.3f} °C for 'temperature stays the same'.")
    cA, cB = st.columns(2)
    cA.write("**What do CNN and LSTM bring?** (ablation)")
    cA.dataframe(pd.DataFrame(meta["ablation"]).T.round(3))
    imp = pd.Series(meta["feature_importance"]).sort_values()
    cB.write("**Feature importance**"); cB.bar_chart(imp)
    for f_, cap in [("fig_rf_tradeoff.png", "Energy saving vs thermal risk"), ("fig_rf_timeline.png", "Test timeline")]:
        if (DATA / f_).exists(): st.image(str(DATA / f_), caption=cap)
    if lstm is not None:
        st.subheader("LSTM – CPU load forecast")
        st.line_chart(lstm[["actual_load", "predicted_load"]].iloc[:576])

# ----------------------------------------------------------------- TAB 5 : about
with tab5:
    st.markdown(f"""
**Pipeline** – CNN classifies the thermal map of the racks (Normal / Warm / Hotspot) → LSTM forecasts the total CPU load →
Random Forest predicts the cooling level needed in {H} minutes and chooses the cheapest safe level.

**Honest limits (put them in the report)**
- Workload traces (2013) and temperature sensors (2018) come from different datacenters: they are linked **by time of day only**.
- The data contain **no HVAC actions**, so the replay is open-loop: temperatures are assumed not to change with the decision.
- Cooling power per level, rated power, CO₂ factor and price are **assumptions** (editable in the sidebar).
- The recording is short (≈30 h): conclusions should be confirmed on longer data.
""")
    st.json({k: meta[k] for k in ["horizon_min", "eps", "thresholds_C", "sklearn_version", "test_period"]})

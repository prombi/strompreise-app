# app.py
import time
from typing import Optional, Tuple, List
import datetime as dt
import pandas as pd
import pytz
import requests
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Day-Ahead Strompreis (DE/LU)", page_icon="⚡", layout="centered")
st.title("⚡ Day-Ahead Strompreise (DE/LU)")
st.caption(
    "Quelle: SMARD / EPEX Day-Ahead. Gebühren werden aus PLZ vorbefüllt (MVP-Heuristik) "
    "und können manuell angepasst werden."
)

# ---------------------------------------------------------
# Gebühren-Heuristik (MVP) + Preisberechnung
# ---------------------------------------------------------
def estimate_fees_from_plz(plz: str):
    prefix = (plz or "").strip()[:2]

    big_cities = {
        "10","11","12","13","14","20","21","22","80","81","82","85","86",
        "50","51","53","60","61","63","65","70","71","72","73","40","41",
        "42","44","45","30","31","32","90","91","01","04"
    }
    mid_cities = {"33","34","47","48","49","54","55","56","66","67","68","69",
                  "74","75","76","77","78","79","83","84","87","88","89"}

    if prefix in big_cities:
        kav = 2.39
    elif prefix in mid_cities:
        kav = 1.99
    else:
        kav = 1.59

    north_east = {"01","02","03","04","10","11","12","13","14","17","18","19","23","24"}
    south = {"80","81","82","83","84","85","86","87","88","89","90","91"}
    if prefix in south:
        netzentgelt = 7.5
    elif prefix in north_east:
        netzentgelt = 9.0
    else:
        netzentgelt = 8.0

    return {
        "stromsteuer_ct": 2.05,
        "umlagen_ct": 2.651,
        "konzessionsabgabe_ct": kav,
        "netzentgelt_ct": netzentgelt,
        "mwst": 19
    }

def compute_price(ct_per_kwh: float, fees: dict, include_fees: bool) -> float:
    if not include_fees:
        return ct_per_kwh
    surcharges = (
        fees["stromsteuer_ct"]
        + fees["umlagen_ct"]
        + fees["konzessionsabgabe_ct"]
        + fees["netzentgelt_ct"]
    )
    net_sum = ct_per_kwh + surcharges
    return net_sum * (1 + fees["mwst"] / 100.0)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Gebühren einstellen")

    if "plz" not in st.session_state:
        st.session_state.plz = "82340"
    if "fees" not in st.session_state:
        st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)

    st.text_input("PLZ", key="plz")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Vorbelegen aus PLZ"):
            st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)
    with c2:
        if st.button("Standard reset"):
            st.session_state.fees = estimate_fees_from_plz("")

    st.subheader("Manuell anpassen")
    st.session_state.fees["stromsteuer_ct"] = st.number_input(
        "Stromsteuer (ct/kWh)", 0.0, 10.0, float(st.session_state.fees.get("stromsteuer_ct", 2.05)), 0.01
    )
    st.session_state.fees["umlagen_ct"] = st.number_input(
        "Umlagen gesamt (ct/kWh)", 0.0, 10.0, float(st.session_state.fees.get("umlagen_ct", 2.651)), 0.001
    )
    st.session_state.fees["konzessionsabgabe_ct"] = st.number_input(
        "Konzessionsabgabe (ct/kWh)", 0.0, 5.0, float(st.session_state.fees.get("konzessionsabgabe_ct", 1.59)), 0.01
    )
    st.session_state.fees["netzentgelt_ct"] = st.number_input(
        "Netzentgelt pauschal (ct/kWh)", 0.0, 50.0, float(st.session_state.fees.get("netzentgelt_ct", 8.0)), 0.1
    )
    st.session_state.fees["mwst"] = st.number_input(
        "MwSt (%)", 0, 25, int(st.session_state.fees.get("mwst", 19)), 1
    )

# ---------------------------------------------------------
# Top controls
# ---------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    try:
        view_mode = st.segmented_control("Ansicht", ["Ohne Gebühren", "Inkl. Gebühren"], default="Ohne Gebühren")
    except Exception:
        view_mode = st.radio("Ansicht", ["Ohne Gebühren", "Inkl. Gebühren"], index=0)
with col2:
    resolution_choice = st.selectbox("Auflösung", ["quarterhour", "hour"], index=0)

include_fees = (view_mode == "Inkl. Gebühren")

# ---------------------------------------------------------
# SMARD Loader
# ---------------------------------------------------------
SERIES_ID = "4169"
REGION_CANDIDATES = ["DE-LU", "DE"]
SMARD_BASE = "https://www.smard.de/app"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit-SMARD/1.0)"}
_last_tried: List[str] = []

class SmardError(Exception):
    pass

def _safe_get_json(url: str, timeout: int = 30) -> dict:
    _last_tried.append(url)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code != 200:
        raise SmardError(f"HTTP {r.status_code} für {url}")
    try:
        return r.json()
    except Exception:
        snippet = (r.text or "")[:160].replace("\n", " ")
        ctype = r.headers.get("Content-Type", "")
        raise SmardError(f"Kein JSON von {url}. Content-Type='{ctype}', Antwort: '{snippet}...'")

def _get_index(region: str, resolution: str) -> list[int]:
    url = f"{SMARD_BASE}/chart_data/{SERIES_ID}/{region}/index_{resolution}.json"
    data = _safe_get_json(url)
    ts = data.get("timestamps") or []
    if not ts:
        raise SmardError(f"Keine timestamps in index_{resolution}.json ({region})")
    return ts

def _try_load_series(region: str, resolution: str, ts: int) -> Optional[pd.DataFrame]:
    for path in ["table_data", "chart_data"]:
        url = f"{SMARD_BASE}/{path}/{SERIES_ID}/{region}/{SERIES_ID}_{region}_{resolution}_{ts}.json"
        try:
            data = _safe_get_json(url)
            series = data.get("series")
            if series:
                return pd.DataFrame(series, columns=["ts_ms", "eur_per_mwh"])
        except SmardError:
            continue
    return None

@st.cache_data(ttl=900)
def load_smard_series(prefer_resolution: str = "quarterhour", max_backsteps: int = 12) -> Tuple[pd.DataFrame, str, str]:
    resolutions = [prefer_resolution] + ([r for r in ["hour"] if r != prefer_resolution])
    for region in REGION_CANDIDATES:
        for resolution in resolutions:
            try:
                idx = _get_index(region, resolution)
            except SmardError:
                continue
            for ts in reversed(idx[-(max_backsteps + 1):]):
                df = _try_load_series(region, resolution, ts)
                if df is not None and not df.empty:
                    return df, resolution, region
                time.sleep(0.15)
    raise SmardError("Keine gültige SMARD-Datei gefunden (region/auflösung/ts).")


# ------------------------------------------------------
# Call API and get data df_raw
# ------------------------------------------------------
try:
    df_raw, used_resolution, used_region = load_smard_series(
        prefer_resolution=resolution_choice, max_backsteps=12
    )
except SmardError as e:
    with st.expander("Diagnose: letzte getestete URLs"):
        for u in _last_tried[-14:]:
            st.code(u)
    st.error(
        "⚠️ Konnte SMARD-Daten nicht laden.\n\n"
        f"Fehler: {e}\n\n"
        "Tipps: später erneut versuchen (frischer Timestamp), ggf. 'Auflösung' auf 'hour' stellen, "
        "oder Firmen-Proxy prüfen."
    )
    st.stop()

# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------
# ---------------------------------------------------------
# Keep ALL valid datapoints, then offer a slider to choose the visible window
# ---------------------------------------------------------
tz_berlin = pytz.timezone("Europe/Berlin")
now = dt.datetime.now(tz=tz_berlin)
today = now.date()
yesterday = today - dt.timedelta(days=1)
tomorrow = today + dt.timedelta(days=1)

# Convert timestamps to timezone-aware datetimes
df_raw["ts"] = pd.to_datetime(df_raw["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")

# Convert €/MWh → ct/kWh (1 €/MWh = 0.1 ct/kWh)
df_raw["ct_per_kwh"] = df_raw["eur_per_mwh"] * 0.1

# 1) Build full dataset with valid prices (no pre-filtering to 12→12)
df_all = (
    df_raw
    .dropna(subset=["ct_per_kwh"])
    .sort_values("ts")
    .copy()
)
if df_all.empty:
    st.info("Keine gültigen Preisdaten verfügbar.")
    st.stop()

min_ts = df_all["ts"].min()
max_ts = df_all["ts"].max()

# 2) Define your *default* Day-Ahead window
noon_today = tz_berlin.localize(dt.datetime.combine(today, dt.time(12, 0)))
noon_yesterday = tz_berlin.localize(dt.datetime.combine(yesterday, dt.time(12, 0)))
noon_tomorrow  = tz_berlin.localize(dt.datetime.combine(tomorrow,  dt.time(12, 0)))

if now < noon_today:
    # before 12:00 → yesterday 12:00 → today 12:00
    start_window = noon_yesterday
    end_window   = noon_today
else:
    # at/after 12:00 → today 12:00 → tomorrow 12:00
    start_window = noon_today
    end_window   = noon_tomorrow

# Clamp defaults to available data (so the slider has valid defaults even if data ends earlier)
default_start = max(min_ts, start_window)
default_end   = min(max_ts, end_window)
if default_start > default_end:
    # Fallback: if DA window not in data at all, default to full available range
    default_start, default_end = min_ts, max_ts

# 3) Slider spans the full available range; defaults are 12→12
step_td = dt.timedelta(minutes=15 if used_resolution == "quarterhour" else 60)
with st.sidebar:
    st.subheader("Sichtbarer Zeitraum")
    t_start, t_end = st.slider(
        "Zeitfenster",
        min_value=min_ts,
        max_value=max_ts,
        value=(default_start, default_end),
        step=step_td,
        format="DD.MM.YY HH:mm",
    )

# 4) Filter ONLY the view to the slider (dataset itself stays complete)
t_start_ts = pd.Timestamp(t_start)
t_end_ts   = pd.Timestamp(t_end)
df = df_all[(df_all["ts"] >= t_start_ts) & (df_all["ts"] <= t_end_ts)].copy()
if df.empty:
    st.warning("Im gewählten Zeitfenster liegen keine Preispunkte vor.")
    st.stop()

# 5) Compute the two chart layers (Börsenstrompreis, Gebühren inkl. MwSt) on the filtered view
fees = st.session_state.fees
df["spot_ct"] = df["ct_per_kwh"]

fees_no_vat = (
    fees["stromsteuer_ct"]
    + fees["umlagen_ct"]
    + fees["konzessionsabgabe_ct"]
    + fees["netzentgelt_ct"]
)
df["fees_incl_vat_ct"] = (fees_no_vat + df["spot_ct"]) * (fees["mwst"] / 100.0) + fees_no_vat
df["total_ct"] = df["spot_ct"] + df["fees_incl_vat_ct"]

# 6) KPIs based on the filtered view (respects slider)
metric_col = "total_ct" if include_fees else "spot_ct"
now_local = pd.Timestamp.now(tz="Europe/Berlin")
current_idx = min(df["ts"].searchsorted(now_local, side="left"), len(df) - 1)
current_price = float(df.iloc[current_idx][metric_col])
m, M, avg = float(df[metric_col].min()), float(df[metric_col].max()), float(df[metric_col].mean())

k1, k2, k3 = st.columns(3)
k1.metric("aktuell", f"{current_price:.2f} ct/kWh")
k2.metric("⭣ min", f"{m:.2f} ct/kWh")
k3.metric("⭡ max", f"{M:.2f} ct/kWh")
st.caption(
    f"Durchschnitt: {avg:.2f} ct/kWh · "
    f"Auflösung: {used_resolution} · Region: {used_region}"
)

# 7) Plotly chart (two layers, step lines, classic colors)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["ts"], y=df["spot_ct"],
    name="Börsenstrompreis",
    mode="lines",
    line_shape="hv",
    line=dict(width=0.8, color="#1f77b4"),
    stackgroup="one",
    hovertemplate=(
        "Zeit: %{x|%d.%m %H:%M}<br>"
        "Börsenstrompreis: %{y:.2f} ct/kWh"
        "<br>Gesamtpreis: %{customdata:.2f} ct/kWh<extra></extra>"
    ),
    customdata=df["total_ct"],
))

fig.add_trace(go.Scatter(
    x=df["ts"], y=df["fees_incl_vat_ct"],
    name="Gebühren (inkl. MwSt)",
    mode="lines",
    line_shape="hv",
    line=dict(width=0.8, color="#ff7f0e"),
    stackgroup="one",
    hovertemplate=(
        "Zeit: %{x|%d.%m %H:%M}<br>"
        "Gebühren (inkl. MwSt): %{y:.2f} ct/kWh"
        "<br>Gesamtpreis: %{customdata:.2f} ct/kWh<extra></extra>"
    ),
    customdata=df["total_ct"],
))

fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Zeit (lokal)",
    yaxis_title="ct/kWh",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25, x=0.0),
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", namelength=-1),
)
st.plotly_chart(fig, use_container_width=True)

# Footer about slider defaults vs full range (optional)
st.caption(
    f"Standard-Fenster: {start_window.strftime('%d.%m %H:%M')} – {(start_window + dt.timedelta(days=1)).strftime('%d.%m %H:%M')} · "
    "Den Schieberegler kannst du auf die komplette verfügbare Datenreichweite ausdehnen."
)

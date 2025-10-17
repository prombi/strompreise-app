# app.py
import datetime as dt
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go


MODULE_NAME = "price_sources"


def _load_price_sources_module():
    for filename in ("price-sources.py", "price_sources.py"):
        module_path = Path(__file__).with_name(filename)
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Konnte price-sources Moduldatei nicht finden (price-sources.py oder price_sources.py).")


price_sources = sys.modules.get(MODULE_NAME) or _load_price_sources_module()


def _is_timezone_aware(values) -> bool:
    dtype = getattr(values, "dtype", None)
    return isinstance(dtype, pd.DatetimeTZDtype)

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Day-Ahead Strompreis (DE/LU)", page_icon="⚡", layout="centered")
st.title("⚡ Day-Ahead Strompreise (DE/LU)")
st.caption(
    "Quelle: Datenquelle wählbar (SMARD.de oder ENTSO-E Day-Ahead). Gebühren werden aus PLZ vorbefüllt "
    "(MVP-Heuristik) und können manuell angepasst werden."
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
SOURCE_OPTIONS = {
    "SMARD (Bundesnetzagentur)": "SMARD",
    "ENTSO-E Transparency": "ENTSOE",
}

col_source, col_view, col_res = st.columns(3)
with col_source:
    selected_source_label = st.selectbox("Datenquelle", list(SOURCE_OPTIONS.keys()), index=0)
    data_source_choice = SOURCE_OPTIONS[selected_source_label]
with col_view:
    try:
        view_mode = st.segmented_control("Ansicht", ["Ohne Gebühren", "Inkl. Gebühren"], default="Ohne Gebühren")
    except Exception:
        view_mode = st.radio("Ansicht", ["Ohne Gebühren", "Inkl. Gebühren"], index=0)
with col_res:
    resolution_disabled = data_source_choice != "SMARD"
    resolution_choice = st.selectbox(
        "Auflösung", ["quarterhour", "hour"], index=0, disabled=resolution_disabled
    )

include_fees = (view_mode == "Inkl. Gebühren")

if "entsoe_token" not in st.session_state:
    st.session_state.entsoe_token = ""

with st.sidebar:
    if data_source_choice == "ENTSOE":
        st.subheader("ENTSO-E Zugang")
        st.session_state.entsoe_token = st.text_input(
            "API Token",
            value=st.session_state.entsoe_token,
            type="password",
            help="Trage hier deinen ENTSO-E Token ein, um die Transparenzplattform abzurufen.",
        )

# ---------------------------------------------------------
# Datenquellen laden
# ---------------------------------------------------------
class PriceDataError(Exception):
    """Raised when a selected data source cannot provide usable data."""


def prepare_price_dataframe(df_raw: pd.DataFrame, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """
    Normalise raw API data into the df_all structure:
    - ensure a timezone-aware 'ts' column (Berlin)
    - ensure 'ct_per_kwh' exists (derive from eur_per_mwh if needed)
    - drop NaNs and sort by timestamp
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["ts", "eur_per_mwh", "ct_per_kwh", "source", "resolution"])

    df = df_raw.copy()

    if "ts" not in df.columns:
        if "ts_ms" in df.columns:
            ts_series = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        else:
            raise PriceDataError("Datensatz enthält keine Spalte 'ts' oder 'ts_ms'.")
    else:
        ts_series = df["ts"]

    if not _is_timezone_aware(ts_series):
        ts_series = pd.to_datetime(ts_series, utc=True)

    if not _is_timezone_aware(ts_series):
        ts_series = ts_series.dt.tz_localize("UTC")

    df["ts"] = ts_series.dt.tz_convert(tz)

    if "ct_per_kwh" not in df.columns:
        if "eur_per_mwh" not in df.columns:
            raise PriceDataError("Datensatz enthält weder 'ct_per_kwh' noch 'eur_per_mwh'.")
        df["ct_per_kwh"] = pd.to_numeric(df["eur_per_mwh"], errors="coerce") * 0.1

    df_prepared = (
        df.dropna(subset=["ts", "ct_per_kwh"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return df_prepared


def normalize_resolution_hint(hint: Optional[str], default: str = "quarterhour") -> str:
    if not hint:
        return default
    hint_lower = str(hint).lower()
    if "15" in hint_lower or "quarter" in hint_lower:
        return "quarterhour"
    if "60" in hint_lower or "hour" in hint_lower:
        return "hour"
    return default


def detect_resolution_label(df: pd.DataFrame, fallback: str = "quarterhour") -> str:
    if df.empty:
        return fallback

    if "resolution" in df.columns:
        res_values = df["resolution"].dropna().astype(str).str.lower()
        if res_values.str.contains("15").any() or res_values.str.contains("quarter").any():
            return "quarterhour"
        if res_values.str.contains("60").any() or res_values.str.contains("hour").any():
            return "hour"

    diffs = df.sort_values("ts")["ts"].diff().dropna()
    if not diffs.empty:
        median_minutes = diffs.dt.total_seconds().median() / 60.0
        if median_minutes <= 30:
            return "quarterhour"
        if median_minutes >= 45:
            return "hour"

    return fallback


@st.cache_data(ttl=900)
def load_price_data(
    source: str,
    resolution: str,
    entsoe_token: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    if source == "SMARD":
        df = price_sources.fetch_smard_day_ahead(resolution=resolution)
        meta = {"region": "DE-LU", "raw_resolution": resolution, "source_id": "SMARD"}
        return df, meta

    if source == "ENTSOE":
        if not entsoe_token:
            raise PriceDataError("Für ENTSO-E wird ein API Token benötigt.")

        now_utc = dt.datetime.now(dt.timezone.utc)
        start_utc = (now_utc - dt.timedelta(days=2)).replace(minute=0, second=0, microsecond=0)
        end_utc = (now_utc + dt.timedelta(days=2)).replace(minute=0, second=0, microsecond=0)

        df = price_sources.fetch_entsoe_day_ahead(
            token=entsoe_token,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        meta = {"region": "DE-LU", "raw_resolution": None, "source_id": "ENTSOE"}
        return df, meta

    raise PriceDataError(f"Unbekannte Datenquelle: {source}")


# ------------------------------------------------------
# Call API and get data df_raw
# ------------------------------------------------------
try:
    df_raw, data_meta = load_price_data(
        data_source_choice, resolution_choice, st.session_state.entsoe_token
    )
except PriceDataError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()
except Exception as exc:
    st.error(f"⚠️ Unerwarteter Fehler beim Laden der Datenquelle: {exc}")
    st.stop()

used_region = data_meta.get("region", "DE-LU")
resolution_fallback = normalize_resolution_hint(
    data_meta.get("raw_resolution"), default=resolution_choice
)

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

# Convert €/MWh → ct/kWh (1 €/MWh = 0.1 ct/kWh)

# 1) Build full dataset with valid prices (no pre-filtering to 12→12)
df_all = prepare_price_dataframe(df_raw)
if df_all.empty:
    st.info("Keine gültigen Preisdaten verfügbar.")
    st.stop()

used_resolution = detect_resolution_label(df_all, fallback=resolution_fallback)

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
    f"Auflösung: {used_resolution} · Region: {used_region} · Quelle: {selected_source_label}"
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

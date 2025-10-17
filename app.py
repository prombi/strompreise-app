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
from streamlit_plotly_events import plotly_events


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
    include_fees = st.toggle("Inkl. Gebühren", value=False)
with col_res:
    resolution_disabled = data_source_choice != "SMARD"
    resolution_is_quarterhour = st.toggle(
        "Viertelstündlich",
        value=True,
        disabled=resolution_disabled,
        help="Schaltet zwischen Viertelstunden- und Stundenauflösung (nur SMARD).",
    )
    resolution_choice = "quarterhour" if resolution_is_quarterhour else "hour"

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

midnight_today = tz_berlin.localize(dt.datetime.combine(today, dt.time(0, 0)))
midnight_tomorrow = midnight_today + dt.timedelta(days=1)

# if now < noon_today:
#     # before 12:00 → yesterday 12:00 → today 12:00
#     start_window = noon_yesterday
#     end_window   = noon_today
# else:
#     # at/after 12:00 → today 12:00 → tomorrow 12:00
#     start_window = noon_today
#     end_window   = noon_tomorrow

start_window = midnight_today
end_window = midnight_tomorrow

# Clamp defaults to available data (so the slider has valid defaults even if data ends earlier)
default_start = max(min_ts, start_window)
default_end   = min(max_ts, end_window)
if default_start > default_end:
    # Fallback: if DA window not in data at all, default to full available range
    default_start, default_end = min_ts, max_ts

# 3) Prepare data for chart and stats
stats_container = st.container()

fees = st.session_state.fees
df_chart = df_all.copy()
df_chart["spot_ct"] = df_chart["ct_per_kwh"]

fees_no_vat = (
    fees["stromsteuer_ct"]
    + fees["umlagen_ct"]
    + fees["konzessionsabgabe_ct"]
    + fees["netzentgelt_ct"]
)
df_chart["fees_incl_vat_ct"] = (fees_no_vat + df_chart["spot_ct"]) * (fees["mwst"] / 100.0) + fees_no_vat
df_chart["total_ct"] = df_chart["spot_ct"] + df_chart["fees_incl_vat_ct"]
spot_series = df_chart["spot_ct"].astype(float)
total_series = df_chart["total_ct"].astype(float)
fees_series = df_chart["fees_incl_vat_ct"].astype(float)
time_series = df_chart["ts"].dt.tz_convert(tz_berlin).dt.tz_localize(None)
# fees_hover = fees_series.to_numpy()
fees_hover = fees_series.round(6).tolist()

initial_view = (default_start, default_end)
if "view_range" not in st.session_state:
    st.session_state.view_range = initial_view

def _normalize_ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = tz_berlin.localize(ts)
    else:
        ts = ts.tz_convert(tz_berlin)
    return ts

view_start, view_end = st.session_state.view_range
view_start = _normalize_ts(view_start)
view_end = _normalize_ts(view_end)
view_start = max(min_ts, view_start)
view_end = min(max_ts, view_end)
if view_start >= view_end:
    view_start, view_end = initial_view
st.session_state.view_range = (view_start, view_end)

# 7) Plotly chart (two layers, step lines, classic colors)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time_series,
    y=spot_series.tolist(),
    name="Börsenstrompreis",
    mode="lines",
    line_shape="hv",
    line=dict(width=0.8, color="#1f77b4"),
    fill="tozeroy",
    fillcolor="rgba(31, 119, 180, 0.25)",
    hovertemplate="Börsenstrompreis: %{y:.1sf} ct/kWh<extra></extra>",
))

fig.add_trace(go.Scatter(
    x=time_series,
    y=total_series.tolist(),
    name="Gesamtpreis",
    mode="lines",
    line_shape="hv",
    line=dict(width=1.2, color="#d62728"),
    fill="tonexty",
    fillcolor="rgba(255, 127, 14, 0.35)",
    customdata=fees_hover,
    hovertemplate=(
        "Gesamtpreis: %{y:.1f} ct/kWh<br>"
        "Gebühren (inkl. MwSt): %{customdata:.1f} ct/kWh<extra></extra>"
    ),
))

# default_day_start = dt.datetime.combine(today, dt.time.min).replace(tzinfo=tz_berlin)
# default_day_end = dt.datetime.combine(today, dt.time.max).replace(tzinfo=tz_berlin)
default_day_start = midnight_today
default_day_end = midnight_tomorrow

day_shapes = []
start_ref = min_ts.tz_convert(tz_berlin) if min_ts.tzinfo is not None else tz_berlin.localize(min_ts)
end_ref = max_ts.tz_convert(tz_berlin) if max_ts.tzinfo is not None else tz_berlin.localize(max_ts)
start_day = start_ref.normalize()
end_day = end_ref.normalize()
for day in pd.date_range(start=start_day, end=end_day, freq="D", tz=tz_berlin):
    day_naive = day.tz_localize(None)
    day_shapes.append(
        dict(
            type="line",
            x0=day_naive,
            x1=day_naive,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="rgba(0,0,0,0.35)", width=1.5),
        )
    )
    noon = day + pd.Timedelta(hours=12)
    if min_ts <= noon <= max_ts:
        noon_naive = noon.tz_localize(None)
        day_shapes.append(
            dict(
                type="line",
                x0=noon_naive,
                x1=noon_naive,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dot"),
            )
        )

now_ts = pd.Timestamp(now)
if min_ts <= now_ts <= max_ts:
    now_naive = now_ts.tz_localize(None) if now_ts.tzinfo else now_ts
    day_shapes.append(
        dict(
            type="line",
            x0=now_naive,
            x1=now_naive,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#d62728", width=2),
        )
)

range_start_ts = pd.Timestamp(view_start)
range_end_ts = pd.Timestamp(view_end)
range_start = range_start_ts.tz_localize(None) if range_start_ts.tzinfo else range_start_ts
range_end = range_end_ts.tz_localize(None) if range_end_ts.tzinfo else range_end_ts
fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(
        title="Zeit (lokal)",
        type="date",
        range=[range_start, range_end],
        rangeslider=dict(visible=True),
    ),
    yaxis_title="ct/kWh",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25, x=0.0),
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", namelength=-1),
    shapes=day_shapes,
)
events = plotly_events(
    fig,
    click_event=False,
    select_event=False,
    hover_event=False,
    override_height=420,
    relayout_event=True,
    key="price_chart",
)

new_start_raw = new_end_raw = None
for event in events or []:
    if not isinstance(event, dict):
        continue
    if "range" in event:
        rng = event["range"]
        if isinstance(rng, dict):
            x_range = rng.get("x")
            if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
                new_start_raw, new_end_raw = x_range
        elif isinstance(rng, (list, tuple)) and len(rng) == 2:
            new_start_raw, new_end_raw = rng
    if new_start_raw is None and "xaxis.range[0]" in event and "xaxis.range[1]" in event:
        new_start_raw = event["xaxis.range[0]"]
        new_end_raw = event["xaxis.range[1]"]
    if new_start_raw is None and "xaxis.range" in event:
        x_range = event["xaxis.range"]
        if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
            new_start_raw, new_end_raw = x_range
    if new_start_raw is None and "data" in event:
        data = event["data"]
        if isinstance(data, dict):
            x_range = data.get("xaxis.range")
            if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
                new_start_raw, new_end_raw = x_range

if new_start_raw and new_end_raw:
    new_start = _normalize_ts(new_start_raw)
    new_end = _normalize_ts(new_end_raw)
    new_start = max(min_ts, new_start)
    new_end = min(max_ts, new_end)
    if new_start < new_end and (new_start, new_end) != st.session_state.view_range:
        st.session_state.view_range = (new_start, new_end)
        st.experimental_rerun()

view_start, view_end = st.session_state.view_range
view_start = _normalize_ts(view_start)
view_end = _normalize_ts(view_end)

df_view = df_chart[(df_chart["ts"] >= view_start) & (df_chart["ts"] <= view_end)].copy()

with stats_container:
    if df_view.empty:
        st.warning("Im ausgewählten Zeitbereich liegen keine Preispunkte vor.")
    else:
        metric_col = "total_ct" if include_fees else "spot_ct"
        now_local = now
        idx = df_view["ts"].searchsorted(now_local, side="left")
        idx = min(max(idx, 0), len(df_view) - 1)
        current_price = float(df_view.iloc[idx][metric_col])
        avg = float(df_view[metric_col].mean())
        min_price = float(df_view[metric_col].min())
        max_price = float(df_view[metric_col].max())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("aktuell", f"{current_price:.1f} ct/kWh")
        c2.metric("average", f"{avg:.1f} ct/kWh")
        c3.metric("⭣ min", f"{min_price:.1f} ct/kWh")
        c4.metric("⭡ max", f"{max_price:.1f} ct/kWh")
        st.caption(
            f"Auflösung: {used_resolution} · Region: {used_region} · Quelle: {selected_source_label}"
        )

# Footer about slider defaults vs full range (optional)
st.caption(
    f"Standard-Fenster: {start_window.strftime('%d.%m %H:%M')} – {(start_window + dt.timedelta(days=1)).strftime('%d.%m %H:%M')} · "
    "Nutze den Range-Slider unter dem Diagramm, um andere Zeitbereiche auszuwählen."
)

# app.py
import datetime as dt
from typing import Optional
import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go
import price_sources

DEFAULT_PLZ = "82340"
FEE_WIDGET_KEYS = {
    "umlagen_ct": ("fee_umlagen", float),
    "konzessionsabgabe_ct": ("fee_konzessionsabgabe", float),
    "netzentgelt_ct": ("fee_netzentgelt", float),
    "mwst": ("fee_mwst", int),
}


def _sync_fee_widget_state() -> None:
    """Keep the fee input widgets aligned with the current fee dictionary."""
    for field, (key, caster) in FEE_WIDGET_KEYS.items():
        st.session_state[key] = caster(st.session_state.fees[field])


def _ensure_fee_widget_defaults() -> None:
    """Populate widget session state with the current fee values when missing."""
    for field, (key, caster) in FEE_WIDGET_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = caster(st.session_state.fees[field])

def _is_timezone_aware(values) -> bool:
    dtype = getattr(values, "dtype", None)
    return isinstance(dtype, pd.DatetimeTZDtype)
# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Day-Ahead Strompreis", page_icon="⚡", layout="centered")
st.title("⚡ Day-Ahead Strompreise")
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
        kav = 1.32              # 82340 Feldafing
    north_east = {"01","02","03","04","10","11","12","13","14","17","18","19","23","24"}
    south = {"80","81","82","83","84","85","86","87","88","89","90","91"}
    if prefix in south:
        netzentgelt = 7.35      # 82340 Feldafing
    elif prefix in north_east:
        netzentgelt = 9.0
    else:
        netzentgelt = 8.0

    return {
        "stromsteuer_ct": 2.05,
        "umlagen_ct": 0.28+0.00+0.00+0.82+1.56,     # KWKG- + Abschalt- + EEG- + Offshore- + §19 StromNEV- Umlagen; previously 2.651,
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
    st.header("Gebühren")
    if "plz" not in st.session_state:
        st.session_state.plz = DEFAULT_PLZ
    if "fees" not in st.session_state:
        st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)
    _ensure_fee_widget_defaults()
    st.text_input("PLZ", key="plz")
    col_apply_fees, col_reset_fees = st.columns(2)
    if col_apply_fees.button("Gebühren aus PLZ", use_container_width=True):
        st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)
        _sync_fee_widget_state()
    if col_reset_fees.button("Zurück-setzen", use_container_width=True):
        st.session_state.fees = estimate_fees_from_plz(DEFAULT_PLZ)
        _sync_fee_widget_state()
    st.session_state.fees["umlagen_ct"] = st.number_input(
        "Umlagen gesamt (ct/kWh)",
        min_value=0.0,
        max_value=10.0,
        value=float(st.session_state["fee_umlagen"]),
        step=0.001,
        key="fee_umlagen",
    )
    st.session_state.fees["konzessionsabgabe_ct"] = st.number_input(
        "Konzessionsabgabe (ct/kWh)",
        min_value=0.0,
        max_value=5.0,
        value=float(st.session_state["fee_konzessionsabgabe"]),
        step=0.01,
        key="fee_konzessionsabgabe",
    )
    st.session_state.fees["netzentgelt_ct"] = st.number_input(
        "Netzentgelt pauschal (ct/kWh)",
        min_value=0.0,
        max_value=50.0,
        value=float(st.session_state["fee_netzentgelt"]),
        step=0.1,
        key="fee_netzentgelt",
    )
    st.session_state.fees["mwst"] = st.number_input(
        "MwSt (%)",
        min_value=0,
        max_value=25,
        value=int(st.session_state["fee_mwst"]),
        step=1,
        key="fee_mwst",
    )
# ---------------------------------------------------------
# Top controls
# ---------------------------------------------------------
SOURCE_OPTIONS = {
    "SMARD (Bundesnetzagentur)": "SMARD",
    "ENTSO-E Combined": "ENTSOE-COMBINED",
    "ENTSO-E - 1 (12:00 Auction)": "ENTSOE-1",
    "ENTSO-E - 2 (10:15 EXAA Auction)": "ENTSOE-2"
}
col_source, col_res = st.columns(2)
with col_source:
    selected_source_label = st.selectbox("Datenquelle", list(SOURCE_OPTIONS.keys()), index=1)
    data_source_choice = SOURCE_OPTIONS[selected_source_label]
with col_res:
    resolution_disabled = data_source_choice != "SMARD"
    resolution_is_quarterhour = st.toggle(
        "Viertelstündlich",
        value=True,
        disabled=resolution_disabled,
        help="Schaltet zwischen Viertelstunden- und Stundenauflösung (nur SMARD).",
    )
    resolution_choice = "PT15M" if resolution_is_quarterhour else "PT60M"
if "entsoe_token" not in st.session_state:
    st.session_state.entsoe_token = st.secrets.get("entsoe_token", "")
if "entsoe_days_back" not in st.session_state:
    st.session_state.entsoe_days_back = 5
if "entsoe_days_forward" not in st.session_state:
    st.session_state.entsoe_days_forward = 2
with st.sidebar:
    if data_source_choice.startswith("ENTSOE"):
        st.subheader("ENTSO-E Einstellungen")
        st.session_state.entsoe_days_back = st.slider(
            "Tage zurück",
            min_value=1,
            max_value=90,
            value=st.session_state.entsoe_days_back,
        )
        st.session_state.entsoe_days_forward = st.slider(
            "Tage voraus",
            min_value=1,
            max_value=7,
            value=st.session_state.entsoe_days_forward,
        )
# ---------------------------------------------------------
# Datenquellen laden
# ---------------------------------------------------------
class PriceDataError(Exception):
    """Raised when a selected data source cannot provide usable data."""

def prepare_price_dataframe(
    df_raw: pd.DataFrame, 
    fees: dict,
    tz: str = "Europe/Berlin"
) -> pd.DataFrame:
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
    # --- Add fee and total price calculations ---
    df_prepared["spot_ct"] = df_prepared["ct_per_kwh"]
    fees_excl_vat_ct = (
        fees["stromsteuer_ct"]
        + fees["umlagen_ct"]
        + fees["konzessionsabgabe_ct"]
        + fees["netzentgelt_ct"]
    )
    df_prepared["fees_excl_vat_ct"] = fees_excl_vat_ct
    vat_on_price_and_fees = (df_prepared["spot_ct"] + fees_excl_vat_ct) * (fees["mwst"] / 100.0)
    df_prepared["vat_ct"] = vat_on_price_and_fees
    df_prepared["fees_incl_vat_ct"] = fees_excl_vat_ct + vat_on_price_and_fees
    df_prepared["total_ct"] = df_prepared["spot_ct"] + df_prepared["fees_incl_vat_ct"]

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
    entsoe_days_back: int,
    entsoe_days_forward: int,
) -> tuple[pd.DataFrame, dict]:
    if source == "SMARD":
        df = price_sources.fetch_smard_day_ahead(resolution=resolution)
        meta = {"region": "DE-LU", "raw_resolution": resolution, "source_id": "SMARD"}
        return df, meta
    
    if (source == "ENTSOE-1") or (source == "ENTSOE-2"):
        if not entsoe_token:
            raise PriceDataError("Für ENTSO-E wird ein API Token benötigt.")
        now_utc = dt.datetime.now(dt.timezone.utc)
        start_utc = (now_utc - dt.timedelta(days=entsoe_days_back)).replace(minute=0, second=0, microsecond=0)
        end_utc = (now_utc + dt.timedelta(days=entsoe_days_forward)).replace(minute=0, second=0, microsecond=0)
        if source == "ENTSOE-2":
            position = '2'
        else:
            position = '1'

        df = price_sources.fetch_entsoe_day_ahead(
            token=entsoe_token,
            start_utc=start_utc,
            end_utc=end_utc,
            position=position
        )
        meta = {"region": "DE-LU", "raw_resolution": None, "source_id": source}
        return df, meta
    
    if source == "ENTSOE-COMBINED":
        if not entsoe_token:
            raise PriceDataError("Für ENTSO-E wird ein API Token benötigt.")
        now_utc = dt.datetime.now(dt.timezone.utc)
        start_utc = (now_utc - dt.timedelta(days=entsoe_days_back)).replace(minute=0, second=0, microsecond=0)
        end_utc = (now_utc + dt.timedelta(days=entsoe_days_forward)).replace(minute=0, second=0, microsecond=0)
        
        # Fetch both positions
        df_both = price_sources.fetch_entsoe_day_ahead(token=entsoe_token, start_utc=start_utc, end_utc=end_utc, position=['1', '2'])
        
        # Prioritize position '1' and use '2' as a fallback
        df_sorted = df_both.sort_values(by=['ts', 'position'], ascending=[True, True])
        df = df_sorted.drop_duplicates(subset=['ts'], keep='first')
        meta = {"region": "DE-LU", "raw_resolution": None, "source_id": source}
        return df, meta
    raise PriceDataError(f"Unbekannte Datenquelle: {source}")

# ------------------------------------------------------
# Call API and get data df_raw
# ------------------------------------------------------
try:
    df_raw, data_meta = load_price_data(
        source=data_source_choice,
        resolution=resolution_choice,
        entsoe_token=st.session_state.entsoe_token,
        entsoe_days_back=st.session_state.entsoe_days_back,
        entsoe_days_forward=st.session_state.entsoe_days_forward,
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

# 1) Build full dataset with valid prices 
df_all = prepare_price_dataframe(df_raw, fees=st.session_state.fees)
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
# 3) Prepare data for chart and stats
df_chart = df_all
spot_series = df_chart["spot_ct"].astype(float)
total_series = df_chart["total_ct"].astype(float)
fees_series = df_chart["fees_incl_vat_ct"].astype(float)
time_series = df_chart["ts"].dt.tz_convert(tz_berlin).dt.tz_localize(None)
position_series = df_chart["position"].astype(str).tolist() if "position" in df_chart else ["1"] * len(df_chart)
# fees_hover = fees_series.to_numpy()
fees_hover = fees_series.round(6).tolist()
initial_view = (default_start, default_end)

def _normalize_ts(value) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = tz_berlin.localize(ts)
    else:
        ts = ts.tz_convert(tz_berlin)
    return ts

local_min = _normalize_ts(min_ts).floor('h')
local_max = _normalize_ts(max_ts).ceil('h')
default_start_local = _normalize_ts(initial_view[0]).floor('h')
default_end_local = _normalize_ts(initial_view[1]).ceil('h')
slider_min = local_min.to_pydatetime().replace(tzinfo=None)
slider_max = local_max.to_pydatetime().replace(tzinfo=None)
slider_default = (
    default_start_local.to_pydatetime().replace(tzinfo=None),
    default_end_local.to_pydatetime().replace(tzinfo=None),
)
slider_key = 'range_slider'
if slider_key not in st.session_state:
    st.session_state[slider_key] = slider_default
pending_key = 'range_slider_pending'
if pending_key in st.session_state:
    pending_start, pending_end = st.session_state.pop(pending_key)
    pending_start = max(slider_min, pending_start)
    pending_end = min(slider_max, pending_end)
    if pending_end < pending_start:
        pending_end = min(slider_max, pending_start + (default_end_local - default_start_local))
    st.session_state[slider_key] = (pending_start, pending_end)
current_start, current_end = st.session_state[slider_key]
current_duration = current_end - current_start
view_start = _normalize_ts(current_start)
view_end = _normalize_ts(current_end)
view_start = max(min_ts, view_start)
view_end = min(max_ts, view_end)
if view_start >= view_end:
    view_end = min(max_ts, view_start + pd.Timedelta(hours=1))
df_view = df_chart[(df_chart['ts'] >= view_start) & (df_chart['ts'] <= view_end)].copy()
range_start_ts = pd.Timestamp(view_start)
range_end_ts = pd.Timestamp(view_end)
range_start = range_start_ts.tz_localize(None) if range_start_ts.tzinfo else range_start_ts
range_end = range_end_ts.tz_localize(None) if range_end_ts.tzinfo else range_end_ts
view_series = df_view['total_ct'] if not df_view.empty else df_chart['total_ct']
if view_series.empty:
    view_series = pd.Series([0.0])
y_max = float(view_series.max())
y_max = max(y_max, 1.0)
padding = y_max * 0.05
fig = go.Figure()

# --- Generate vertical lines for days, midday, and now ---
day_shapes = []

# Iterate through each day in the visible range to create day/midday markers
loop_day_start = view_start.replace(hour=0, minute=0, second=0, microsecond=0)

current_ts = loop_day_start
while current_ts <= view_end:
    # Midnight line (solid)
    day_shapes.append(go.layout.Shape(
        type="line",
        x0=current_ts, x1=current_ts, y0=0, y1=1, yref="paper",
        line=dict(color="rgba(128, 128, 128, 0.5)", width=1)
    ))
    # Midday line (dashed)
    midday_ts = current_ts.replace(hour=12)
    if midday_ts < view_end:
        day_shapes.append(go.layout.Shape(
            type="line",
            x0=midday_ts, x1=midday_ts, y0=0, y1=1, yref="paper",
            line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash")
        ))
    current_ts += dt.timedelta(days=1)

# Add a special line for the current time ("now")
now_local = dt.datetime.now(tz=tz_berlin)
if view_start <= now_local <= view_end:
    day_shapes.append(go.layout.Shape(
        type="line",
        x0=now_local, x1=now_local, y0=0, y1=1, yref="paper",
        line=dict(color="purple", width=1.5, dash="dot"),
        name="Jetzt"
    ))
# --- End of line generation ---

# --- Plotting Logic ---

# Base hovertemplate for spot price
spot_hovertemplate = "Börsenstrompreis: %{y:.1f} ct/kWh<extra></extra>"
custom_data_spot = None

# If the source is ENTSOE-related, add the position to the hover text
if data_source_choice.startswith("ENTSOE"):
    spot_hovertemplate = (
        "Börsenstrompreis: %{y:.1f} ct/kWh<br>" +
        "ENTSO-E Position: %{customdata}<extra></extra>"
    )
    custom_data_spot = position_series

fig.add_trace(go.Scatter(
    x=time_series, 
    y=spot_series, 
    name="Börsenstrompreis", 
    mode="lines",
    line_shape="hv", 
    line=dict(width=0.8, color="#1f77b4"),
    fill="tozeroy", 
    fillcolor="rgba(31, 119, 180, 0.25)",
    customdata=custom_data_spot,
    hovertemplate=spot_hovertemplate,
))

fig.add_trace(go.Scatter(
    x=time_series, y=total_series, name="Gesamtpreis", mode="lines",
    line_shape="hv", line=dict(width=1.2, color="#d62728"),
    fill="tonexty", fillcolor="rgba(255, 127, 14, 0.35)",
    customdata=fees_hover,
    hovertemplate="Gesamtpreis: %{y:.1f} ct/kWh<br>Gebühren: %{customdata:.1f} ct/kWh<extra></extra>",
))

fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(
        title="Zeit (lokal)",
        type="date",
        range=[range_start, range_end],
        rangeslider=dict(visible=True),
        fixedrange=True,
    ),
    yaxis=dict(
        title="ct/kWh",
        range=[0.0, y_max + padding],
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
    ),
    hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", namelength=-1),
    shapes=day_shapes,
)
st.plotly_chart(fig, use_container_width=True)
start_naive, end_naive = st.slider(
    'Sichtbaren Zeitraum wählen',
    min_value=slider_min,
    max_value=slider_max,
    value=st.session_state[slider_key],
    step=dt.timedelta(hours=1),
    format='DD.MM.YYYY HH:mm',
    key=slider_key,
)
current_start = start_naive
current_end = end_naive
current_duration = current_end - current_start
col_prev, col_reset, col_next = st.columns(3)
if col_prev.button('⟲ –24h'):
    new_start = current_start - dt.timedelta(hours=24)
    new_end = current_end - dt.timedelta(hours=24)
    if new_start < slider_min:
        new_start = slider_min
        new_end = min(slider_max, new_start + current_duration)
    st.session_state['range_slider_pending'] = (new_start, new_end)
    st.rerun()
if col_reset.button('Standardfenster'):
    # Re-calculate the ideal default window and clamp it to the slider's bounds
    ideal_start = tz_berlin.localize(dt.datetime.combine(today, dt.time(0, 0)))
    ideal_end = ideal_start + dt.timedelta(days=1)
    
    clamped_start = max(slider_min, ideal_start.replace(tzinfo=None))
    clamped_end = min(slider_max, ideal_end.replace(tzinfo=None))
    st.session_state['range_slider_pending'] = (clamped_start, clamped_end)
    st.rerun()
if col_next.button('+24h ⟳'):
    new_start = current_start + dt.timedelta(hours=24)
    new_end = current_end + dt.timedelta(hours=24)
    if new_end > slider_max:
        new_end = slider_max
        new_start = max(slider_min, new_end - current_duration)
    st.session_state['range_slider_pending'] = (new_start, new_end)
    st.rerun()
current_start, current_end = st.session_state[slider_key]
view_start = _normalize_ts(current_start)
view_end = _normalize_ts(current_end)
view_start = max(min_ts, view_start)
view_end = min(max_ts, view_end)
if view_start >= view_end:
    view_end = min(max_ts, view_start + pd.Timedelta(hours=1))

st.subheader("Statistik")
include_fees = st.toggle("Inkl. Gebühren", value=True, key="include_fees_toggle")

with st.container():
    if df_view.empty:
        st.warning("Im ausgewählten Zeitbereich liegen keine Preispunkte vor.")
    else:
        metric_col = "total_ct" if include_fees else "spot_ct"
        now_local = now
        idx = df_view['ts'].searchsorted(now_local, side="left")
        idx = min(max(idx, 0), len(df_view) - 1)
        current_price = float(df_view.iloc[idx][metric_col])
        avg = float(df_view[metric_col].mean())
        min_price = float(df_view[metric_col].min())
        max_price = float(df_view[metric_col].max())
        c1, c2 = st.columns(2)
        c1.metric("aktuell", f"{current_price:.1f} ct/kWh")
        c2.metric("Durchschnitt", f"{avg:.1f} ct/kWh")
        c3, c4 = st.columns(2)
        c3.metric("min", f"{min_price:.1f} ct/kWh")
        c4.metric("max", f"{max_price:.1f} ct/kWh")
        st.caption(
            f"Sichtbarer Bereich: {view_start.strftime('%d.%m %H:%M')} – {view_end.strftime('%d.%m %H:%M')}"
        )
        st.caption(
            f"Auflösung: {used_resolution} · Region: {used_region} · Quelle: {selected_source_label}"
        )
st.caption(
    f"Standard-Fenster: {start_window.strftime('%d.%m %H:%M')} – {(start_window + dt.timedelta(days=1)).strftime('%d.%m %H:%M')} "
    "Nutze den Range-Slider unter dem Diagramm, um andere Zeitbereiche auszuwählen."
)

import requests, pandas as pd, datetime as dt
import pytz
import streamlit as st
import altair as alt
import time
from typing import Optional

# ------------------------------ #
# Seite & Grund-Setup
# ------------------------------ #
st.set_page_config(page_title="Day-Ahead Strompreis (DE/LU)", page_icon="⚡", layout="centered")
st.title("⚡ Day-Ahead Strompreise (DE/LU)")
st.caption("Quelle: SMARD / EPEX Day-Ahead. Gebühren werden aus PLZ vorbefüllt (Heuristik, MVP) und können manuell angepasst werden.")

# ------------------------------ #
# Hilfsfunktionen
# ------------------------------ #
def estimate_fees_from_plz(plz: str):
    """
    MVP-Heuristik für Vorbelegung:
    - Konzessionsabgabe (KAV) grob nach 'Großstadt' / 'Mittelstadt' / sonst
    - Netzentgelt: sehr grobe Regionalpauschale (nur als Platzhalter!)
    Alle Werte in ct/kWh, MwSt in %.
    """
    # sehr grobe Großstadt- und Mittelstadt-Erkennung über PLZ-Prefix
    prefix = plz.strip()[:2]
    big_cities = {
        "10","11","12","13","14",  # Berlin
        "20","21","22",            # Hamburg
        "80","81","82","85","86",  # München/OBB
        "50","51","53",            # Köln/Bonn
        "60","61","63","65",       # Frankfurt/Rhein-Main
        "70","71","72","73",       # Stuttgart-Region
        "40","41","42","44","45",  # Düsseldorf/Ruhr
        "30","31","32",            # Hannover/Umfeld
        "90","91",                 # Nürnberg/Fürth/ER
        "01","04"                  # Dresden/Leipzig
    }
    mid_cities = {"33","34","47","48","49","54","55","56","66","67","68","69","74","75","76","77","78","79","83","84","87","88","89"}
    # KAV Staffel (gesetzlich festgelegt) – hier als simple Zuordnung
    if prefix in big_cities:
        kav = 2.39
    elif prefix in mid_cities:
        kav = 1.99
    else:
        kav = 1.59  # Standard-Annahme für viele Gemeinden (Haushalte)

    # extrem grobe, rein illustrative Netzentgelt-Pauschalen (MVP!)
    north_east = {"01","02","03","04","10","11","12","13","14","17","18","19","23","24","17","18","19"}
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

def compute_price(ct_per_kwh, fees, include_fees: bool):
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

# ------------------------------ #
# Sidebar: PLZ + manuelle Gebühren
# ------------------------------ #
with st.sidebar:
    st.header("Gebühren einstellen")
    # Session-State Defaults
    if "plz" not in st.session_state:
        st.session_state.plz = "82340"
    if "fees" not in st.session_state:
        st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)

    st.text_input("PLZ", key="plz")

    colb1, colb2 = st.columns([1,1])
    with colb1:
        if st.button("Vorbelegen aus PLZ"):
            st.session_state.fees = estimate_fees_from_plz(st.session_state.plz)
    with colb2:
        reset = st.button("Standard reset")
        if reset:
            st.session_state.fees = estimate_fees_from_plz("00000")  # führt zu Default-Werten

    # Manuelle Eingaben – immer editierbar, vorbefüllt aus st.session_state.fees
    st.subheader("Manuell anpassen")
    st.session_state.fees["stromsteuer_ct"] = st.number_input(
        "Stromsteuer (ct/kWh)", min_value=0.0, max_value=10.0,
        value=float(st.session_state.fees.get("stromsteuer_ct", 2.05)), step=0.01
    )
    st.session_state.fees["umlagen_ct"] = st.number_input(
        "Umlagen gesamt (ct/kWh)", min_value=0.0, max_value=10.0,
        value=float(st.session_state.fees.get("umlagen_ct", 2.651)), step=0.001
    )
    st.session_state.fees["konzessionsabgabe_ct"] = st.number_input(
        "Konzessionsabgabe (ct/kWh)", min_value=0.0, max_value=5.0,
        value=float(st.session_state.fees.get("konzessionsabgabe_ct", 1.59)), step=0.01
    )
    st.session_state.fees["netzentgelt_ct"] = st.number_input(
        "Netzentgelt pauschal (ct/kWh)", min_value=0.0, max_value=50.0,
        value=float(st.session_state.fees.get("netzentgelt_ct", 8.0)), step=0.1
    )
    st.session_state.fees["mwst"] = st.number_input(
        "MwSt (%)", min_value=0, max_value=25,
        value=int(st.session_state.fees.get("mwst", 19)), step=1
    )

# ------------------------------ #
# Haupt-UI
# ------------------------------ #
top1, top2 = st.columns(2)
with top1:
    view_mode = st.segmented_control("Ansicht", options=["Ohne Gebühren", "Inkl. Gebühren"], default="Ohne Gebühren")
with top2:
    resolution = st.selectbox("Auflösung", ["quarterhour", "hour"], index=0)

# ------------------------------ #
# Daten holen (SMARD 4169)
# ------------------------------ #
# ------ ROBUSTER SMARD-LOADER (ersetzt deinen bisherigen Ladeblock) ------
import time
from typing import Optional, Tuple, List

SERIES_ID = "4169"             # Day-Ahead Marktpreis
REGION_CANDIDATES = ["DE-LU", "DE"]  # beide probieren
SMARD_BASE = "https://www.smard.de/app"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit-SMARD/1.0)"}

class SmardError(Exception): ...
_last_tried: List[str] = []     # sammelt zuletzt getestete URLs für Diagnose

def _safe_get_json(url: str, timeout: int = 30) -> dict:
    _last_tried.append(url)
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code != 200:
        raise SmardError(f"HTTP {r.status_code} für {url}")
    ctype = r.headers.get("Content-Type", "")
    try:
        return r.json()
    except Exception:
        snippet = (r.text or "")[:120].replace("\n", " ")
        raise SmardError(f"Kein JSON von {url}. Content-Type='{ctype}', Antwort: '{snippet}...'")

def _get_index(region: str, resolution: str) -> list[int]:
    url = f"{SMARD_BASE}/chart_data/{SERIES_ID}/{region}/index_{resolution}.json"
    data = _safe_get_json(url)
    ts = data.get("timestamps") or []
    if not ts:
        raise SmardError(f"Keine timestamps in index_{resolution}.json ({region})")
    return ts

def _try_load_series(region: str, resolution: str, ts: int) -> Optional[pd.DataFrame]:
    # 1) table_data (bevorzugt laut Beispielen)
    url_table = f"{SMARD_BASE}/table_data/{SERIES_ID}/{region}/{SERIES_ID}_{region}_{resolution}_{ts}.json"
    try:
        data = _safe_get_json(url_table)
        series = data.get("series")
        if series:
            return pd.DataFrame(series, columns=["ts_ms", "eur_per_mwh"])
    except SmardError:
        pass
    # 2) chart_data als Fallback (hat teils früher Daten)
    url_chart = f"{SMARD_BASE}/chart_data/{SERIES_ID}/{region}/{SERIES_ID}_{region}_{resolution}_{ts}.json"
    try:
        data = _safe_get_json(url_chart)
        series = data.get("series")
        if series:
            return pd.DataFrame(series, columns=["ts_ms", "eur_per_mwh"])
    except SmardError:
        return None
    return None

@st.cache_data(ttl=900)
def load_smard_series(prefer_resolution: str = "quarterhour", max_backsteps: int = 12) -> Tuple[pd.DataFrame, str, str]:
    # versuche quarterhour -> hour
    resolutions = [prefer_resolution] + ([r for r in ["hour"] if r != prefer_resolution])
    for region in REGION_CANDIDATES:
        for resolution in resolutions:
            try:
                idx = _get_index(region, resolution)
            except SmardError:
                continue
            # die letzten N Timestamps rückwärts probieren
            for ts in reversed(idx[-(max_backsteps+1):]):
                df = _try_load_series(region, resolution, ts)
                if df is not None and not df.empty:
                    return df, resolution, region
                time.sleep(0.15)
    raise SmardError("Keine gültige SMARD-Datei gefunden (region/auflösung/ts).")

# --- Aufruf im Hauptfluss (ersetze deinen bisherigen try/except) ---
try:
    df_raw, used_resolution, used_region = load_smard_series(prefer_resolution="quarterhour", max_backsteps=12)
except SmardError as e:
    # zeige die letzten getesteten URLs an – super hilfreich beim Debuggen
    with st.expander("Diagnose: letzte getestete URLs"):
        for u in _last_tried[-12:]:
            st.code(u)
    st.error(
        "⚠️ Konnte SMARD-Daten nicht laden.\n\n"
        f"Fehler: {e}\n\n"
        "Tipps: 1) Später erneut versuchen (frischer Timestamp), "
        "2) 'Auflösung' in der UI auf 'hour' stellen, "
        "3) Unternehmens-Proxy prüfen."
    )
    st.stop()
# ------ ENDE ROBUSTER LOADER ------


# ---------- Benutzung im Hauptcode ----------
try:
    df_raw, used_resolution, used_region = load_smard_series(prefer_resolution="quarterhour", max_backsteps=6)
except SmardError as e:
    st.error(
        "⚠️ Konnte SMARD-Daten nicht laden.\n\n"
        f"Fehler: {e}\n\n"
        "Probier es in ein paar Minuten erneut. Fällt das häufiger auf, schalte in der UI auf 'hour' "
        "oder verringere 'max_backsteps'."
    )
    st.stop()

# ab hier wie gehabt weiterrechnen:
import pytz, datetime as dt
df_raw["ts"] = pd.to_datetime(df_raw["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
df_raw["ct_per_kwh"] = df_raw["eur_per_mwh"] * 0.1

# Heute bis Ende der Prognose
today = dt.datetime.now(tz=pytz.timezone("Europe/Berlin")).date()
df = df_raw[df_raw["ts"].dt.date >= today].copy()




# Preisansicht berechnen
include_fees = (view_mode == "Inkl. Gebühren")
df["price_view"] = df["ct_per_kwh"].apply(lambda x: compute_price(x, st.session_state.fees, include_fees))
label = "Spot (ct/kWh)" if not include_fees else "Spot inkl. Gebühren (ct/kWh)"

# Kennzahlen
if not df.empty:
    now_local = pd.Timestamp.now(tz="Europe/Berlin")
    current_idx = df["ts"].searchsorted(now_local, side="left")
    current_price = df.iloc[min(current_idx, len(df)-1)]["price_view"]
    m, M, avg = df["price_view"].min(), df["price_view"].max(), df["price_view"].mean()
else:
    current_price = m = M = avg = None

k1, k2, k3 = st.columns(3)
k1.metric("aktuell", f"{current_price:.2f} ct/kWh" if current_price is not None else "–")
k2.metric("⭣ min", f"{m:.2f} ct/kWh" if m is not None else "–")
k3.metric("⭡ max", f"{M:.2f} ct/kWh" if M is not None else "–")
st.caption(f"Durchschnitt heute: {avg:.2f} ct/kWh" if avg is not None else "")

# Chart
chart = (
    alt.Chart(df)
    .mark_area(interpolate="step-after", opacity=0.5)
    .encode(
        x=alt.X("ts:T", title="Zeit (lokal)"),
        y=alt.Y("price_view:Q", title=label),
        tooltip=[
            alt.Tooltip("ts:T", title="Zeit"),
            alt.Tooltip("price_view:Q", title=label, format=".2f")
        ]
    )
    .properties(height=260)
)
st.altair_chart(chart, use_container_width=True)

# Hinweis-Text
st.caption(
    f"PLZ: {st.session_state.plz} · Gebühren aktuell: "
    f"Stromsteuer {st.session_state.fees['stromsteuer_ct']:.3g} ct, "
    f"Umlagen {st.session_state.fees['umlagen_ct']:.3g} ct, "
    f"Konzessionsabgabe {st.session_state.fees['konzessionsabgabe_ct']:.3g} ct, "
    f"Netzentgelt {st.session_state.fees['netzentgelt_ct']:.3g} ct, "
    f"MwSt {st.session_state.fees['mwst']} %."
)

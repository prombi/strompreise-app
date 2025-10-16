import requests, pandas as pd, datetime as dt
import pytz
import streamlit as st
import altair as alt

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
BASE = "https://www.smard.de/app"
FILTER = "4169"  # Marktpreis
REGION = "DE"
index_url = f"{BASE}/chart_data/{FILTER}/{REGION}/index_{resolution}.json"
idx = requests.get(index_url, timeout=30).json()
last_ts = idx["timestamps"][-1]
data_url = f"{BASE}/table_data/{FILTER}/{REGION}/{FILTER}_{REGION}_{resolution}_{last_ts}.json"
raw = requests.get(data_url, timeout=30).json()

series = raw["series"]  # [[ts_ms, eur_per_mwh], ...]
df = pd.DataFrame(series, columns=["ts_ms", "eur_per_mwh"])
df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
df["ct_per_kwh"] = df["eur_per_mwh"] * 0.1  # €/MWh → ct/kWh

# Heute bis Ende der Prognose
today = dt.datetime.now(tz=pytz.timezone("Europe/Berlin")).date()
df = df[df["ts"].dt.date >= today].copy()

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

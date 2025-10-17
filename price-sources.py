from __future__ import annotations

import datetime as dt
import math
from typing import Literal, Optional

import pandas as pd
import pytz
import requests
from xml.etree import ElementTree as ET

TZ_BERLIN = pytz.timezone("Europe/Berlin")

# -----------------------------
# 1) SMARD (Bundesnetzagentur)
# -----------------------------
# region codes (SMARD chart data): DE-LU is "DE"
# filters: 4169 = Day-Ahead, market price €/MWh (since Oct 2025: quarterhour available)
# resolution: "quarterhour" or "hour"
# Docs & parameter list: https://github.com/bundesAPI/smard-api and OpenAPI view
# -----------------------------

SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_FILTER_DA = "4169"  # Day-Ahead price
SMARD_REGION_DE = "DE"

def fetch_smard_day_ahead(
    resolution: Literal["quarterhour", "hour"] = "quarterhour",
    region: str = SMARD_REGION_DE,
    max_backsteps: int = 12,
) -> pd.DataFrame:
    """
    Fetch Day-Ahead market prices from SMARD for DE/LU.
    Returns a DataFrame with:
      ts (tz-aware Europe/Berlin), eur_per_mwh (float), ct_per_kwh (float), source (str="SMARD"), resolution
    Will walk back through available timestamp indices until it finds a table file.
    """
    # Step 1: get timestamp index list for this filter/region/resolution
    idx_url = f"{SMARD_BASE}/{SMARD_FILTER_DA}/{region}/index_{resolution}.json"
    idx = requests.get(idx_url, timeout=30)
    idx.raise_for_status()
    idx_json = idx.json()

    # The "timestamps" array lists available dataset anchors (ms since epoch)
    stamps = idx_json.get("timestamps") or []
    if not stamps:
        raise RuntimeError("SMARD: no timestamps found for Day-Ahead price index")

    # Step 2: walk backwards to find a valid table_data file
    errors = []
    for i, ts_ms in enumerate(sorted(stamps, reverse=True)[:max_backsteps]):
        # table_data/{filter}/{region}/{filter}_{region}_{resolution}_{timestamp}.json
        tbl_url = (
            f"{SMARD_BASE}/table_data/{SMARD_FILTER_DA}/{region}/"
            f"{SMARD_FILTER_DA}_{region}_{resolution}_{ts_ms}.json"
        )
        try:
            r = requests.get(tbl_url, timeout=30)
            r.raise_for_status()
            data = r.json()
            series = data.get("series", [])
            if not series:
                errors.append(f"empty series at {i} ({tbl_url})")
                continue

            # SMARD emits [ [timestamp_ms, value_eur_per_mwh], ... ]
            rows = []
            for row in series:
                if not isinstance(row, list) or len(row) < 2:
                    continue
                ts_ms_i, value = row[0], row[1]
                if value is None:
                    continue
                ts = pd.to_datetime(ts_ms_i, unit="ms", utc=True).tz_convert(TZ_BERLIN)
                rows.append({"ts": ts, "eur_per_mwh": float(value)})

            if not rows:
                errors.append(f"no non-null rows at {i} ({tbl_url})")
                continue

            df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
            df["ct_per_kwh"] = df["eur_per_mwh"] * 0.1
            df["source"] = "SMARD"
            df["resolution"] = resolution
            return df

        except Exception as ex:
            errors.append(f"{type(ex).__name__} at {i}: {ex}")

    raise RuntimeError(
        "SMARD: Could not find a valid Day-Ahead table_data file. "
        f"Tried {len(errors)} candidates; last errors: {errors[-3:]}"
    )


# ----------------------------------------
# 2) ENTSO-E Transparency Platform (REST)
# ----------------------------------------
# Official data item: Day-ahead Prices [12.1.D]
# documentType=A44, in_Domain=EIC, out_Domain=EIC (same value), periodStart/End UTC (yyyyMMddHHmm)
# Response is XML Publication_MarketDocument with Periods containing resolution (PT60M or PT15M)
# and Points with <position> and <price.amount>.
# User must register & get API token: https://transparency.entsoe.eu/  (Guide)
# ----------------------------------------

ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"
DE_LU_EIC = "10Y1001A1001A82H"  # Bidding zone code for DE/LU (BZN)

def _fmt_utc(ts: dt.datetime) -> str:
    """Format timezone-aware dt as ENTSO-E yyyyMMddHHmm in UTC."""
    if ts.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")
    ts_utc = ts.astimezone(dt.timezone.utc)
    return ts_utc.strftime("%Y%m%d%H%M")

def _res_to_minutes(res_str: str) -> int:
    # Expect "PT15M", "PT60M", etc.
    if not res_str.startswith("PT") or not res_str.endswith("M"):
        raise ValueError(f"Unexpected ENTSO-E resolution: {res_str}")
    return int(res_str[2:-1])

def fetch_entsoe_day_ahead(
    token: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
    eic_bzn: str = DE_LU_EIC,
) -> pd.DataFrame:
    """
    Fetch Day-Ahead prices via ENTSO-E REST (XML), returning:
      ts (Europe/Berlin), eur_per_mwh, ct_per_kwh, source="ENTSOE", resolution ("PT15M"/"PT60M").
    start_utc / end_utc must be tz-aware in any tz (converted to UTC for the query).
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc and end_utc must be timezone-aware")

    params = {
        "securityToken": token,
        "documentType": "A44",
        "in_Domain": eic_bzn,
        "out_Domain": eic_bzn,
        "periodStart": _fmt_utc(start_utc),
        "periodEnd": _fmt_utc(end_utc),
    }

    resp = requests.get(ENTSOE_BASE, params=params, timeout=45)
    if resp.status_code == 401:
        raise PermissionError("ENTSO-E: Unauthorized (check token)")
    resp.raise_for_status()

    # "No matching data" comes back as an Acknowledgement document (HTTP 200)
    if b"Acknowledgement_MarketDocument" in resp.content and b"No matching data" in resp.content:
        return pd.DataFrame(columns=["ts", "eur_per_mwh", "ct_per_kwh", "source", "resolution"])

    # Parse XML
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as ex:
        raise RuntimeError(f"ENTSO-E: XML parse error: {ex}")

    rows = []
    # Day-ahead doc can include multiple TimeSeries
    for ts_node in root.findall(".//ns:TimeSeries", ns):
        # Resolution per Period (could differ)
        for period in ts_node.findall(".//ns:Period", ns):
            # Time interval is UTC in the response (per docs)
            start_txt = period.findtext("ns:timeInterval/ns:start", namespaces=ns)
            res_txt = period.findtext("ns:resolution", namespaces=ns)
            if not (start_txt and res_txt):
                continue
            try:
                res_min = _res_to_minutes(res_txt)
            except Exception:
                continue

            start_dt_utc = pd.to_datetime(start_txt, utc=True)

            for p in period.findall("ns:Point", ns):
                pos_txt = p.findtext("ns:position", namespaces=ns)
                val_txt = p.findtext("ns:price.amount", namespaces=ns)
                if not (pos_txt and val_txt):
                    continue
                try:
                    pos = int(pos_txt)
                    price = float(val_txt)  # Currency per MWh (usually EUR/MWh)
                except Exception:
                    continue

                # position is 1-based
                ts_utc = start_dt_utc + pd.Timedelta(minutes=(pos - 1) * res_min)
                ts_local = ts_utc.tz_convert(TZ_BERLIN)
                rows.append(
                    {
                        "ts": ts_local,
                        "eur_per_mwh": price,
                        "ct_per_kwh": price * 0.1,
                        "source": "ENTSOE",
                        "resolution": f"PT{res_min}M",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["ts", "eur_per_mwh", "ct_per_kwh", "source", "resolution"])

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


# ---------------------------------------------------------
# 3) smartENERGY (AT) – light client (public docs are sparse)
# ---------------------------------------------------------
# Their page states: 15-min data, and “ab 17:00 Preise für den nächsten Tag zusätzlich”.
# Endpoint details / auth are not formally documented on the public page; some
# community projects use their JSON endpoints (may change).
# We expose a tiny generic fetcher so you can plug in the actual URL & params.
# ---------------------------------------------------------

def fetch_smartenergy(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    ts_field: str = "start_timestamp",  # adjust to real field name
    price_field: str = "price_eur_per_mwh",  # adjust to real field name
    tz: str = "Europe/Vienna",  # AT site; convert to Berlin by default below
) -> pd.DataFrame:
    """
    Generic fetcher for smartENERGY’s JSON.
    You must pass the correct endpoint URL (and any params/headers if required).
    Returns ts (Berlin), eur_per_mwh, ct_per_kwh, source="SMARTENERGY", resolution guessed (15 min).
    """
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Expect either a list of records or {"data":[...]}
    records = data.get("data", data)
    if not isinstance(records, list):
        raise RuntimeError("smartENERGY: unexpected JSON structure; provide a custom parser")

    tz_src = pytz.timezone(tz)
    out = []
    for rec in records:
        if rec.get(price_field) is None or rec.get(ts_field) is None:
            continue
        # try both ISO and epoch seconds
        ts_val = rec[ts_field]
        if isinstance(ts_val, (int, float)):
            ts_local = pd.to_datetime(ts_val, unit="s", utc=True).tz_convert(tz_src)
        else:
            ts_local = pd.to_datetime(ts_val)
            if ts_local.tzinfo is None:
                ts_local = tz_src.localize(ts_local)

        ts_berlin = ts_local.tz_convert(TZ_BERLIN)
        eur_per_mwh = float(rec[price_field])
        out.append(
            {
                "ts": ts_berlin,
                "eur_per_mwh": eur_per_mwh,
                "ct_per_kwh": eur_per_mwh * 0.1,
                "source": "SMARTENERGY",
                "resolution": "PT15M",
            }
        )

    df = pd.DataFrame(out).sort_values("ts").reset_index(drop=True)
    return df

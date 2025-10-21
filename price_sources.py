from __future__ import annotations

import datetime as dt
import math
from typing import Literal, Optional

import pandas as pd
import pytz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from xml.etree import ElementTree as ET

TZ_BERLIN = pytz.timezone("Europe/Berlin")

# -----------------------------
# 1) SMARD (Bundesnetzagentur)
# -----------------------------
# filters: 4169 = Day-Ahead market price (€/MWh)
# regions used by SMARD here: use "DE-LU" first, then "DE"
SMARD_ROOT = "https://www.smard.de/app"
SMARD_CHART = f"{SMARD_ROOT}/chart_data"
SMARD_TABLE = f"{SMARD_ROOT}/table_data"
SMARD_FILTER_DA = "4169"
SMARD_REGIONS = ["DE-LU", "DE"]
SMARD_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Streamlit-SMARD/1.0)"}

def _smard_get_json(url: str, timeout: int = 30) -> dict:
    r = requests.get(url, headers=SMARD_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_smard_day_ahead(
    resolution: Literal["quarterhour", "hour"] = "quarterhour",
    max_backsteps: int = 16,
) -> pd.DataFrame:
    """
    Fetch Day-Ahead market prices from SMARD for DE/LU.
    Version 0.0.2
    Returns DataFrame: ts (Europe/Berlin), eur_per_mwh, ct_per_kwh, source="SMARD", resolution.
    Strategy:
      - try regions in order: DE-LU, DE
      - get index_{resolution}.json from chart_data
      - walk timestamps backwards, try table_data file first, then chart_data file
      - if quarterhour fails entirely, fall back to hour
    """
    errors = []

    def _try_one_resolution(res: str) -> Optional[pd.DataFrame]:
        for region in SMARD_REGIONS:
            # 1) index list
            idx_url = f"{SMARD_CHART}/{SMARD_FILTER_DA}/{region}/index_{res}.json"
            try:
                idx_json = _smard_get_json(idx_url)
                stamps = idx_json.get("timestamps") or []
                if not stamps:
                    errors.append(f"no timestamps for {region}/{res}")
                    continue
            except Exception as ex:
                errors.append(f"index err {region}/{res}: {ex}")
                continue

            # 2) walk back through last N candidates
            for ts_ms in sorted(stamps, reverse=True)[:max_backsteps]:
                # (a) table_data (preferred)
                tbl = f"{SMARD_TABLE}/{SMARD_FILTER_DA}/{region}/{SMARD_FILTER_DA}_{region}_{res}_{ts_ms}.json"
                # (b) chart_data (fallback)
                ch  = f"{SMARD_CHART}/{SMARD_FILTER_DA}/{region}/{SMARD_FILTER_DA}_{region}_{res}_{ts_ms}.json"
                for url in (tbl, ch):
                    try:
                        data = _smard_get_json(url)
                        series = data.get("series") or []
                        if not series:
                            errors.append(f"empty series {region}/{res}/{ts_ms} ({'table' if url==tbl else 'chart'})")
                            continue
                        rows = []
                        for row in series:
                            if not isinstance(row, list) or len(row) < 2:
                                continue
                            ts_i, val = row[0], row[1]
                            if val is None:
                                continue
                            ts = pd.to_datetime(ts_i, unit="ms", utc=True).tz_convert(TZ_BERLIN)
                            rows.append({"ts": ts, "eur_per_mwh": float(val)})
                        if not rows:
                            errors.append(f"no valid rows {region}/{res}/{ts_ms}")
                            continue
                        df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
                        df["ct_per_kwh"] = df["eur_per_mwh"] * 0.1
                        df["source"] = "SMARD"
                        df["resolution"] = res
                        return df
                    except Exception as ex:
                        errors.append(f"{type(ex).__name__} {region}/{res}/{ts_ms}: {ex}")
                        continue
        return None

    # Try preferred resolution, then fallback
    for res in [resolution, "hour"] if resolution != "hour" else ["hour"]:
        df = _try_one_resolution(res)
        if df is not None:
            return df

    raise RuntimeError(
        "SMARD: Could not find a valid Day-Ahead file. "
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
    mrid: Optional[str | list[str]] = None,
    position: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """
    Fetch Day-Ahead prices via ENTSO-E REST (XML), returning:
      ts (Europe/Berlin), eur_per_mwh, ct_per_kwh, source="ENTSOE",
      resolution ("PT15M"/"PT60M"), and mrid.
    start_utc / end_utc must be tz-aware in any tz (converted to UTC for the query).
    eic_bzn: Bidding zone EIC code (e.g., DE-LU).
    mrid: If specified, filters for the TimeSeries with this specific mRID or list of mRIDs.
    position: If specified, filters for the TimeSeries with this specific
              classificationSequence_AttributeInstanceComponent.position.
    """
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("start_utc and end_utc must be timezone-aware")

    # --- Retry logic for network resilience ---
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=1,  # Wait 1s, 2s, 4s between retries
        status_forcelist=[429, 500, 502, 503, 504], # Status codes to retry on
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)

    # --- Chunking logic to handle large date ranges ---
    # The ENTSO-E API often times out on requests for long periods.
    # We split the total period into smaller chunks (e.g., 7 days) and fetch them sequentially.
    all_dfs = []
    current_start = start_utc
    chunk_size_days = 7

    while current_start < end_utc:
        chunk_end = min(current_start + dt.timedelta(days=chunk_size_days), end_utc)

        params = {
            "securityToken": token,
            "documentType": "A44",
            "in_Domain": eic_bzn,
            "out_Domain": eic_bzn,
            "periodStart": _fmt_utc(current_start),
            "periodEnd": _fmt_utc(chunk_end),
        }

        try:
            resp = session.get(ENTSOE_BASE, params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            # If even a single chunk fails after retries, we raise an error.
            raise RuntimeError(f"ENTSO-E: Network request for chunk {current_start.date()} failed: {e}")

        if resp.status_code == 401:
            raise PermissionError("ENTSO-E: Unauthorized (check token)")
        resp.raise_for_status()

        # "No matching data" comes back as an Acknowledgement document (HTTP 200)
        if b"Acknowledgement_MarketDocument" in resp.content and b"No matching data" in resp.content:
            # This chunk is empty, continue to the next one
            current_start = chunk_end
            continue

        # If we get here, we have valid XML data for the chunk
        all_dfs.append(resp.content)
        current_start = chunk_end

    if not all_dfs:
        # If all chunks were empty, return an empty DataFrame
        return pd.DataFrame(columns=["ts", "eur_per_mwh", "ct_per_kwh", "source", "resolution", "mrid"])

    # --- End of chunking logic ---

    # Now, parse all the collected XML documents
    all_rows = []
    for xml_content in all_dfs:
        chunk_rows = _parse_entsoe_xml(xml_content, mrid, position)
        all_rows.extend(chunk_rows)

    if not all_rows:
        return pd.DataFrame(columns=["ts", "eur_per_mwh", "ct_per_kwh", "source", "resolution", "mrid"])

    df = pd.DataFrame(all_rows).sort_values("ts").drop_duplicates().reset_index(drop=True)
    return df


def _parse_entsoe_xml(
    xml_content: bytes,
    mrid: Optional[str | list[str]] = None,
    position: Optional[str | list[str]] = None,
) -> list[dict]:
    """Helper function to parse a single ENTSO-E XML response."""
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as ex:
        raise RuntimeError(f"ENTSO-E: XML parse error: {ex}")

    # Construct XPath to select TimeSeries nodes, filtering by mRID and/or position if provided.
    timeseries_xpath = ".//ns:TimeSeries"
    filters = []

    # Helper to create a list from a string or list
    def _to_list(value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    mrid_list = _to_list(mrid)
    if mrid_list:
        mrid_conditions = " or ".join([f"ns:mRID='{i}'" for i in mrid_list])
        filters.append(f"({mrid_conditions})")

    position_list = _to_list(position)
    if position_list:
        # The tag name is unusual, but this XPath should work with ElementTree
        pos_conditions = " or ".join([f"ns:classificationSequence_AttributeInstanceComponent.position='{p}'" for p in position_list])
        filters.append(f"({pos_conditions})")

    if filters:
        combined_filters = " and ".join(filters)
        timeseries_xpath = f".//ns:TimeSeries[{combined_filters}]"
    
    rows = []
    # Day-ahead doc can include multiple TimeSeries

    for ts_node in root.findall(timeseries_xpath, ns):
        # Resolution per Period (could differ)
        ts_mrid = ts_node.findtext("ns:mRID", namespaces=ns)
        if not ts_mrid:
            continue
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
                        "mrid": ts_mrid,
                    }
                )
    return rows


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

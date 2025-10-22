import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Mapping, Dict, Any, Iterable, Callable, Union

def plot_segments_by_category(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    value_col: str = "value",
    category_col: str = "category",
    params_by_category: Mapping[str, Dict[str, Any]],
    scatter_defaults: Optional[Dict[str, Any]] = None,  # global defaults for every trace
    sort_by_time: bool = True,
    layout: Optional[Dict[str, Any]] = None,
    fig: Optional[go.Figure] = None
) -> go.Figure:
    """
    Render a time series as contiguous segments per `category`, each segment as its own go.Scatter.

    - Per-category params deep-merge over `scatter_defaults`.
    - Robust customdata per segment (columns, vectors, callables).
    - `fill="tonexty"` support using `tonexty_anchor`:
        * "zero" | "prev" | "<colname>" | callable(seg_df)->array
      The function inserts a hidden baseline trace immediately before the filled trace.
    - NEW: `tonexty_anchor_line_shape` (per-category) lets you set the baseline's `line_shape`.
      If omitted, it inherits the segment's `line_shape` (recommended).

    Per-category special keys (consumed by this function, not passed to go.Scatter):
      - customdata
      - tonexty_anchor
      - tonexty_anchor_line_shape
    """

    # ---------------- helpers ----------------
    CustomDataSpec = Union[None, str, Iterable[str], pd.Series, np.ndarray, Callable[[pd.DataFrame], np.ndarray]]

    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base or {})
        for k, v in (override or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def _resolve_segment_customdata(seg: pd.DataFrame, full_df: pd.DataFrame, 
                                    spec: Union[None, str, Iterable[str], pd.Series, np.ndarray, Callable[[pd.DataFrame], np.ndarray]]):
        if spec is None:
            return None
        if callable(spec):
            arr = spec(seg)
            return np.asarray(arr) if arr is not None else None
        if isinstance(spec, str):
            col = spec[4:] if spec.startswith("col:") else spec
            if col not in seg.columns:
                raise KeyError(f"customdata column '{col}' not in DataFrame segment.")
            return seg[[col]].to_numpy()
        if isinstance(spec, (list, tuple)) and all(isinstance(c, str) for c in spec):
            missing = [c for c in spec if c not in seg.columns]
            if missing:
                raise KeyError(f"customdata columns missing: {missing}")
            return seg[list(spec)].to_numpy()
        if isinstance(spec, pd.Series):
            vec = spec.to_numpy()
        elif isinstance(spec, np.ndarray):
            vec = spec
        else:
            raise TypeError("Unsupported customdata spec.")
        if vec.shape[0] != len(full_df):
            raise ValueError(f"customdata vector must have length {len(full_df)} (got {vec.shape[0]}).")
        pos_idx = seg["_rowid"].to_numpy()
        arr = vec[pos_idx]
        return arr.reshape(-1, 1) if arr.ndim == 1 else np.asarray(arr)

    TonextyAnchor = Union[str, Callable[[pd.DataFrame], Iterable[float]]]

    def _resolve_tonexty_baseline(seg: pd.DataFrame, *, 
                                  anchor: Union[str, Callable[[pd.DataFrame], Iterable[float]]], 
                                  last_cat_value: Optional[float]) -> Optional[np.ndarray]:
        n = len(seg)
        if anchor is None:
            return None
        if isinstance(anchor, str):
            if anchor == "zero":
                return np.zeros(n, dtype=float)
            if anchor == "prev":
                base = 0.0 if last_cat_value is None else float(last_cat_value)
                return np.full(n, base, dtype=float)
            if anchor not in seg.columns:
                raise KeyError(f"tonexty_anchor column '{anchor}' not in segment.")
            return seg[anchor].to_numpy(dtype=float)
        arr = np.asarray(anchor(seg), dtype=float)
        if arr.shape[0] != n:
            raise ValueError("tonexty_anchor callable must return array of same length as segment.")
        return arr

    # ---------------- validation & prep ----------------
    for c in (time_col, value_col, category_col):
        if c not in df.columns:
            raise ValueError(f"DataFrame missing column: {c}")

    _df = df.dropna(subset=[time_col, value_col]).copy()
    if sort_by_time:
        _df = _df.sort_values(time_col)
    _df[category_col] = _df[category_col].astype(str)

    # Stable positional id for alignment (0..n-1)
    _df = _df.reset_index(drop=True)
    _df["_rowid"] = np.arange(len(_df))

    # Contiguous segments
    _df["_segment_id"] = (_df[category_col] != _df[category_col].shift(1)).cumsum()

    fig = fig or go.Figure()
    legend_seen = {getattr(tr, "name", None) for tr in fig.data if hasattr(tr, "name")}

    # Track last value per category (for anchor='prev')
    last_value_by_cat: Dict[str, float] = {}

    # ---------------- build traces ----------------
    for _, seg in _df.groupby("_segment_id", sort=True):
        cat = seg[category_col].iloc[0]
        per_cat = params_by_category.get(cat, {})
        if not (per_cat or scatter_defaults):
            continue

        merged = _deep_merge(scatter_defaults or {}, per_cat or {})

        # Pull special (non-Plotly) keys
        customdata_spec = merged.pop("customdata", None)
        tonexty_anchor = merged.pop("tonexty_anchor", None)
        # NEW: optional explicit baseline line_shape; else inherit from segment's merged line_shape
        tonexty_anchor_line_shape = merged.pop("tonexty_anchor_line_shape", merged.get("line_shape", None))

        # Defaults: name & legend
        merged.setdefault("name", cat)
        if "showlegend" not in merged:
            merged["showlegend"] = (merged["name"] not in legend_seen)
        legend_seen.add(merged["name"])

        # If tonexty requested, insert a baseline with matching line_shape BEFORE the filled trace
        wants_tonexty = (merged.get("fill") == "tonexty") and (tonexty_anchor is not None)
        if wants_tonexty:
            baseline_y = _resolve_tonexty_baseline(
                seg,
                anchor=tonexty_anchor,
                last_cat_value=last_value_by_cat.get(cat)
            )
            if baseline_y is not None:
                baseline_kwargs = dict(
                    x=seg[time_col],
                    y=baseline_y,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"__baseline__{merged['name']}",
                    legendgroup=merged.get("legendgroup", None),
                )
                if tonexty_anchor_line_shape is not None:
                    baseline_kwargs["line_shape"] = tonexty_anchor_line_shape
                fig.add_trace(go.Scatter(**baseline_kwargs))

        # Segment trace (can be step/hv/vh/etc.)
        seg_customdata = _resolve_segment_customdata(seg, _df, customdata_spec)
        fig.add_trace(go.Scatter(
            x=seg[time_col],
            y=seg[value_col],
            customdata=seg_customdata,
            **merged
        ))

        last_value_by_cat[cat] = float(seg[value_col].iloc[-1])

    # Layout
    fig.update_layout(xaxis_title=time_col, yaxis_title=value_col)
    if layout:
        fig.update_layout(**layout)
    return fig

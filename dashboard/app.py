import warnings
from datetime import datetime, date
from itertools import cycle

import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import colors as pc
import json
import os

DEFAULT_CSV_PATH = "./data/final_merged_data_with_cities.csv"
DEFAULT_COLUMNS_MAP_PATH = "./data/columns_map.json"

# -----------------------------
# EXPECTED INPUTS (provide these)
# -----------------------------
# 1) Provide your columns_map somewhere (import or paste).
#    Example shape: {'load': {'units': 'kW', 'description': '...'}, ...}
# columns_map = {...}  # <- import or define

# 2) Provide your data as a DataFrame with at least:
#    ['equipmentId', 'dateTime', ... your series ...]
#    Either import it or enable the uploader below.

st.set_page_config(page_title="Grid Dashboard", layout="wide")

st.title("⚡ Equipment Time Series Dashboard")

# --- Load data ---
@st.cache_data
def load_csv(file):
    # Try to parse dateTime automatically
    df = pd.read_csv(file)
    if 'dateTime' in df.columns:
        df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
    return df

uploaded = st.sidebar.file_uploader("Upload CSV (must include 'equipmentId' and 'dateTime')", type=["csv"])
data_holder = st.session_state.get("alldf")

if uploaded is not None:
    alldf = load_csv(uploaded)
    st.session_state["alldf"] = alldf
elif data_holder is not None:
    alldf = data_holder
elif os.path.exists(DEFAULT_CSV_PATH):
    alldf = load_csv(DEFAULT_CSV_PATH)
else:
    st.info("Upload a CSV to begin, or modify the script to import your existing DataFrame as `alldf`.")
    st.stop()

# Load columns_map from JSON (uploader > local file > session_state)
@st.cache_data
def load_columns_map_json(file_or_path):
    """Load columns_map from JSON (UploadedFile or filesystem path)."""
    if hasattr(file_or_path, "read"):           # Streamlit UploadedFile
        obj = json.load(file_or_path)
    else:                                       # path on disk
        with open(file_or_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

    # Basic validation / normalization
    if not isinstance(obj, dict):
        raise ValueError("columns_map JSON must be a JSON object (dict).")

    for col, meta in obj.items():
        if not isinstance(meta, dict) or "units" not in meta:
            raise ValueError(f"Entry for '{col}' must be an object with at least a 'units' field.")
        meta.setdefault("description", "")       # optional in your JSON

    return obj
columns_map_file = st.sidebar.file_uploader("Upload columns_map.json", type=["json"])
columns_map = {}
if columns_map_file is not None:
    columns_map = load_columns_map_json(columns_map_file)
elif os.path.exists(DEFAULT_COLUMNS_MAP_PATH):
    columns_map = load_columns_map_json(DEFAULT_COLUMNS_MAP_PATH)
elif 'columns_map' in st.session_state:
    columns_map = st.session_state['columns_map']

if not columns_map:
    st.warning("No columns_map loaded. Upload a JSON (column -> {units, description}).")
else:
    st.session_state['columns_map'] = columns_map  # persist for reruns

# Guard rails
required_cols = {'equipmentId', 'dateTime'}
missing = required_cols - set(alldf.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Ensure datetime type
alldf['dateTime'] = pd.to_datetime(alldf['dateTime'], errors='coerce')
alldf = alldf.dropna(subset=['dateTime']).sort_values('dateTime')

# -----------------------------
# Sidebar controls
# -----------------------------
left, right = st.columns([2, 3])

with st.sidebar:
    equip_ids = sorted(alldf['equipmentId'].dropna().unique().tolist())
    equipment_id = st.selectbox("Equipment ID", equip_ids)

    # Candidate columns: numeric time series available in df (and referenced in columns_map if provided)
    numeric_cols = (
        alldf.select_dtypes(include=['number', 'float', 'int'])
             .columns.drop(['equipmentId'], errors='ignore')
             .tolist()
    )
    # Keep those present in columns_map if any units defined; otherwise allow any numeric columns
    if columns_map:
        allowed = [c for c in numeric_cols if c in columns_map]
    else:
        allowed = numeric_cols

    default_cols = [c for c in ['load', 'hotspotTemperature', 'temperature'] if c in allowed]
    columns = st.multiselect("Columns to plot", options=allowed, default=default_cols)

    st.markdown("**Time window**")
    min_dt = alldf['dateTime'].min().date()
    max_dt = alldf['dateTime'].max().date()
    start_date = st.date_input("Start", value=max(min_dt, date(2023, 1, 1)), min_value=min_dt, max_value=max_dt)
    end_date   = st.date_input("End",   value=min(max_dt, date(2023, 12, 31)), min_value=min_dt, max_value=max_dt)


    go_btn = st.button("Plot")

def _to_ts(x, is_end=False):
    """Accepts date/datetime/str; returns pandas.Timestamp.
       If 'YYYY-MM' string provided, end expands to month-end."""
    if isinstance(x, (datetime, date)):
        return pd.to_datetime(x)
    ts = pd.to_datetime(x, errors='coerce')
    if ts is pd.NaT:
        raise ValueError(f"Invalid date: {x}")
    # Month-only expansion for end bound (if you ever feed 'YYYY-MM')
    if isinstance(x, str) and len(x) == 7 and x.count('-') == 1:
        if is_end:
            ts = ts + pd.offsets.MonthEnd(1)
    return ts

def plot_by_units(df, equipment_id, columns, columns_map, start_dt, end_dt):
    """Plots time series grouped by units with dual y-axes and distinct colors."""
    msgs = []

    # Filter equipment + time window
    df = df[df['equipmentId'] == equipment_id].copy()
    if df.empty:
        st.warning(f"No data for equipmentId={equipment_id}.")
        return

    start_ts = _to_ts(start_dt, is_end=False)
    end_ts   = _to_ts(end_dt,   is_end=True)
    df = df[(df['dateTime'] >= start_ts) & (df['dateTime'] <= end_ts)]
    if df.empty:
        st.warning(f"No data in selected window: {start_ts.date()} → {end_ts.date()}.")
        return

    # Resolve units, skip '-' and missing
    resolved = []
    for col in columns:
        if col not in df.columns:
            msgs.append(f"• Column '{col}' not in data → skipped.")
            continue
        unit = (columns_map.get(col) or {}).get('units') if columns_map else None
        if unit is None:
            msgs.append(f"• No unit defined for '{col}' → skipped.")
            continue
        if unit == '-':
            msgs.append(f"• Unit for '{col}' is '-' (non-physical) → skipped.")
            continue
        resolved.append((col, unit))

    if not resolved:
        st.warning("No plottable columns after unit checks.")
        if msgs: st.info("\n".join(msgs))
        return

    # Enforce at most two units
    units_in_order = []
    for _, u in resolved:
        if u not in units_in_order:
            units_in_order.append(u)
    if len(units_in_order) > 2:
        msgs.append(f"• More than two units requested {units_in_order} → only first two kept.")
        allowed_units = set(units_in_order[:2])
        resolved = [(c, u) for (c, u) in resolved if u in allowed_units]
        units_in_order = units_in_order[:2]

    # Group by unit (preserve order)
    by_unit = {u: [] for u in units_in_order}
    for c, u in resolved:
        by_unit[u].append(c)

    # Distinct colors across ALL series
    palette = pc.qualitative.Plotly or pc.qualitative.D3
    needed = sum(len(v) for v in by_unit.values())
    color_cycle = cycle(palette * (1 + needed // len(palette)))

    # Build figure (dual y-axis if two units)
    secondary = True if len(units_in_order) == 2 else False
    fig = make_subplots(specs=[[{"secondary_y": secondary}]])
    fig.update_layout(template="plotly_white", hovermode="x unified")

    # Left axis (first unit)
    left_unit = units_in_order[0]
    for col in by_unit[left_unit]:
        fig.add_trace(
            go.Scatter(
                x=df['dateTime'], y=df[col],
                mode='lines',
                name=f"{col} [{left_unit}]",
                line=dict(color=next(color_cycle))
            ),
            secondary_y=False
        )

    # Right axis (second unit if present)
    if secondary:
        right_unit = units_in_order[1]
        for col in by_unit[right_unit]:
            fig.add_trace(
                go.Scatter(
                    x=df['dateTime'], y=df[col],
                    mode='lines',
                    name=f"{col} [{right_unit}]",
                    line=dict(color=next(color_cycle))
                ),
                secondary_y=True
            )

    # Axes labels
    fig.update_xaxes(title_text="DateTime")
    fig.update_yaxes(title_text=left_unit, secondary_y=False)
    if secondary:
        fig.update_yaxes(title_text=right_unit, secondary_y=True)

    title_cols = ", ".join(columns)
    fig.update_layout(
        title=f"Equipment {equipment_id} — {title_cols}<br>"
              f"<sub>{start_ts.date()} → {end_ts.date()}</sub>",
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=40, r=40, t=90, b=40),
        height=500
    )

    if msgs:
        st.warning("\n".join(msgs))
    st.plotly_chart(fig, use_container_width=True)

with left:
    st.subheader("Inputs")
    st.write(f"**Equipment:** `{equipment_id}`")
    st.write(f"**Columns:** {', '.join(columns) if columns else '(none)'}")
    st.write(f"**Window:** {start_date} → {end_date}")

with right:
    if st.checkbox("Show filtered data sample"):
        mask = (
            (alldf['equipmentId'] == equipment_id) &
            (alldf['dateTime'] >= pd.to_datetime(start_date)) &
            (alldf['dateTime'] <= pd.to_datetime(end_date))
        )
        st.dataframe(alldf.loc[mask].head(200))

if go_btn:
    plot_by_units(alldf, equipment_id, columns, columns_map, start_date, end_date)
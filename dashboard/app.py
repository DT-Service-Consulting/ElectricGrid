import warnings
from datetime import datetime, date
from itertools import cycle
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly import colors as pc
import json
import os

from fft_utils import compute_fft, load_fft_from_cache, save_fft_to_cache


# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent.resolve()
FFT_CACHE_DIR = (BASE_DIR / "fft_cache").resolve()
DEFAULT_CSV_PATH = (BASE_DIR / ".." / "data" / "final_merged_data_with_cities.csv").resolve()
DEFAULT_COLUMNS_MAP_PATH = (BASE_DIR / ".." / "data" / "columns_map.json").resolve()
# === NEW === optionally a separate default for equipment CSV (can be the same)
DEFAULT_EQUIPMENT_CSV_PATH = DEFAULT_CSV_PATH

st.set_page_config(page_title="Grid Dashboard", layout="wide")
st.title("⚡ Electric Grid Dashboard")

# --- Loaders ---
@st.cache_data(show_spinner=False)
def load_csv(file):
    df = pd.read_csv(file)
    if 'dateTime' in df.columns:
        df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
    return df

@st.cache_data(show_spinner=False)
def load_columns_map_json(file_or_path):
    if hasattr(file_or_path, "read"):
        obj = json.load(file_or_path)
    else:
        with open(file_or_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("columns_map JSON must be a JSON object (dict).")
    for col, meta in obj.items():
        if not isinstance(meta, dict) or "units" not in meta:
            raise ValueError(f"Entry for '{col}' must include a 'units' field.")
        meta.setdefault("description", "")
    return obj

@st.cache_data(show_spinner=False)
def resolve_alldf(uploaded, default_path):
    if uploaded is not None:
        return load_csv(uploaded)
    if 'alldf' in st.session_state:
        return st.session_state['alldf']
    if Path(default_path).exists():
        return load_csv(str(default_path))
    return None

# --- Sidebar: plot type switch (NEW) ---
plot_type = st.sidebar.radio("Plot type", ["Time series", "Equipment plot", "FFT"], horizontal=True)

# --- Common: columns_map ---
columns_map_file = st.sidebar.file_uploader("Upload columns_map.json", type=["json"], key="columns_map_up")
columns_map = {}
if columns_map_file is not None:
    columns_map = load_columns_map_json(columns_map_file)
elif Path(DEFAULT_COLUMNS_MAP_PATH).exists():
    columns_map = load_columns_map_json(str(DEFAULT_COLUMNS_MAP_PATH))
elif 'columns_map' in st.session_state:
    columns_map = st.session_state['columns_map']
if not columns_map:
    st.warning("No columns_map loaded. Upload a JSON (column -> {units, description}).")
else:
    st.session_state['columns_map'] = columns_map

# ----------------------------
# TIME SERIES MODE (existing)
# ----------------------------
if plot_type == "Time series":

    uploaded = st.sidebar.file_uploader(
        "Upload CSV (must include 'equipmentId' and 'dateTime')",
        type=["csv"], key="ts_csv_up"
    )
    alldf = resolve_alldf(uploaded, DEFAULT_CSV_PATH)

    if alldf is None:
        st.info("Upload a CSV to begin, or provide a valid DEFAULT_CSV_PATH.")
        st.stop()

    st.session_state['alldf'] = alldf

    # Guard rails
    required_cols = {'equipmentId', 'dateTime'}
    missing = required_cols - set(alldf.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Ensure datetime type
    alldf['dateTime'] = pd.to_datetime(alldf['dateTime'], errors='coerce')
    alldf = alldf.dropna(subset=['dateTime']).sort_values('dateTime')

    left, right = st.columns([2, 3])

    with st.sidebar:
        equip_ids = sorted(alldf['equipmentId'].dropna().unique().tolist())
        equipment_id = st.selectbox("Equipment ID", equip_ids)

        numeric_cols = (
            alldf.select_dtypes(include=['number', 'float', 'int'])
                 .columns.drop(['equipmentId'], errors='ignore')
                 .tolist()
        )
        # exclude variables with units '-' from being plotted
        if columns_map:
            allowed = [
                c for c in numeric_cols
                if c in columns_map and columns_map[c].get('units') != '-'
            ]
        else:
            allowed = numeric_cols
        default_cols = [c for c in ['load', 'hotspotTemperature', 'temperature'] if c in allowed]
        columns = st.multiselect("Columns to plot", options=allowed, default=default_cols)

        st.markdown("**Time window**")
        min_dt = alldf['dateTime'].min().date()
        max_dt = alldf['dateTime'].max().date()
        start_date = st.date_input("Start", value=max(min_dt, date(2023, 1, 1)), min_value=min_dt, max_value=max_dt)
        end_date   = st.date_input("End",   value=min(max_dt, date(2023, 12, 31)), min_value=min_dt, max_value=max_dt)
        go_btn_ts = st.button("Plot time series")

    def _to_ts(x, is_end=False):
        if isinstance(x, (datetime, date)):
            return pd.to_datetime(x)
        ts = pd.to_datetime(x, errors='coerce')
        if ts is pd.NaT:
            raise ValueError(f"Invalid date: {x}")
        if isinstance(x, str) and len(x) == 7 and x.count('-') == 1 and is_end:
            ts = ts + pd.offsets.MonthEnd(1)
        return ts

    def plot_by_units(df, equipment_id, columns, columns_map, start_dt, end_dt):
        msgs = []
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

        units_in_order = []
        for _, u in resolved:
            if u not in units_in_order:
                units_in_order.append(u)
        if len(units_in_order) > 2:
            msgs.append(f"• More than two units requested {units_in_order} → only first two kept.")
            allowed_units = set(units_in_order[:2])
            resolved = [(c, u) for (c, u) in resolved if u in allowed_units]
            units_in_order = units_in_order[:2]

        by_unit = {u: [] for u in units_in_order}
        for c, u in resolved:
            by_unit[u].append(c)

        palette = pc.qualitative.Plotly or pc.qualitative.D3
        needed = sum(len(v) for v in by_unit.values())
        color_cycle = cycle(palette * (1 + needed // len(palette)))

        secondary = True if len(units_in_order) == 2 else False
        fig = make_subplots(specs=[[{"secondary_y": secondary}]])
        fig.update_layout(template="plotly_white", hovermode="x unified")

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

    left.subheader("Inputs")
    left.write(f"**Equipment:** `{equipment_id}`")
    left.write(f"**Columns:** {', '.join(columns) if columns else '(none)'}")
    left.write(f"**Window:** {start_date} → {end_date}")

    with right:
        if st.checkbox("Show filtered data sample"):
            mask = (
                (alldf['equipmentId'] == equipment_id) &
                (alldf['dateTime'] >= pd.to_datetime(start_date)) &
                (alldf['dateTime'] <= pd.to_datetime(end_date))
            )
            st.dataframe(alldf.loc[mask].head(200))

    if go_btn_ts:
        plot_by_units(alldf, equipment_id, columns, columns_map, start_date, end_date)

# ----------------------------
# FFT PLOT MODE
# ----------------------------

elif plot_type == "FFT":
    st.subheader("Frequency analysis (FFT)")

    # Use the same time-series CSV loader as Time series tab
    uploaded = st.sidebar.file_uploader(
        "Upload CSV (must include 'equipmentId' and 'dateTime')",
        type=["csv"], key="fft_csv_up"
    )
    alldf_fft = resolve_alldf(uploaded, DEFAULT_CSV_PATH)
    if alldf_fft is None:
        st.info("Upload a CSV to begin, or provide a valid DEFAULT_CSV_PATH.")
        st.stop()

    # Ensure datetime col
    alldf_fft['dateTime'] = pd.to_datetime(alldf_fft['dateTime'], errors='coerce')
    alldf_fft = alldf_fft.dropna(subset=['dateTime']).sort_values('dateTime')

    # Identify numeric plottable columns (exclude units == '-')
    numeric_cols = (
        alldf_fft.select_dtypes(include=['number', 'float', 'int'])
                .columns.drop(['equipmentId'], errors='ignore')
                .tolist()
    )
    if columns_map:
        fft_candidates = [c for c in numeric_cols if c in columns_map and columns_map[c].get('units') != '-']
    else:
        fft_candidates = numeric_cols

    equip_ids = sorted(alldf_fft['equipmentId'].dropna().unique().tolist())

    # Two panels (A and B) for comparison
    st.markdown("Choose two series to compare (each panel: equipment + variable).")
    colA, colB = st.columns(2, gap="large")

    with colA:
        st.markdown("**Panel A**")
        eqA = st.selectbox("Equipment ID (A)", equip_ids, key="fft_eq_a")
        varA = st.selectbox("Variable (A)", fft_candidates, key="fft_var_a")
        forceA = st.checkbox("Recompute FFT (A)", value=False, key="fft_force_a")

    with colB:
        st.markdown("**Panel B**")
        eqB = st.selectbox("Equipment ID (B)", equip_ids, key="fft_eq_b")
        varB = st.selectbox("Variable (B)", fft_candidates, key="fft_var_b")
        forceB = st.checkbox("Recompute FFT (B)", value=False, key="fft_force_b")

    # Frequency axis options
    axis_unit = st.selectbox("Frequency unit", ["cycles/day", "Hz"], index=0)
    y_scale_log = st.checkbox("Log scale (amplitude)", value=False)
    max_freq = st.number_input(
        f"Max frequency ({'cycles/day' if axis_unit=='cycles/day' else 'Hz'})",
        min_value=0.0, value=0.0, help="0 = auto (Nyquist)"
    )

    plot_btn = st.button("Compute / Plot FFTs")

    # Difference plot options
    show_diff = st.checkbox("Show difference plot (A − B)", value=False)
    diff_on = st.selectbox("Difference on", ["Amplitude", "Power"], index=0)

    # def _source_tag(df: pd.DataFrame) -> str:
    #     """A simple tag for cache partitioning; clear cache if data changes."""
    #     # Try to derive from file path, otherwise a stable shape-based tag
    #     try:
    #         # If you have a known path, use it; else fall back to size hash
    #         return "default_source"
    #     except Exception:
    #         h = hashlib.sha1(str((len(df), tuple(df.columns))).encode("utf-8")).hexdigest()[:8]
    #         return f"mem_{h}"

    def _get_fft_cached(df_all: pd.DataFrame, eq, var, force=False):
        src = "default_source"  # keep simple; clear folder if your CSV changes
        cached = None if force else load_fft_from_cache(FFT_CACHE_DIR, src, eq, var)
        if cached is not None:
            return cached, True

        # Build per-equipment full series (no time window)
        sub = df_all.loc[df_all['equipmentId'] == eq, ['dateTime', var]].dropna()
        if sub.empty:
            st.warning(f"No data for equipmentId={eq}, variable={var}.")
            return None, False

        try:
            fft_df = compute_fft(sub, time_col='dateTime', value_col=var,
                                 resample_seconds=None, detrend=True, window="hann")
        except Exception as e:
            st.error(f"FFT failed for {eq}-{var}: {e}")
            return None, False

        save_fft_to_cache(FFT_CACHE_DIR, src, eq, var, fft_df)
        return fft_df, False

    def _plot_fft(fft_df: pd.DataFrame, title: str):
        if axis_unit == "cycles/day":
            x = fft_df['frequency_per_day']
            x_title = "Frequency (cycles/day)"
        else:
            x = fft_df['frequency_hz']
            x_title = "Frequency (Hz)"

        y = fft_df['amplitude']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Amplitude'))
        fig.update_layout(
            template="plotly_white",
            title=title,
            xaxis_title=x_title,
            yaxis_title="Amplitude",
            height=450,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        if y_scale_log:
            fig.update_yaxes(type="log")
        # Max frequency
        if max_freq and max_freq > 0:
            fig.update_xaxes(range=[0, max_freq])
        st.plotly_chart(fig, use_container_width=True)

    def _align_and_diff(fft_a: pd.DataFrame, fft_b: pd.DataFrame, use_hz: bool, diff_kind: str):
        """
        Interpolate both spectra onto a common frequency axis over the overlapping range,
        then return (x_grid, y_diff) where y_diff = A - B for chosen metric.
        diff_kind: "Amplitude" or "Power"
        """
        xa = fft_a['frequency_hz'].to_numpy() if use_hz else fft_a['frequency_per_day'].to_numpy()
        xb = fft_b['frequency_hz'].to_numpy() if use_hz else fft_b['frequency_per_day'].to_numpy()

        if diff_kind == "Power":
            ya = fft_a['power'].to_numpy()
            yb = fft_b['power'].to_numpy()
        else:
            ya = fft_a['amplitude'].to_numpy()
            yb = fft_b['amplitude'].to_numpy()

        # Overlap range
        left = max(np.nanmin(xa), np.nanmin(xb))
        right = min(np.nanmax(xa), np.nanmax(xb))
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            return None, None, "No overlapping frequency range to compare."

        # Choose grid size ~ smaller of the two (keeps structure but avoids upsampling too much)
        n = int(min(len(xa), len(xb)))
        if n < 8:
            return None, None, "Too few common points to compute a meaningful difference."

        x_grid = np.linspace(left, right, n)
        # Ensure strictly increasing for interp
        sort_a = np.argsort(xa)
        sort_b = np.argsort(xb)
        yai = np.interp(x_grid, xa[sort_a], ya[sort_a])
        ybi = np.interp(x_grid, xb[sort_b], yb[sort_b])

        return x_grid, (yai - ybi), None

    if plot_btn:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            fftA, from_cacheA = _get_fft_cached(alldf_fft, eqA, varA, force=forceA)
            if fftA is not None:
                unitA = (columns_map.get(varA) or {}).get('units', '') if columns_map else ''
                titleA = f"A — FFT: {eqA} · {varA} [{unitA}] " + ("(cached)" if from_cacheA else "(computed)")
                _plot_fft(fftA, titleA)
        with c2:
            fftB, from_cacheB = _get_fft_cached(alldf_fft, eqB, varB, force=forceB)
            if fftB is not None:
                unitB = (columns_map.get(varB) or {}).get('units', '') if columns_map else ''
                titleB = f"B — FFT: {eqB} · {varB} [{unitB}] " + ("(cached)" if from_cacheB else "(computed)")
                _plot_fft(fftB, titleB)

        if show_diff and (fftA is not None) and (fftB is not None):
            use_hz = (axis_unit == "Hz")
            xg, yd, err = _align_and_diff(fftA, fftB, use_hz=use_hz, diff_kind=diff_on)
            if err:
                st.warning(err)
            else:
                x_title = "Frequency (Hz)" if use_hz else "Frequency (cycles/day)"
                y_title = f"Δ{diff_on} (A − B)"
                figd = go.Figure()
                figd.add_trace(go.Scatter(x=xg, y=yd, mode='lines', name=f"Δ{diff_on}"))
                figd.update_layout(
                    template="plotly_white",
                    title=f"Difference: Panel A − Panel B ({diff_on})",
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    height=450,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                if y_scale_log and diff_on == "Power":
                    # Log of signed differences is undefined; keep linear for signed data.
                    st.info("Log scale disabled on difference of Power (can be negative).")
                elif y_scale_log and diff_on == "Amplitude":
                    # Amplitude difference can be negative too; log would be invalid.
                    st.info("Log scale disabled on signed differences.")
                st.plotly_chart(figd, use_container_width=True)

    # Optional: quick table preview for either selection
    if st.checkbox("Show FFT table (A)"):
        fftA, _ = _get_fft_cached(alldf_fft, eqA, varA, force=False)
        if fftA is not None:
            st.dataframe(fftA.head(500))
    if st.checkbox("Show FFT table (B)"):
        fftB, _ = _get_fft_cached(alldf_fft, eqB, varB, force=False)
        if fftB is not None:
            st.dataframe(fftB.head(500))

# ----------------------------
# EQUIPMENT PLOT MODE
# ----------------------------
else:
    st.subheader("Equipment scatter plot")

    equip_file = st.sidebar.file_uploader(
        "Upload equipment CSV (can be same as time series)",
        type=["csv"], key="equip_csv_up"
    )
    if equip_file is not None:
        equip_df = load_csv(equip_file)
    elif Path(DEFAULT_EQUIPMENT_CSV_PATH).exists():
        equip_df = load_csv(str(DEFAULT_EQUIPMENT_CSV_PATH))
    elif 'alldf' in st.session_state:
        equip_df = st.session_state['alldf'].copy()
    else:
        st.info("Upload an equipment CSV or set DEFAULT_EQUIPMENT_CSV_PATH.")
        st.stop()

    if 'equipmentId' not in equip_df.columns:
        st.error("Equipment CSV must include 'equipmentId'.")
        st.stop()

    # --- Build candidates ---
    # X/Y candidates: numeric columns present in columns_map with units != '-'
    numeric_cols_eq = equip_df.select_dtypes(include=['number', 'float', 'int']).columns.tolist()
    if columns_map:
        xy_candidates = [
            c for c in numeric_cols_eq
            if c in columns_map and columns_map[c].get('units') != '-'
        ]
    else:
        xy_candidates = numeric_cols_eq

    if len(xy_candidates) < 2:
        st.error("Need at least two numeric columns (with units other than '-') to plot.")
        st.stop()

    # Color-by candidates: variables with units '-'
    colorable_vars = [
        c for c, meta in columns_map.items()
        if isinstance(meta, dict) and meta.get('units') == '-' and c in equip_df.columns
    ] if columns_map else []
    colorable_options = ['(none)'] + sorted(colorable_vars)

    # --- Defaults ---
    pref = ['nominalLoad', 'heatRunTest_copperLosses', 'heatRunTest_noLoadLosses']
    default_x = next((c for c in pref if c in xy_candidates), xy_candidates[0])
    default_y = next((c for c in pref if c in xy_candidates and c != default_x), xy_candidates[1])

    x_col = st.sidebar.selectbox("X variable", xy_candidates, index=xy_candidates.index(default_x))
    y_col = st.sidebar.selectbox("Y variable", xy_candidates, index=xy_candidates.index(default_y))
    color_by = st.sidebar.selectbox("Color by: ", colorable_options, index=0)
    show_labels = st.sidebar.checkbox("Show equipmentId labels", value=False)

    def _axis_label(col):
        unit = (columns_map.get(col) or {}).get('units') if columns_map else None
        return f"{col} [{unit}]" if unit else col

    # --- Construct per-equipment table ---
    # If color_by selected, include it (unless it's 'equipmentId' which is already present)
    group_cols = ['equipmentId', x_col, y_col]
    if color_by != '(none)' and color_by != 'equipmentId':
        group_cols.append(color_by)

    # Ensure uniqueness while preserving order (just in case)
    group_cols = list(dict.fromkeys(group_cols))

    work = equip_df[group_cols + (['dateTime'] if 'dateTime' in equip_df.columns else [])].copy()
    if 'dateTime' in work.columns:
        work = work.sort_values('dateTime')

    # Warn if variables vary over time per equipment (they shouldn't)
    nunq = work.groupby('equipmentId')[[x_col, y_col]].nunique(dropna=True)
    inconsistent = nunq[(nunq[x_col] > 1) | (nunq[y_col] > 1)]
    if not inconsistent.empty:
        st.warning(
            f"{len(inconsistent)} equipment have varying values for X/Y across time. "
            "Using the first non-null value per equipment."
        )

    # Reduce to 1 row per equipment
    first_non_null = lambda s: s.dropna().iloc[0] if s.dropna().size else None
    agg_spec = {x_col: first_non_null, y_col: first_non_null}
    if color_by != '(none)' and color_by != 'equipmentId':
        agg_spec[color_by] = first_non_null

    agg = work.groupby('equipmentId', as_index=False).agg(agg_spec)

    # If coloring by equipmentId, derive it directly (one category per equipment)
    if color_by == 'equipmentId':
        agg['__color_cat__'] = agg['equipmentId'].astype(str)
        cat_label_name = 'equipmentId'
    elif color_by != '(none)':
        agg['__color_cat__'] = agg[color_by].fillna("Unknown").astype(str)
        cat_label_name = color_by
    else:
        cat_label_name = None

    # Coerce numeric and drop missing for axes
    agg[x_col] = pd.to_numeric(agg[x_col], errors='coerce')
    agg[y_col] = pd.to_numeric(agg[y_col], errors='coerce')
    agg = agg.dropna(subset=[x_col, y_col])

    x_label = _axis_label(x_col)
    y_label = _axis_label(y_col)

    fig = go.Figure()

    if cat_label_name is None:
        # Single trace
        fig.add_trace(
            go.Scatter(
                x=agg[x_col],
                y=agg[y_col],
                mode='markers+text' if show_labels else 'markers',
                text=agg['equipmentId'] if show_labels else None,
                textposition='top center',
                name='equipment',
                hovertemplate=(
                    "equipmentId=%{text}<br>" if show_labels else "equipmentId=%{customdata[0]}<br>"
                ) + f"{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>",
                customdata=None if show_labels else agg[['equipmentId']].values
            )
        )
    else:
        # Color by categorical (__color_cat__)
        palette = pc.qualitative.Plotly or pc.qualitative.D3
        categories = agg['__color_cat__']
        uniq = categories.unique().tolist()
        colors = (palette * (1 + len(uniq) // len(palette)))[:len(uniq)]
        color_map = dict(zip(uniq, colors))

        for cat in uniq:
            mask = (categories == cat)
            fig.add_trace(
                go.Scatter(
                    x=agg.loc[mask, x_col],
                    y=agg.loc[mask, y_col],
                    mode='markers+text' if show_labels else 'markers',
                    text=agg.loc[mask, 'equipmentId'] if show_labels else None,
                    textposition='top center',
                    name=f"{cat_label_name}={cat}",
                    marker=dict(color=color_map[cat]),
                    hovertemplate=(
                        ("equipmentId=%{text}<br>" if show_labels else "equipmentId=%{customdata[0]}<br>")
                        + f"{cat_label_name}={cat}<br>{x_label}=%{{x}}<br>{y_label}=%{{y}}<extra></extra>"
                    ),
                    customdata=None if show_labels else agg.loc[mask, ['equipmentId']].values
                )
            )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        title=f"Equipment scatter: {x_col} vs {y_col}"
              + ("" if color_by == '(none)' else f" — colored by {color_by}")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: table preview
    if st.checkbox("Show per-equipment values"):
        show_cols = ['equipmentId', x_col, y_col] + ([] if color_by == '(none)' else [color_by])
        st.dataframe(agg[show_cols].sort_values('equipmentId').reset_index(drop=True))
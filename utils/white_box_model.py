from pyexpat import model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import json
from config import DATA_DIR, TFO_PARAMETERS_FILE
from scipy.optimize import least_squares

RENAME_MAP = {
        'deltaTopOil':     'heatRunTest_deltaTopOil',
        'deltaHotspot':    'heatRunTest_deltaHotspot',
        'x':               'heatRunTest_x',
        'y':               'heatRunTest_y',
        'h':               'heatRunTest_h',
        'noLoadLosses':    'heatRunTest_noLoadLosses',
        'copperLosses':    'heatRunTest_copperLosses',
        'nominalLoad':     'nominalLoad',
        'pf':              'pf',
        's':               's',
        'ambient_bias':    'ambient_bias',
    }

METRIC_COLUMNS = ['MAE', 'RMSE', 'Bias', 'R2', 'NRMSE%_IQR', 'P95_abs_err', 'Max_abs_err']
K_BINS = [0, 0.5, 0.9, 1.1, np.inf]
K_LABELS = ['<0.5','0.5-0.9','0.9-1.1','>1.1']
# K_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, np.inf]
# K_LABELS = ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','1-1.1','>1.1']
# add to metric_columns the by-K metrics
for k in K_LABELS:
    for metric in ['n', 'MAE', 'RMSE', 'Bias']:
        METRIC_COLUMNS.append(f'K_{k}_{metric}')


def load_base_params_by_eid(wb_model):
    tfo_parameters = pd.read_csv(TFO_PARAMETERS_FILE)
    tfo_parameters.rename(columns=RENAME_MAP, inplace=True)

    base_params_by_eid = {}
    for _, r in tfo_parameters.iterrows():
        eid = r['equipmentId']
        d = {}
        for k in wb_model.all_param_keys:
            if k in r and pd.notna(r[k]):
                d[k] = float(r[k])
        base_params_by_eid[eid] = d
    return base_params_by_eid

def _get_series_or_param(d, params, param_key, df_col):
    if param_key in params:
        return np.full(len(d), float(params[param_key]), dtype=float)
    return d[df_col].to_numpy(dtype=float)

def build_x0_bounds(wb_model, selected_keys, base_params=None, init_overrides=None, bounds_overrides=None):
    x0, lb, ub = [], [], []
    for k in selected_keys:
        try: b_l, b_u = wb_model.params_registry[k]['bounds']
        except KeyError:
            raise ValueError(f"Parameter '{k}' not found in wb_model.params_registry")
        if bounds_overrides and k in bounds_overrides:
            b_l, b_u = bounds_overrides[k]

        if init_overrides and k in init_overrides:
            val = init_overrides[k]
        elif base_params and k in base_params:
            val = base_params[k]
        else:
            val = wb_model.params_registry[k]['default']
            if val is None:
                val = (b_l + b_u) / 2.0

        x0.append(float(val))
        lb.append(float(b_l))
        ub.append(float(b_u))
    return np.array(x0), np.array(lb), np.array(ub)

def divide_by_k_df_into_columns(byK):
    # Convert byK to individual columns
    byK = byK.to_dict(orient='index')
    keys = list(byK.keys())
    values = list(byK.values())
    for k, v in zip(keys, values):
        for metric in ['n', 'MAE', 'RMSE', 'Bias']:
            byK[f'K_{k}_{metric}'] = v[metric]
        del byK[k]
    return byK

def merge_by_k_columns_into_df(metrics_dict, labels=None, index_name='K_bin'):
    byK = {k: v for k, v in metrics_dict.items() if k.startswith('K_')}
    if not byK:
        return pd.DataFrame(columns=['n','MAE','RMSE','Bias'])

    # rebuild nested dict: {k_bin: {metric: value}}
    nested = {}
    for key, val in byK.items():
        _, k_bin, metric = key.split('_', 2)  # "K_<bin>_<metric>"
        nested.setdefault(k_bin, {})[metric] = val

    df = pd.DataFrame.from_dict(nested, orient='index')[['n','MAE','RMSE','Bias']]

    # keep your label order if provided
    if labels is not None:
        df = df.reindex(labels)

    # restore the index name so print shows "K_bin" above the bins
    df.index.name = index_name
    return df

def get_metrics(d, y_true_vals, y_pred_vals, load_col='load', nominal_load_col='nominalLoad', k_bins_and_labels=[K_BINS, K_LABELS]):
    # errors as an index-aligned Series
    k_bins, k_labels = k_bins_and_labels

    err = pd.Series(y_pred_vals - y_true_vals, index=d.index, dtype=float)
    d = d.copy()
    d['err'] = err
    d['abs_err'] = err.abs()
    d['sq_err'] = err**2

    mae  = d['abs_err'].mean()
    rmse = np.sqrt(d['sq_err'].mean())
    mbe  = d['err'].mean()

    # Robust NRMSE (% of IQR)
    q75, q25 = np.percentile(y_true_vals, 75), np.percentile(y_true_vals, 25)
    iqr = q75 - q25 if (q75 - q25) > 0 else (np.max(y_true_vals) - np.min(y_true_vals))
    nrmse_pct = 100 * rmse / iqr if iqr > 0 else np.nan
    r2 = 1 - d['sq_err'].sum() / np.sum((y_true_vals - np.mean(y_true_vals))**2)

    # Bin by K and aggregate **per bin**
    d['K_bin'] = pd.cut(
        d[load_col] / (d[nominal_load_col] * 1000),
        bins=k_bins, labels=k_labels, include_lowest=True
    )

    byK = d.groupby('K_bin', observed=False).agg(
        n=('err', 'size'),
        MAE=('abs_err', 'mean'),
        RMSE_mean_sq=('sq_err', 'mean'),
        Bias=('err', 'mean'),
    )
    byK['RMSE'] = np.sqrt(byK['RMSE_mean_sq'])
    byK = byK[['n','MAE','RMSE','Bias']]

    # (optional) keep the header "K_bin" when printing
    byK.index.name = 'K_bin'

    # flatten to dict if you still need the K_* columns
    byK_flat = divide_by_k_df_into_columns(byK)

    metrics_dict = {
        'MAE': mae, 'RMSE': rmse, 'Bias': mbe,
        'NRMSE%_IQR': nrmse_pct, 'R2': r2,
        'P95_abs_err': d['abs_err'].quantile(0.95),
        'Max_abs_err': d['abs_err'].max(),
    }
    metrics_dict.update(byK_flat)
    return metrics_dict

def eval_wb(df, y_true='hotspotTemperature', y_pred='wb_pred', load_col='load', nominal_load_col='nominalLoad', bias_correction= 0, print_results=False, k_bins_and_labels = [K_BINS, K_LABELS]):
    d = df[[y_true, y_pred, load_col, nominal_load_col]].dropna().copy()
    y_pred_vals = d[y_pred] + bias_correction
    y_true_vals = d[y_true]
    metrics_dict = get_metrics(d, y_true_vals, y_pred_vals, load_col, nominal_load_col, k_bins_and_labels=k_bins_and_labels)
    if print_results:
        print_metrics_cols(metrics_dict)
    return metrics_dict

def get_preds_and_metrics(wb_model, d, params_hat, base_params=None, load_col='load', nominal_load_col='nominalLoad'):
    merged = {**(base_params or {}), **params_hat}
    pred = wb_model.predict(d, merged)
    tgt  = d['hotspotTemperature'].to_numpy(dtype=float)
    metrics_dict = get_metrics(d, tgt, pred, load_col=load_col, nominal_load_col=nominal_load_col)
    return metrics_dict, pred

def print_metrics_cols(results, metric_cols = None):
    if metric_cols is None:
        metric_cols = METRIC_COLUMNS

    if isinstance(results, dict):
        results = pd.DataFrame([results])

    mean_vals = results[metric_cols].mean(numeric_only=True)
    # Print those that dont start with 'K'
    for col_fit in metric_cols:
        if not col_fit.startswith('K'):
            print(f"{col_fit}: {mean_vals[col_fit]:.3f}")
    # Print K-bin metrics separately
    kbin_cols = [c for c in metric_cols if c.startswith('K')]
    if kbin_cols:
        kbin_dict = {}
        for col_fit in kbin_cols:
            col_name = '_'.join(col_fit.split('_')[0:3])
            kbin_dict[col_name] = mean_vals[col_fit]
        kbin_df = merge_by_k_columns_into_df(kbin_dict)
        print(kbin_df)

def residuals_vec(theta, d, wb_model, selected_keys, base_params=None, priors=None, prior_weight=0.0):
    fitted = {k: float(v) for k, v in zip(selected_keys, theta)}
    merged = {**(base_params or {}), **fitted}
    pred = wb_model.predict(d, merged)
    res  = pred - d['hotspotTemperature'].to_numpy(dtype=float)
    if priors and prior_weight > 0:
        reg = []
        for k, mu in priors.items():
            if k in fitted:
                reg.append(prior_weight * (fitted[k] - mu))
        if reg:
            res = np.concatenate([res, np.array(reg, dtype=float)])
    return res

# --------------------------------------------------------------------
# Fit transformers functions
# --------------------------------------------------------------------
def fit_transformer(wb_model,
                    df_equipment,
                    selected_keys,
                    base_params=None,
                    init_overrides=None,
                    bounds_overrides=None,
                    priors=None,
                    prior_weight=0.0):

    needed = ['temperature','hotspotTemperature','load','nominalLoad',
            'heatRunTest_copperLosses','heatRunTest_noLoadLosses']
    d = df_equipment[needed].dropna().copy()
    if d.empty:
        raise ValueError("No valid rows after dropna.")

    x0, lb, ub = build_x0_bounds(wb_model, selected_keys, base_params, init_overrides, bounds_overrides)
    sol = least_squares(residuals_vec, x0,
                        args=(d, wb_model, selected_keys, base_params, priors, prior_weight),
                        bounds=(lb, ub), loss='soft_l1', f_scale=5.0, max_nfev=500)

    params_hat = {k: float(v) for k, v in zip(selected_keys, sol.x)}
    metrics, pred = get_preds_and_metrics(wb_model, d, params_hat)
    return params_hat, metrics, pred

def materialize_params(wb_model, p_hat, base_params=None, all_param_keys=None):
    """Build a full {param: value} covering ALL_PARAM_KEYS using:
    1) fitted values (p_hat), then 2) base_params (tfo), then 3) registry default."""
    out = {}
    if all_param_keys is None:
        try: all_param_keys = wb_model.all_param_keys
        except AttributeError: raise ValueError("all_param_keys must be provided if wb_model has no all_param_keys attribute")
    if base_params is None:
        try: base_params = wb_model.base_params
        except AttributeError: raise ValueError("base_params must be provided if wb_model has no base_params attribute")
    for k in all_param_keys:
        if k in p_hat:
            out[k] = float(p_hat[k])
        elif base_params and k in base_params:
            out[k] = float(base_params[k])
        else:
            out[k] = wb_model.params_registry[k]['default']
    return out

def fit_all_transformers(wb_model, df, id_col='equipmentId',
                        selected_keys=None,
                        base_params_by_eid=None,
                        init_overrides_by_eid=None,
                        bounds_overrides=None,
                        priors=None, prior_weight=0.0,
                        og_comparison=False,
                        savefile=None, print_metrics=False,
                        update_model_base=True):

    if selected_keys is None:
        selected_keys = ['s','pf','heatRunTest_deltaTopOil','heatRunTest_deltaHotspot',
                        'heatRunTest_x','heatRunTest_y','heatRunTest_h']
        
    if base_params_by_eid is None:
        base_params_by_eid = wb_model.base_params

    results, preds = [], []

    for eid, g in df.groupby(id_col):
        try:
            base_params = (base_params_by_eid or {}).get(eid, {})
            init_over   = (init_overrides_by_eid or {}).get(eid, None)

            # -------- FITTED RUN --------
            p_hat, metrics_fit, pred_fit = fit_transformer(
                wb_model,
                g,
                selected_keys,
                base_params=base_params,
                init_overrides=init_over,
                bounds_overrides=bounds_overrides,
                priors=priors, prior_weight=prior_weight
            )

            params_out_fit = materialize_params(wb_model, p_hat, base_params)
            year_fit = 2024

            if update_model_base:
                if not hasattr(wb_model, 'base_params') or wb_model.base_params is None:
                    wb_model.base_params = {}
                # copy to avoid aliasing issues
                wb_model.base_params = wb_model.base_params.copy()
                wb_model.base_params[eid] = params_out_fit

            # Collect preds (long, with source)
            tmp_fit = g[['dateTime', id_col, 'hotspotTemperature']].copy()
            tmp_fit['wb_pred'] = pred_fit
            tmp_fit['source'] = 'fit'
            preds.append(tmp_fit)

            # -------- OG (TFO) COMPARISON --------
            params_out_og = {k: np.nan for k in wb_model.all_param_keys}
            metrics_og = {k: np.nan for k in METRIC_COLUMNS}
            year_og = np.nan

            if og_comparison:
                # Take OG params for selected keys from base_params (no optimization)
                og_params = {k: float(base_params[k]) for k in selected_keys if k in base_params}
                metrics_og, pred_og = get_preds_and_metrics(wb_model, g, og_params, base_params=base_params)
                params_out_og = materialize_params(wb_model, og_params, base_params)
                # pick an OG year if present
                if 'manufactureYear' in g.columns and g['manufactureYear'].notna().any():
                    year_og = g['manufactureYear'].dropna().iloc[0]

                # preds for OG
                tmp_og = g[['dateTime', id_col, 'hotspotTemperature']].copy()
                tmp_og['wb_pred'] = pred_og
                tmp_og['source'] = 'og'
                preds.append(tmp_og)

            # -------- BUILD ONE WIDE ROW --------
            wide_row = {
                'equipmentId': eid,
                'year_fit': year_fit,
                'year_og': year_og,
                'year_diff': (year_fit - year_og) if pd.notna(year_og) else np.nan,
                'n': int(len(g)),
            }

            # suffix all params + metrics
            for k, v in params_out_fit.items():
                wide_row[f'{k}_fit'] = v
            for k, v in metrics_fit.items():
                wide_row[f'{k}_fit'] = v

            for k, v in params_out_og.items():
                wide_row[f'{k}_og'] = v
            for k, v in metrics_og.items():
                wide_row[f'{k}_og'] = v

            results.append(wide_row)

        except Exception as e:
            results.append({'equipmentId': eid, 'error': str(e)})

    res_df = pd.DataFrame(results)
    pred_df = pd.concat(preds, ignore_index=True) if preds else None

    if savefile:
        res_df.to_csv(f"wb_{savefile}_parameters.csv", index=False)
        if pred_df is not None:
            pred_df.to_csv(f"wb_{savefile}_fitted.csv", index=False)

    if print_metrics and not res_df.empty:
        # Average only FIT metrics in the wide table
        fit_metric_cols = [c for c in [f'{metric_col}_fit' for metric_col in METRIC_COLUMNS] if c in res_df.columns]
        if fit_metric_cols:
            print_metrics_cols(res_df, fit_metric_cols)

    return res_df, pred_df

# --------------------------------------------------------------------
# Functions for differences
# --------------------------------------------------------------------

def compute_differences_vs_og(df_wide, relative=True, keep_years=False):
    """
    Takes the wide table produced by fit_all_transformers (one row per equipmentId)
    and returns only the *_diff columns (fit - og).
    
    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide table with *_fit and *_og columns.
    keep_years : bool
        If True, keep year_fit, year_og, year_diff for reference.
    """
    diffs = pd.DataFrame()
    diffs['equipmentId'] = df_wide['equipmentId']


    # collect only *_diff
    fit_cols = [c for c in df_wide.columns if c.endswith('_fit')]
    for c_fit in fit_cols:
        base = c_fit[:-4]
        c_og = base + '_og'
        if c_og in df_wide.columns:
            if pd.api.types.is_numeric_dtype(df_wide[c_fit]) and pd.api.types.is_numeric_dtype(df_wide[c_og]):
                if relative:
                    # relative diff
                    diffs[base + '_diff'] = (df_wide[c_fit] - df_wide[c_og]) / df_wide[c_og].replace(0, np.nan)
                else:
                    # absolute diff
                    diffs[base + '_diff'] = df_wide[c_fit] - df_wide[c_og]

    if 'year_fit' in df_wide and 'year_og' in df_wide:
        diffs['year_diff'] = df_wide['year_fit'] - df_wide['year_og']
        if keep_years:
            diffs['year_fit'] = df_wide['year_fit']
            diffs['year_og']  = df_wide['year_og']

    return diffs

def plot_differences(diffs, column, title=None, force_zero_intercept=False):
    """
    Scatter plot of year_diff (x) vs. a difference column (y),
    labeling each point by equipmentId. Skips rows where the
    column is 0 or NaN. Adds a linear fit and R^2 to the figure.
    
    Parameters
    ----------
    diffs : pd.DataFrame
        Differences table from compute_differences_vs_og
    column : str
        The difference column to plot (e.g. 'RMSE_diff')
    title : str, optional
        Plot title
    force_zero_intercept : bool, default False
        If True, fit line is forced through the origin (y = m*x).
    """
    if column not in diffs.columns:
        raise ValueError(f"{column} not in DataFrame")

    df_plot = diffs.copy()
    df_plot = df_plot[df_plot[column].notna() & (df_plot[column] != 0)]

    if df_plot.empty:
        print(f"No non-zero values to plot for column '{column}'.")
        return

    x = df_plot['year_diff'].to_numpy(dtype=float)
    y = df_plot[column].to_numpy(dtype=float)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)

    # Add labels for each point
    for _, row in df_plot.iterrows():
        plt.text(float(row['year_diff']), float(row[column]), str(row['equipmentId']),
                 fontsize=8, ha='right', va='bottom')

    r2_text = "R²: n/a"
    if len(x) >= 2 and np.nanstd(x) > 0:
        if force_zero_intercept:
            # Solve least squares for slope only
            m = np.sum(x * y) / np.sum(x * x)
            b = 0.0
        else:
            m, b = np.polyfit(x, y, 1)

        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ys = m * xs + b
        plt.plot(xs, ys, linewidth=1.5)

        # Compute R²
        y_hat = m * x + b
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if force_zero_intercept:
            eq = f"y = {m:.3g}·x"
        else:
            eq = f"y = {m:.3g}·x + {b:.3g}"
        r2_text = f"{eq}\nR² = {r2:.3f}" if np.isfinite(r2) else f"{eq}\nR²: n/a"

    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.xlabel("Year Difference")
    plt.ylabel(column)
    plt.title(title if title else f"Year Difference vs {column}")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.gca().text(0.02, 0.98, r2_text, transform=plt.gca().transAxes,
                   fontsize=9, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', alpha=0.1))

    plt.tight_layout()
    plt.show()
    
def show_metric_diff(diff_df, metric="RMSE", tol=1e-4, verbose=True, known_params=None):
    """
    Print and summarize how a chosen metric changed vs OG, grouped by the combination
    of variables that were adjusted (|param_diff| >= tol).

    Parameters
    ----------
    diff_df : pd.DataFrame
        Output from compute_differences_vs_og(...) – must contain:
        - 'equipmentId'
        - 'year_diff' (optional but useful)
        - '{metric}_diff' column for the chosen metric
        - parameter diff columns like 's_diff', 'pf_diff', 'heatRunTest_x_diff', etc.
    metric : str
        Metric name, e.g., 'RMSE', 'MAE', 'Bias', 'R2'.
    tol : float
        Absolute tolerance to decide if a variable is considered "adjusted".
        Variables with |diff| >= tol are treated as adjusted.
    verbose : bool
        If True, prints a line per equipmentId.
    known_params : list[str] or None
        If provided, only consider these names (without the '_diff' suffix) as parameters.
        If None, all *_diff columns except the chosen metric and year_diff are treated as params.

    Returns
    -------
    details_df : pd.DataFrame
        Columns: equipmentId, year_diff (if present), metric_diff, adjusted_vars
    summary_df : pd.DataFrame
        Grouped by adjusted_vars with mean_metric_diff, std_metric_diff, count
    """
    import pandas as pd
    import numpy as np

    metric_col = f"{metric}_diff"
    if metric_col not in diff_df.columns:
        raise ValueError(f"Column '{metric_col}' not found in diff_df.")

    # Identify parameter diff columns
    if known_params is not None:
        param_diff_cols = [f"{p}_diff" for p in known_params if f"{p}_diff" in diff_df.columns]
    else:
        # take all *_diff except the chosen metric and year_diff
        param_diff_cols = [
            c for c in diff_df.columns
            if c.endswith("_diff") and c not in {metric_col, "year_diff"}
        ]

    # Build details rows
    records = []
    for _, row in diff_df.iterrows():
        # which params are "adjusted"?
        adjusted = []
        for c in param_diff_cols:
            val = row[c]
            if pd.notna(val) and abs(val) >= tol:
                adjusted.append(c[:-5])  # strip '_diff' suffix

        adjusted_vars = ",".join(sorted(adjusted)) if adjusted else "(none)"
        rec = {
            "equipmentId": row["equipmentId"],
            "metric_diff": row[metric_col],
            "adjusted_vars": adjusted_vars
        }
        if "year_diff" in diff_df.columns:
            rec["year_diff"] = row["year_diff"]
        records.append(rec)

        if verbose:
            yd_txt = f", Δyears={int(row['year_diff'])}" if "year_diff" in rec and pd.notna(rec["year_diff"]) else ""
            print(f"{row['equipmentId']}: Δ{metric}={row[metric_col]:.4f}{yd_txt} | adjusted: {adjusted_vars}")

    details_df = pd.DataFrame(records)

    # Summary by combination
    agg_dict = {"metric_diff": ["mean", "std", "count"]}
    summary = (
        details_df.groupby("adjusted_vars", dropna=False)["metric_diff"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_metric_diff", "std": "std_metric_diff", "count": "count"})
        .sort_values(["count", "mean_metric_diff"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return details_df, summary


# --------------------------------------------------------------------
# Bias functions
# --------------------------------------------------------------------



# --------------------------------------------------------------------
# Model classes
# --------------------------------------------------------------------

# class DefaultWBModel:
#     def __init__(self, params_registry = None):
#         if params_registry is not None:
#             self.params_registry = params_registry
#         else:
#             self.params_registry = {
#                 's':                        {'default': 0.0, 'bounds': (0.00, 0.25)},
#                 'pf':                       {'default': 1, 'bounds': (0.80, 1.00)},
#                 'ambient_bias':             {'default': 0.0,  'bounds': (-10.0, 10.0)},
#                 'heatRunTest_deltaTopOil':  {'default': 50.0, 'bounds': (5.0, 80.0)},
#                 'heatRunTest_deltaHotspot': {'default': 63.0, 'bounds': (10.0, 120.0)},
#                 'heatRunTest_x':            {'default': 0.8,  'bounds': (0.5, 1.2)},
#                 'heatRunTest_y':            {'default': 1.5,  'bounds': (0.8, 2.2)},
#                 'heatRunTest_h':            {'default': 1.0,  'bounds': (0.7, 1.5)},
#                 'heatRunTest_copperLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'heatRunTest_noLoadLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'nominalLoad':              {'default': None, 'bounds': (1.0, 1e6)},
#             }

#         self.all_param_keys = list(self.params_registry.keys())
#         self.base_params = load_base_params_by_eid(self)


#     def predict(self, d, params):
#         s   = params.get('s', self.params_registry['s']['default'])
#         pf  = params.get('pf', self.params_registry['pf']['default'])
#         amb_bias = params.get('ambient_bias', self.params_registry['ambient_bias']['default'])

#         dTO = params.get('heatRunTest_deltaTopOil', self.params_registry['heatRunTest_deltaTopOil']['default'])
#         dH  = params.get('heatRunTest_deltaHotspot', self.params_registry['heatRunTest_deltaHotspot']['default'])
#         x   = params.get('heatRunTest_x', self.params_registry['heatRunTest_x']['default'])
#         y   = params.get('heatRunTest_y', self.params_registry['heatRunTest_y']['default'])
#         h   = params.get('heatRunTest_h', self.params_registry['heatRunTest_h']['default'])

#         Pcu_ref_W = _get_series_or_param(d, params, 'heatRunTest_copperLosses', 'heatRunTest_copperLosses')
#         P0_W      = _get_series_or_param(d, params, 'heatRunTest_noLoadLosses', 'heatRunTest_noLoadLosses')
#         S_namepl  = _get_series_or_param(d, params, 'nominalLoad', 'nominalLoad')

#         Pload_ref = Pcu_ref_W / (1.0 - s)
#         R = Pload_ref / P0_W

#         P_kW = d['load'].to_numpy(dtype=float)
#         S_base = S_namepl * 1000.0  # MVA → kVA
#         K = np.clip(P_kW / (pf * S_base), 0.0, 2.5)

#         top_oil_rise = dTO * np.power((1.0 + R * (K**2)) / (1.0 + R), x)
#         hs_rise      = h * (K**y) * (dH - dTO)

#         return d['temperature'].to_numpy(dtype=float) + amb_bias + top_oil_rise + hs_rise

# class WBModel_NoAmbientBias:
#     '''
#     WB model without ambient_bias parameter.
#     '''
#     def __init__(self, params_registry = None):
#         if params_registry is not None:
#             self.params_registry = params_registry
#         else:
#             self.params_registry = {
#                 's':                        {'default': 0.0, 'bounds': (0.00, 0.25)},
#                 'pf':                       {'default': 1, 'bounds': (0.80, 1.00)},
#                 'heatRunTest_deltaTopOil':  {'default': 50.0, 'bounds': (5.0, 80.0)},
#                 'heatRunTest_deltaHotspot': {'default': 63.0, 'bounds': (10.0, 120.0)},
#                 'heatRunTest_x':            {'default': 0.8,  'bounds': (0.5, 1.2)},
#                 'heatRunTest_y':            {'default': 1.5,  'bounds': (0.8, 2.2)},
#                 'heatRunTest_h':            {'default': 1.0,  'bounds': (0.7, 1.5)},
#                 'heatRunTest_copperLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'heatRunTest_noLoadLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'nominalLoad':              {'default': None, 'bounds': (1.0, 1e6)},
#             }

#         self.all_param_keys = list(self.params_registry.keys())
#         self.base_params = load_base_params_by_eid(self)


#     def predict(self, d, params):
#         s   = params.get('s', self.params_registry['s']['default'])
#         pf  = params.get('pf', self.params_registry['pf']['default'])

#         dTO = params.get('heatRunTest_deltaTopOil', self.params_registry['heatRunTest_deltaTopOil']['default'])
#         dH  = params.get('heatRunTest_deltaHotspot', self.params_registry['heatRunTest_deltaHotspot']['default'])
#         x   = params.get('heatRunTest_x', self.params_registry['heatRunTest_x']['default'])
#         y   = params.get('heatRunTest_y', self.params_registry['heatRunTest_y']['default'])
#         h   = params.get('heatRunTest_h', self.params_registry['heatRunTest_h']['default'])

#         Pcu_ref_W = _get_series_or_param(d, params, 'heatRunTest_copperLosses', 'heatRunTest_copperLosses')
#         P0_W      = _get_series_or_param(d, params, 'heatRunTest_noLoadLosses', 'heatRunTest_noLoadLosses')
#         S_namepl  = _get_series_or_param(d, params, 'nominalLoad', 'nominalLoad')

#         Pload_ref = Pcu_ref_W / (1.0 - s)
#         R = Pload_ref / P0_W

#         P_kW = d['load'].to_numpy(dtype=float)
#         S_base = S_namepl * 1000.0  # MVA → kVA
#         K = np.clip(P_kW / (pf * S_base), 0.0, 2.5)

#         top_oil_rise = dTO * np.power((1.0 + R * (K**2)) / (1.0 + R), x)
#         hs_rise      = h * (K**y) * (dH - dTO)

#         return d['temperature'].to_numpy(dtype=float) + top_oil_rise + hs_rise


# class WBModel_BiasCorrection:
#     '''
#     This model includes a bias correction term that is a linear function of K to ensure zero bias across K bins:
#         bias = m*K + b
#     B is always calculated per equipmentId. The slope m can be calculated globally or per equipmentId.
#     '''
#     def __init__(self, params_registry=None, df=None, base_params=None, correction_type: str = None, bias_correction=None):
#         if params_registry is not None:
#             self.params_registry = params_registry
#         else:
#             self.params_registry = {
#                 's':                        {'default': 0.0, 'bounds': (0.00, 0.25)},
#                 'pf':                       {'default': 1, 'bounds': (0.80, 1.00)},
#                 'heatRunTest_deltaTopOil':  {'default': 50.0, 'bounds': (5.0, 80.0)},
#                 'heatRunTest_deltaHotspot': {'default': 63.0, 'bounds': (10.0, 120.0)},
#                 'heatRunTest_x':            {'default': 0.8,  'bounds': (0.5, 1.2)},
#                 'heatRunTest_y':            {'default': 1.5,  'bounds': (0.8, 2.2)},
#                 'heatRunTest_h':            {'default': 1.0,  'bounds': (0.7, 1.5)},
#                 'heatRunTest_copperLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'heatRunTest_noLoadLosses': {'default': None, 'bounds': (1, 1e6)},
#                 'nominalLoad':              {'default': None, 'bounds': (1.0, 1e6)},
#                 'bias_intercept_main':      {'default': 0.0,  'bounds': (-10.0, 10.0)},
#                 'bias_slope_fit':           {'default': 0.0,  'bounds': (-10.0, 10.0)},
#                 'bias_intercept_fit':       {'default': 0.0,  'bounds': (-10.0, 10.0)},
#             }

#         self.all_param_keys = list(self.params_registry.keys())
#         if base_params is not None:
#             self.base_params = base_params
#         else:
#             self.base_params = load_base_params_by_eid(self)

#         if bias_correction is not None:
#             self.bias_correction = bias_correction
#             if df is not None:
#                 print("Warning: Both bias_correction and df provided; using bias_correction.")
#         elif df is not None:
#             self.bias_correction = {}
#             if correction_type not in {'global', 'per_equipment', 'k_per_equipment', 'k_global'}:
#                 raise ValueError("correction_type must be one of {'global', 'per_equipment', 'k_per_equipment', 'k_global'} for fitting.")
#             elif correction_type == 'global':
#                 # Fit a single global bias correction (intercept only)
#                 eids = df['equipmentId'].unique()
#                 all_results = eval_wb(df)
#                 overall_bias = all_results['Bias']
#                 for eid in eids:
#                     self.bias_correction[eid] = {'intercept_main': overall_bias, 'slope_fit': 0.0, 'intercept_fit': 0.0}
#                 print(f"Fitted global bias correction: intercept_main={overall_bias:.3f}")
#             else:
#                 main_biasses = get_bias_by_transformer(df, print_results=False)
#                 if correction_type == 'per_equipment':
#                     for eid, main_bias in main_biasses.items():
#                         self.bias_correction[eid] = {'intercept_main': main_bias, 'slope_fit': 0.0, 'intercept_fit': 0.0}
#                     print(f"Fitted per-equipment bias corrections (intercept only) for {len(main_biasses)} equipmentIds.")
#                 elif correction_type == 'k_global':
#                     alldf_bias = add_bias_to_df(df, main_biasses, id_col='equipmentId')
#                     results = eval_wb(alldf_bias, y_pred='bias_predicted')
#                     fit = get_metric_fit(results, metric='Bias', plot_graphs=False)
#                     slope = fit.c[0] if len(fit.c) > 1 else 0.0
#                     intercept = fit.c[1] if len(fit.c) > 1 else fit.c[0]
#                     for eid, main_bias in main_biasses.items():
#                         self.bias_correction[eid] = {'intercept_main': main_bias, 'slope_fit': slope, 'intercept_fit': intercept}
#                 elif correction_type == 'k_per_equipment':
#                     bias_fits = get_bias_fits_by_transformer(df, metric='Bias', id_col='equipmentId')
#                     for eid, fit_fn in bias_fits[0].items():
#                         slope = fit_fn.c[0] if len(fit_fn.c) > 1 else 0.0
#                         intercept = fit_fn.c[1] if len(fit_fn.c) > 1 else fit_fn.c[0]
#                         main_bias = main_biasses.get(eid, 0.0)
#                         self.bias_correction[eid] = {'intercept_main': main_bias, 'slope_fit': slope, 'intercept_fit': intercept}
#         else:
#             raise ValueError("Either bias_correction or df with correction_type must be provided.")

#     def predict(self, d, params):
#         s   = params.get('s', self.params_registry['s']['default'])
#         pf  = params.get('pf', self.params_registry['pf']['default'])

#         dTO = params.get('heatRunTest_deltaTopOil', self.params_registry['heatRunTest_deltaTopOil']['default'])
#         dH  = params.get('heatRunTest_deltaHotspot', self.params_registry['heatRunTest_deltaHotspot']['default'])
#         x   = params.get('heatRunTest_x', self.params_registry['heatRunTest_x']['default'])
#         y   = params.get('heatRunTest_y', self.params_registry['heatRunTest_y']['default'])
#         h   = params.get('heatRunTest_h', self.params_registry['heatRunTest_h']['default'])

#         Pcu_ref_W = _get_series_or_param(d, params, 'heatRunTest_copperLosses', 'heatRunTest_copperLosses')
#         P0_W      = _get_series_or_param(d, params, 'heatRunTest_noLoadLosses', 'heatRunTest_noLoadLosses')
#         S_namepl  = _get_series_or_param(d, params, 'nominalLoad', 'nominalLoad')

#         intercept_main = params.get('bias_intercept_main', self.params_registry['bias_intercept_main']['default'])
#         slope_fit     = params.get('bias_slope_fit', self.params_registry['bias_slope_fit']['default'])
#         intercept_fit = params.get('bias_intercept_fit', self.params_registry['bias_intercept_fit']['default'])

#         Pload_ref = Pcu_ref_W / (1.0 - s)
#         R = Pload_ref / P0_W

#         P_kW = d['load'].to_numpy(dtype=float)
#         S_base = S_namepl * 1000.0  # MVA → kVA
#         K = P_kW / (pf * S_base), 0.0, 2.5

#         top_oil_rise = dTO * np.power((1.0 + R * (K**2)) / (1.0 + R), x)
#         hs_rise      = h * (K**y) * (dH - dTO)

#         bias_correction = intercept_main + slope_fit * K + intercept_fit

#         return d['temperature'].to_numpy(dtype=float) + top_oil_rise + hs_rise + bias_correction


# ---------------------------------------------------------------------
# Base class (Template Method pattern)
# ---------------------------------------------------------------------
class BaseWBModel:
    """
    White-box transformer thermal model with extension hooks for bias terms.
    Subclasses override the bias hooks to customize the final prediction.
    """

    DEFAULT_REGISTRY = {
        's':                        {'default': 0.0, 'bounds': (0.00, 0.25)},
        'pf':                       {'default': 1.0, 'bounds': (0.80, 1.00)},
        'heatRunTest_deltaTopOil':  {'default': 50.0, 'bounds': (5.0, 80.0)},
        'heatRunTest_deltaHotspot': {'default': 63.0, 'bounds': (10.0, 120.0)},
        'heatRunTest_x':            {'default': 0.8,  'bounds': (0.5, 1.2)},
        'heatRunTest_y':            {'default': 1.5,  'bounds': (0.8, 2.2)},
        'heatRunTest_h':            {'default': 1.0,  'bounds': (0.7, 1.5)},
        'heatRunTest_copperLosses': {'default': None, 'bounds': (1, 1e6)},
        'heatRunTest_noLoadLosses': {'default': None, 'bounds': (1, 1e6)},
        'nominalLoad':              {'default': None, 'bounds': (1.0, 1e6)},
        # Subclasses may extend this (e.g., ambient_bias, bias_*).
    }

    def __init__(self, params_registry: dict | None = None, base_params: dict | None = None):
        # Compose registry
        self.params_registry = (params_registry or self.DEFAULT_REGISTRY).copy()
        self.all_param_keys = list(self.params_registry.keys())

        # Load base parameters per equipment
        self.base_params = base_params if base_params is not None else load_base_params_by_eid(self)

    # ------------------ internal helpers ------------------
    def get_param(self, params: dict, key: str) -> float:
        reg = self.params_registry[key]
        return params.get(key, reg['default'])

    def compute_K(self, d: pd.DataFrame, params: dict) -> np.ndarray:
        pf       = float(self.get_param(params, 'pf'))
        S_namepl = _get_series_or_param(d, params, 'nominalLoad', 'nominalLoad')  # MVA
        P_kW     = d['load'].to_numpy(dtype=float)
        S_base   = S_namepl * 1000.0  # MVA → kVA
        denom = np.maximum(pf * S_base, 1e-9)
        return np.clip(P_kW / denom, 0.0, 2.5)

    def thermal_core(self, d: pd.DataFrame, params: dict, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s   = float(self.get_param(params, 's'))
        dTO = float(self.get_param(params, 'heatRunTest_deltaTopOil'))
        dH  = float(self.get_param(params, 'heatRunTest_deltaHotspot'))
        x   = float(self.get_param(params, 'heatRunTest_x'))
        y   = float(self.get_param(params, 'heatRunTest_y'))
        h   = float(self.get_param(params, 'heatRunTest_h'))

        Pcu_ref_W = _get_series_or_param(d, params, 'heatRunTest_copperLosses', 'heatRunTest_copperLosses')
        P0_W      = _get_series_or_param(d, params, 'heatRunTest_noLoadLosses', 'heatRunTest_noLoadLosses')

        Pload_ref = Pcu_ref_W / (1.0 - s)
        R = Pload_ref / P0_W

        top_oil_rise = dTO * np.power((1.0 + R * (K**2)) / (1.0 + R), x)
        hs_rise      = h * (K**y) * (dH - dTO)
        return top_oil_rise, hs_rise

    # ------------------ bias hooks (override in subclasses) ------------------
    def _bias_constants_from_params(self, params: dict):
        """Return constants for bias term to avoid dict lookups in hot loops."""
        return None

    def _bias_eval_numpy(self, K: np.ndarray, b_consts, ambient_array: np.ndarray) -> np.ndarray:
        """NumPy bias evaluation. Default: zero."""
        return np.zeros_like(ambient_array, dtype=float)

    def _bias_eval_torch(self, K, b_consts, ambient_tensor):
        """Torch bias evaluation. Default: zero."""
        import torch
        return torch.zeros_like(ambient_tensor)

    # ------------------ public API ------------------
    def predict(self, d: pd.DataFrame, params: dict) -> np.ndarray:
        """
        Vectorized prediction in NumPy.
        Requires d to contain at least: ['temperature','load','heatRunTest_*','nominalLoad'].
        """
        K = self.compute_K(d, params)
        top_oil_rise, hs_rise = self.thermal_core(d, params, K)
        bias_consts = self._bias_constants_from_params(params)
        ambient = d['temperature'].to_numpy(dtype=float)
        bias = self._bias_eval_numpy(K, bias_consts, ambient)
        return ambient + top_oil_rise + hs_rise + bias
    
    def add_preds_to_df(self, df, column_name = 'wb_preds'):
        records = []
        for eid, g in df.groupby('equipmentId'):
            params = self.materialize_params_for_eid(eid)
            preds = self.predict(g, params)
            rec = g.copy()
            rec[column_name] = preds
            records.append(rec)
        return pd.concat(records, ignore_index=True)
    
    def eval_model(self, df, column_name = 'wb_preds', print_results=True):
        """
        Evaluate model on a DataFrame with 'equipmentId', 'temperature', 'load', and optional 'heatRunTest_*' and 'nominalLoad'.
        Uses base_params for each equipmentId.
        Returns a DataFrame with added 'wb_pred' column.
        """
        df_res = self.add_preds_to_df(df, column_name=column_name)
        eval_results = eval_wb(df_res, y_pred=column_name, print_results=print_results)
        return df_res, eval_results


    # ---------- PINN-friendly: bind params once & return fast callables ----------
    def materialize_params_for_eid(self, equipment_id: str, p_hat: dict | None = None) -> dict:
        """
        Compose full params for a given equipment:
           fitted (p_hat) > base_params[eid] > registry defaults
        """
        p_hat = p_hat or {}
        base = (self.base_params or {}).get(equipment_id, {})
        out = {}
        for k in self.all_param_keys:
            if k in p_hat:
                out[k] = float(p_hat[k])
            elif k in base:
                out[k] = float(base[k])
            else:
                out[k] = self.params_registry[k]['default']
        return out

    def make_pinn_predictor(self, equipment_id: str, p_hat: dict | None = None,
                            framework: str = 'torch', device=None):
        """
        Returns (predict_fn, loss_fn) specialized to one equipment + params.
        - predict_fn(temperature, load, copperLosses=None, noLoadLosses=None, nominalLoad=None) -> y_pred
        - loss_fn(y_true, temperature, load, ...) -> scalar loss
        If copper/noLoad/nominal are not provided at call time, they must be present as scalars in params.
        """
        params = self.materialize_params_for_eid(equipment_id, p_hat=p_hat)

        # Bind scalars once (avoid dict lookups in the hot path)
        s   = float(self.get_param(params, 's'))
        pf  = float(self.get_param(params, 'pf'))
        dTO = float(self.get_param(params, 'heatRunTest_deltaTopOil'))
        dH  = float(self.get_param(params, 'heatRunTest_deltaHotspot'))
        x   = float(self.get_param(params, 'heatRunTest_x'))
        y   = float(self.get_param(params, 'heatRunTest_y'))
        h   = float(self.get_param(params, 'heatRunTest_h'))

        Pcu_ref_const = params.get('heatRunTest_copperLosses')
        P0_const      = params.get('heatRunTest_noLoadLosses')
        Snamepl_const = params.get('nominalLoad')  # MVA

        bias_consts = self._bias_constants_from_params(params)

        if framework == 'torch':
            import torch

            def _ensure_tensor(x, like, name):
                if x is None:
                    return None
                if not torch.is_tensor(x):
                    x = torch.as_tensor(x, dtype=like.dtype, device=like.device)
                return x

            def predict_fn(temperature, load, copperLosses=None, noLoadLosses=None, nominalLoad=None):
                temperature = temperature if torch.is_tensor(temperature) else torch.as_tensor(temperature, device=device)
                if device is not None:
                    temperature = temperature.to(device)
                load = load if torch.is_tensor(load) else torch.as_tensor(load, dtype=temperature.dtype, device=temperature.device)

                Pcu = _ensure_tensor(copperLosses if copperLosses is not None else Pcu_ref_const, temperature, 'copperLosses')
                P0  = _ensure_tensor(noLoadLosses if noLoadLosses is not None else P0_const,      temperature, 'noLoadLosses')
                Snp = _ensure_tensor(nominalLoad  if nominalLoad  is not None else Snamepl_const, temperature, 'nominalLoad')

                if (Pcu is None) or (P0 is None) or (Snp is None):
                    raise ValueError("copperLosses, noLoadLosses, and nominalLoad must be provided either as scalars in params or as tensors at call time.")

                Pload_ref = Pcu / (1.0 - s)
                R = Pload_ref / P0
                denom = torch.clamp(pf * Snp * 1000.0, min=1e-9)
                K = torch.clamp(load / denom, min=0.0, max=2.5)

                top_oil_rise = dTO * torch.pow((1.0 + R * (K**2)) / (1.0 + R), x)
                hs_rise      = h * torch.pow(K, y) * (dH - dTO)

                bias = self._bias_eval_torch(K, bias_consts, temperature)
                return temperature + top_oil_rise + hs_rise + bias

            def loss_fn(y_true, temperature, load, copperLosses=None, noLoadLosses=None, nominalLoad=None, reduction='mean'):
                y_true = y_true if torch.is_tensor(y_true) else torch.as_tensor(y_true, dtype=temperature.dtype, device=temperature.device)
                y_pred = predict_fn(temperature, load, copperLosses, noLoadLosses, nominalLoad)
                err = y_pred - y_true
                if reduction == 'mean':
                    return torch.mean(err**2)
                elif reduction == 'sum':
                    return torch.sum(err**2)
                else:
                    return err**2

            return predict_fn, loss_fn

        # -------- NumPy version (useful for quick vectorized tests) --------
        def predict_fn_np(temperature, load, copperLosses=None, noLoadLosses=None, nominalLoad=None):
            temperature = np.asarray(temperature, dtype=float)
            load        = np.asarray(load, dtype=float)

            Pcu = np.asarray(copperLosses if copperLosses is not None else Pcu_ref_const, dtype=float) if (copperLosses is not None or Pcu_ref_const is not None) else None
            P0  = np.asarray(noLoadLosses if noLoadLosses is not None else P0_const, dtype=float) if (noLoadLosses is not None or P0_const is not None) else None
            Snp = np.asarray(nominalLoad  if nominalLoad  is not None else Snamepl_const, dtype=float) if (nominalLoad  is not None or Snamepl_const is not None) else None

            if (Pcu is None) or (P0 is None) or (Snp is None):
                raise ValueError("copperLosses, noLoadLosses, and nominalLoad must be provided either as scalars in params or as arrays at call time.")

            Pload_ref = Pcu / (1.0 - s)
            R = Pload_ref / P0
            denom = np.maximum(pf * Snp * 1000.0, 1e-9)
            K = np.clip(load / denom, 0.0, 2.5)

            top_oil_rise = dTO * np.power((1.0 + R * (K**2)) / (1.0 + R), x)
            hs_rise      = h * np.power(K, y) * (dH - dTO)

            bias = self._bias_eval_numpy(K, bias_consts, temperature)
            return temperature + top_oil_rise + hs_rise + bias

        def loss_fn_np(y_true, temperature, load, copperLosses=None, noLoadLosses=None, nominalLoad=None, reduction='mean'):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = predict_fn_np(temperature, load, copperLosses, noLoadLosses, nominalLoad)
            err2 = (y_pred - y_true)**2
            if reduction == 'mean':
                return float(err2.mean())
            elif reduction == 'sum':
                return float(err2.sum())
            else:
                return err2

        return predict_fn_np, loss_fn_np

# ---------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------
class DefaultWBModel(BaseWBModel):
    """Thermal core + constant ambient_bias added to ambient temperature."""
    def __init__(self, params_registry: dict | None = None, base_params: dict | None = None):
        reg = (params_registry or BaseWBModel.DEFAULT_REGISTRY).copy()
        reg.update({
            'ambient_bias': {'default': 0.0, 'bounds': (-10.0, 10.0)}
        })
        super().__init__(reg, base_params)

    def _bias_constants_from_params(self, params: dict):
        return float(params.get('ambient_bias', 0.0))

    def _bias_eval_numpy(self, K: np.ndarray, b_const, ambient_array: np.ndarray) -> np.ndarray:
        return np.full_like(ambient_array, fill_value=b_const, dtype=float)

    def _bias_eval_torch(self, K, b_const, ambient_tensor):
        import torch
        return torch.full_like(ambient_tensor, fill_value=b_const)

class WBModel_NoAmbientBias(BaseWBModel):
    """Thermal core only; zero bias."""
    # Inherits defaults: zero bias hooks
    pass

class WBModel_BiasCorrection(BaseWBModel):
    """
    Adds linear bias with intercepts:
        bias = bias_intercept_main + bias_slope_fit * K + bias_intercept_fit
    """
    def __init__(self, bias_correction = None, params_registry: dict | None = None, base_params: dict | None = None):
        reg = (params_registry or BaseWBModel.DEFAULT_REGISTRY).copy()
        reg.update({
            'bias_intercept_main': {'default': 0.0, 'bounds': (-10.0, 10.0)},
            'bias_slope_fit':      {'default': 0.0, 'bounds': (-10.0, 10.0)},
            'bias_intercept_fit':  {'default': 0.0, 'bounds': (-10.0, 10.0)},
        })
        self.k_bins_fit = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, np.inf]
        self.k_labels_fit = ['0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1','1-1.1','1.1-inf']
        if bias_correction is not None:
            self.bias_correction = bias_correction
        else:
            print('Bias correction was not provided. Run fit_bias_correction() and select correction_type (global, per_equipment, k_per_equipment, k_global).')
        super().__init__(reg, base_params)

    def _bias_constants_from_params(self, params: dict):
        return (
            float(params.get('bias_intercept_main', 0.0)),
            float(params.get('bias_slope_fit', 0.0)),
            float(params.get('bias_intercept_fit', 0.0)),
        )

    def _bias_eval_numpy(self, K: np.ndarray, b_consts, ambient_array: np.ndarray) -> np.ndarray:
        b_main, m_fit, b_fit = b_consts
        return -(b_main + m_fit * K + b_fit)

    def _bias_eval_torch(self, K, b_consts, ambient_tensor):
        import torch
        b_main, m_fit, b_fit = b_consts
        return -(b_main + m_fit * K + b_fit)
    
    def fit_bias_correction(self, df, correction_type):
        self.bias_correction = {}
        if correction_type not in {'global', 'per_equipment', 'k_per_equipment', 'k_global'}:
            raise ValueError("correction_type must be one of {'global', 'per_equipment', 'k_per_equipment', 'k_global'} for fitting.")
        elif correction_type == 'global':
            # Fit a single global bias correction (intercept only)
            eids = df['equipmentId'].unique()
            df_base = self.add_preds_to_df(df, column_name='wb_pred')
            all_results = eval_wb(df_base, y_pred='wb_pred')
            overall_bias = all_results['Bias']
            for eid in eids:
                self.bias_correction[int(eid)] = {'bias_intercept_main': float(overall_bias), 'bias_slope_fit': 0, 'bias_intercept_fit': 0}
            print(f"Fitted global bias correction: intercept_main={overall_bias:.3f}")
        else:
            main_biasses = self._get_bias_by_transformer(df, print_results=False)
            main_bias_col_name = 'bias_main_pred'
            alldf_bias = self._add_bias_to_df(df, main_biasses, id_col='equipmentId', column_name=main_bias_col_name)
            if correction_type == 'per_equipment':
                for eid, main_bias in main_biasses.items():
                    self.bias_correction[eid] = {'bias_intercept_main': float(main_bias), 'bias_slope_fit': 0, 'bias_intercept_fit': 0}
                print(f"Fitted per-equipment bias corrections (intercept only) for {len(main_biasses)} equipmentIds.")
            elif correction_type == 'k_global':
                results = eval_wb(alldf_bias, y_pred=main_bias_col_name, k_bins_and_labels=[self.k_bins_fit, self.k_labels_fit])
                fit = self._get_metric_fit(results, metric='Bias', plot_graphs=False)
                slope = fit.c[0] if len(fit.c) > 1 else 0.0
                intercept = fit.c[1] if len(fit.c) > 1 else fit.c[0]
                for eid, main_bias in main_biasses.items():
                    self.bias_correction[eid] = {'bias_intercept_main': float(main_bias), 'bias_slope_fit': float(slope), 'bias_intercept_fit': float(intercept)}
            elif correction_type == 'k_per_equipment':
                bias_fits = self._get_bias_fits_by_transformer(alldf_bias, base_column=main_bias_col_name, metric='Bias', id_col='equipmentId')
                for eid, fit_fn in bias_fits.items():
                    slope = fit_fn.c[0] if len(fit_fn.c) > 1 else 0.0
                    intercept = fit_fn.c[1] if len(fit_fn.c) > 1 else fit_fn.c[0]
                    main_bias = main_biasses.get(eid, 0.0)
                    self.bias_correction[eid] = {'bias_intercept_main': float(main_bias), 'bias_slope_fit': float(slope), 'bias_intercept_fit': float(intercept)}
        for eid, bc in self.bias_correction.items():
            self.base_params[eid].update(bc)

    def _get_bias_by_transformer(self, df, id_col='equipmentId', print_results=False, column_name='wb_pred'):
        biasses = {}
        for eid, g in df.groupby(id_col):
            g_res = self.add_preds_to_df(g, column_name=column_name)
            result = eval_wb(g_res, y_pred=column_name)
            if print_results: 
                print(f"Evaluating equipmentId={eid} with {len(g)} rows")
                print(result)
            biasses[eid] = result['Bias']
        return biasses

    def _add_bias_to_df(self, df, bias_map, id_col='equipmentId', column_name = 'bias_predicted'):
        df = df.copy()
        df['wb_bias'] = df[id_col].map(bias_map)
        df[column_name] = df['wb_pred'] - df['wb_bias']
        return df

    def _add_K_bin_bias_to_df(self, df, fit_fn, pred_col='wb_pred'):
        df = df.copy()
        K = df['load'] / (df['nominalLoad'] * 1000.0)
        correction = fit_fn(K)
        df['k_bin_pred'] = df[pred_col] - correction
        return df

    def _get_metric_fit(self, result, metric, plot_graphs=False):
        k_columns = [k for k in result.keys() if k.startswith('K_')]
        filtered_columns = [c for c in k_columns if c.endswith(f'_{metric}')]
        x_values = [(float(c.split('_')[1].split('-')[1])+float(c.split('_')[1].split('-')[0]))/2 for c in filtered_columns]
        y_values = [result[c] for c in filtered_columns]
        mask = ~np.isnan(y_values)
        x_values = np.array(x_values)[mask]
        y_values = np.array(y_values)[mask]
        fit = np.polyfit(x_values, y_values, 1)
        fit_fn = np.poly1d(fit)
        if plot_graphs:
            plt.figure(figsize=(8,6))
            plt.plot(x_values, y_values, marker='o')
            plt.plot(x_values, fit_fn(x_values), linestyle='--')
        print(fit_fn.c)
        return fit_fn

    def _get_bias_fits_by_transformer(self, df, metric='Bias', id_col='equipmentId', base_column='bias_predicted', print_results=False, plot_graphs=False):
        bias_fits = {}
        for eid, g in df.groupby(id_col):
            result = eval_wb(g, y_pred=base_column, k_bins_and_labels=[self.k_bins_fit, self.k_labels_fit])
            if print_results:
                print(f"Evaluating equipmentId={eid} with {len(g)} rows")
                print_metrics_cols(result)
            bias_fits[int(eid)] = self._get_metric_fit(result, metric, plot_graphs=plot_graphs)
        return bias_fits

    def _add_bias_fits_to_df(self, df, bias_map,
                    id_col='equipmentId',
                    k_col='K',
                    load_col='load',
                    nominal_load_col='nominalLoad'):
        """
        Adds:
        - wb_bias: bias estimated from the per-equipment fit evaluated at K
        - bias_predicted: wb_pred - wb_bias
        If K isn't present, it's computed as load / (nominalLoad * 1000).
        """
        out = df.copy()

        # Ensure K exists
        if k_col not in out.columns:
            out[k_col] = out[load_col] / (out[nominal_load_col] * 1000)

        # Prepare output column
        out['wb_bias'] = np.nan

        # Apply each equipment's fit to its K values
        for eid, idx in out.groupby(id_col).groups.items():
            fit_fn = bias_map.get(eid)  # this is your np.poly1d from get_bias_fits_by_transformer
            if fit_fn is None:
                continue
            k_vals = out.loc[idx, k_col].astype(float).to_numpy()
            # Handle missing K gracefully
            mask = ~np.isnan(k_vals)
            if mask.any():
                out.loc[idx[mask], 'wb_bias'] = fit_fn(k_vals[mask])

        # Final column
        out['bias_predicted'] = out['wb_pred'] - out['wb_bias']
        return out
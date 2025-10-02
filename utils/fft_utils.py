from __future__ import annotations
import os
import math
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

def _infer_dt_seconds(t: pd.Series) -> float:
    """Infer a regular sampling step (seconds) from a datetime Series."""
    t = pd.to_datetime(t).sort_values().dropna().drop_duplicates()
    if len(t) < 2:
        raise ValueError("Not enough timestamps to infer sampling interval.")
    deltas = t.diff().dt.total_seconds().iloc[1:]
    dt = float(np.median(deltas))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer a positive sampling interval.")
    return dt

def _rule_from_seconds(dt: float) -> str:
    """Convert seconds -> pandas resample rule (nearest second)."""
    return f"{max(1, int(round(dt)))}S"

def _regularize_series(df: pd.DataFrame, time_col: str, value_col: str,
                       resample_seconds: Optional[float]) -> Tuple[pd.DatetimeIndex, np.ndarray, float]:
    """Return (index, values, dt_seconds) as a uniformly sampled series (linear interp)."""
    s = df[[time_col, value_col]].copy()
    s[time_col] = pd.to_datetime(s[time_col], errors="coerce")
    s = s.dropna(subset=[time_col]).sort_values(time_col)
    if s.empty:
        raise ValueError("No valid time/value data.")
    dt = resample_seconds if resample_seconds else _infer_dt_seconds(s[time_col])
    rule = _rule_from_seconds(dt)
    ser = (s.set_index(time_col)[value_col]
             .astype(float)
             .resample(rule)
             .interpolate("time"))
    ser = ser.dropna()
    return ser.index, ser.values.astype(float), float(dt)


def _prepare_uniform_series(df: pd.DataFrame,
                            time_col: str,
                            value_col: str,
                            resample_seconds: Optional[float],
                            detrend: bool,
                            window: Optional[str]) -> Tuple[np.ndarray, float]:
    """
    Return uniformly-sampled, windowed series xw and dt (seconds).
    """
    idx, x, dt = _regularize_series(df, time_col, value_col, resample_seconds)
    if x.size < 4:
        raise ValueError("Too few samples for FFT.")

    if detrend:
        x = x - np.nanmean(x)

    if window == "hann":
        w = np.hanning(x.size)
    elif window in (None, False):
        w = np.ones(x.size, dtype=float)
    else:
        raise ValueError(f"Unsupported window: {window}")

    xw = x * w
    return xw, float(dt)

def _fft_complex(xw: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complex rFFT and frequency axis in Hz.
    """
    Y = np.fft.rfft(xw)                  # complex spectrum
    f = np.fft.rfftfreq(xw.size, d=dt)   # Hz
    return f, Y

def _package_spectrum(f_hz: np.ndarray,
                      Y: np.ndarray,
                      dt: float,
                      window_energy: Optional[float] = None,
                      amplitude_norm: str = "onesided") -> pd.DataFrame:
    """
    Build a tidy DataFrame with complex spectrum + amplitude/power/phase.
    """
    n = Y.size if f_hz.size == Y.size else (2*(Y.size-1))
    # Default window energy = N if not provided
    if window_energy is None:
        window_energy = float(n)

    # amplitude scaling consistent with compute_fft()
    if amplitude_norm == "onesided":
        scale = 2.0 / window_energy
        amp = np.abs(Y) * scale
        if amp.size > 0:
            amp[0] = amp[0] / 2.0
            if (n % 2) == 0:
                amp[-1] = amp[-1] / 2.0
    else:
        scale = 1.0 / window_energy
        amp = np.abs(Y) * scale

    out = pd.DataFrame({
        "frequency_hz": f_hz,
        "frequency_per_day": f_hz * 86400.0,
        "real": Y.real,
        "imag": Y.imag,
        "amplitude": amp,
        "power": amp**2,
        "phase_rad": np.angle(Y)
    })
    out["n_samples"] = int(round(n))
    out["dt_seconds"] = float(dt)
    return out

def compute_fft_complex(df: pd.DataFrame,
                        time_col: str = "dateTime",
                        value_col: str = "load",
                        resample_seconds: Optional[float] = None,
                        detrend: bool = True,
                        window: Optional[str] = "hann",
                        amplitude_norm: str = "onesided") -> pd.DataFrame:
    """
    Return the complex FFT spectrum of a single series, plus amplitude/power/phase.
    """
    # prepare windowed series and dt
    idx, x, dt = _regularize_series(df, time_col, value_col, resample_seconds)
    if x.size < 4:
        raise ValueError("Too few samples for FFT.")
    if detrend:
        x = x - np.nanmean(x)
    if window == "hann":
        w = np.hanning(x.size)
    elif window in (None, False):
        w = np.ones(x.size, dtype=float)
    else:
        raise ValueError(f"Unsupported window: {window}")

    xw = x * w
    f_hz, Y = _fft_complex(xw, dt)
    return _package_spectrum(f_hz, Y, dt, window_energy=float(np.sum(w)), amplitude_norm=amplitude_norm)

def compute_fft_diff_complex(df_a: pd.DataFrame,
                             df_b: pd.DataFrame,
                             time_col: str = "dateTime",
                             value_col: str = "load",
                             resample_seconds: Optional[float] = None,
                             detrend: bool = True,
                             window: Optional[str] = "hann",
                             amplitude_norm: str = "onesided") -> pd.DataFrame:
    """
    FFT of the time-domain difference (A - B), using identical resampling/windowing.
    This is equivalent to subtracting complex FFTs after **identical** preprocessing.
    """
    # Build uniform, windowed series for A and B on **their own** grids
    # Then align to a **common** grid: choose the smaller dt and the overlapping time span.
    # We reuse _regularize_series twice, then resample both to the same rule.

    # 1) Uniform series (unwindowed) so we can align lengths exactly
    idx_a, xa, dt_a = _regularize_series(df_a, time_col, value_col, resample_seconds)
    idx_b, xb, dt_b = _regularize_series(df_b, time_col, value_col, resample_seconds)

    # 2) choose common dt (finer of the two) and overlap range
    dt = float(min(dt_a, dt_b))
    rule = f"{max(1, int(round(dt)))}S"
    t0 = max(idx_a.min(), idx_b.min())
    t1 = min(idx_a.max(), idx_b.max())
    if not (pd.notna(t0) and pd.notna(t1) and t1 > t0):
        raise ValueError("No overlapping time range between the two series.")

    idx_common = pd.date_range(t0.floor("S"), t1.ceil("S"), freq=rule)

    # 3) reindex/interpolate both onto the common grid
    sa = pd.Series(xa, index=idx_a).reindex(idx_common).interpolate("time")
    sb = pd.Series(xb, index=idx_b).reindex(idx_common).interpolate("time")

    x = sa.values.astype(float)
    y = sb.values.astype(float)
    if detrend:
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    # 4) identical window
    n = min(x.size, y.size)
    x = x[:n]; y = y[:n]
    if window == "hann":
        w = np.hanning(n)
    elif window in (None, False):
        w = np.ones(n, dtype=float)
    else:
        raise ValueError(f"Unsupported window: {window}")

    diffw = (x - y) * w

    # 5) FFT of the difference
    f_hz, Ydiff = _fft_complex(diffw, dt)
    return _package_spectrum(f_hz, Ydiff, dt, window_energy=float(np.sum(w)), amplitude_norm=amplitude_norm)


def compute_fft(df: pd.DataFrame,
                time_col: str = "dateTime",
                value_col: str = "load",
                resample_seconds: Optional[float] = None,
                detrend: bool = True,
                window: Optional[str] = "hann",
                amplitude_norm: str = "onesided") -> pd.DataFrame:
    """
    Compute single-sided FFT magnitude (and power) for a time series.
    - Resamples to a regular grid (median dt if resample_seconds is None)
    - Detrends by removing mean if detrend=True
    - Window supports: None, "hann"
    - amplitude_norm "onesided": 2/N * |FFT| (except DC & Nyquist)
    Returns columns: ['frequency_hz','frequency_per_day','amplitude','power','n_samples','dt_seconds']
    """
    idx, x, dt = _regularize_series(df, time_col, value_col, resample_seconds)
    n = x.size
    if n < 4:
        raise ValueError("Too few samples for FFT.")

    # Detrend (remove mean)
    if detrend:
        x = x - np.nanmean(x)

    # Window
    if window == "hann":
        w = np.hanning(n)
    elif window in (None, False):
        w = np.ones(n, dtype=float)
    else:
        raise ValueError(f"Unsupported window: {window}")

    xw = x * w

    # FFT
    y = np.fft.rfft(xw)
    freqs_hz = np.fft.rfftfreq(n, d=dt)

    # Basic amplitude normalization (single-sided)
    # Scale by window energy to keep magnitudes comparable when windowing
    scale = 2.0 / np.sum(w) if amplitude_norm == "onesided" else 1.0 / np.sum(w)
    amp = np.abs(y) * scale
    # Fix DC and Nyquist not to double
    if amplitude_norm == "onesided":
        amp[0] = amp[0] / 2.0
        if n % 2 == 0:  # Nyquist present
            amp[-1] = amp[-1] / 2.0

    power = amp ** 2
    cpd = freqs_hz * 86400.0

    out = pd.DataFrame({
        "frequency_hz": freqs_hz,
        "frequency_per_day": cpd,
        "amplitude": amp,
        "power": power
    })
    out["n_samples"] = n
    out["dt_seconds"] = dt
    return out

def _cache_key(source_tag: str, equipment_id, column: str) -> str:
    raw = f"{source_tag}::{equipment_id}::{column}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def load_fft_from_cache(cache_dir: Path, source_tag: str, equipment_id, column: str) -> Optional[pd.DataFrame]:
    cache_dir = Path(cache_dir)
    key = _cache_key(source_tag, equipment_id, column)
    pq = cache_dir / f"{key}.parquet"
    csv = cache_dir / f"{key}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    return None

def save_fft_to_cache(cache_dir: Path, source_tag: str, equipment_id, column: str, df_fft: pd.DataFrame) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(source_tag, equipment_id, column)
    path = cache_dir / f"{key}.parquet"
    try:
        df_fft.to_parquet(path, index=False)
    except Exception:
        path = cache_dir / f"{key}.csv"
        df_fft.to_csv(path, index=False)
    return path

import numpy as np

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

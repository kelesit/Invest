from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import bottleneck as _bn
    _HAS_BN = True
except ImportError:
    _HAS_BN = False


class FeatureTransformError(RuntimeError):
    """Raised when the transform config references unknown primitives."""


@dataclass(frozen=True)
class CrossCombinationSpec:
    """Combine two primitives (using their cs_rank versions) with one operation."""

    left: str
    right: str
    op: str   # "mul" | "sub" | "div"
    name: str


@dataclass(frozen=True)
class TransformConfig:
    keep_identity: bool = True
    apply_cs_rank: bool = True
    ts_zscore_windows: tuple[int, ...] = (20, 60, 120)
    ts_change_windows: tuple[int, ...] = (5, 20, 60)
    apply_second_order: bool = True
    cross_combinations: tuple[CrossCombinationSpec, ...] = ()
    output_prefix: str = "feat_"


def compute_feature_frame(
    primitives: dict[str, pd.DataFrame],
    config: TransformConfig | None = None,
) -> pd.DataFrame:
    """Transform primitive matrices into a flat (date, symbol, feat_*) DataFrame.

    Memory strategy: primitives are processed one at a time. Each primitive's
    transform matrices are computed, immediately flattened to 1-D float32 arrays,
    and then released. The final DataFrame is constructed from the accumulated
    column dict in a single pass.

    Second-order transforms reuse the first-order intermediate arrays to avoid
    redundant computation.
    """
    cfg = config or TransformConfig()
    _validate_config(cfg, primitives)

    ref = next(iter(primitives.values()))
    n_dates, n_symbols = ref.shape
    date_arr = np.repeat(ref.index.to_numpy(), n_symbols)
    symbol_arr = np.tile(ref.columns.to_numpy(), n_dates)

    columns: dict[str, np.ndarray] = {}

    for key, matrix in primitives.items():
        _transform_primitive(key, matrix.to_numpy(dtype=np.float32), cfg, columns)

    _apply_cross_combinations(primitives, cfg, columns)

    return pd.DataFrame({"date": date_arr, "symbol": symbol_arr, **columns})


# ---------------------------------------------------------------------------
# Per-primitive transform
# ---------------------------------------------------------------------------

def _transform_primitive(
    key: str,
    mat: np.ndarray,
    cfg: TransformConfig,
    out: dict[str, np.ndarray],
) -> None:
    p = cfg.output_prefix

    if cfg.keep_identity:
        out[f"{p}{key}"] = mat.ravel()

    # First-order: compute once, store intermediate for second-order reuse
    csr = _cs_rank(mat) if cfg.apply_cs_rank else None
    if csr is not None:
        out[f"{p}{key}_csr"] = csr.ravel()

    tsz: dict[int, np.ndarray] = {}
    for w in cfg.ts_zscore_windows:
        tsz[w] = _ts_zscore(mat, w)
        out[f"{p}{key}_tsz{w}"] = tsz[w].ravel()

    tc: dict[int, np.ndarray] = {}
    for w in cfg.ts_change_windows:
        tc[w] = _ts_change(mat, w)
        out[f"{p}{key}_tc{w}"] = tc[w].ravel()

    if not cfg.apply_second_order:
        return

    # Second-order: reuse intermediates, no recomputation
    for w, arr in tsz.items():
        out[f"{p}{key}_tsz{w}_csr"] = _cs_rank(arr).ravel()

    for w, arr in tc.items():
        out[f"{p}{key}_tc{w}_csr"] = _cs_rank(arr).ravel()

    if csr is not None:
        for w in cfg.ts_zscore_windows:
            out[f"{p}{key}_csr_tsz{w}"] = _ts_zscore(csr, w).ravel()
        for w in cfg.ts_change_windows:
            out[f"{p}{key}_csr_tc{w}"] = _ts_change(csr, w).ravel()


# ---------------------------------------------------------------------------
# Cross-combinations (second pass, after all primitives are processed)
# ---------------------------------------------------------------------------

def _apply_cross_combinations(
    primitives: dict[str, pd.DataFrame],
    cfg: TransformConfig,
    out: dict[str, np.ndarray],
) -> None:
    p = cfg.output_prefix
    for spec in cfg.cross_combinations:
        left_col = f"{p}{spec.left}_csr"
        right_col = f"{p}{spec.right}_csr"
        if left_col not in out or right_col not in out:
            raise FeatureTransformError(
                f"Cross-combination '{spec.name}' needs cs_rank of '{spec.left}' and "
                f"'{spec.right}'. Enable apply_cs_rank=True."
            )
        left = out[left_col]
        right = out[right_col]
        if spec.op == "mul":
            result = left * right
        elif spec.op == "sub":
            result = left - right
        elif spec.op == "div":
            with np.errstate(invalid="ignore", divide="ignore"):
                result = np.where(right != 0, left / right, np.nan)
        else:
            raise FeatureTransformError(f"Unknown cross-combination op: {spec.op!r}")
        out[f"{p}{spec.name}"] = result.astype(np.float32)


# ---------------------------------------------------------------------------
# Primitive transforms (numpy-level, float32 output)
# ---------------------------------------------------------------------------

def _cs_rank(mat: np.ndarray) -> np.ndarray:
    """Percentile rank across the symbol axis (axis=1) for each date, NaN-aware.

    Uses double-argsort (fully vectorized, no Python loops).
    NaN inputs are pushed to -inf so they sort first, then masked out.
    Tie handling: ordinal (not average). Float inputs rarely tie exactly,
    so the difference from average-rank is negligible in practice.
    """
    nan_mask = ~np.isfinite(mat)
    # Replace NaN with -inf so they sort first along axis=1
    temp = np.where(nan_mask, -np.inf, mat.astype(np.float64))

    # argsort twice: first gives sort order, second gives rank (0-indexed)
    order = np.argsort(temp, axis=1)
    ranks = np.argsort(order, axis=1).astype(np.float32)

    # Number of NaN per row — NaN values received the lowest ranks (0,1,...,n_nans-1)
    n_nans = nan_mask.sum(axis=1, keepdims=True).astype(np.float32)
    n_valid = mat.shape[1] - n_nans

    # Shift ranks so valid values are 1-indexed among themselves, then normalise
    result = (ranks - n_nans + 1.0) / np.maximum(n_valid, 1.0)
    result[nan_mask] = np.nan
    return result.astype(np.float32)


def _ts_zscore(mat: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score along the time axis (axis=0), one series per symbol.

    Uses bottleneck when available (3-4x faster). Falls back to prefix-sum.
    A window is valid only when all `window` values are finite (min_periods=window).
    """
    if _HAS_BN:
        m = _bn.move_mean(mat, window=window, min_count=window, axis=0)
        s = _bn.move_std(mat, window=window, min_count=window, ddof=1, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(s > 0, (mat - m) / s, np.nan)
        return z.astype(np.float32)

    n, cols = mat.shape
    nan_mask = ~np.isfinite(mat)
    x = np.where(nan_mask, 0.0, mat.astype(np.float64))

    def _prefix(a: np.ndarray) -> np.ndarray:
        return np.vstack([np.zeros((1, cols), dtype=a.dtype), np.cumsum(a, axis=0)])

    ps  = _prefix(x)
    ps2 = _prefix(x ** 2)
    pn  = _prefix((~nan_mask).astype(np.float64))

    roll_s  = ps[window:]  - ps[:-window]
    roll_s2 = ps2[window:] - ps2[:-window]
    roll_n  = pn[window:]  - pn[:-window]

    all_valid = roll_n == window
    mean = roll_s / window
    var  = np.maximum((roll_s2 - roll_s ** 2 / window) / (window - 1), 0.0)
    std  = np.sqrt(var)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(all_valid & (std > 0), (x[window - 1:] - mean) / std, np.nan)

    result = np.full((n, cols), np.nan, dtype=np.float32)
    result[window - 1:] = z.astype(np.float32)
    return result


def _ts_change(mat: np.ndarray, window: int) -> np.ndarray:
    """Absolute difference from `window` periods ago along the time axis."""
    result = np.empty_like(mat, dtype=np.float32)
    result[:window] = np.nan
    result[window:] = (mat[window:] - mat[:-window]).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(cfg: TransformConfig, primitives: dict[str, pd.DataFrame]) -> None:
    if not primitives:
        raise FeatureTransformError("primitives dict is empty")
    for spec in cfg.cross_combinations:
        for attr in (spec.left, spec.right):
            if attr not in primitives:
                raise FeatureTransformError(
                    f"Cross-combination '{spec.name}' references unknown primitive: {attr!r}"
                )

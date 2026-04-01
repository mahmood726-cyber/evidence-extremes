"""
EvidenceExtremes: Extreme Value Theory for Cochrane Meta-Analysis Trust Scores.

Fits GEV to block minima (domain-level) and GPD to exceedances below threshold 50,
then computes return levels and compares tail indices across domains.
"""

import io
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto
from typing import Optional

# ── stdout safety (Windows cp1252) ─────────────────────────────────────────
# Only redirect when running as a script (not under pytest capture)
if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── DATA LOADING ────────────────────────────────────────────────────────────

DATA_SCORES = "C:/Models/EvidenceScore/results/scores.csv"
DATA_GROUPS = "C:/Models/TrustGate/data/review_groups.csv"

THRESHOLD = 50.0  # F-grade boundary


def load_data(
    scores_path: str = DATA_SCORES,
    groups_path: str = DATA_GROUPS,
) -> pd.DataFrame:
    """Load and join scores with domain labels.

    Returns DataFrame with columns: review_id, final_score, review_group.
    Each row is one meta-analysis.
    """
    scores = pd.read_csv(scores_path)
    groups = pd.read_csv(groups_path)

    # Merge on review_id: groups has review_id_prefix (= CD-number stem)
    # scores has review_id which IS the CD-number
    merged = scores.merge(
        groups.rename(columns={"review_id_prefix": "review_id"}),
        on="review_id",
        how="left",
    )
    merged["review_group"] = merged["review_group"].fillna("Other")
    return merged[["review_id", "final_score", "review_group"]]


def get_block_minima(df: pd.DataFrame) -> pd.Series:
    """Return minimum trust score per review_group (domain-level block minima)."""
    return df.groupby("review_group")["final_score"].min()


# ── GEV FITTING ─────────────────────────────────────────────────────────────

def fit_gev(block_minima: np.ndarray) -> dict:
    """Fit GEV distribution to block minima (lower tail).

    scipy.stats.genextreme fits MAXIMA by convention, so we negate the data
    (i.e. fit -block_minima), then recover the location for the original scale.

    Returns dict with keys: shape (xi), loc (mu_min), scale (sigma), success.
    shape > 0 => Frechet (heavy lower tail)
    shape < 0 => Weibull (bounded)
    shape ~ 0 => Gumbel
    """
    arr = np.asarray(block_minima, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 3:
        return {"shape": np.nan, "loc": np.nan, "scale": np.nan, "success": False,
                "n": len(arr)}

    neg = -arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, loc, scale = genextreme.fit(neg)

    # mu_min is the location on the ORIGINAL (non-negated) scale
    mu_min = -loc

    return {
        "shape": float(shape),   # xi
        "loc": float(mu_min),    # mu on original scale
        "scale": float(scale),   # sigma
        "success": True,
        "n": len(arr),
    }


# ── GPD FITTING ─────────────────────────────────────────────────────────────

def compute_exceedances(scores: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Return exceedances: u - score for all scores < threshold (positive values)."""
    arr = np.asarray(scores, dtype=float)
    below = arr[arr < threshold]
    return threshold - below  # positive


def fit_gpd(exceedances: np.ndarray) -> dict:
    """Fit GPD to exceedances (already u - x, so >= 0).

    Uses floc=0 (threshold already subtracted).
    Returns dict with keys: shape, scale, success, n.
    """
    arr = np.asarray(exceedances, dtype=float)
    arr = arr[arr > 0]  # must be strictly positive

    if len(arr) < 3:
        return {"shape": np.nan, "scale": np.nan, "success": False, "n": len(arr)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shape, loc, scale = genpareto.fit(arr, floc=0)

    return {
        "shape": float(shape),
        "scale": float(scale),
        "success": True,
        "n": len(arr),
    }


# ── RETURN LEVELS ────────────────────────────────────────────────────────────

def return_level_gev(
    shape: float,
    loc: float,
    scale: float,
    m: float,
    gumbel_tol: float = 1e-6,
) -> float:
    """Compute GEV return level for MINIMA for return period m.

    Derivation: we fit GEV on -block_minima (maxima convention).
    The m-period return level for minima is x_m = -w_m where w_m is the
    m-period maximum return level of the negated data.

    For MAXIMA: w_m = loc_neg + sigma/xi*(y^(-xi) - 1), y = -log(1-1/m)
    For MINIMA: x_m = -w_m = -loc_neg - sigma/xi*(y^(-xi) - 1)
                           = mu + sigma/xi*(1 - y^(-xi))   [since mu = -loc_neg]

    Wait — that still increases with m. Let me be precise:
    x_m = -loc_neg - sigma/xi*(y^(-xi) - 1) = mu - sigma/xi*(y^(-xi) - 1)
        = mu + sigma/xi*(1 - y^(-xi))
    As m -> inf, y -> 0, y^(-xi) -> inf for xi>0, so 1-y^(-xi) -> -inf => x_m -> -inf. CORRECT.

    Gumbel special case (|xi| < tol):
    w_m = loc_neg + sigma*log(y), so x_m = -loc_neg - sigma*log(y) = mu - sigma*log(y)
    Wait: mu = -loc_neg, so x_m = mu - sigma*log(y).
    Actually: Gumbel ppf(1-1/m) = loc_neg - sigma*log(-log(1-1/m)) = loc_neg - sigma*log(y)
    x_m = -(loc_neg - sigma*log(y)) = -loc_neg + sigma*log(y) = mu + sigma*log(y)
    As m->inf, y->0, log(y)->-inf => x_m -> -inf. CORRECT.

    Returns the return level (a trust score; lower is more extreme for minima).
    """
    if not (np.isfinite(shape) and np.isfinite(loc) and np.isfinite(scale)):
        return np.nan

    p = 1.0 / m  # probability of being that extreme
    y = -np.log(1.0 - p)  # reduced variate (positive for p in (0,1), small for large m)

    if abs(shape) < gumbel_tol:
        # Gumbel minima: x_m = mu + sigma*log(y)  [log(y) < 0 for m > ~2.7]
        x_m = loc + scale * np.log(y)
    else:
        # Frechet/Weibull minima: x_m = mu + sigma/xi*(1 - y^(-xi))
        # For xi>0: y^(-xi)->inf as y->0 => x_m -> -inf as m->inf. CORRECT.
        x_m = loc + (scale / shape) * (1.0 - y ** (-shape))

    return float(x_m)


def return_levels_all(
    shape: float,
    loc: float,
    scale: float,
    periods: tuple = (50, 100, 500),
) -> dict:
    """Return dict of {m: return_level} for requested periods."""
    return {m: return_level_gev(shape, loc, scale, m) for m in periods}


# ── TAIL INDEX COMPARISON ────────────────────────────────────────────────────

def tail_index_by_domain(
    df: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Fit GPD to exceedances per domain, return shape (tail index) + CI.

    Higher xi (shape) => heavier lower tail => more extreme low-quality outliers.

    Returns DataFrame with columns: domain, xi, scale, n, ci_lo, ci_hi.
    The CI is a simple normal approximation using SE ~ sqrt(2/n) for xi.
    """
    records = []
    for domain, grp in df.groupby("review_group"):
        exc = compute_exceedances(grp["final_score"].values, threshold)
        result = fit_gpd(exc)
        n = result["n"]
        xi = result["shape"]
        sc = result["scale"]

        # Simple SE approximation: var(xi) ≈ 2/n for GPD (asymptotically)
        if result["success"] and n >= 3:
            se_xi = np.sqrt(2.0 / n)
            ci_lo = xi - 1.96 * se_xi
            ci_hi = xi + 1.96 * se_xi
        else:
            se_xi = np.nan
            ci_lo = np.nan
            ci_hi = np.nan

        records.append({
            "domain": domain,
            "xi": xi,
            "scale": sc,
            "n": n,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "success": result["success"],
        })

    return pd.DataFrame(records).sort_values("xi", ascending=False)


# ── QQ PLOT DATA ─────────────────────────────────────────────────────────────

def qq_plot_data(scores: np.ndarray, shape: float, loc: float, scale: float) -> dict:
    """Compute empirical vs theoretical quantiles for GEV QQ plot.

    The GEV was fitted on negated data (-scores) in maxima convention.
    For the QQ plot on the ORIGINAL scale:
    - Empirical: sorted scores ascending (smallest first = most extreme minima first)
    - Theoretical: the quantiles of the MINIMA distribution at the same plotting positions.

    Since the minima CDF corresponds to the complementary probability of the maxima CDF
    on the negated scale, we use (1-pp) when querying genextreme.ppf on the negated data,
    then negate back to get the minima quantiles on the original scale.

    Returns dict with 'empirical' and 'theoretical' arrays (both ascending).
    """
    arr = np.sort(np.asarray(scores, dtype=float))
    n = len(arr)
    if n < 2:
        return {"empirical": arr.tolist(), "theoretical": []}

    # Gringorten plotting positions
    pp = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
    pp = np.clip(pp, 1e-6, 1 - 1e-6)

    # Theoretical: use 1-pp on the negated distribution, then negate back
    neg_loc = -loc  # loc_neg = -mu_min (from fit on -scores)
    theoretical_neg = genextreme.ppf(1.0 - pp, shape, loc=neg_loc, scale=scale)
    theoretical = -theoretical_neg  # back to original score scale (ascending)

    return {
        "empirical": arr.tolist(),
        "theoretical": theoretical.tolist(),
    }


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(
    scores_path: str = DATA_SCORES,
    groups_path: str = DATA_GROUPS,
    threshold: float = THRESHOLD,
    return_periods: tuple = (50, 100, 500),
) -> dict:
    """End-to-end pipeline: load data, fit GEV + GPD, compute return levels.

    Returns a results dict suitable for dashboard generation.
    """
    df = load_data(scores_path, groups_path)

    n_total = len(df)
    n_below_50 = int((df["final_score"] < threshold).sum())

    # --- Block minima (one per domain) ---
    block_min = get_block_minima(df)
    gev_result = fit_gev(block_min.values)

    # --- Return levels ---
    rl = return_levels_all(
        gev_result["shape"],
        gev_result["loc"],
        gev_result["scale"],
        periods=return_periods,
    )

    # --- GPD on overall exceedances ---
    all_exc = compute_exceedances(df["final_score"].values, threshold)
    gpd_result = fit_gpd(all_exc)

    # --- Tail index by domain ---
    tail_df = tail_index_by_domain(df, threshold)

    # --- QQ data (overall) ---
    qq = qq_plot_data(
        df["final_score"].values,
        gev_result["shape"],
        gev_result["loc"],
        gev_result["scale"],
    )

    # --- Heaviest tail domain ---
    valid_tail = tail_df[tail_df["success"]]
    heaviest_domain = (
        valid_tail.iloc[0]["domain"] if not valid_tail.empty else "N/A"
    )
    heaviest_xi = (
        float(valid_tail.iloc[0]["xi"]) if not valid_tail.empty else np.nan
    )

    return {
        "n_total": n_total,
        "n_below_50": n_below_50,
        "mean_score": float(df["final_score"].mean()),
        "gev": gev_result,
        "return_levels": rl,
        "gpd": gpd_result,
        "tail_by_domain": tail_df.to_dict(orient="records"),
        "block_minima": dict(block_min),
        "heaviest_domain": heaviest_domain,
        "heaviest_xi": heaviest_xi,
        "qq": qq,
    }


if __name__ == "__main__":
    import json

    print("Running EvidenceExtremes pipeline...")
    results = run_pipeline()

    print(f"  n_total         = {results['n_total']}")
    print(f"  n_below_50      = {results['n_below_50']}")
    print(f"  GEV xi (shape)  = {results['gev']['shape']:.4f}")
    print(f"  GEV mu (loc)    = {results['gev']['loc']:.4f}")
    print(f"  GEV sigma       = {results['gev']['scale']:.4f}")
    print(f"  Return level m=50  : {results['return_levels'][50]:.2f}")
    print(f"  Return level m=100 : {results['return_levels'][100]:.2f}")
    print(f"  Return level m=500 : {results['return_levels'][500]:.2f}")
    print(f"  Heaviest tail domain: {results['heaviest_domain']} (xi={results['heaviest_xi']:.4f})")

    # Save results
    out_path = "C:/Models/EvidenceExtremes/results.json"
    save_results = {k: v for k, v in results.items() if k != "qq"}
    # Convert block_minima keys to strings for JSON
    save_results["block_minima"] = {str(k): v for k, v in save_results["block_minima"].items()}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

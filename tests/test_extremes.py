"""
25 tests for EvidenceExtremes.

T1-T4:   fit_gev
T5-T8:   fit_gpd
T9-T12:  return_level_gev
T13-T16: tail_index_by_domain
T17-T20: qq_plot_data
T21-T25: pipeline integration
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import pytest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extremes_engine import (
    fit_gev,
    fit_gpd,
    compute_exceedances,
    return_level_gev,
    return_levels_all,
    tail_index_by_domain,
    qq_plot_data,
    run_pipeline,
    load_data,
    get_block_minima,
    THRESHOLD,
)

RNG = np.random.default_rng(42)

# ─── Tolerance ───────────────────────────────────────────────────────────────
RTOL = 0.15   # 15% relative tolerance for EVT parameter estimates
ATOL = 3.0    # absolute tolerance for return levels (score units)


# ═══════════════════════════════════════════════════════════════════════════════
# T1-T4: fit_gev
# ═══════════════════════════════════════════════════════════════════════════════

class TestFitGEV:
    """T1-T4: GEV fitting tests."""

    def test_t1_known_gev_shape_recovered(self):
        """T1: Fit on synthetic GEV minima recovers shape within tolerance."""
        from scipy.stats import genextreme
        # Generate block minima from a known GEV (xi=0.2, mu=50, sigma=5)
        xi_true, mu_true, sigma_true = 0.2, 50.0, 5.0
        # genextreme uses MAXIMA convention; for minima we negate
        # So sample from GEV maxima and negate to get minima
        np.random.seed(0)
        maxima = genextreme.rvs(xi_true, loc=-mu_true, scale=sigma_true, size=200, random_state=0)
        block_minima = -maxima  # these are the "minima" on original scale
        result = fit_gev(block_minima)
        assert result["success"], "fit_gev should succeed on 200 samples"
        # shape should be close to xi_true
        assert abs(result["shape"] - xi_true) < 0.3, (
            f"Shape {result['shape']:.3f} far from true {xi_true}"
        )

    def test_t2_negative_data_handling(self):
        """T2: fit_gev negates internally; loc returned on original scale."""
        # If we pass minima in range [30,60], loc (mu_min) should be in that range
        block_minima = np.array([35.0, 40.0, 42.0, 38.0, 36.0, 45.0, 33.0, 41.0])
        result = fit_gev(block_minima)
        assert result["success"]
        # mu_min should be close to the actual data range
        assert 20.0 < result["loc"] < 70.0, (
            f"loc={result['loc']:.2f} outside expected range for input {block_minima}"
        )
        assert result["scale"] > 0, "scale must be positive"

    def test_t3_small_sample_returns_failure(self):
        """T3: Small sample (n<3) returns success=False."""
        result = fit_gev(np.array([40.0, 35.0]))  # only 2 values
        assert not result["success"], "Should fail with n=2"
        assert result["n"] == 2

    def test_t4_parameter_ranges(self):
        """T4: Shape/scale from typical trust score minima are physically plausible."""
        # Realistic block minima from 14 domains (scores 19-50)
        block_minima = np.array([19, 24, 27, 30, 32, 35, 38, 39, 40, 41, 42, 44, 46, 48], dtype=float)
        result = fit_gev(block_minima)
        assert result["success"]
        assert result["scale"] > 0, "Scale must be positive"
        assert -5.0 < result["shape"] < 5.0, f"Shape {result['shape']} implausible"
        assert np.isfinite(result["loc"]), "loc must be finite"


# ═══════════════════════════════════════════════════════════════════════════════
# T5-T8: fit_gpd
# ═══════════════════════════════════════════════════════════════════════════════

class TestFitGPD:
    """T5-T8: GPD fitting tests."""

    def test_t5_known_pareto_shape_recovered(self):
        """T5: Fit on synthetic Pareto data recovers shape within tolerance."""
        from scipy.stats import genpareto
        xi_true, sigma_true = 0.3, 8.0
        np.random.seed(1)
        exc = genpareto.rvs(xi_true, loc=0, scale=sigma_true, size=300, random_state=1)
        exc = exc[exc > 0]
        result = fit_gpd(exc)
        assert result["success"]
        assert abs(result["shape"] - xi_true) < 0.3, (
            f"GPD shape {result['shape']:.3f} far from true {xi_true}"
        )

    def test_t6_threshold_exceedances_positive(self):
        """T6: compute_exceedances returns positive values for scores below threshold."""
        scores = np.array([30.0, 45.0, 55.0, 60.0, 48.0, 35.0])
        exc = compute_exceedances(scores, threshold=50.0)
        # Below 50: 30, 45, 48, 35 => 4 exceedances
        assert len(exc) == 4, f"Expected 4 exceedances, got {len(exc)}"
        assert all(e > 0 for e in exc), "All exceedances must be positive"
        # Check values: 50-30=20, 50-45=5, 50-48=2, 50-35=15
        expected = sorted([20.0, 5.0, 2.0, 15.0])
        assert np.allclose(sorted(exc), expected), f"Got {sorted(exc)}, expected {expected}"

    def test_t7_small_sample_gpd_fails(self):
        """T7: GPD fit with n<3 returns success=False."""
        result = fit_gpd(np.array([2.0, 5.0]))
        assert not result["success"]
        assert result["n"] == 2

    def test_t8_zero_exceedances_fails(self):
        """T8: All scores above threshold => 0 exceedances => fit_gpd fails."""
        scores = np.array([55.0, 60.0, 70.0, 80.0, 90.0])
        exc = compute_exceedances(scores, threshold=50.0)
        assert len(exc) == 0, "No exceedances expected"
        result = fit_gpd(exc)
        assert not result["success"]
        assert result["n"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# T9-T12: return_level_gev
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnLevelGEV:
    """T9-T12: Return level computation tests."""

    def test_t9_known_params_return_level(self):
        """T9: Known GEV params produce expected return level for m=2 (Gumbel)."""
        # xi=0 (Gumbel minima), mu=40, sigma=5
        # Minima return level: x_m = mu + sigma*log(y) where y = -log(1-1/m)
        # m=2: y = -log(0.5) = 0.6931, log(y) = -0.3665 => x_2 = 40 + 5*(-0.3665) = 38.167
        xi, loc, sigma = 0.0, 40.0, 5.0
        rl = return_level_gev(xi, loc, sigma, m=2)
        y = -np.log(1.0 - 0.5)
        expected = 40.0 + 5.0 * np.log(y)
        assert abs(rl - expected) < 1e-8, f"Got {rl}, expected {expected}"

    def test_t10_gumbel_case_xi_zero(self):
        """T10: Gumbel case (xi=0): x_100 < x_50 (longer period = more extreme minimum)."""
        xi, loc, sigma = 0.0, 50.0, 8.0
        rl100 = return_level_gev(xi, loc, sigma, m=100)
        rl50 = return_level_gev(xi, loc, sigma, m=50)
        assert np.isfinite(rl100), f"Return level should be finite, got {rl100}"
        # For minima: 1-in-100 year minimum should be LOWER than 1-in-50 year minimum
        assert rl100 < rl50, (
            f"1-in-100 minimum ({rl100:.2f}) should be lower than 1-in-50 ({rl50:.2f})"
        )

    def test_t11_return_levels_m100(self):
        """T11: return_levels_all returns dict with m=100 key."""
        xi, loc, sigma = 0.1, 45.0, 6.0
        rl_dict = return_levels_all(xi, loc, sigma, periods=(50, 100, 500))
        assert 100 in rl_dict, "Should have m=100 key"
        assert np.isfinite(rl_dict[100]), "m=100 return level should be finite"

    def test_t12_return_levels_m500(self):
        """T12: m=500 return level is lower (more extreme) than m=100."""
        xi, loc, sigma = 0.1, 45.0, 6.0
        rl_dict = return_levels_all(xi, loc, sigma, periods=(100, 500))
        # For minima: longer return period = more extreme = LOWER score
        assert rl_dict[500] < rl_dict[100], (
            f"1-in-500 min ({rl_dict[500]:.2f}) should be < 1-in-100 min ({rl_dict[100]:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T13-T16: tail_index_by_domain
# ═══════════════════════════════════════════════════════════════════════════════

class TestTailIndex:
    """T13-T16: Tail index comparison tests."""

    def _make_df(self, domain_scores: dict) -> pd.DataFrame:
        rows = []
        for domain, scores in domain_scores.items():
            for s in scores:
                rows.append({"review_id": f"CD{hash(domain+str(s))%9999:04d}",
                              "final_score": s, "review_group": domain})
        return pd.DataFrame(rows)

    def test_t13_multiple_domains_returned(self):
        """T13: tail_index_by_domain returns one row per domain."""
        df = self._make_df({
            "CardioA": [30, 35, 40, 45, 48, 55, 60, 65, 70] * 5,
            "RespB":   [25, 32, 38, 44, 47, 50, 58, 62, 72] * 5,
            "PainC":   [28, 33, 41, 46, 48, 52, 61, 67, 75] * 5,
        })
        result = tail_index_by_domain(df, threshold=50.0)
        assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
        assert set(result["domain"]) == {"CardioA", "RespB", "PainC"}

    def test_t14_single_domain(self):
        """T14: Single-domain input returns one-row DataFrame."""
        scores = list(range(25, 75))  # includes exceedances below 50
        df = self._make_df({"OnlyDomain": scores})
        result = tail_index_by_domain(df, threshold=50.0)
        assert len(result) == 1
        assert result.iloc[0]["domain"] == "OnlyDomain"

    def test_t15_no_exceedances_domain_flagged(self):
        """T15: Domain with all scores above threshold gets success=False."""
        df = self._make_df({
            "Good": [55, 60, 65, 70, 75, 80],   # no exceedances
            "Bad":  [25, 30, 35, 40, 45, 48] * 3,
        })
        result = tail_index_by_domain(df, threshold=50.0)
        good_row = result[result["domain"] == "Good"].iloc[0]
        assert not good_row["success"], "Domain with no exceedances should have success=False"

    def test_t16_ci_width_decreases_with_n(self):
        """T16: Larger sample => narrower xi CI."""
        np.random.seed(42)
        scores_small = list(np.random.uniform(20, 55, 20))
        scores_large = list(np.random.uniform(20, 55, 200))
        df_small = self._make_df({"D": scores_small})
        df_large = self._make_df({"D": scores_large})
        res_small = tail_index_by_domain(df_small, threshold=50.0)
        res_large = tail_index_by_domain(df_large, threshold=50.0)
        small_row = res_small[res_small["domain"] == "D"]
        large_row = res_large[res_large["domain"] == "D"]
        if small_row.iloc[0]["success"] and large_row.iloc[0]["success"]:
            ci_small = small_row.iloc[0]["ci_hi"] - small_row.iloc[0]["ci_lo"]
            ci_large = large_row.iloc[0]["ci_hi"] - large_row.iloc[0]["ci_lo"]
            assert ci_large < ci_small, "Larger n should give narrower CI"


# ═══════════════════════════════════════════════════════════════════════════════
# T17-T20: qq_plot_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestQQPlot:
    """T17-T20: QQ plot data tests."""

    def test_t17_empirical_vs_theoretical_lengths_match(self):
        """T17: Empirical and theoretical arrays have same length."""
        scores = np.arange(30, 80, dtype=float)
        result = qq_plot_data(scores, shape=0.1, loc=50.0, scale=8.0)
        assert len(result["empirical"]) == len(result["theoretical"]), (
            "Empirical and theoretical must have same length"
        )

    def test_t18_empirical_is_sorted(self):
        """T18: Empirical quantiles are sorted ascending."""
        np.random.seed(5)
        scores = np.random.normal(65, 10, 100)
        result = qq_plot_data(scores, shape=0.0, loc=65.0, scale=10.0)
        emp = result["empirical"]
        assert emp == sorted(emp), "Empirical should be sorted"

    def test_t19_theoretical_finite_for_valid_params(self):
        """T19: All theoretical quantiles are finite for valid GEV params."""
        np.random.seed(6)
        scores = np.random.uniform(35, 98, 50)
        result = qq_plot_data(scores, shape=0.15, loc=55.0, scale=7.0)
        assert all(np.isfinite(q) for q in result["theoretical"]), (
            "All theoretical quantiles should be finite"
        )

    def test_t20_generated_gev_qq_linear(self):
        """T20: QQ plot for data generated from same GEV should be ~linear."""
        from scipy.stats import genextreme
        # Generate 100 samples from GEV(0.1, 60, 8) minima
        np.random.seed(7)
        xi, mu, sigma = 0.1, 60.0, 8.0
        # Samples from GEV for MAXIMA with negated params give minima
        maxima = genextreme.rvs(xi, loc=-mu, scale=sigma, size=100, random_state=7)
        scores = -maxima  # original scale
        result = qq_plot_data(scores, shape=xi, loc=mu, scale=sigma)
        emp = np.array(result["empirical"])
        the = np.array(result["theoretical"])
        # Pearson r should be > 0.9 if QQ is roughly linear
        corr = np.corrcoef(emp, the)[0, 1]
        assert corr > 0.85, f"QQ correlation too low: {corr:.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# T21-T25: Pipeline integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    """T21-T25: Full pipeline integration tests."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Run pipeline once for all integration tests."""
        return run_pipeline()

    def test_t21_pipeline_runs_on_real_data(self, pipeline_result):
        """T21: Pipeline runs successfully on real data."""
        r = pipeline_result
        assert r["n_total"] == 6229, f"Expected 6229 MAs, got {r['n_total']}"
        assert r["n_below_50"] == 271, f"Expected 271 below threshold, got {r['n_below_50']}"

    def test_t22_gev_output_valid(self, pipeline_result):
        """T22: GEV result has valid shape, loc, scale."""
        gev = pipeline_result["gev"]
        assert gev["success"], "GEV fit should succeed on real data"
        assert np.isfinite(gev["shape"]), "GEV shape must be finite"
        assert np.isfinite(gev["loc"]), "GEV loc must be finite"
        assert gev["scale"] > 0, "GEV scale must be positive"

    def test_t23_all_return_levels_finite(self, pipeline_result):
        """T23: All return levels (m=50,100,500) are finite."""
        rl = pipeline_result["return_levels"]
        for m, val in rl.items():
            assert np.isfinite(val), f"Return level m={m} is not finite: {val}"

    def test_t24_tail_domain_summary_consistent(self, pipeline_result):
        """T24: Tail domain results are internally consistent."""
        tail = pipeline_result["tail_by_domain"]
        assert len(tail) > 0, "Should have at least one domain"
        # heaviest domain is the first (sorted desc by xi)
        valid = [r for r in tail if r["success"]]
        if valid:
            top = valid[0]
            assert pipeline_result["heaviest_domain"] == top["domain"]
            assert abs(pipeline_result["heaviest_xi"] - top["xi"]) < 1e-9

    def test_t25_return_level_ordering(self, pipeline_result):
        """T25: 1-in-500 minimum <= 1-in-100 minimum <= 1-in-50 minimum."""
        rl = pipeline_result["return_levels"]
        # For minima: longer return period means MORE extreme (lower) score
        assert rl[500] <= rl[100] <= rl[50], (
            f"Return level ordering violated: rl50={rl[50]:.2f}, "
            f"rl100={rl[100]:.2f}, rl500={rl[500]:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

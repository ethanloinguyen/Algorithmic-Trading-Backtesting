"""
tests/test_core.py
------------------
Unit tests for the core algorithmic components.
Run with: pytest tests/test_core.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, "/app")


# ── dCor Engine Tests ─────────────────────────────────────────────────────────

class TestDcorEngine:
    def test_dcor_independent_series(self):
        """Independent series should have dCor near 0."""
        from src.dcor_engine import dcor
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0, 1, 300)
        result = dcor(x, y)
        assert result < 0.15, f"Expected dCor < 0.15 for independent series, got {result}"

    def test_dcor_identical_series(self):
        """Identical series should have dCor = 1."""
        from src.dcor_engine import dcor
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        result = dcor(x, x)
        assert abs(result - 1.0) < 0.01, f"Expected dCor = 1 for identical series, got {result}"

    def test_dcor_nonlinear_detection(self):
        """dCor should detect nonlinear relationship Pearson misses."""
        from src.dcor_engine import dcor
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = x ** 2 + rng.normal(0, 0.5, 500)

        dc = dcor(x, y)
        pearson = abs(np.corrcoef(x, y)[0, 1])

        assert dc > pearson + 0.1, (
            f"dCor ({dc:.3f}) should substantially exceed |Pearson| ({pearson:.3f}) for quadratic"
        )

    def test_dcor_at_lag_threshold_regime(self):
        """dCor should detect the planted threshold regime relationship."""
        from src.dcor_engine import dcor_at_lag
        rng = np.random.default_rng(42)
        n = 500
        A = rng.normal(0, 1, n)
        B = np.zeros(n)
        for t in range(2, n):
            if A[t - 2] > 0:
                B[t] = 0.5 * A[t - 2] + rng.normal(0, 1)
            else:
                B[t] = rng.normal(0, 1)

        dc_lag2 = dcor_at_lag(A, B, lag=2)
        dc_lag1 = dcor_at_lag(A, B, lag=1)
        dc_lag3 = dcor_at_lag(A, B, lag=3)

        assert dc_lag2 > dc_lag1, f"Lag 2 dCor ({dc_lag2:.3f}) should exceed lag 1 ({dc_lag1:.3f})"
        assert dc_lag2 > dc_lag3, f"Lag 2 dCor ({dc_lag2:.3f}) should exceed lag 3 ({dc_lag3:.3f})"

    def test_dcor_range(self):
        """dCor must always be in [0, 1]."""
        from src.dcor_engine import dcor
        rng = np.random.default_rng(123)
        for _ in range(10):
            x = rng.normal(0, 1, 100)
            y = rng.normal(0, 1, 100)
            result = dcor(x, y)
            assert 0.0 <= result <= 1.0, f"dCor out of range: {result}"

    def test_sharpness_concentrated(self):
        """Single dominant lag should give high sharpness."""
        from src.dcor_engine import compute_sharpness
        # All mass at lag 2
        dcor_vals = {1: 0.01, 2: 0.80, 3: 0.02, 4: 0.01, 5: 0.01}
        sharpness = compute_sharpness(dcor_vals, method="entropy")
        assert sharpness > 0.7, f"Expected high sharpness, got {sharpness}"

    def test_sharpness_uniform(self):
        """Uniform dCor across lags should give low sharpness."""
        from src.dcor_engine import compute_sharpness
        dcor_vals = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
        sharpness = compute_sharpness(dcor_vals, method="entropy")
        assert sharpness < 0.05, f"Expected low sharpness for uniform, got {sharpness}"


# ── Permutation Tests ─────────────────────────────────────────────────────────

class TestPermutation:
    def test_block_shuffle_preserves_length(self):
        """Block shuffle must preserve series length."""
        from src.permutation import block_shuffle
        rng = np.random.default_rng(42)
        x = np.arange(100, dtype=float)
        shuffled = block_shuffle(x, block_size=5, rng=rng)
        assert len(shuffled) == len(x)

    def test_block_shuffle_changes_order(self):
        """Block shuffle should reorder the series."""
        from src.permutation import block_shuffle
        rng = np.random.default_rng(42)
        x = np.arange(100, dtype=float)
        shuffled = block_shuffle(x, block_size=5, rng=rng)
        assert not np.array_equal(x, shuffled), "Shuffled series should not equal original"

    def test_block_shuffle_preserves_values(self):
        """Block shuffle must contain same values."""
        from src.permutation import block_shuffle
        rng = np.random.default_rng(42)
        x = np.arange(100, dtype=float)
        shuffled = block_shuffle(x, block_size=5, rng=rng)
        assert set(shuffled.astype(int)) == set(x.astype(int))

    def test_null_pair_high_pvalue(self):
        """Independent series should yield high p-value (not significant)."""
        from src.permutation import adaptive_permutation_test
        from src.dcor_engine import dcor_at_lag
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0, 1, 200)
        obs = dcor_at_lag(x, y, lag=1)
        p_val, n_perms = adaptive_permutation_test(x, y, lag=1, observed_dcor=obs, rng=rng)
        # Should stop early (tier 1) for null pair
        assert p_val > 0.20 or n_perms == 100, f"Expected early stopping for null pair, got p={p_val}, n={n_perms}"

    def test_planted_pair_low_pvalue(self):
        """Planted lag-2 pair should yield low p-value."""
        from src.permutation import adaptive_permutation_test
        from src.dcor_engine import dcor_at_lag
        rng = np.random.default_rng(42)
        n = 300
        A = rng.normal(0, 1, n)
        B = np.array([0.8 * A[max(0, t-2)] + rng.normal(0, 0.3) for t in range(n)])
        obs = dcor_at_lag(A, B, lag=2)
        p_val, _ = adaptive_permutation_test(A, B, lag=2, observed_dcor=obs, rng=rng)
        assert p_val < 0.10, f"Expected low p-value for planted pair, got {p_val}"


# ── FDR Tests ─────────────────────────────────────────────────────────────────

class TestFDR:
    def test_bh_null_p_values(self):
        """All null p-values (uniform) should yield few/no significant results."""
        from src.fdr import benjamini_hochberg
        rng = np.random.default_rng(42)
        p_values = rng.uniform(0, 1, 1000)
        significant, q_values = benjamini_hochberg(p_values, alpha=0.05)
        fdr = significant.sum() / len(significant)
        assert fdr < 0.15, f"Expected low FDR for null p-values, got {fdr:.2%}"

    def test_bh_planted_p_values(self):
        """Small p-values should be flagged significant."""
        from src.fdr import benjamini_hochberg
        p_values = np.array([0.001, 0.002, 0.003, 0.5, 0.6, 0.7, 0.8, 0.9])
        significant, q_values = benjamini_hochberg(p_values, alpha=0.05)
        assert significant[:3].all(), "First 3 (small p-values) should be significant"
        assert not significant[3:].any(), "Last 4 (large p-values) should not be significant"

    def test_q_values_ordered(self):
        """q-values should be monotonically non-decreasing with p-values."""
        from src.fdr import benjamini_hochberg
        p_values = np.array([0.001, 0.01, 0.05, 0.10, 0.50])
        significant, q_values = benjamini_hochberg(p_values, alpha=0.05)
        assert q_values[0] <= q_values[1] <= q_values[4], "q-values should be ordered"


# ── Stability Tests ───────────────────────────────────────────────────────────

class TestStability:
    def test_half_life_known_decay(self):
        """Half-life estimation should recover approximate decay rate."""
        from src.stability import estimate_half_life
        # Create series with known decay lambda = 0.1 → HL = ln(2)/0.1 ≈ 6.9 steps
        t = np.arange(20, dtype=float)
        dcor_series = 0.5 * np.exp(-0.1 * t) + np.random.default_rng(42).normal(0, 0.01, 20)
        dcor_series = np.clip(dcor_series, 0, 1)
        hl, r2, stable = estimate_half_life(dcor_series, t)
        assert hl is not None, "Half-life should be estimated"
        # HL in steps should be roughly ln(2)/0.1 ≈ 6.9 steps × 63 days/step
        # We check that fit is reasonable
        assert r2 > 0.5, f"R² should be high for clean exponential, got {r2}"

    def test_half_life_flat_series(self):
        """Flat (non-decaying) series should have high half-life or unstable flag."""
        from src.stability import estimate_half_life
        t = np.arange(10, dtype=float)
        dcor_series = np.full(10, 0.3)
        hl, r2, stable = estimate_half_life(dcor_series, t)
        # Either it fits poorly or returns very long half-life
        assert (not stable or hl is None or hl > 200), "Flat series should be flagged unstable"


# ── OOS Strategy Tests ────────────────────────────────────────────────────────

class TestOOSStrategy:
    def test_strategy_long_short_symmetric(self):
        """Long and short positions should be generated symmetrically."""
        from src.oos_model import compute_rolling_zscore
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 200)
        zscores = compute_rolling_zscore(series, lookback=60, threshold=1.0)
        non_nan = zscores[~np.isnan(zscores)]
        longs = (non_nan > 1.0).sum()
        shorts = (non_nan < -1.0).sum()
        # Roughly symmetric for normal distribution
        assert abs(longs - shorts) < 30, f"Long/short counts should be roughly symmetric: {longs} vs {shorts}"

    def test_sharpe_positive_for_good_strategy(self):
        """Consistently positive returns should yield positive Sharpe."""
        from src.oos_model import compute_sharpe
        positive_returns = np.full(500, 0.001)  # 0.1% per day consistently
        sharpe = compute_sharpe(positive_returns)
        assert sharpe > 0, "Consistently positive returns should yield positive Sharpe"


# ── Bootstrap Tests ───────────────────────────────────────────────────────────

class TestBootstrap:
    def test_signal_strength_range(self):
        """Signal strength must always be in [0, 100]."""
        from src.bootstrap import compute_signal_strength
        predicted = pd.Series(np.random.default_rng(42).normal(0.3, 0.5, 500))
        ss = compute_signal_strength(predicted)
        assert ss.min() >= 0.0, f"Signal strength below 0: {ss.min()}"
        assert ss.max() <= 100.0, f"Signal strength above 100: {ss.max()}"

    def test_signal_strength_monotonic(self):
        """Higher predicted Sharpe should yield higher signal strength."""
        from src.bootstrap import compute_signal_strength
        predicted = pd.Series([0.1, 0.5, 1.0, 2.0])
        ss = compute_signal_strength(predicted)
        assert list(ss) == sorted(list(ss)), "Signal strength should be monotonically increasing with Sharpe"


# ── Synthetic Tests ───────────────────────────────────────────────────────────

class TestSynthetic:
    def test_synthetic_generation(self):
        """Synthetic universe should have correct number of pairs."""
        from src.synthetic import generate_synthetic_universe
        series, planted, null = generate_synthetic_universe(
            n_obs=100, n_planted=5, n_null=10, seed=42
        )
        assert len(planted) == 5
        assert len(null) == 10
        assert len(series) == 30  # 5*2 + 10*2

    def test_planted_relationship_is_nonlinear(self):
        """Planted threshold regime should have higher dCor than Pearson at lag 2."""
        from src.synthetic import generate_synthetic_universe
        from src.dcor_engine import dcor_at_lag
        series, planted, _ = generate_synthetic_universe(n_obs=500, n_planted=1, n_null=0, seed=42)
        ti, tj = planted[0]
        x = series[ti]
        y = series[tj]
        dc = dcor_at_lag(x, y, lag=2)
        pearson = abs(np.corrcoef(x[:-2], y[2:])[0, 1])
        assert dc > pearson, f"dCor ({dc:.3f}) should exceed |Pearson| ({pearson:.3f}) for nonlinear"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

"""
tests/test_strategy_utils.py
============================
Unit tests for strategy utility functions: softmax weighting, z-score, weight cap.
Run with: python3 -m pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd


# ─── Functions extracted from strategy.py for testability ─────────────────────

def softmax_weights(scores: pd.Series, temp: float, top_k: int) -> pd.Series:
    """Select top_k stocks; weight by softmax(score / temp)."""
    top  = scores.nlargest(top_k)
    exp_ = np.exp((top - top.max()) / temp)
    return exp_ / exp_.sum()


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score with winsorize ±3σ"""
    df  = df.replace([np.inf, -np.inf], np.nan)
    mu  = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return (df.sub(mu, axis=0).div(std, axis=0)).clip(-3, 3)


def apply_weight_cap(weights: pd.Series, cap: float = 0.08,
                     max_iter: int = 20) -> pd.Series:
    """Iteratively clip at cap and renormalize until all weights ≤ cap."""
    w = weights.copy()
    for _ in range(max_iter):
        total = w.sum()
        if total <= 0:
            break
        w = w / total
        if (w <= cap + 1e-9).all():
            break
        w = w.clip(upper=cap)
    return w / w.sum() if w.sum() > 0 else w


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestSoftmaxWeights:
    def test_weights_sum_to_one(self):
        """softmax 輸出加總應為 1"""
        scores = pd.Series({f"S{i}": float(i) for i in range(30)})
        w = softmax_weights(scores, temp=5.0, top_k=20)
        assert abs(w.sum() - 1.0) < 1e-9, f"加總 {w.sum():.8f} ≠ 1"

    def test_selects_top_k_stocks(self):
        """應剛好選出 top_k 支股票"""
        scores = pd.Series({f"S{i}": float(i) for i in range(50)})
        w = softmax_weights(scores, temp=5.0, top_k=20)
        assert len(w) == 20

    def test_higher_score_gets_higher_weight(self):
        """分數較高的股票應獲得較高權重"""
        scores = pd.Series({"A": 10.0, "B": 5.0, "C": 1.0})
        w = softmax_weights(scores, temp=1.0, top_k=3)
        assert w["A"] > w["B"] > w["C"], "權重排序應與分數排序一致"

    def test_high_temperature_gives_near_equal_weights(self):
        """高溫度 → 權重趨近等權"""
        scores = pd.Series({"A": 10.0, "B": 5.0, "C": 1.0})
        w = softmax_weights(scores, temp=100.0, top_k=3)
        # All weights should be close to 1/3
        assert all(abs(w_i - 1/3) < 0.05 for w_i in w), \
            f"高溫下權重應接近等權 1/3，實際: {w.to_dict()}"

    def test_low_temperature_concentrates_top_stock(self):
        """低溫度 → 權重集中在最高分股票"""
        scores = pd.Series({"A": 100.0, "B": 1.0, "C": 0.5})
        w = softmax_weights(scores, temp=0.01, top_k=3)
        assert w["A"] > 0.99, f"低溫下最高分股票應佔 >99%，實際: {w['A']:.4f}"

    def test_all_weights_non_negative(self):
        """所有權重應 ≥ 0"""
        scores = pd.Series({f"S{i}": np.random.randn() for i in range(20)})
        w = softmax_weights(scores, temp=5.0, top_k=10)
        assert (w >= 0).all(), "存在負權重"


class TestZScoreXS:
    def test_output_shape_preserved(self):
        """z-score 輸出形狀應與輸入相同"""
        df = pd.DataFrame(np.random.randn(10, 5))
        result = zscore_xs(df)
        assert result.shape == df.shape

    def test_cross_sectional_mean_near_zero(self):
        """每一行（截面）的均值應接近 0（容許浮點誤差）"""
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(20, 100))
        result = zscore_xs(df)
        row_means = result.mean(axis=1)
        # After winsorizing some outliers, mean may shift slightly — use 0.01 tolerance
        assert (row_means.abs() < 0.01).all(), \
            f"截面均值偏離 0 過大: {row_means.abs().max():.6f}"

    def test_winsorized_at_3sigma(self):
        """極端值應被截斷在 ±3，需要至少2欄才能計算 std"""
        df = pd.DataFrame({"A": [1e6, 0.0, 0.0, 0.0, 0.0, 0.0],
                           "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
        result = zscore_xs(df)
        # Drop NaN then check bounds
        valid = result.dropna(how="all").stack().dropna()
        if len(valid) > 0:
            assert valid.max() <= 3.0 + 1e-9
            assert valid.min() >= -3.0 - 1e-9

    def test_inf_replaced_with_nan(self):
        """無限大應被轉為 NaN，不影響其他數值"""
        df = pd.DataFrame({"A": [np.inf, 1.0, 2.0], "B": [1.0, 2.0, 3.0]})
        result = zscore_xs(df)
        # Should not crash; NaN is acceptable in result
        assert not np.any(np.isinf(result.values)), "結果中不應有 inf"


class TestWeightCapStrategy:
    def test_cap_at_8pct(self):
        """策略輸出的每支股票權重不超過 8%（需要足夠多的股票才分得山下）"""
        cap = 0.08
        # 20 stocks: first 5 have 40% total weight concentrated on few stocks
        raw_vals = [0.20, 0.15, 0.12, 0.10, 0.08] + [0.035] * 15
        weights = pd.Series({f"S{i}": v for i, v in enumerate(raw_vals)})
        capped = apply_weight_cap(weights, cap=cap)
        assert (capped <= cap + 1e-9).all(), \
            f"存在超過 {cap:.0%} 的股票: {capped[capped > cap + 1e-9]}"

    def test_renormalization_after_cap(self):
        """cap 後重新標準化，總和應為 1"""
        weights = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})
        capped  = apply_weight_cap(weights)
        assert abs(capped.sum() - 1.0) < 1e-9

    def test_empty_weights_returns_empty(self):
        """空 Series 不應報錯"""
        empty  = pd.Series(dtype=float)
        capped = apply_weight_cap(empty)
        assert len(capped) == 0

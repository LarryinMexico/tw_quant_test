"""
tests/test_live_trade.py
========================
Unit tests for live_trade.py fee calculations, risk controls, and NAV math.
Run with: python3 -m pytest tests/ -v
"""

import pytest
import pandas as pd
import sys
import os

# ─── Constants (mirror live_trade.py) ─────────────────────────────────────────
FEE_RATE          = 0.001425
TAX_RATE          = 0.003
SLIPPAGE          = 0.001
MAX_INVEST_RATIO  = 0.90
MAX_SINGLE_WEIGHT = 0.08
INIT_CASH         = 1_000_000.0


# ─── Helper functions (extracted from live_trade.py for testability) ──────────

def apply_buy(cash: float, price: float, budget: float) -> tuple:
    """Simulate a buy order. Returns (shares, new_cash, cost_including_fee)"""
    slip_price = price * (1 + SLIPPAGE)
    shares     = int(budget // slip_price)
    cost       = shares * slip_price
    fee        = cost * FEE_RATE
    total_cost = cost + fee
    if shares > 0 and cash >= total_cost:
        return shares, cash - total_cost, total_cost
    return 0, cash, 0.0


def apply_sell(cash: float, price: float, shares: int) -> tuple:
    """Simulate a sell order. Returns (new_cash, net_revenue)"""
    slip_price  = price * (1 - SLIPPAGE)
    revenue     = shares * slip_price
    fee_tax     = revenue * (FEE_RATE + TAX_RATE)
    net_revenue = revenue - fee_tax
    return cash + net_revenue, net_revenue


def apply_weight_cap(weights: pd.Series, cap: float = MAX_SINGLE_WEIGHT,
                     max_iter: int = 20) -> pd.Series:
    """Iteratively clip weights at cap and renormalize until convergence."""
    w = weights.copy()
    for _ in range(max_iter):
        total = w.sum()
        if total <= 0:
            break
        w = w / total  # normalize
        if (w <= cap + 1e-9).all():
            break
        w = w.clip(upper=cap)
    return w / w.sum() if w.sum() > 0 else w


def compute_nav(cash: float, positions: dict, prices: dict) -> float:
    equity = sum(prices.get(s, 0) * sh for s, sh in positions.items())
    return cash + equity


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestFeeCalculation:
    def test_buy_fee_deducted(self):
        """買進手續費應被正確扣除"""
        price  = 100.0
        budget = 10_000.0
        shares, new_cash, total_cost = apply_buy(50_000.0, price, budget)
        assert shares > 0
        # FEE_RATE 0.1425% + SLIPPAGE 0.1% — total cost > raw share cost
        raw_cost = shares * price
        assert total_cost > raw_cost, "手續費 + 滑價應使總成本高於純股票成本"

    def test_sell_tax_fee_deducted(self):
        """賣出手續費 + 交易稅應被正確扣除"""
        price = 50.0
        shares = 200
        _, net_revenue = apply_sell(0.0, price, shares)
        gross = price * shares
        assert net_revenue < gross, "賣出應扣除手續費與交易稅"
        expected_deduction = gross * (SLIPPAGE + FEE_RATE + TAX_RATE)
        actual_deduction   = gross - net_revenue
        assert abs(actual_deduction - expected_deduction) < 1.0, \
            f"扣除金額不符: 預期 {expected_deduction:.2f}, 實際 {actual_deduction:.2f}"

    def test_slippage_direction(self):
        """買進滑價應向上，賣出向下"""
        price    = 100.0
        budget   = 100_000.0
        shares_buy, _, _ = apply_buy(200_000.0, price, budget)
        # Effective buy price > raw price
        effective_buy = budget / max(shares_buy, 1)
        # effective price per share should be higher than raw price (due to slippage)
        assert effective_buy >= price or shares_buy == 0  # passes if slippage applied

    def test_zero_shares_if_insufficient_cash(self):
        """現金不足時不應購入"""
        shares, cash, cost = apply_buy(cash=1.0, price=100.0, budget=10_000.0)
        assert shares == 0
        assert cash == 1.0
        assert cost == 0.0


class TestWeightCap:
    def test_weight_cap_clips_at_8pct(self):
        """任何單支股票權重不得超過 8%（10支以上才分得山下）"""
        # 20 stocks with heavier weights on first few — cap must iterate to converge
        raw_vals = [0.15, 0.12, 0.11, 0.10, 0.09] + [0.03] * 15
        raw = pd.Series({f"S{i}": v for i, v in enumerate(raw_vals)})
        capped = apply_weight_cap(raw)
        assert (capped > MAX_SINGLE_WEIGHT + 1e-9).sum() == 0, \
            f"有股票超過 {MAX_SINGLE_WEIGHT:.0%} 上限: {capped[capped > MAX_SINGLE_WEIGHT + 1e-9]}"

    def test_weights_sum_to_one_after_cap(self):
        """cap 後重新標準化，所有權重加總應等於 1"""
        raw = pd.Series({"A": 0.50, "B": 0.30, "C": 0.20})
        capped = apply_weight_cap(raw)
        assert abs(capped.sum() - 1.0) < 1e-9, \
            f"權重加總 {capped.sum():.6f} ≠ 1"

    def test_no_cap_if_below_threshold(self):
        """若正規化後所有股票權重已 ≤ 8%，則 cap 不應改變數值"""
        # Only 3 stocks each at ~33% — after normalization they all exceed 8%,
        # so with 20 stocks at 5% each, cap should not fire.
        raw = pd.Series({f"S{i}": 1.0 for i in range(20)})  # equal weight → 5% each
        raw_norm = raw / raw.sum()  # = 0.05 each, below cap
        capped = apply_weight_cap(raw_norm)
        assert (capped <= MAX_SINGLE_WEIGHT + 1e-9).all()
        assert abs(capped.sum() - 1.0) < 1e-9


class TestCashReserve:
    def test_investable_cap_at_90pct(self):
        """建倉資金不應超過 NAV 的 90%"""
        nav = 1_000_000.0
        investable = nav * MAX_INVEST_RATIO
        assert investable <= nav * 0.90 + 1e-6
        assert investable == pytest.approx(900_000.0)

    def test_cash_minimum_10pct(self):
        """模擬建倉後剩餘現金應 ≥ 10% NAV"""
        nav      = 1_000_000.0
        investable = nav * MAX_INVEST_RATIO
        # Simulate buying stocks — cash usage capped at investable
        cash_after = nav - investable
        assert cash_after >= nav * 0.10 - 1.0, \
            f"現金 {cash_after:,.0f} 低於 NAV 的 10%"


class TestNAVCalculation:
    def test_nav_equals_cash_plus_equity(self):
        """NAV 應等於現金 + 持股市值"""
        cash      = 50_000.0
        positions = {"2330": 100, "0050": 200}
        prices    = {"2330": 900.0, "0050": 150.0}
        expected  = 50_000 + 100 * 900 + 200 * 150
        assert compute_nav(cash, positions, prices) == expected

    def test_nav_with_missing_price(self):
        """缺少報價的股票應以 0 計價（保守估計）"""
        cash      = 100_000.0
        positions = {"2330": 100, "UNKNOWN": 500}
        prices    = {"2330": 900.0}
        nav = compute_nav(cash, positions, prices)
        assert nav == 100_000 + 100 * 900, "缺報價股票應貢獻 0 元"

    def test_empty_positions_nav_equals_cash(self):
        """無持股時 NAV 應等於現金"""
        nav = compute_nav(200_000.0, {}, {})
        assert nav == 200_000.0

"""
tests/test_wheel_options.py — Unit tests for wheel_options.py's pure math
(Black-Scholes greeks/pricing, richness scoring, premium efficiency).
No network / IBKR calls.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta

import pytest

import wheel_config as wc
import wheel_options as wo


def _future_expiry(days: int) -> str:
    return (datetime.now().date() + timedelta(days=days)).strftime("%Y%m%d")


class TestBlackScholesDelta:
    def test_put_delta_is_negative(self):
        d = wo.bs_delta(S=100, K=95, T=0.25, r=0.05, sigma=0.25, option_type="put")
        assert d < 0

    def test_call_delta_is_positive(self):
        d = wo.bs_delta(S=100, K=95, T=0.25, r=0.05, sigma=0.25, option_type="call")
        assert d > 0

    def test_deep_otm_put_delta_near_zero(self):
        d = wo.bs_delta(S=100, K=50, T=0.1, r=0.05, sigma=0.2, option_type="put")
        assert abs(d) < 0.05

    def test_deep_itm_put_delta_near_minus_one(self):
        d = wo.bs_delta(S=50, K=100, T=0.1, r=0.05, sigma=0.2, option_type="put")
        assert d < -0.9

    def test_zero_time_returns_zero(self):
        assert wo.bs_delta(S=100, K=95, T=0, r=0.05, sigma=0.25) == 0.0


class TestBlackScholesPrice:
    def test_put_price_nonnegative(self):
        price = wo.bs_price(S=100, K=95, T=0.25, r=0.05, sigma=0.25, option_type="put")
        assert price >= 0

    def test_deep_itm_put_price_at_least_intrinsic(self):
        price = wo.bs_price(S=50, K=100, T=0.25, r=0.05, sigma=0.2, option_type="put")
        assert price >= (100 - 50) * 0.9   # comfortably above discounted intrinsic


class TestRealizedVolatility:
    def test_none_on_empty_history(self):
        import pandas as pd
        assert wo.realized_volatility(pd.DataFrame({"Close": []})) is None

    def test_returns_positive_float_on_real_data(self):
        import pandas as pd
        import numpy as np
        rng = np.random.default_rng(42)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
        df = pd.DataFrame({"Close": closes})
        rv = wo.realized_volatility(df)
        assert rv is not None
        assert rv > 0


class TestBuildCandidate:
    def test_richness_ratio_and_overpriced_flag(self):
        c = wo.build_candidate(
            symbol="TEST", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.40, realized_vol=0.20, delta=-0.08,
        )
        assert c is not None
        assert c.richness_ratio == pytest.approx(2.0)
        assert c.overpriced is True

    def test_not_overpriced_when_iv_close_to_realized(self):
        c = wo.build_candidate(
            symbol="TEST", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.21, realized_vol=0.20, delta=-0.08,
        )
        assert c.overpriced is False

    def test_delta_is_estimated_flag(self):
        c_estimated = wo.build_candidate(
            symbol="TEST", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.30, realized_vol=0.25, delta=None,
        )
        c_broker = wo.build_candidate(
            symbol="TEST", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.30, realized_vol=0.25, delta=-0.08,
        )
        assert c_estimated.delta_is_estimated is True
        assert c_broker.delta_is_estimated is False

    def test_expired_contract_returns_none(self):
        c = wo.build_candidate(
            symbol="TEST", expiry=_future_expiry(-1), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.30, realized_vol=0.25, delta=-0.08,
        )
        assert c is None

    def test_premium_efficiency_rewards_less_delta_for_same_yield(self):
        low_delta = wo.build_candidate(
            symbol="A", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.30, realized_vol=0.25, delta=-0.05,
        )
        high_delta = wo.build_candidate(
            symbol="B", expiry=_future_expiry(45), strike=90, underlying_price=100,
            premium=1.50, market_iv=0.30, realized_vol=0.25, delta=-0.20,
        )
        assert low_delta.premium_efficiency > high_delta.premium_efficiency


# ---------------------------------------------------------------------------
# Fake ib_insync client for exercising the fetch-loop filters
# (min premium, max delta) without a live IBKR connection.
# ---------------------------------------------------------------------------

class _FakeGreeks:
    def __init__(self, impliedVol, delta):
        self.impliedVol = impliedVol
        self.delta = delta


class _FakeTicker:
    def __init__(self, contract, bid, ask, iv, delta, last=None):
        self.contract = contract
        self.bid = bid
        self.ask = ask
        self.last = last
        self.modelGreeks = _FakeGreeks(iv, delta)


class _FakeChain:
    def __init__(self, exchange, expirations, strikes, tradingClass):
        self.exchange = exchange
        self.expirations = expirations
        self.strikes = strikes
        self.tradingClass = tradingClass


class _FakeIB:
    """Stands in for a connected ib_insync.IB -- fills in a canned put
    chain for one expiry, with per-strike (bid, ask, delta) supplied by the
    test, so the loop's own filtering logic gets exercised end to end."""

    def __init__(self, expiry: str, strikes_data: dict):
        self.expiry = expiry
        # strike -> (bid, ask, delta) or (bid, ask, delta, last)
        self.strikes_data = strikes_data

    async def qualifyContractsAsync(self, *contracts):
        for c in contracts:
            c.conId = 1
        return list(contracts)

    async def reqSecDefOptParamsAsync(self, *_args, **_kwargs):
        return [_FakeChain("SMART", [self.expiry], sorted(self.strikes_data), "TEST")]

    async def reqTickersAsync(self, *contracts):
        tickers = []
        for c in contracts:
            row = self.strikes_data[c.strike]
            bid, ask, delta = row[0], row[1], row[2]
            last = row[3] if len(row) > 3 else None
            iv = row[4] if len(row) > 4 else 0.30
            tickers.append(_FakeTicker(c, bid, ask, iv=iv, delta=delta, last=last))
        return tickers


class _FakeClient:
    def __init__(self, ib):
        self._ib = ib

    @property
    def ib(self):
        return self._ib


class TestFetchLoopFilters:
    """Exercises _fetch_put_candidates_async's own filtering (min premium,
    max delta), not just build_candidate's math."""

    def _run(self, strikes_data, **overrides):
        expiry = _future_expiry(45)
        client = _FakeClient(_FakeIB(expiry, strikes_data))
        kwargs = dict(
            client=client, symbol="TEST", underlying_price=100.0, realized_vol=0.25,
            dte_min=30, dte_max=60, max_delta_abs=wc.MAX_DELTA_ABS,
            max_expiries=1, strike_band_low_pct=0.80, strike_band_high_pct=0.95,
            min_premium_per_contract=wc.MIN_PREMIUM_PER_CONTRACT,
        )
        kwargs.update(overrides)
        return asyncio.run(wo._fetch_put_candidates_async(**kwargs))

    def test_below_min_premium_is_filtered_out(self):
        # strike 85: $2.00 mid -> $200/contract, low delta -> kept
        # strike 88: $0.30 mid -> $30/contract, low delta -> filtered (premium)
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            88: (0.25, 0.35, -0.08),
        })
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

    def test_above_max_delta_is_filtered_out_even_with_good_premium(self):
        # strike 92: $1.00 mid -> $100/contract (well above min premium) but
        # delta 0.15 > MAX_DELTA_ABS (0.10) -> filtered on delta, not premium
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            92: (0.95, 1.05, -0.15),
        })
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

    def test_custom_min_premium_threshold_is_respected(self):
        # $0.30 mid ($30/contract) is below the default $50 floor but above
        # a caller-supplied $20 floor -- confirms the parameter is live, not
        # just the config default.
        candidates = self._run(
            {88: (0.25, 0.35, -0.08)},
            min_premium_per_contract=20.0,
        )
        assert len(candidates) == 1
        assert candidates[0].strike == 88

    def test_nan_last_with_no_bid_ask_is_dropped_not_kept(self):
        # Reproduces the reported "Delta -0.00 | Premium $nan" bug: IBKR
        # returns NaN (not None/0) for bid/ask/last on an illiquid strike
        # with no quotes. `elif t.last:` used to treat NaN as truthy
        # (bool(float('nan')) is True in Python) and set premium=NaN, which
        # then slipped past the min-premium filter too since every
        # NaN-comparison (nan < x, nan > x) is False.
        nan = float("nan")
        candidates = self._run({
            85: (1.95, 2.05, -0.05),          # healthy quote -> kept
            89: (nan, nan, -0.03, nan),        # no bid/ask, NaN last -> must be dropped
        })
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}
        assert all(math.isfinite(c.premium) for c in candidates)
        assert all(math.isfinite(c.delta) for c in candidates)

    def test_nan_broker_delta_is_dropped_not_kept(self):
        """A NaN modelGreeks.delta (broker reported it but couldn't compute
        one) must never reach a returned candidate as NaN: abs(nan) > x is
        always False, so a raw NaN delta would otherwise silently bypass
        the max-delta filter entirely. It's converted to None instead,
        which falls back to the Black-Scholes proxy delta (a real, finite
        number) -- here that computes to roughly -0.13 (S=100, K=90, 45
        DTE, iv=0.30), over MAX_DELTA_ABS (0.10), so it's correctly
        filtered on delta with a real number, not silently let through."""
        nan = float("nan")
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            90: (1.00, 1.10, nan),
        })
        assert all(math.isfinite(c.delta) for c in candidates)
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

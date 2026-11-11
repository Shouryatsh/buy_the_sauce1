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
    def __init__(self, contract, bid, ask, iv, delta, last=None, market_data_type=1):
        self.contract = contract
        self.bid = bid
        self.ask = ask
        self.last = last
        self.modelGreeks = _FakeGreeks(iv, delta)
        self.marketDataType = market_data_type


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

    def __init__(self, expiry: str, strikes_data: dict, quotes_from_type: int = 2, expirations=None):
        self.expiry = expiry
        self.expirations = expirations or [expiry]
        self.expiries_quoted: set[str] = set()
        # strike -> (bid, ask, delta) or (bid, ask, delta, last)
        self.strikes_data = strikes_data
        # Market data types requested, in order. Quotes are only served once a
        # type >= quotes_from_type is requested (4 = account has no live
        # options entitlement, so only delayed data comes back); before that
        # IBKR's "no quote" -1 is returned, as on the live account after hours.
        self.quotes_from_type = quotes_from_type
        self.types_requested: list[int] = []

    def reqMarketDataType(self, t):
        self.types_requested.append(t)

    # Streaming open interest (generic tick 101). Row index 5 sets a strike's
    # put OI (default 500); None simulates IBKR never sending the tick.
    def reqMktData(self, contract, generic_ticks="", snapshot=False, regulatory=False):
        row = self.strikes_data[contract.strike]
        oi = row[5] if len(row) > 5 else 500
        return type("OITicker", (), {"putOpenInterest": oi})()

    def cancelMktData(self, contract):
        pass

    async def qualifyContractsAsync(self, *contracts):
        for c in contracts:
            c.conId = 1
        return list(contracts)

    async def reqSecDefOptParamsAsync(self, *_args, **_kwargs):
        return [_FakeChain("SMART", self.expirations, sorted(self.strikes_data), "TEST")]

    async def reqTickersAsync(self, *contracts):
        mdt = self.types_requested[-1] if self.types_requested else 1
        tickers = []
        for c in contracts:
            self.expiries_quoted.add(c.lastTradeDateOrContractMonth)
            row = self.strikes_data[c.strike]
            bid, ask, delta = row[0], row[1], row[2]
            last = row[3] if len(row) > 3 else None
            iv = row[4] if len(row) > 4 else 0.30
            if mdt < self.quotes_from_type:
                bid = ask = -1.0
            tickers.append(_FakeTicker(c, bid, ask, iv=iv, delta=delta, last=last, market_data_type=mdt))
        return tickers


class _FakeClient:
    def __init__(self, ib):
        self._ib = ib

    @property
    def ib(self):
        return self._ib

    # Same contract as WheelIBKRClient.run / run_many, on a throwaway loop.
    def run(self, factory, timeout=30.0):
        async def _go():
            r = factory()
            return await r if asyncio.iscoroutine(r) else r
        return asyncio.run(_go())

    def run_many(self, factories, timeout=120.0):
        async def _one(f):
            try:
                return (await f(), None)
            except Exception as exc:
                return (None, exc)

        async def _all():
            return await asyncio.gather(*(_one(f) for f in factories))
        return asyncio.run(_all())


class TestFetchLoopFilters:
    """Exercises _fetch_put_candidates_async's own filtering (min premium,
    max delta), not just build_candidate's math."""

    @pytest.fixture(autouse=True)
    def _no_oi_wait(self, monkeypatch):
        monkeypatch.setattr(wc, "OI_WAIT_SECONDS", 0)

    def _run(self, strikes_data, quotes_from_type=2, expirations=None, **overrides):
        expiry = _future_expiry(45)
        self.fake_ib = _FakeIB(expiry, strikes_data, quotes_from_type=quotes_from_type, expirations=expirations)
        client = _FakeClient(self.fake_ib)
        kwargs = dict(
            client=client, symbol="TEST", underlying_price=100.0, realized_vol=0.25,
            dte_min=30, dte_max=60, max_delta_abs=wc.MAX_DELTA_ABS,
            max_expiries=1, strike_band_low_pct=0.80, strike_band_high_pct=0.95,
            min_premium_per_contract=wc.MIN_PREMIUM_PER_CONTRACT,
            monthly_only=False,   # the single synthetic expiry is rarely a third Friday
        )
        kwargs.update(overrides)
        return asyncio.run(wo._fetch_put_candidates_async(**kwargs))

    def test_below_min_premium_is_filtered_out(self):
        # strike 85: $2.00 mid -> $200/contract, low delta -> kept
        # strike 88: $0.30 mid -> $30/contract, low delta -> filtered (premium)
        # (tight 0.29/0.31 quote so it isn't the spread filter doing the work)
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            88: (0.29, 0.31, -0.08),
        })
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

    def test_above_max_delta_is_filtered_out_even_with_good_premium(self):
        # strike 92: $1.00 mid -> $100/contract (well above min premium) but
        # delta 0.20 > MAX_DELTA_ABS (0.15) -> filtered on delta, not premium
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            92: (0.95, 1.05, -0.20),
        })
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

    def test_custom_min_premium_threshold_is_respected(self):
        # $0.30 mid ($30/contract) is below the default $50 floor but above
        # a caller-supplied $20 floor -- confirms the parameter is live, not
        # just the config default.
        candidates = self._run(
            {88: (0.29, 0.31, -0.08)},
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
        DTE, iv=0.30), over the 0.10 cap passed below, so it's correctly
        filtered on delta with a real number, not silently let through."""
        nan = float("nan")
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            90: (1.00, 1.10, nan),
        }, max_delta_abs=0.10)
        assert all(math.isfinite(c.delta) for c in candidates)
        strikes_kept = {c.strike for c in candidates}
        assert strikes_kept == {85}

    def test_last_trade_without_bid_ask_is_never_used_as_premium(self):
        """The WDAY $140 put case: IBKR returned bid/ask -1 (no quote) and a
        last trade of 0.75, while the real market was 0.30 / 1.95. The last
        trade must not become the premium — no two-sided quote, no candidate."""
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            90: (-1.0, -1.0, -0.04, 0.75),
        })
        assert {c.strike for c in candidates} == {85}

    def test_wide_spread_is_rejected(self):
        # 0.30 / 1.95 → mid 1.125, spread 147% of mid: the mid isn't a real price.
        candidates = self._run({
            85: (1.95, 2.05, -0.05),
            90: (0.30, 1.95, -0.04),
        }, max_delta_abs=0.5, min_premium_per_contract=0)
        assert {c.strike for c in candidates} == {85}

    def test_quote_details_are_recorded(self):
        [c] = self._run({85: (1.95, 2.05, -0.05)})
        assert (c.bid, c.ask) == (1.95, 2.05)
        assert c.premium == pytest.approx(2.00)
        assert c.spread_pct == pytest.approx(0.05)
        assert c.quote_type == "frozen"      # served under type 2 in the fake
        # quotes on type 2, then the open-interest pass on type 1
        assert self.fake_ib.types_requested == [2, 1]

    def test_falls_back_to_delayed_when_no_live_quotes(self):
        """Account without live options data: type 2 yields nothing, so the
        fetch must retry with type 4 (delayed) and label the quote as such."""
        [c] = self._run({85: (1.95, 2.05, -0.05)}, quotes_from_type=4)
        assert self.fake_ib.types_requested == [2, 4, 1]   # live → delayed fallback → open interest
        assert c.quote_type == "delayed-frozen"
        assert c.is_delayed

    def test_low_open_interest_is_dropped(self):
        # strike 85: OI 500 (kept); 86: OI 40 < 100 (dead chain); 87: no OI tick at all
        candidates = self._run({
            85: (1.95, 2.05, -0.05, None, 0.30, 500),
            86: (1.75, 1.85, -0.05, None, 0.30, 40),
            87: (1.55, 1.65, -0.05, None, 0.30, None),
        })
        assert {c.strike for c in candidates} == {85}
        assert candidates[0].open_interest == 500

    def test_open_interest_threshold_is_a_parameter(self):
        candidates = self._run({86: (1.75, 1.85, -0.05, None, 0.30, 40)}, min_open_interest=25)
        assert [c.open_interest for c in candidates] == [40]

    def test_monthly_only_skips_weeklies(self):
        """With weeklies listed around it, only the third-Friday expiry in
        the DTE window is quoted."""
        today = datetime.now().date()
        monthly = None
        for months_ahead in range(1, 4):
            y, m = divmod(today.month - 1 + months_ahead, 12)
            tf = wo.third_friday(today.year + y, m + 1)
            if 30 <= (tf - today).days <= 60:
                monthly = tf
                break
        assert monthly is not None
        weeklies = [monthly - timedelta(days=7), monthly + timedelta(days=7)]
        exps = sorted(d.strftime("%Y%m%d") for d in [monthly, *weeklies])
        candidates = self._run({85: (1.95, 2.05, -0.05)}, expirations=exps,
                               monthly_only=True, max_expiries=3)
        assert self.fake_ib.expiries_quoted == {monthly.strftime("%Y%m%d")}
        assert {c.expiry for c in candidates} == {monthly.strftime("%Y%m%d")}


class TestBatchWaves:
    """fetch_put_candidates_batch never mixes market data types while
    requests are in flight: all quotes on one type, then one OI pass."""

    @pytest.fixture(autouse=True)
    def _no_oi_wait(self, monkeypatch):
        monkeypatch.setattr(wc, "OI_WAIT_SECONDS", 0)
        monkeypatch.setattr(wc, "MONTHLY_EXPIRIES_ONLY", False)

    def _batch(self, strikes_data, quotes_from_type=2):
        ib = _FakeIB(_future_expiry(45), strikes_data, quotes_from_type=quotes_from_type)
        res = wo.fetch_put_candidates_batch(_FakeClient(ib), [("AAA", 100.0, 0.25), ("BBB", 100.0, 0.25)],
                                            strike_band_low_pct=0.80, strike_band_high_pct=0.95)
        return ib, res

    def test_single_type_then_open_interest(self):
        ib, res = self._batch({85: (1.95, 2.05, -0.05, None, 0.30, 500),
                               86: (1.75, 1.85, -0.05, None, 0.30, 40)})
        assert set(ib.types_requested) == {2, 1}
        assert ib.types_requested[-1] == 1 and 4 not in ib.types_requested
        assert {s: [c.strike for c in cs] for s, cs in res.items()} == {"AAA": [85], "BBB": [85]}

    def test_delayed_wave_only_when_nothing_quoted(self):
        ib, res = self._batch({85: (1.95, 2.05, -0.05)}, quotes_from_type=4)
        first4 = ib.types_requested.index(4)
        assert set(ib.types_requested[:first4]) == {2}          # whole first wave on type 2
        assert 2 not in ib.types_requested[first4:]              # no type-2 requests mixed into wave 2
        assert ib.types_requested[-1] == 1
        assert all(cs and cs[0].quote_type == "delayed-frozen" for cs in res.values())


class TestMonthlyExpiry:
    def test_third_friday(self):
        assert wo.third_friday(2026, 11) == datetime(2026, 11, 20).date()
        assert wo.third_friday(2026, 10) == datetime(2026, 10, 16).date()

    def test_weekly_is_not_monthly(self):
        assert wo.is_monthly_expiry("20261120", {"20261120"})
        assert not wo.is_monthly_expiry("20261113", {"20261113", "20261120"})
        assert not wo.is_monthly_expiry("20261127", {"20261120", "20261127"})

    def test_good_friday_thursday_monthly(self):
        # Apr 18 2025 was Good Friday (market closed); April's monthly expired Thu Apr 17.
        assert wo.is_monthly_expiry("20250417", {"20250411", "20250417", "20250425"})
        # A Thursday is not monthly when the third Friday itself is listed.
        assert not wo.is_monthly_expiry("20261119", {"20261119", "20261120"})

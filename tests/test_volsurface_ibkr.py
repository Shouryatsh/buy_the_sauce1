"""
tests/test_volsurface_ibkr.py — the IBKR quote source, run end to end against a FAKE broker session (real ib_insync contract
classes, fake network): option-class choice, per-expiry contract lists, strike thinning, batched streaming quotes under the line
limit, the live -> delayed fallback, unknown symbols. It proves the control flow and the table it produces; the live behaviour
(TWS answers, timing) was checked separately against a running TWS.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ib_insync import Option

from tests.volsurface_fixtures import ASOF, SPOT, make_quotes
from volsurface import data, model, surface
from volsurface.surface import QUOTE_COLUMNS

DAYS = (14, 30, 60)
EXPIRIES = [(ASOF.tz_localize(None).normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in DAYS]


class FakeIB:
    def __init__(self, known=True, live=True, delayed=True, price_ok=True, history=True):
        self.q = make_quotes(DAYS)
        self.book = {(r.expiry.replace("-", ""), float(r.strike), r.right): (r.bid, r.ask) for r in self.q.itertuples()}
        self.known, self.live, self.delayed, self.price_ok, self.history = known, live, delayed, price_ok, history
        self.mdt_calls: list[int] = []
        self._mdt = 0
        self._n = 0
        self.open_lines = 0
        self.max_lines = 0
        self.details_requests: list = []

    async def qualifyContractsAsync(self, *contracts):
        out = []
        for c in contracts:
            if not self.known:
                return []
            self._n += 1
            c.conId = 1000 + self._n
            out.append(c)
        return out

    async def reqSecDefOptParamsAsync(self, symbol, fut, sec_type, con_id):
        expiries = sorted({e.replace("-", "") for e in self.q["expiry"]})
        strikes = sorted(set(self.q["strike"]) | {1.0, 99999.0})            # the strike list is a union: includes strikes no expiry has
        return [SimpleNamespace(tradingClass="SPX", exchange="SMART", expirations=expiries, strikes=strikes, multiplier="100"),
                SimpleNamespace(tradingClass="SPXW", exchange="SMART", expirations=expiries, strikes=strikes, multiplier="100")]

    async def reqContractDetailsAsync(self, proto):
        self.details_requests.append((proto.lastTradeDateOrContractMonth, proto.tradingClass))
        out = []
        for (e, k, r), _ in self.book.items():
            if e == proto.lastTradeDateOrContractMonth and proto.tradingClass == "SPXW":
                self._n += 1
                out.append(SimpleNamespace(contract=Option("SPX", e, k, r, "SMART", "100", "USD", tradingClass="SPXW", conId=5000 + self._n)))
        return out

    def reqMarketDataType(self, t):
        self._mdt = t
        self.mdt_calls.append(t)

    async def reqTickersAsync(self, *contracts):
        usable = self.price_ok
        return [SimpleNamespace(contract=c, bid=-1.0, ask=-1.0, volume=0.0, marketDataType=2,
                                marketPrice=(lambda: SPOT) if usable else (lambda: float("nan"))) for c in contracts]

    def reqMktData(self, c, generic="", snapshot=False, reg=False):
        self.open_lines += 1
        self.max_lines = max(self.max_lines, self.open_lines)
        usable = (self._mdt == 2 and self.live) or (self._mdt == 4 and self.delayed)
        bid, ask = self.book.get((c.lastTradeDateOrContractMonth, float(c.strike), c.right), (-1.0, -1.0)) if usable else (-1.0, -1.0)
        return SimpleNamespace(contract=c, bid=bid, ask=ask, volume=5.0, marketDataType=self._mdt)

    def cancelMktData(self, c):
        self.open_lines -= 1

    async def reqHistoricalDataAsync(self, *a, **k):
        return [SimpleNamespace(close=SPOT)] if self.history else []


class FakeClient:
    def __init__(self, ib):
        self.ib = ib

    def run(self, factory, timeout=30.0):
        return asyncio.run(factory())


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(data, "now_et", lambda: ASOF)
    orig = data.stream_quotes

    async def fast(ib, contracts, mdt, two_sided, batch=80, max_wait=0.2, settle=0.05):
        return await orig(ib, contracts, mdt, two_sided, batch, max_wait, settle)
    monkeypatch.setattr(data, "stream_quotes", fast)


def fetch(ib, expiries=EXPIRIES, **kw):
    return data.fetch_ibkr(FakeClient(ib), "SPX", expiries, spot=SPOT, **kw)


def test_fetch_returns_the_standard_quote_table_from_the_pm_settled_class():
    ib = FakeIB()
    qs = fetch(ib)
    assert list(qs.quotes.columns) == QUOTE_COLUMNS and len(qs.quotes) > 100
    assert set(qs.quotes["root"]) == {"SPXW"} and set(qs.quotes["right"]) == {"P", "C"} and set(qs.quotes["expiry"]) == set(EXPIRIES)
    assert qs.source == "IBKR (frozen)" and qs.spot == SPOT and qs.notes == []
    assert (qs.quotes["bid"] > 0).sum() > 100
    both = qs.quotes.dropna(subset=["bid", "ask"])
    assert (both["ask"] >= both["bid"]).all()
    assert sorted(ib.details_requests) == sorted((e.replace("-", ""), "SPXW") for e in EXPIRIES)       # one contract list per expiry, PM class


def test_only_real_contracts_are_quoted_and_only_the_otm_half_plus_a_few_atm_pairs():
    qs = fetch(FakeIB())
    assert not (qs.quotes["strike"] == 99999.0).any() and not (qs.quotes["strike"] == 1.0).any()      # strikes from the union that no expiry lists
    puts, calls = qs.quotes[qs.quotes["right"] == "P"], qs.quotes[qs.quotes["right"] == "C"]
    assert puts["strike"].max() <= 1.041 * SPOT and calls["strike"].min() >= 0.959 * SPOT
    assert puts["strike"].min() < 0.75 * SPOT and calls["strike"].max() > 1.1 * SPOT
    both = set(puts["strike"]) & set(calls["strike"])
    assert len(both) >= 4                                                          # overlap around the money: parity pairs for the forward


def test_streaming_never_holds_more_market_data_lines_than_a_batch():
    ib = FakeIB()
    fetch(ib)
    assert ib.max_lines <= 80 and ib.open_lines == 0                              # every line is cancelled again


def test_the_ibkr_quotes_flow_through_the_whole_pipeline():
    qs = fetch(FakeIB())
    surf = surface.build_surface(qs.quotes, qs.spot, ASOF, qs.symbol, qs.source)
    assert len(surf.slices) == 3 and all(s.forward_source == "parity" for s in surf.slices)
    fs = model.fit_surface(surf, model.FitSettings(weights="equal", nstarts=6))
    assert all(f.fit is not None and f.fit.rmse_iv < 0.004 for f in fs.slices)


def test_it_falls_back_to_delayed_data_when_there_are_no_live_quotes():
    ib = FakeIB(live=False, delayed=True)
    qs = fetch(ib)
    assert ib.mdt_calls[0] == 2 and 4 in ib.mdt_calls
    assert any("15-minute delayed" in n for n in qs.notes) and qs.source == "IBKR (delayed-frozen)" and (qs.quotes["bid"] > 0).sum() > 100


def test_no_quotes_at_all_yields_missing_values_never_invented_ones():
    qs = fetch(FakeIB(live=False, delayed=False))
    assert qs.quotes["bid"].isna().all() and qs.quotes["ask"].isna().all()
    assert not surface.build_surface(qs.quotes, qs.spot, ASOF).slices              # nothing usable; the page says so instead of fitting air


def test_an_expiry_ibkr_does_not_list_is_reported_and_the_rest_still_load():
    bogus = (ASOF.tz_localize(None).normalize() + pd.Timedelta(days=500)).strftime("%Y-%m-%d")
    qs = fetch(FakeIB(), EXPIRIES + [bogus])
    assert any(bogus in n and "not in IBKR's option chain" in n for n in qs.notes) and set(qs.quotes["expiry"]) == set(EXPIRIES)
    two = [(ASOF.tz_localize(None).normalize() + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (500, 600, 700, 800, 900)]
    note = [n for n in fetch(FakeIB(), EXPIRIES + two).notes if "not in IBKR's option chain" in n]
    assert len(note) == 1 and "5 expiries" in note[0] and "+1 more" in note[0]          # one note, not five
    with pytest.raises(ValueError, match="no matching option contracts"):
        fetch(FakeIB(), [bogus])


def test_unknown_symbol_and_missing_connection_are_clear_errors():
    with pytest.raises(ValueError, match="does not know"):
        data.fetch_ibkr(FakeClient(FakeIB(known=False)), "NOPE", EXPIRIES, spot=100.0)
    with pytest.raises(ValueError, match="not connected"):
        data.fetch_ibkr(None, "SPX")


def test_spot_comes_from_a_ticker_then_from_history_and_failing_both_is_an_error():
    assert data.fetch_ibkr(FakeClient(FakeIB()), "SPX", EXPIRIES).spot == SPOT
    assert data.fetch_ibkr(FakeClient(FakeIB(price_ok=False)), "SPX", EXPIRIES).spot == SPOT       # no price on the ticker -> daily history
    with pytest.raises(ValueError, match="could not determine a price"):
        data.fetch_ibkr(FakeClient(FakeIB(price_ok=False, history=False)), "SPX", EXPIRIES)


def test_index_symbols_use_an_index_contract_and_equities_a_stock_contract():
    seen = []

    class Spy(FakeIB):
        async def qualifyContractsAsync(self, *contracts):
            seen.append((contracts[0].secType, contracts[0].exchange))
            return await super().qualifyContractsAsync(*contracts)
    data.fetch_ibkr(FakeClient(Spy()), "SPX", EXPIRIES, spot=SPOT)
    data.fetch_ibkr(FakeClient(Spy()), "SPY", EXPIRIES, spot=SPOT)
    assert seen[0] == ("IND", "CBOE") and seen[1][0] == "STK"


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def params_fixture():
    return [{"tradingClass": "SPX", "exchange": "SMART", "multiplier": "100", "expirations": ["20261120", "20261230"], "strikes": []},
            {"tradingClass": "SPXW", "exchange": "SMART", "multiplier": "100", "expirations": ["20261120", "20261125"], "strikes": []}]


def test_trading_class_prefers_the_pm_settled_class_when_listed():
    ps = params_fixture()
    assert data.pick_trading_class(ps, "20261120")["tradingClass"] == "SPXW"
    assert data.pick_trading_class(ps, "20261230")["tradingClass"] == "SPX"          # only the AM class lists it
    assert data.pick_trading_class(ps, "20261125")["tradingClass"] == "SPXW"
    assert data.pick_trading_class(ps, "20270618") is None
    assert data.pick_trading_class(ps, "20261120", pm_preferred=False)["tradingClass"] in ("SPX", "SPXW")


def test_thin_strikes_keeps_exact_strikes_dense_near_the_money_and_inside_the_band():
    strikes = list(range(3000, 12001, 5))
    out = data.thin_strikes(strikes, 7700.0, (0.7, 1.3), 80)
    assert set(out) <= set(map(float, strikes)) and out == sorted(out) and 60 <= len(out) <= 84
    assert min(out) >= 0.7 * 7700 and max(out) <= 1.3 * 7700
    core = [k for k in out if 0.88 * 7700 <= k <= 1.06 * 7700]
    assert len(core) >= 0.6 * len(out) and len(core) / (0.18 * 7700 / 5) > len([k for k in out if k not in core]) / (0.42 * 7700 / 5)
    assert data.thin_strikes([7600.0, 7700.0, 7800.0], 7700.0, (0.7, 1.3), 80) == [7600.0, 7700.0, 7800.0]       # under budget: untouched
    assert data.thin_strikes([1.0, 99999.0], 7700.0, (0.7, 1.3), 80) == []

"""
Stock screener filters for fundamentally strong companies.

This module provides filter functions to identify fundamentally strong
companies based on key financial metrics such as free cash flow yield,
debt-to-equity ratio, return on equity, and more.
"""


def free_cash_flow_yield(free_cash_flow, market_cap):
    """Calculate free cash flow yield as a percentage.

    Free cash flow yield = (Free Cash Flow / Market Cap) * 100

    Args:
        free_cash_flow: Free cash flow (in any currency unit).
        market_cap: Market capitalisation (same currency unit as free_cash_flow).

    Returns:
        Free cash flow yield as a percentage (float).

    Raises:
        ValueError: If market_cap is zero or negative.
    """
    if market_cap <= 0:
        raise ValueError("market_cap must be a positive number")
    return (free_cash_flow / market_cap) * 100


def has_strong_fcf_yield(free_cash_flow, market_cap, min_yield_pct=5.0):
    """Return True if the company's free cash flow yield exceeds the minimum threshold.

    Args:
        free_cash_flow: Free cash flow (in any currency unit).
        market_cap: Market capitalisation (same currency unit as free_cash_flow).
        min_yield_pct: Minimum free cash flow yield percentage (default 5.0).

    Returns:
        True if FCF yield >= min_yield_pct, False otherwise.
    """
    return free_cash_flow_yield(free_cash_flow, market_cap) >= min_yield_pct


def has_low_debt_to_equity(debt, equity, max_ratio=1.0):
    """Return True if the company's debt-to-equity ratio is below the maximum threshold.

    Args:
        debt: Total debt.
        equity: Total shareholders' equity.
        max_ratio: Maximum acceptable debt-to-equity ratio (default 1.0).

    Returns:
        True if D/E ratio <= max_ratio, False otherwise.

    Raises:
        ValueError: If equity is zero.
    """
    if equity == 0:
        raise ValueError("equity must not be zero")
    return (debt / equity) <= max_ratio


def has_strong_roe(net_income, equity, min_roe_pct=15.0):
    """Return True if the company's return on equity exceeds the minimum threshold.

    Args:
        net_income: Net income (profit after tax).
        equity: Total shareholders' equity.
        min_roe_pct: Minimum ROE percentage (default 15.0).

    Returns:
        True if ROE >= min_roe_pct, False otherwise.

    Raises:
        ValueError: If equity is zero.
    """
    if equity == 0:
        raise ValueError("equity must not be zero")
    roe = (net_income / equity) * 100
    return roe >= min_roe_pct


def has_positive_revenue_growth(current_revenue, previous_revenue):
    """Return True if revenue has grown compared to the previous period.

    Args:
        current_revenue: Revenue for the current period.
        previous_revenue: Revenue for the previous period.

    Returns:
        True if current_revenue > previous_revenue, False otherwise.

    Raises:
        ValueError: If previous_revenue is zero or negative.
    """
    if previous_revenue <= 0:
        raise ValueError("previous_revenue must be a positive number")
    return current_revenue > previous_revenue


def is_fundamentally_strong(
    free_cash_flow,
    market_cap,
    debt,
    equity,
    net_income,
    current_revenue,
    previous_revenue,
    min_fcf_yield_pct=5.0,
    max_de_ratio=1.0,
    min_roe_pct=15.0,
):
    """Return True if the company passes all fundamental strength filters.

    A company is considered fundamentally strong if it satisfies ALL of:
      - Free cash flow yield >= min_fcf_yield_pct (default 5%)
      - Debt-to-equity ratio <= max_de_ratio (default 1.0)
      - Return on equity >= min_roe_pct (default 15%)
      - Revenue growth is positive (current > previous)

    Args:
        free_cash_flow: Free cash flow.
        market_cap: Market capitalisation.
        debt: Total debt.
        equity: Shareholders' equity.
        net_income: Net income.
        current_revenue: Revenue for the current period.
        previous_revenue: Revenue for the previous period.
        min_fcf_yield_pct: Minimum FCF yield % (default 5.0).
        max_de_ratio: Maximum D/E ratio (default 1.0).
        min_roe_pct: Minimum ROE % (default 15.0).

    Returns:
        True if all filters pass, False otherwise.
    """
    return (
        has_strong_fcf_yield(free_cash_flow, market_cap, min_fcf_yield_pct)
        and has_low_debt_to_equity(debt, equity, max_de_ratio)
        and has_strong_roe(net_income, equity, min_roe_pct)
        and has_positive_revenue_growth(current_revenue, previous_revenue)
    )


def filter_strong_companies(companies, min_fcf_yield_pct=5.0, max_de_ratio=1.0, min_roe_pct=15.0):
    """Filter a list of company dicts and return those that are fundamentally strong.

    Each company dict must contain the following keys:
        - name (str)
        - free_cash_flow (float)
        - market_cap (float)
        - debt (float)
        - equity (float)
        - net_income (float)
        - current_revenue (float)
        - previous_revenue (float)

    Args:
        companies: List of company dicts.
        min_fcf_yield_pct: Minimum FCF yield % (default 5.0).
        max_de_ratio: Maximum D/E ratio (default 1.0).
        min_roe_pct: Minimum ROE % (default 15.0).

    Returns:
        List of company dicts that pass all fundamental filters.
    """
    return [
        company
        for company in companies
        if is_fundamentally_strong(
            free_cash_flow=company["free_cash_flow"],
            market_cap=company["market_cap"],
            debt=company["debt"],
            equity=company["equity"],
            net_income=company["net_income"],
            current_revenue=company["current_revenue"],
            previous_revenue=company["previous_revenue"],
            min_fcf_yield_pct=min_fcf_yield_pct,
            max_de_ratio=max_de_ratio,
            min_roe_pct=min_roe_pct,
        )
    ]

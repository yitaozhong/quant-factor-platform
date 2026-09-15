"""
Best-effort equation registry for the 171 P1-capable factors.

This is a first-pass representation layer, not an execution engine.

Design:
- every P1 factor gets an entry
- `equation` is a symbolic expression when it is clear from code or repo docs
- otherwise `equation` falls back to the repo's `SignalDoc.csv` detailed definition
- `detailed_definition` always preserves the repo metadata verbatim

Notation used in symbolic equations:
- lag(x, n): n-month lag unless otherwise noted
- delta(x, n): x - lag(x, n)
- avg(x, y): (x + y) / 2
- cumret(ret, a, b): prod_{i=a..b}(1 + lag(ret, i)) - 1
- mean_w(x, w): rolling mean over window w
- std_w(x, w): rolling standard deviation over window w
- max_w(x, w): rolling maximum over window w
- rank_pct(x, by=month): cross-sectional percentile rank within month
- qcut5(x, by=month): cross-sectional quintile within month
- resid(y ~ x1 + x2 + ... | group): residual from cross-sectional regression
- beta_ols(y ~ x, window, min_obs): rolling OLS beta
- r2(y ~ x1 + ...): regression R-squared

The purpose of this file is to make the factor library inspectable before
implementation. It does not attempt to be a formal parser or a full CAS.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
P1_MANIFEST_PATH = ROOT / "p1_core_available_factors.csv"
SIGNALDOC_PATH = ROOT / "Open_Source_Asset_Pricing" / "SignalDoc.csv"


@dataclass(frozen=True)
class FactorEquation:
    factor: str
    equation: str
    script: str
    source_group: str
    equation_source: str
    detailed_definition: str


MANUAL_SYMBOLIC_EQUATIONS: dict[str, str] = {
    "AbnormalAccruals": (
        "resid(tempAccruals ~ 1 / lag(at, 12) + delta(sale, 12) / lag(at, 12) + "
        "ppegt / lag(at, 12) | by=(fyear, sic2))"
    ),
    "Accruals": (
        "(delta(act, 12) - delta(che, 12) - (delta(lct, 12) - delta(dlc, 12) - "
        "delta(txp_filled0, 12)) - dp) / avg(at, lag(at, 12))"
    ),
    "AccrualsBM": (
        "1{qcut5(Accruals, by=month)=5 and qcut5(BM, by=month)=1} + "
        "0{qcut5(Accruals, by=month)=1 and qcut5(BM, by=month)=5}"
    ),
    "AM": "log(at / mve_permco)",
    "AnalystRevision": "meanest / lag(meanest, 1)",
    "AnalystValue": (
        "(ceq1 + (FROE2 - r) * ceq1 / (1 + r) + (FROE3 - r) * ceq2 / ((1 + r) * r)) "
        "/ mve_permco, with r = 0.12"
    ),
    "AOP": "(AnalystValue - IntrinsicValue) / abs(IntrinsicValue)",
    "AssetGrowth": "at / lag(at, 12) - 1",
    "Beta": "beta_ols((ret - rf) ~ (vwretd - rf), window=60, min_obs=20)",
    "BetaLiquidityPS": "beta_ols(ret ~ ps_innov, window=36, min_obs=12)",
    "BM": "log(ceqt / me_datadate)",
    "BMdec": "log(ceq / me_december)",
    "BookLeverage": "log(at / ceq)",
    "CashProd": "(mve_c + dltt - at) / che",
    "CF": "(ib + dp) / mve_permco",
    "cfp": "oancf / mve_permco",
    "ChEQ": "ceq / lag(ceq, 12) - 1",
    "ChForecastAccrual": "1{rank_pct(Accruals, by=month) > 0.5 and delta(meanest, 1) > 0}",
    "ChNAnalyst": "numest / lag(numest, 1) - 1",
    "CompEquIss": "log(ME / lag(ME, 60)) - log(1 + cumret(ret, 1, 60))",
    "CompositeDebtIssuance": "log(dltt / lag(dltt, 60)) - log(dltt / lag(dltt, 12))",
    "ConsRecomm": "1{mean(ireccd) > 3} + 0{mean(ireccd) <= 1.5}",
    "CoskewACX": (
        "E[(r_i - mean(r_i)) * (r_m - mean(r_m))^2] / "
        "(std(r_i) * std(r_m)^2) using daily data"
    ),
    "Coskewness": (
        "E[(r_i - mean(r_i)) * (r_m - mean(r_m))^2] / "
        "(std(r_i) * std(r_m)^2) using 60 monthly observations"
    ),
    "DivInit": "1{no qualifying dividend in prior 12 months and qualifying dividend this month}",
    "DivOmit": "1{qualifying dividend in seasonal pattern historically and no expected dividend now}",
    "DivSeason": "expected_dividend_month_indicator based on prior-year same-month dividend timing",
    "DivYieldST": (
        "bucket(Ediv1 / abs(prc)); Ediv1 uses lag 2, 5, or 11 months based on distcd timing code"
    ),
    "DolVol": "mean_w(vol * abs(prc), 2)",
    "DownRecomm": "1{delta(mean(ireccd), 1) < 0}",
    "EarningsForecastDisparity": "forecast_level - forecast_implied_by_historical_accounting_relation",
    "EarningsStreak": "surp if sign(surp) = sign(lag(surp, 1)); surp = (actual - meanest) / price",
    "EP": "ib / mve_permco",
    "ExclExp": "1{actual EPS < analyst expectation threshold} or scaled unexpected exclusion event proxy",
    "FEPS": "meanest(fpi=1)",
    "ForecastDispersion": "stdev / abs(meanest)",
    "GP": "(revt - cogs) / at",
    "grcapx": "(capx - lag(capx, 24)) / lag(capx, 24)",
    "grcapx3y": "3 * capx / (lag(capx, 12) + lag(capx, 24) + lag(capx, 36))",
    "High52": "prc / max_w(prc, 12)",
    "Illiquidity": "mean_daily(abs(ret_d) / dollar_volume_d) over prior 12 months",
    "Investment": "(capx / revt) / mean_w(capx / revt, 36)",
    "Leverage": "lt / mve_permco",
    "LRreversal": "cumret(ret, 13, 60)",
    "MaxRet": "max(ret_d) within month",
    "Mom6m": "cumret(ret, 1, 6)",
    "Mom12m": "cumret(ret, 1, 11)",
    "Mom12mOffSeason": "mean(nonseasonal monthly returns over the prior 12 months)",
    "MomSeason": "mean(lag(ret, 12), lag(ret, 24), lag(ret, 36), lag(ret, 48), lag(ret, 60))",
    "MomSeasonShort": "mean(lag(ret, 12), lag(ret, 24))",
    "MRreversal": "cumret(ret, 1, 24)",
    "NetDebtFinance": "(dltis - dltr - dlcch_filled0) / avg(at, lag(at, 12))",
    "NetEquityFinance": "(sstk - prstkc) / avg(at, lag(at, 12))",
    "NOA": "(operating_assets - operating_liabilities) / lag(at, 12)",
    "OperProf": "(revt - cogs - xsga - xint) / ceq",
    "PctAcc": "(ib - oancf_fallback) / max(abs(ib), 0.01)",
    "PctTotAcc": "(ni - (prstkcc - sstk + dvt + oancf + fincf + ivncf)) / abs(ni)",
    "PredictedFE": (
        "b0 + b1 * rankSG + b2 * rankBM + b3 * rankAOP + b4 * rankLTG, "
        "where b is estimated from cross-sectional regressions of FErr"
    ),
    "Price": "log(abs(prc))",
    "PriceDelayRsq": (
        "1 - r2(ret_d ~ mktrf_d) / r2(ret_d ~ mktrf_d + lag(mktrf_d, 1) + "
        "lag(mktrf_d, 2) + lag(mktrf_d, 3) + lag(mktrf_d, 4))"
    ),
    "PriceDelaySlope": (
        "(1*b1 + 2*b2 + 3*b3 + 4*b4) / (b0 + b1 + b2 + b3 + b4), "
        "where ret_d ~ mktrf_d + lag(mktrf_d, 1:4)"
    ),
    "PriceDelayTstat": (
        "(1*t1 + 2*t2 + 3*t3 + 4*t4) / (t0 + t1 + t2 + t3 + t4), "
        "where t_j are t-stats on mktrf lags in the unrestricted regression"
    ),
    "RealizedVol": "std(ret_d) over daily window mapped to month",
    "ResidualMomentum": (
        "mean_w(lag(resid(retrf ~ mktrf + hml + smb, rolling 36m), 1), 11) / "
        "std_w(lag(resid(retrf ~ mktrf + hml + smb, rolling 36m), 1), 11)"
    ),
    "REV6": "sum_{j=0..5} ((lag(meanest, j) - lag(meanest, j + 1)) / lag(price, j + 1))",
    "RevenueSurprise": "(saleq - lag(saleq, 12)) / mve or price-scaled surprise benchmark",
    "roaq": "ibq / lag(atq, 3)",
    "RoE": "ib / ceq",
    "ShareIss1Y": "(adj_shares_{t-6} / adj_shares_{t-18}) - 1, adj_shares = shrout / cfacshr",
    "ShareIss5Y": "(adj_shares / lag(adj_shares, 60)) - 1",
    "ShareRepurchase": "scstkc / lag(mve_c, 12)",
    "Size": "log(abs(prc) * shrout)",
    "SP": "sale / mve_permco",
    "STreversal": "lag(ret, 1)",
    "Tax": "tax_numerator / (tax_rate * ib), with piecewise tax_rate and cap at 1 when ib < 0",
    "TotalAccruals": (
        "pre-1988: (delta(NWC) + delta(NNCOA) + delta(NFA)) / lag(at, 12); "
        "1988+: (ni - oancf - ivncf - fincf + sstk - prstkc - dv) / lag(at, 12)"
    ),
    "UpRecomm": "1{delta(mean(ireccd), 1) > 0}",
    "VolMkt": "mean_w(vol * abs(prc), 12) / ME",
    "VolumeTrend": "slope(volume ~ time, rolling 60m, min_obs=30) / mean_w(volume, 60)",
    "XFIN": "(net_equity_financing + net_debt_financing) / avg(at, lag(at, 12))",
    "zerotrade1M": "standardized_zero_trade_measure using daily no-trade frequency over 1 month",
    "zerotrade6M": "standardized_zero_trade_measure using daily no-trade frequency over 6 months",
    "zerotrade12M": "standardized_zero_trade_measure using daily no-trade frequency over 12 months",
}


def _load_signaldoc_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    with SIGNALDOC_PATH.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            definitions[row["Acronym"]] = row["Detailed Definition"].strip()
    return definitions


def _load_p1_manifest() -> list[dict[str, str]]:
    with P1_MANIFEST_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_p1_factor_equations() -> dict[str, FactorEquation]:
    definitions = _load_signaldoc_definitions()
    rows = _load_p1_manifest()
    registry: dict[str, FactorEquation] = {}

    for row in rows:
        factor = row["factor"]
        detailed_definition = definitions[factor]
        equation = MANUAL_SYMBOLIC_EQUATIONS.get(factor, detailed_definition)
        equation_source = (
            "manual_symbolic"
            if factor in MANUAL_SYMBOLIC_EQUATIONS
            else "SignalDoc detailed definition"
        )
        registry[factor] = FactorEquation(
            factor=factor,
            equation=equation,
            script=row["script"],
            source_group=row["source_group"],
            equation_source=equation_source,
            detailed_definition=detailed_definition,
        )

    return dict(sorted(registry.items(), key=lambda item: item[0].lower()))


P1_FACTOR_EQUATIONS = build_p1_factor_equations()


def get_factor_equation(factor: str) -> FactorEquation:
    return P1_FACTOR_EQUATIONS[factor]


if __name__ == "__main__":
    print(f"P1 factor equations loaded: {len(P1_FACTOR_EQUATIONS)}")
    manual_count = sum(
        1
        for spec in P1_FACTOR_EQUATIONS.values()
        if spec.equation_source == "manual_symbolic"
    )
    print(f"Manual symbolic equations: {manual_count}")
    print(f"SignalDoc fallback equations: {len(P1_FACTOR_EQUATIONS) - manual_count}")

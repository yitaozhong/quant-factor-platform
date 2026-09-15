from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
P1_MANIFEST = ROOT / "p1_core_available_factors.csv"
SIGNALDOC = ROOT / "Open_Source_Asset_Pricing" / "SignalDoc.csv"
DL_MANIFEST = ROOT / "signals_download_manifest.csv"
Expr = dict[str, Any]


@dataclass(frozen=True)
class FactorSpec:
    """Public factor definition for a single P1 factor.

    The user-facing contract is intentionally small: each factor exposes a
    human-readable `description` and a machine-readable `expression` tree.
    """

    description: str
    expression: Expr


def V(dataset: str, name: str, where: dict[str, Any] | None = None) -> Expr:
    """Create a variable leaf node.

    This is the canonical way to reference a column from a specific dataset
    inside an expression tree.

    ``where`` selects raw rows before pivoting ({column: value}; the sentinel
    "__notna__" keeps non-null rows). Required for IBES summary files where one
    (ticker, month) carries several forecast horizons (fpi) — a leaf without a
    filter silently pivots whichever row sorts last.
    """

    node: Expr = {"kind": "var", "dataset": dataset, "name": name}
    if where:
        node["filter"] = where
    return node


def C(value: Any) -> Expr:
    """Create a constant leaf node for literals such as numbers or flags."""

    return {"kind": "const", "value": value}


def O(op: str, *args: Expr, **params: Any) -> Expr:
    """Create a generic operator node.

    `args` are positional child expressions. `params` stores operator-specific
    metadata such as rolling window length, lag units, or filter conditions.
    """

    return {"kind": "op", "op": op, "args": list(args), "params": params}


def ADD(*args: Expr) -> Expr:
    """Convenience wrapper for n-ary addition."""

    return O("add", *args)


def SUB(a: Expr, b: Expr) -> Expr:
    """Convenience wrapper for binary subtraction."""

    return O("sub", a, b)


def MUL(*args: Expr) -> Expr:
    """Convenience wrapper for n-ary multiplication."""

    return O("mul", *args)


def DIV(a: Expr, b: Expr) -> Expr:
    """Convenience wrapper for division."""

    return O("div", a, b)


def LOG(x: Expr) -> Expr:
    """Wrap an expression in a natural-log transform."""

    return O("log", x)


def ABS(x: Expr) -> Expr:
    """Wrap an expression in an absolute-value transform."""

    return O("abs", x)


def LAG(x: Expr, n: int, unit: str = "months") -> Expr:
    """Create a lag operator over a time-indexed input series.

    The `unit` parameter keeps the tree explicit about whether the lag is in
    months, days, or another time scale.
    """

    return O("lag", x, C(n), unit=unit)


def DELTA(x: Expr, n: int, unit: str = "months") -> Expr:
    """Create a first-difference operator relative to an `n`-period lag."""

    return O("delta", x, C(n), unit=unit)


def AVG(a: Expr, b: Expr) -> Expr:
    """Construct the arithmetic average of two expressions."""

    return DIV(ADD(a, b), C(2))


def MEAN_LAGS(x: Expr, lags: list[int], fill_missing: Any | None = None) -> Expr:
    """Average a selected set of lagged observations from the same input.

    This is used heavily by seasonal and off-season momentum definitions where
    the relevant history is a sparse collection of monthly lags rather than one
    contiguous rolling window.
    """

    params = {"lags": lags}
    if fill_missing is not None:
        params["fill_missing"] = fill_missing
    return O("mean_of_lags", x, **params)


def COMPOUND(x: Expr, lags: list[int], fill_missing: Any | None = None) -> Expr:
    """Compound returns across a selected set of lags.

    The operator is written symbolically here; the runtime engine can later map
    it to a log-sum or product-of-(1+r) implementation.
    """

    params = {"lags": lags}
    if fill_missing is not None:
        params["fill_missing"] = fill_missing
    return O("compound_return", x, **params)


def GROWTH(x: Expr, n: int) -> Expr:
    """Construct the standard percentage-growth expression `(x - lag(x)) / lag(x)`."""

    return DIV(SUB(x, LAG(x, n)), LAG(x, n))


def FILLNA(x: Expr, value: Any) -> Expr:
    """Replace missing values in an expression with a constant."""

    return O("fillna", x, value=value)


def ROLLING_MEAN(x: Expr, window_months: int, min_obs: int | None = None) -> Expr:
    """Create a rolling monthly mean operator."""

    params: dict[str, Any] = {"window_months": window_months}
    if min_obs is not None:
        params["min_obs"] = min_obs
    return O("rolling_mean", x, **params)


def ROLLING_STD(x: Expr, window_months: int, min_obs: int | None = None) -> Expr:
    """Create a rolling monthly standard-deviation operator."""

    params: dict[str, Any] = {"window_months": window_months}
    if min_obs is not None:
        params["min_obs"] = min_obs
    return O("rolling_std", x, **params)


def POSITIVE_ONLY(x: Expr) -> Expr:
    """Mask non-positive values."""

    return O("positive_only", x)


def NEGATIVE_TO_NULL(x: Expr) -> Expr:
    """Mask negative values."""

    return O("negative_to_null", x)


def ZERO_TO_NULL(x: Expr) -> Expr:
    """Mask exact zero values."""

    return O("zero_to_null", x)


def NONFINANCIAL_ONLY(x: Expr, sic: Expr) -> Expr:
    """Keep observations outside SIC financial industries 6000-6999."""

    return O("nonfinancial_only", x, sic)


def MANUFACTURING_ONLY(x: Expr, sic: Expr) -> Expr:
    """Keep observations in SIC manufacturing industries 2000-3999."""

    return O("manufacturing_only", x, sic)


def POW(x: Expr, power: float) -> Expr:
    """Raise an expression to a scalar power."""

    return O("pow", x, C(power))


def _snake(name: str) -> str:
    """Convert a factor name to a stable snake-case operator label.

    Fallback trees use this so even factors without manual symbolic trees still
    get deterministic, parseable operator names.
    """

    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).replace("-", "_").lower()


def _same_month_lags(start_year: int, end_year: int) -> list[int]:
    """Return lags for the same calendar month in prior years.

    Example: for years 2 through 5, this yields 23, 35, 47, 59 months.
    Those are the seasonal monthly lags used by the momentum-seasonality
    factors in the original library.
    """

    return [12 * y - 1 for y in range(start_year, end_year + 1)]


def _offseason_lags(start_year: int, end_year: int) -> list[int]:
    """Return all monthly lags in a year range except the same-month lags.

    This is the complement of `_same_month_lags` over the same horizon and is
    used for off-season momentum definitions.
    """

    lag_start = (start_year - 1) * 12
    lag_end = end_year * 12
    return [lag for lag in range(lag_start, lag_end) if (lag + 1) % 12 != 0]


def _load_defs() -> dict[str, str]:
    """Load the detailed human-readable factor definitions from `SignalDoc.csv`."""

    with SIGNALDOC.open(encoding="utf-8") as f:
        return {row["Acronym"]: row["Detailed Definition"].strip() for row in csv.DictReader(f)}


def _load_manifest() -> list[dict[str, str]]:
    """Load the P1 factor manifest assembled earlier in the project."""

    with P1_MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_vars(text: str) -> tuple[str, ...]:
    """Extract variable names from the free-form manifest column text.

    The download manifest is descriptive rather than normalized, so this helper
    strips labels and punctuation, keeps identifier-like tokens, and removes a
    small set of prose words that are not true variable names.
    """

    cleaned = []
    for part in text.split(";"):
        part = part.strip()
        if ":" in part:
            part = part.split(":", 1)[1]
        cleaned.extend(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", part))
    out, seen = [], set()
    for token in cleaned:
        if token.lower() in {"full", "table", "currently", "selected", "code", "important", "downstream", "fields"}:
            continue
        if token not in seen:
            seen.add(token)
            out.append(token)
    return tuple(out)


def _build_dataset_variables() -> dict[str, tuple[str, ...]]:
    """Build the dataset-to-variable catalog used by the factor specs.

    Most entries come from the download manifest, then key P1 intermediate
    tables are overridden manually where the manifest is too coarse. The
    catalog is intentionally limited to raw P1 datasets; derived helper tables
    such as `SignalMasterTable.parquet` are excluded on purpose.
    """

    out: dict[str, tuple[str, ...]] = {}
    with DL_MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vars_tuple = _parse_vars(row["variables"])
            for part in row["output_file"].split(";"):
                name = Path(part.strip()).name
                if name:
                    out[name] = vars_tuple
    out["monthlyCRSP.parquet"] = (
        "permno", "permco", "ticker", "exchcd", "shrcd", "sicCRSP",
        "time_avail_m", "ret", "retx", "vol", "shrout", "prc",
        "cfacshr", "bidlo", "askhi", "mve_c", "mve_permco", "dlret", "dlstcd",
    )
    out["dailyCRSP.parquet"] = ("permno", "time_d", "ret", "vol", "shrout", "prc", "cfacshr", "cfacpr")
    out["monthlyFF.parquet"] = ("time_avail_m", "mktrf", "smb", "hml", "rf", "umd")
    out["dailyFF.parquet"] = ("time_d", "mktrf", "smb", "hml", "rf", "umd")
    out["monthlyMarket.parquet"] = ("time_avail_m", "vwretd", "ewretd", "usdval")
    out["monthlyLiquidity.parquet"] = ("time_avail_m", "ps_innov")
    return out


DATASET_VARIABLES = _build_dataset_variables()


RAW_DATASET_ALIASES = {
    "SignalMasterTable.parquet": ("monthlyCRSP.parquet",),
}


def _normalize_dataset_name(dataset: str) -> tuple[str, ...]:
    """Map derived-table names onto raw dataset names.

    The factor expressions are intended to start from raw downloaded inputs.
    When the earlier manifest mentions a convenience table such as
    `SignalMasterTable.parquet`, we rewrite that dependency to the closest raw
    dataset representation used by the new registry.
    """

    return RAW_DATASET_ALIASES.get(dataset, (dataset,))


def _normalize_dependencies(deps: tuple[str, ...]) -> tuple[str, ...]:
    """Expand and deduplicate dependency names after alias normalization."""

    out, seen = [], set()
    for dep in deps:
        for normalized in _normalize_dataset_name(dep):
            if normalized not in seen:
                seen.add(normalized)
                out.append(normalized)
    return tuple(out)


DEFAULT_DEP_VARS = {
    "monthlyCRSP.parquet": ("ret", "prc", "shrout", "vol"),
    "dailyCRSP.parquet": ("ret", "prc", "vol"),
    "m_aCompustat.parquet": ("at", "sale", "ib", "ceq"),
    "a_aCompustat.parquet": ("at", "sale", "ib", "ceq"),
    "m_QCompustat.parquet": ("atq", "saleq", "ibq", "rdq"),
    "CompustatQuarterly.parquet": ("atq", "saleq", "ibq", "rdq"),
    "monthlyFF.parquet": ("mktrf", "smb", "hml", "rf"),
    "dailyFF.parquet": ("mktrf", "smb", "hml", "rf"),
    "monthlyMarket.parquet": ("vwretd", "ewretd"),
    "monthlyLiquidity.parquet": ("ps_innov",),
    "IBES_EPS_Unadj.parquet": ("meanest", "medest", "stdev", "numest", "fpi"),
    "IBES_EPS_Adj.parquet": ("meanest", "actual", "price", "fpi"),
    "IBES_Recommendations.parquet": ("ireccd", "anndats", "actdats"),
    "IBES_UnadjustedActuals.parquet": ("int0a", "fy0a", "curr_price"),
    "CRSPdistributions.parquet": ("divamt", "distcd", "exdt", "paydt"),
    "CCMLinkingTable.parquet": ("gvkey", "permno", "timeLinkStart_d", "timeLinkEnd_d"),
}


def _manual_trees() -> dict[str, Expr]:
    """Hand-code symbolic trees for factors with clear algebraic structure.

    This is the high-value part of the registry: factors here do not merely get
    a placeholder operator label, they get an explicit tree made from variables,
    constants, and compositional operators. The shared local aliases defined at
    the top keep the individual factor expressions compact and readable.
    """

    m_ret = V("monthlyCRSP.parquet", "ret")
    m_prc = V("monthlyCRSP.parquet", "prc")
    m_mve = V("monthlyCRSP.parquet", "mve_permco")
    m_mve_c = V("monthlyCRSP.parquet", "mve_c")
    # Return series with legacy-correct fill domain (see deviation register / P4):
    # zero-filled ONLY inside each permno's SignalMasterTable listing span — gap
    # months act as 0 (matching fill_date_gaps + fillna(0)); months outside the
    # span stay NaN so lags/products propagate NaN instead of manufacturing values.
    m_ret_span0 = O("span_fill", m_ret, value=0.0)
    # Zero-filled on SMT rows only — interior gap months stay NaN. For scripts
    # that fillna(0) BEFORE gap-filling (Mom6m/Mom12m/MomSeason family), a lag
    # landing on a gap month must be NaN (skipped by means, poisoning products),
    # not 0.
    m_ret_smt0 = O("smt_fill", m_ret, value=0.0)

    def SPAN_OUT(x: Expr) -> Expr:
        """Restrict output to the listing span (legacy gap-filled frame rows)."""
        return O("span_only", x)

    def SMT_OUT(x: Expr) -> Expr:
        """Restrict output to SignalMasterTable rows (scripts without gap-filling)."""
        return O("smt_only", x)

    def ZF(var_name: str) -> Expr:
        """Zero-filled on SMT rows only (script fillna(0) domain), NaN elsewhere."""
        return O("smt_fill", V("m_aCompustat.parquet", var_name), value=0.0)
    m_exchcd = V("monthlyCRSP.parquet", "exchcd")
    m_siccrsp = V("monthlyCRSP.parquet", "sicCRSP")
    m_shrcd = V("monthlyCRSP.parquet", "shrcd")
    m_shrout = V("monthlyCRSP.parquet", "shrout")
    m_cfacshr = V("monthlyCRSP.parquet", "cfacshr")
    m_vol = V("monthlyCRSP.parquet", "vol")
    d_ret = V("dailyCRSP.parquet", "ret")
    d_prc = V("dailyCRSP.parquet", "prc")
    d_vol = V("dailyCRSP.parquet", "vol")
    ff_m = V("monthlyFF.parquet", "mktrf")
    rf_m = V("monthlyFF.parquet", "rf")
    ff_d = V("dailyFF.parquet", "mktrf")
    rf_d = V("dailyFF.parquet", "rf")
    vw = V("monthlyMarket.parquet", "vwretd")
    liq = V("monthlyLiquidity.parquet", "ps_innov")
    act = V("m_aCompustat.parquet", "act")
    che = V("m_aCompustat.parquet", "che")
    lct = V("m_aCompustat.parquet", "lct")
    dlc = V("m_aCompustat.parquet", "dlc")
    at = V("m_aCompustat.parquet", "at")
    txp = V("m_aCompustat.parquet", "txp")
    dp = V("m_aCompustat.parquet", "dp")
    ceq = V("m_aCompustat.parquet", "ceq")
    ceqt = V("m_aCompustat.parquet", "ceqt")
    ib = V("m_aCompustat.parquet", "ib")
    ni = V("m_aCompustat.parquet", "ni")
    sale = V("m_aCompustat.parquet", "sale")
    capx = V("m_aCompustat.parquet", "capx")
    dltt = V("m_aCompustat.parquet", "dltt")
    dltis = V("m_aCompustat.parquet", "dltis")
    dltr = V("m_aCompustat.parquet", "dltr")
    dlcch = V("m_aCompustat.parquet", "dlcch")
    sstk = V("m_aCompustat.parquet", "sstk")
    prstkc = V("m_aCompustat.parquet", "prstkc")
    dvt = V("m_aCompustat.parquet", "dvt")
    fopt = V("m_aCompustat.parquet", "fopt")
    oancf = V("m_aCompustat.parquet", "oancf")
    ivncf = V("m_aCompustat.parquet", "ivncf")
    fincf = V("m_aCompustat.parquet", "fincf")
    txfo = V("m_aCompustat.parquet", "txfo")
    txfed = V("m_aCompustat.parquet", "txfed")
    txt = V("m_aCompustat.parquet", "txt")
    txdi = V("m_aCompustat.parquet", "txdi")
    meanest = V("IBES_EPS_Unadj.parquet", "meanest")
    # fpi='1' = annual FY1 consensus — the horizon the analyst scripts use.
    # ForecastDispersion.py additionally requires a non-null forecast period end.
    meanest_fy1 = V("IBES_EPS_Unadj.parquet", "meanest", where={"fpi": "1"})
    meanest_fy1_fpe = V("IBES_EPS_Unadj.parquet", "meanest", where={"fpi": "1", "fpedats": "__notna__"})
    stdev_fy1_fpe = V("IBES_EPS_Unadj.parquet", "stdev", where={"fpi": "1", "fpedats": "__notna__"})
    medest = V("IBES_EPS_Unadj.parquet", "medest")
    numest = V("IBES_EPS_Unadj.parquet", "numest")
    stdev = V("IBES_EPS_Unadj.parquet", "stdev")
    rec = V("IBES_Recommendations.parquet", "ireccd")
    int0a = V("IBES_UnadjustedActuals.parquet", "int0a")
    fy0a = V("IBES_UnadjustedActuals.parquet", "fy0a")
    divamt = V("CRSPdistributions.parquet", "divamt")
    distcd = V("CRSPdistributions.parquet", "distcd")
    sic = V("m_aCompustat.parquet", "sic")
    # Point-in-time industry code (registered deviation): 'sic' is the header
    # code as of the data download — using it reclassifies a firm's entire past
    # when its industry changes (look-ahead). 'sich' is the historical code per
    # fiscal year; it is missing in early years, where the header code is the
    # only information available (constant thereafter, documented).
    sic_pit = O("coalesce", V("m_aCompustat.parquet", "sich"), sic)
    revt = V("m_aCompustat.parquet", "revt")
    cogs = V("m_aCompustat.parquet", "cogs")
    xad = V("m_aCompustat.parquet", "xad")
    xsga = V("m_aCompustat.parquet", "xsga")
    xint = V("m_aCompustat.parquet", "xint")
    oibdp = V("m_aCompustat.parquet", "oibdp")
    intan = V("m_aCompustat.parquet", "intan")
    rect = V("m_aCompustat.parquet", "rect")
    invt = V("m_aCompustat.parquet", "invt")
    aco = V("m_aCompustat.parquet", "aco")
    ao = V("m_aCompustat.parquet", "ao")
    ppent = V("m_aCompustat.parquet", "ppent")
    ppegt = V("m_aCompustat.parquet", "ppegt")
    ap = V("m_aCompustat.parquet", "ap")
    drc = V("m_aCompustat.parquet", "drc")
    ivao = V("m_aCompustat.parquet", "ivao")
    lt = V("m_aCompustat.parquet", "lt")
    lco = V("m_aCompustat.parquet", "lco")
    lo = V("m_aCompustat.parquet", "lo")
    pstk = V("m_aCompustat.parquet", "pstk")
    pstkl = V("m_aCompustat.parquet", "pstkl")
    seq = V("m_aCompustat.parquet", "seq")
    ivst = V("m_aCompustat.parquet", "ivst")
    mib = V("m_aCompustat.parquet", "mib")
    dc = V("m_aCompustat.parquet", "dc")
    cshrc = V("m_aCompustat.parquet", "cshrc")
    ob = V("m_aCompustat.parquet", "ob")
    emp = V("m_aCompustat.parquet", "emp")
    dvc = V("m_aCompustat.parquet", "dvc")
    dv = V("m_aCompustat.parquet", "dv")
    prstkcc = V("m_aCompustat.parquet", "prstkcc")
    pstkrv = V("m_aCompustat.parquet", "pstkrv")
    dvpa = V("m_aCompustat.parquet", "dvpa")
    tstkp = V("m_aCompustat.parquet", "tstkp")
    ppenb = V("m_aCompustat.parquet", "ppenb")
    ppenls = V("m_aCompustat.parquet", "ppenls")
    fatb = V("m_aCompustat.parquet", "fatb")
    fatl = V("m_aCompustat.parquet", "fatl")
    xrd = V("m_aCompustat.parquet", "xrd")
    epspx = V("m_aCompustat.parquet", "epspx")
    txditc = V("m_aCompustat.parquet", "txditc")
    q_che = V("m_QCompustat.parquet", "cheq")
    q_at = V("m_QCompustat.parquet", "atq")
    q_ib = V("m_QCompustat.parquet", "ibq")
    q_txt = V("m_QCompustat.parquet", "txtq")
    q_epspx = V("m_QCompustat.parquet", "epspxq")
    q_revt = V("m_QCompustat.parquet", "revtq")
    q_cshpr = V("m_QCompustat.parquet", "cshprq")
    a_sic = V("a_aCompustat.parquet", "sic")
    a_at = V("a_aCompustat.parquet", "at")
    a_ceq = V("a_aCompustat.parquet", "ceq")
    a_ib = V("a_aCompustat.parquet", "ib")
    a_sale = V("a_aCompustat.parquet", "sale")
    a_prcc_f = V("a_aCompustat.parquet", "prcc_f")
    a_csho = V("a_aCompustat.parquet", "csho")
    a_xad = V("a_aCompustat.parquet", "xad")
    a_xad0 = V("a_aCompustat.parquet", "xad0")
    a_oancf = V("a_aCompustat.parquet", "oancf")
    a_fopt = V("a_aCompustat.parquet", "fopt")
    a_act = V("a_aCompustat.parquet", "act")
    a_che = V("a_aCompustat.parquet", "che")
    a_lct = V("a_aCompustat.parquet", "lct")
    a_dlc = V("a_aCompustat.parquet", "dlc")
    a_ni = V("a_aCompustat.parquet", "ni")
    a_ppegt = V("a_aCompustat.parquet", "ppegt")

    avg_at = AVG(at, LAG(at, 12))
    # ChNWC: at<=0 -> NaN gate INSIDE the ratio (pre-DELTA), matching the script's
    # pre-lag null; kills the inf extras from zero/negative assets
    nwc_ratio = DIV(SUB(SUB(act, che), SUB(lct, dlc)), POSITIVE_ONLY(at))
    net_noncurrent_operating_assets = DIV(SUB(SUB(SUB(at, act), ivao), SUB(SUB(lt, dlc), dltt)), at)
    current_operating_assets = SUB(act, che)
    current_operating_liabilities = SUB(lct, dlc)
    operating_assets = SUB(at, che)
    operating_liabilities = SUB(SUB(SUB(SUB(at, dltt), mib), dc), ceq)
    financial_liabilities = ADD(dltt, dlc, FILLNA(pstk, 0))
    net_financial_assets = SUB(ADD(ivst, ivao), financial_liabilities)
    # ChAssetTurnover: ppent forward-filled per permno on the wide grid (the
    # script ffills present rows, carrying across m_aCompustat coverage gaps)
    total_assets_soliman = SUB(ADD(rect, invt, aco, O("ffill", ppent), intan), ADD(ap, lco, lo))
    asset_turnover = NEGATIVE_TO_NULL(DIV(sale, AVG(total_assets_soliman, LAG(total_assets_soliman, 12))))
    enterprise_adjustment = ADD(SUB(SUB(SUB(che, dltt), dlc), dc), SUB(tstkp, dvpa))
    enterprise_bm = DIV(ADD(ceq, enterprise_adjustment), ADD(m_mve, enterprise_adjustment))
    book_to_price = DIV(ADD(SUB(ceq, dvpa), tstkp), m_mve)
    real_estate_ratio = O("coalesce", DIV(ADD(fatb, fatl), ppegt), DIV(ADD(ppenb, ppenls), ppent))
    capx_filled = O("coalesce", capx, DELTA(ppent, 12))
    avg_lag_capx = AVG(LAG(capx_filled, 12), LAG(capx_filled, 24))
    # zero denominators emit NaN (falling through to the next branch / final NaN),
    # never inf — inf ratios otherwise poison the industry mean
    pchcapx = O(
        "coalesce",
        DIV(SUB(capx_filled, avg_lag_capx), ZERO_TO_NULL(avg_lag_capx)),
        DIV(SUB(capx_filled, LAG(capx_filled, 12)), ZERO_TO_NULL(LAG(capx_filled, 12))),
    )
    # SMT-frame variant (ChInvIA): the script's whole capx chain operates on SMT
    # rows, so off-universe Compustat values must never enter the coalesce/lags
    capx_smt = O("smt_only", capx)
    ppent_smt = O("smt_only", ppent)
    capx_filled_smt = O("coalesce", capx_smt, DELTA(ppent_smt, 12))
    avg_lag_capx_smt = AVG(LAG(capx_filled_smt, 12), LAG(capx_filled_smt, 24))
    pchcapx_smt = O(
        "coalesce",
        DIV(SUB(capx_filled_smt, avg_lag_capx_smt), ZERO_TO_NULL(avg_lag_capx_smt)),
        DIV(SUB(capx_filled_smt, LAG(capx_filled_smt, 12)), ZERO_TO_NULL(LAG(capx_filled_smt, 12))),
    )
    # grcapx family: the ppent-delta fallback only fires for firms aged >= 24
    # months (script gate) — the MUL(C(0), POSITIVE_ONLY(age-23.5)) term adds 0
    # when the gate holds and NaN otherwise. Age counts the SMT ∩ m_aCompustat
    # MERGED panel rows (the script's cumcount runs AFTER the SMT inner merge —
    # SMT-only counting patched capx where Compustat coverage started late), and
    # the whole capx chain lives on SMT rows (the script's frame): off-universe
    # Compustat values never enter the coalesce or the lags. Shared by grcapx
    # and grcapx3y.
    grcapx_age = O("firm_age", O("smt_only", at), require_compustat_row=True)
    grcapx_capx = O(
        "coalesce",
        O("smt_only", capx),
        ADD(DELTA(O("smt_only", ppent), 12), MUL(C(0.0), POSITIVE_ONLY(SUB(grcapx_age, C(23.5))))),
    )
    book_equity = SUB(
        ADD(
            O(
                "coalesce",
                V("m_aCompustat.parquet", "seq"),
                ADD(ceq, O("coalesce", pstk, pstkrv, V("m_aCompustat.parquet", "pstkl"))),
                SUB(at, lt),
            ),
            FILLNA(txditc, 0),
        ),
        O("coalesce", pstk, pstkrv, V("m_aCompustat.parquet", "pstkl")),
    )

    out: dict[str, Expr] = {
        # Sloan 1996: working-capital accruals MINUS DEPRECIATION over average assets
        # (the dp term was missing — proven cell-exact on IBM 1990)
        "Accruals": DIV(SUB(SUB(SUB(DELTA(act, 12), DELTA(che, 12)), SUB(SUB(DELTA(lct, 12), DELTA(dlc, 12)), DELTA(O("fillna", txp, value=0), 12))), dp), AVG(at, LAG(at, 12))),
        "AdExp": POSITIVE_ONLY(DIV(xad, m_mve)),
        "AM": DIV(at, m_mve),
        # lag(at)==0 emits NaN, never inf (same guard as dNoa / InvestPPEInv)
        "AssetGrowth": DIV(SUB(at, LAG(at, 12)), ZERO_TO_NULL(LAG(at, 12))),
        "BrandInvest": O("brand_invest_signal", a_xad, a_xad0, a_at, a_sic),
        "BM": SMT_OUT(LOG(DIV(ceqt, O("market_equity_matched_to_datadate", m_mve, V("m_aCompustat.parquet", "datadate"), lag_months=6)))),
        # output masked to monthlyCRSP ROW presence (crsp_only): the script
        # inner-joins CRSP without the SMT shrcd/exchcd filter
        "BMdec": O("crsp_only", O(
            "bmdec_signal",
            ceq,
            V("m_aCompustat.parquet", "txditc"),
            V("m_aCompustat.parquet", "seq"),
            at,
            lt,
            pstk,
            pstkrv,
            V("m_aCompustat.parquet", "pstkl"),
            V("monthlyCRSP.parquet", "prc"),
            m_shrout,
        )),
        # was log(at/ceq): reference is at/BE with the standard book-equity fallback
        # chain and NO log (despite its FF1992 comment); BE==0 -> NaN. PS NaN makes
        # the final BE NaN even when SE came from seq — SUB(..., PS) reproduces that.
        "BookLeverage": DIV(
            at,
            ZERO_TO_NULL(SUB(
                ADD(
                    O("coalesce", seq, ADD(ceq, O("coalesce", pstk, pstkrv, pstkl)), SUB(at, lt)),
                    FILLNA(txditc, 0),
                ),
                O("coalesce", pstk, pstkrv, pstkl),
            )),
        ),
        # was cheq/atq on the datadate+3-month stamp; reference times availability
        # by the earnings ANNOUNCEMENT (rdq) — adopted per principled ruling.
        # Kernel also registers a deviation: singleton (gvkey, rdq) groups are
        # retained where the reference's Stata dedup artifact drops them.
        "Cash": O("cash_rdq_signal"),
        "CashProd": DIV(SUB(m_mve, at), che),
        # full cash-based rebuild (Ball et al.): zero-filled components on the SMT
        # fill domain, calendar 12-month deltas, gates: shrcd<=11, mve_c present,
        # log(ceq/mve) defined, non-financial by sicCRSP
        "CBOperProf": SMT_OUT(ADD(
            O("nonfinancial_only", DIV(ADD(
                SUB(SUB(ZF("revt"), ZF("cogs")), SUB(ZF("xsga"), ZF("xrd"))),
                MUL(C(-1.0), DELTA(ZF("rect"), 12)),
                MUL(C(-1.0), DELTA(ZF("invt"), 12)),
                MUL(C(-1.0), DELTA(ZF("xpp"), 12)),
                DELTA(ADD(ZF("drc"), ZF("drlt")), 12),
                DELTA(ZF("ap"), 12),
                DELTA(ZF("xacc"), 12),
            ), at), m_siccrsp),
            MUL(C(0.0), NEGATIVE_TO_NULL(SUB(C(11.0), m_shrcd))),
            MUL(C(0.0), m_mve_c),
            MUL(C(0.0), NEGATIVE_TO_NULL(DIV(ceq, m_mve))),
        )),
        "CF": DIV(ADD(ib, dp), m_mve),
        # oancf with accrual-adjusted-ib fallback; lagged accrual inputs SMT-masked
        # (the script's calendar self-merge lags resolve only against SMT∩Compustat
        # rows); output on SMT rows
        "cfp": SMT_OUT(DIV(
            O("coalesce", oancf, SUB(ib, SUB(
                SUB(DELTA(O("smt_only", act), 12), DELTA(O("smt_only", che), 12)),
                SUB(SUB(SUB(DELTA(O("smt_only", lct), 12), DELTA(O("smt_only", dlc), 12)), DELTA(O("smt_only", txp), 12)), dp),
            ))),
            m_mve,
        )),
        # the WHOLE capx chain lives on SMT rows (the script's frame): coalesce,
        # ppent fallback, and lags all read SMT-masked inputs, not raw Compustat
        "ChInvIA": O("industry_adjusted_mean", pchcapx_smt, m_siccrsp),
        "ChAssetTurnover": DELTA(asset_turnover, 12),
        # was the growth RATE; reference is the growth RATIO ceq_t/ceq_{t-12}, both
        # sides required positive (also removes the inf cells from lag==0)
        "ChEQ": DIV(POSITIVE_ONLY(ceq), POSITIVE_ONLY(LAG(ceq, 12))),
        "ChInv": DIV(DELTA(invt, 12), avg_at),
        "ChNNCOA": DELTA(net_noncurrent_operating_assets, 12),
        "ChNWC": DELTA(nwc_ratio, 12),
        # denominator is ANNUAL assets (m_aCompustat.at), not quarterly atq
        "ChTax": DIV(DELTA(q_txt, 12), LAG(at, 12)),
        # was log(1+compound(lags 1-60)) on mve_permco: reference subtracts the RAW
        # compounded return over lags 0-59 and uses mve_c (company market equity)
        # mve_c on SMT rows only (script frame lacks it at gap months). The
        # script's cumulative index SKIPS interior missing-ret months (cumprod
        # skipna == our zero-fill) but its ENDPOINTS must be real observations:
        # tempIdx is NaN wherever ret itself is missing, so both ret_t and
        # ret_{t-60} must exist. The MUL(0, ret) terms add 0 when the endpoint
        # exists and NaN when it does not.
        "CompEquIss": SPAN_OUT(ADD(
            SUB(
                LOG(DIV(O("smt_only", m_mve_c), LAG(O("smt_only", m_mve_c), 60))),
                COMPOUND(m_ret_span0, list(range(0, 60))),
            ),
            MUL(C(0.0), O("smt_only", m_ret)),
            MUL(C(0.0), LAG(O("smt_only", m_ret), 60)),
        )),
        # log((dltt+dlc)_t/(dltt+dlc)_{t-60}), div/log UNGUARDED: lag==0 -> +inf and
        # tempBD==0 -> -inf survive (oracle keeps them); raw m_aCompustat universe
        "CompositeDebtIssuance": LOG(DIV(ADD(dltt, dlc), LAG(ADD(dltt, dlc), 60))),
        # 1.0 if dc OR cshrc present-and-nonzero (script ne(0), not gt(0)); kernel
        # defaults 0.0 on every observed m_aCompustat row
        "ConvDebt": O("convertible_debt_indicator", dc, cshrc),
        # script gates: SMT rows, shrcd<=11 (null shrcd kept), and log(ceq/mve)
        # defined — ceq==0 -> BM=-inf is KEPT (a 0*LOG term would wrongly drop it)
        "DebtIssuance": SMT_OUT(O(
            "shrcd_le11",
            O("bm_defined_gate", O("positive_indicator", dltis), ceq, O("smt_only", m_mve)),
            m_shrcd,
        )),
        "DelCOA": DIV(DELTA(current_operating_assets, 12), avg_at),
        "DelCOL": DIV(DELTA(current_operating_liabilities, 12), avg_at),
        # script exclusions (each leg NaN-keeps: null ceq/sale/sic never excludes):
        # ceq<=0 | (drc==0 & signal==0) | sale<5 | financial SIC 6000-6999
        "DelDRC": O("deldrc_sample_filter", DIV(DELTA(drc, 12), avg_at), drc, ceq, sale, sic),
        "DelEqu": DIV(DELTA(ceq, 12), avg_at),
        "DelFINL": DIV(DELTA(financial_liabilities, 12), avg_at),
        "DelLTI": DIV(DELTA(ivao, 12), avg_at),
        "DelNetFin": DIV(DELTA(net_financial_assets, 12), avg_at),
        # reference OL: at - dltt0 - mib0 - dlc0 - pstk0 - ceq (was using 'dc' and
        # omitting pstk); debt/preferred zero-filled, at/ceq NaN-propagating; and
        # lag(at)==0 emits NaN, never inf
        "dNoa": DIV(
            DELTA(SUB(
                operating_assets,
                SUB(SUB(SUB(SUB(SUB(at, FILLNA(dltt, 0)), FILLNA(mib, 0)), FILLNA(dlc, 0)), FILLNA(pstk, 0)), ceq),
            ), 12),
            ZERO_TO_NULL(LAG(at, 12)),
        ),
        # SMT-row output; the kernel's screen is exclusion-form so missing ceq passes
        "EntMult": SMT_OUT(O("enterprise_multiple", m_mve, dltt, dlc, dc, che, oibdp, ceq)),
        "EBM": enterprise_bm,
        "BPEBM": SUB(book_to_price, enterprise_bm),
        # SMT-row output; lag source SMT-masked (the script's self-merge lag exists
        # only when t-6 is an SMT row); >=0 kept so EP==0 rows survive
        "EP": SMT_OUT(NEGATIVE_TO_NULL(DIV(ib, LAG(O("smt_only", m_mve), 6)))),
        # was unfiltered stdev/meanest; reference uses FY1 rows with non-null
        # forecast period end date
        "ForecastDispersion": SMT_OUT(DIV(stdev_fy1_fpe, ABS(meanest_fy1_fpe))),
        "Frontier": O(
            "frontier_signal",
            at,
            ceq,
            dltt,
            capx,
            sale,
            xrd,
            xad,
            ppent,
            V("m_aCompustat.parquet", "ebitda"),
            m_mve,
            m_siccrsp,
        ),
        "GP": NONFINANCIAL_ONLY(DIV(SUB(revt, cogs), at), sic_pit),
        # size screen = SMT mve_c; deciles via qcut-equivalent breakpoints over the
        # m_aCompustat-row population inside the op; NaN size/decile PASSES; no SMT_OUT
        # (the script's universe is m_aCompustat-driven)
        "GrAdExp": O("size_decile_filtered", O("threshold_min", DELTA(LOG(xad), 12), xad, min_value=0.1), O("smt_only", m_mve_c)),
        "GrLTNOA": O("gr_ltnoa", rect, invt, ppent, aco, intan, ao, ap, lco, lo, at, dp),
        # pure-DSL: 2yr-average growth difference with 0-denominators -> NaN (never
        # inf), calendar-lag 12m fallback; raw m_aCompustat universe (DolVol pattern)
        "GrSaleToGrInv": O(
            "coalesce",
            SUB(
                DIV(SUB(sale, AVG(LAG(sale, 12), LAG(sale, 24))), ZERO_TO_NULL(AVG(LAG(sale, 12), LAG(sale, 24)))),
                DIV(SUB(invt, AVG(LAG(invt, 12), LAG(invt, 24))), ZERO_TO_NULL(AVG(LAG(invt, 12), LAG(invt, 24)))),
            ),
            SUB(
                DIV(SUB(sale, LAG(sale, 12)), ZERO_TO_NULL(LAG(sale, 12))),
                DIV(SUB(invt, LAG(invt, 12)), ZERO_TO_NULL(LAG(invt, 12))),
            ),
        ),
        # same structure as GrSaleToGrInv with xsga as the cost leg
        "GrSaleToGrOverhead": O(
            "coalesce",
            SUB(
                DIV(SUB(sale, AVG(LAG(sale, 12), LAG(sale, 24))), ZERO_TO_NULL(AVG(LAG(sale, 12), LAG(sale, 24)))),
                DIV(SUB(xsga, AVG(LAG(xsga, 12), LAG(xsga, 24))), ZERO_TO_NULL(AVG(LAG(xsga, 12), LAG(xsga, 24)))),
            ),
            SUB(
                DIV(SUB(sale, LAG(sale, 12)), ZERO_TO_NULL(LAG(sale, 12))),
                DIV(SUB(xsga, LAG(xsga, 12)), ZERO_TO_NULL(LAG(xsga, 12))),
            ),
        ),
        # SMT-masked capx with the age-gated ppent fallback; l24==0 keeps inf on
        # both sides (script does); SMT_OUT support; calendar-lag deviation registered
        "grcapx": SMT_OUT(DIV(SUB(grcapx_capx, LAG(grcapx_capx, 24)), LAG(grcapx_capx, 24))),
        # same SMT-masked, merged-panel-age-gated capx chain as grcapx (raw-grid
        # lags let calendar lags read months outside the SMT span); the
        # denominator deliberately keeps inf (the script does)
        "grcapx3y": SMT_OUT(DIV(MUL(C(3), grcapx_capx), ADD(LAG(grcapx_capx, 12), LAG(grcapx_capx, 24), LAG(grcapx_capx, 36)))),
        # zero-fill confined to m_aCompustat row support (inside the kernel) plus the
        # script's >=1965 year gate; no SMT/CRSP masking (script never touches them)
        "hire": O("year_filtered", O("hire_growth", emp), min_year=1965),
        "Investment": O("investment_to_sales_scaled", capx, revt, window_months=36, min_obs=24),
        "Leverage": DIV(lt, m_mve),
        # Momentum/reversal family: span-limited fill (P4) + NaN-skipping means /
        # all-lags products. Formula corrections per p1_factor_reconciliation.json.
        # LRreversal script shifts POSITIONALLY over gappy rows (a defect — see
        # deviation register); we use true calendar lags. Output on SMT rows only
        # (its script never gap-fills, so gap months have no output row).
        "LRreversal": SMT_OUT(COMPOUND(m_ret_smt0, list(range(13, 37)))),  # smt_fill: its script zero-fills BEFORE any gap handling
        "Mom6m": SPAN_OUT(COMPOUND(m_ret_smt0, [1, 2, 3, 4, 5])),
        "Mom12m": SPAN_OUT(COMPOUND(m_ret_smt0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])),
        "Mom12mOffSeason": SPAN_OUT(MEAN_LAGS(m_ret_span0, list(range(1, 11)))),
        "MomSeason": SPAN_OUT(MEAN_LAGS(m_ret_smt0, _same_month_lags(2, 5))),
        "MomSeason06YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_smt0, _same_month_lags(6, 10))),
        "MomSeason11YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_smt0, _same_month_lags(11, 15))),
        "MomSeason16YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_smt0, _same_month_lags(16, 20))),
        "MomSeasonShort": SPAN_OUT(LAG(m_ret_smt0, 11)),  # was mean of lags [11,23]: reference is the single 11-month lag
        "MomOffSeason": SPAN_OUT(MEAN_LAGS(m_ret_span0, _offseason_lags(2, 5))),
        "MomOffSeason06YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_span0, _offseason_lags(6, 10))),
        "MomOffSeason11YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_span0, _offseason_lags(11, 15))),
        "MomOffSeason16YrPlus": SPAN_OUT(MEAN_LAGS(m_ret_span0, _offseason_lags(16, 20))),
        # was lags 1-24: reference compounds months t-13..t-18 only, zero-filling
        # missing lags but requiring at least one real observation among them.
        # Positional-shift script (no gap rows) -> SMT-row output.
        "MRreversal": SMT_OUT(O(
            "require_any_lag",
            COMPOUND(m_ret_span0, [13, 14, 15, 16, 17, 18], fill_missing=0),
            m_ret,
            lags=[13, 14, 15, 16, 17, 18],
        )),
        # numerator ADDS the dlcch term (was subtracted); |x|>1 -> NaN via the
        # CompEquIss-pattern gate
        "NetDebtFinance": ADD(
            DIV(ADD(SUB(dltis, dltr), FILLNA(dlcch, 0)), AVG(at, LAG(at, 12))),
            MUL(C(0.0), NEGATIVE_TO_NULL(SUB(C(1.0), ABS(
                DIV(ADD(SUB(dltis, dltr), FILLNA(dlcch, 0)), AVG(at, LAG(at, 12)))
            )))),
        ),
        # LEVEL of net debt over mve (not a 12m change over lagged ME), nulled in
        # the bottom-2 BM quintiles via Stata-fastxtile breakpoints; the BM key is
        # deliberately NOT gated by the nonfinancial/missing-var masks (financial
        # and missing-var firms still shape the breakpoints); the 0*ADD(...) term
        # replicates the script's dropna over at/ib/csho/ceq/prcc_f
        "NetDebtPrice": O("smt_only", O(
            "monthly_fastxtile_exclude",
            ADD(
                NONFINANCIAL_ONLY(DIV(SUB(ADD(dltt, dlc, pstk, dvpa), ADD(tstkp, che)), m_mve), sic),
                MUL(C(0.0), ADD(at, ib, V("m_aCompustat.parquet", "csho"), ceq, V("m_aCompustat.parquet", "prcc_f"))),
            ),
            O("smt_only", LOG(DIV(ceq, m_mve))),
            n=5,
            max_excluded=2,
        )),
        # numerator SUBTRACTS dv; |x|>1 -> NaN (script trim); raw m_aCompustat
        # universe (no SMT wrap — DolVol pattern), calendar-lag deviation registered
        "NetEquityFinance": O("abs_cap_to_null", DIV(SUB(SUB(sstk, prstkc), dv), AVG(at, LAG(at, 12))), max_abs=1.0),
        # bespoke kernel: SMT∩Compustat universe, SMT∩Compustat-masked mve lag,
        # all-components-zero drop (true 0.0 kept), SIC/ceq screens, >=24-row gate
        "NetPayoutYield": O("netpayout_yield_signal", dvc, prstkc, sstk, m_mve, sic, ceq),
        "NOA": DIV(SUB(operating_assets, operating_liabilities), LAG(at, 12)),
        # size = SMT mve_c on SMT∩m_aCompustat rows; qcut-equivalent tercile edges
        # in the kernel keep NaN-size rows; output on SMT rows
        "OperProf": SMT_OUT(O(
            "size_filtered",
            DIV(SUB(SUB(SUB(revt, cogs), xsga), xint), ceq),
            O("compustat_rows_only", O("smt_only", m_mve_c)),
            min_tercile=2,
        )),
        # SMT rows + shrcd<=11 + financial-SIC exclusion (NaN sicCRSP KEPT) +
        # require ceq and mve_c present (0*x terms add 0 when present, NaN when not)
        "OperProfRD": SMT_OUT(O(
            "shrcd_le11",
            O("financial_excluded", ADD(
                DIV(ADD(SUB(SUB(revt, cogs), xsga), FILLNA(xrd, 0)), at),
                MUL(C(0.0), ceq),
                MUL(C(0.0), m_mve_c),
            ), m_siccrsp),
            m_shrcd,
        )),
        "OPLeverage": DIV(ADD(FILLNA(xsga, 0), cogs), at),
        "OrderBacklog": ZERO_TO_NULL(DIV(ob, avg_at)),
        "OrderBacklogChg": DELTA(ZERO_TO_NULL(DIV(ob, avg_at)), 12),
        # bespoke kernel: SMT∩Compustat universe, SMT∩Compustat-masked mve lag,
        # <=0 -> NaN (row stays in count), SIC/ceq screens, >=24-row gate
        "PayoutYield": O("payout_yield_signal", dvc, prstkc, pstkrv, m_mve, sic, ceq),
        # numerator = ib-oancf with the accrual fallback (incl. the txp term) only
        # when oancf is missing; denominator = |ib| with 0.01 ONLY at ib==0
        # (ZERO_TO_NULL+FILLNA) and NaN when ib is NaN — the MUL(C(0), ib) gate
        # re-poisons the FILLNA (CompEquIss pattern); a NaN first coalesce arg can
        # only leak to the fallback when ib is NaN, where the denominator is NaN anyway
        "PctAcc": DIV(
            O(
                "coalesce",
                SUB(ib, oancf),
                SUB(
                    SUB(DELTA(act, 12), DELTA(che, 12)),
                    SUB(SUB(SUB(DELTA(lct, 12), DELTA(dlc, 12)), DELTA(txp, 12)), dp),
                ),
            ),
            ADD(FILLNA(ZERO_TO_NULL(ABS(ib)), 0.01), MUL(C(0.0), ib)),
        ),
        "PctTotAcc": DIV(SUB(ni, ADD(SUB(V("m_aCompustat.parquet", "prstkcc"), sstk), dvt, oancf, fincf, ivncf)), ABS(ni)),
        "Price": LOG(ABS(m_prc)),
        "Size": SMT_OUT(LOG(V("monthlyCRSP.parquet", "mve_c"))),  # company ME, SMT rows
        # was LAG(ret,1): reference is the CURRENT month's return (proven cell-exact
        # vs oracle), zero-filled on existing SignalMasterTable rows only
        "STreversal": O("smt_fill", m_ret, value=0.0),
        # SMT-row output with SMT-masked lag sources: the script's self-merge lags
        # exist only when the lag month is itself an SMT row
        "ShareIss1Y": SMT_OUT(DIV(
            SUB(O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=6), O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=18)),
            O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=18),
        )),
        "ShareIss5Y": SMT_OUT(DIV(
            SUB(O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=5), O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=65)),
            O("time_lagged_value", O("smt_only", MUL(m_shrout, m_cfacshr)), months=65),
        )),
        # was unfiltered meanest (pivot picked the fpi='6' quarterly row); the
        # reference uses the annual FY1 consensus. Output on SMT rows.
        # lag input masked to SMT rows: the script never merges off-universe IBES
        # values, so the month after a listing gap must not divide by one
        "AnalystRevision": SMT_OUT(DIV(O("smt_only", meanest_fy1), LAG(O("smt_only", meanest_fy1), 1))),
        "ChNAnalyst": SUB(DIV(numest, LAG(numest, 1)), C(1)),
        "FEPS": O("feps_signal", meanest),
        "FirmAge": O("firm_age", m_prc),
        # kernel reworked to script semantics: age = cumulative SMT row count,
        # ret zero-filled on SMT rows with post-filter lag gaps, qcut age quintiles;
        # no grid-wide FILLNA(ret,0); output on SMT rows
        "FirmAgeMom": SMT_OUT(O("firm_age_momentum", m_ret, m_prc)),
        "REV6": O("sum_scaled_revisions", meanest, V("IBES_EPS_Adj.parquet", "price"), months=6),
        "RD": DIV(xrd, m_mve),
        "RDAbility": O("rd_ability_signal", V("a_aCompustat.parquet", "xrd"), V("a_aCompustat.parquet", "sale")),
        # the script groups by point-in-time sicCRSP; the header 'sic' here was an
        # engine-side look-ahead (match_reference fix, not a deviation). The
        # MUL(C(0), at) gate mirrors script L55 dropna(at) (data no-op today); the
        # third positional arg is the tempN count basis — non-null at per
        # (sic2, month) on the SMT∩Compustat universe, counted BEFORE any
        # ratio-validity filtering (script L51)
        "realestate": O(
            "industry_adjusted_mean",
            O("smt_only", ADD(real_estate_ratio, MUL(C(0.0), at))),
            m_siccrsp,
            O("smt_only", at),
            min_industry_obs=5,
        ),
        # SMT-row output (script emits on the SMT-merged panel); keep the true
        # calendar t-3 atq lag
        "roaq": SMT_OUT(DIV(q_ib, LAG(q_at, 3))),
        "RoE": DIV(ni, ceq),
        "ShareRepurchase": O("positive_indicator", prstkc),
        # inputs SMT-masked so turnover and its calendar lags see the script's SMT rows
        "ShareVol": O("share_volume_signal", O("smt_only", m_vol), O("smt_only", m_shrout)),
        "SP": DIV(sale, m_mve),
        "Tax": O("piecewise_tax_ratio", txfo, txfed, txt, txdi, ib),
        "tang": MANUFACTURING_ONLY(DIV(ADD(che, MUL(C(0.715), rect), MUL(C(0.547), invt), MUL(C(0.535), ppegt)), at), sic_pit),
        "TotalAccruals": O(
            "total_accruals_signal",
            V("m_aCompustat.parquet", "ivao"),
            V("m_aCompustat.parquet", "ivst"),
            dltt,
            dlc,
            pstk,
            sstk,
            prstkc,
            dv,
            act,
            che,
            lct,
            at,
            lt,
            ni,
            oancf,
            ivncf,
            fincf,
        ),
        "XFIN": DIV(ADD(SUB(SUB(sstk, dv), prstkc), SUB(dltis, dltr), FILLNA(dlcch, 0)), at),
        "ConsRecomm": O("binary_recommendation_level", rec, low_threshold=1.5, high_threshold=3.0),
        "DownRecomm": O("recommendation_change_indicator", rec, direction="down", lag_months=1),
        "UpRecomm": O("recommendation_change_indicator", rec, direction="up", lag_months=1),
        "AOP": O("analyst_optimism_ratio", V("m_aCompustat.parquet", "ceq"), V("m_aCompustat.parquet", "ibcom"), sale, meanest, m_prc),
        # both cross-sectional kernel inputs SMT-masked so quintile breakpoints and
        # emission match the script's SMT sample (no extra output wrap needed)
        "AccrualsBM": O(
            "accruals_bm_signal",
            O("smt_only", LOG(DIV(ceq, m_mve))),
            O("smt_only", DIV(
                SUB(
                    SUB(DELTA(act, 12), DELTA(che, 12)),
                    SUB(SUB(DELTA(lct, 12), DELTA(dlc, 12)), DELTA(txp, 12)),
                ),
                avg_at,
            )),
            ceq,
        ),
        "AbnormalAccruals": O(
            "abnormal_accruals_signal",
            a_oancf,
            a_fopt,
            a_act,
            a_che,
            a_lct,
            a_dlc,
            a_ib,
            a_sale,
            a_ppegt,
            a_ni,
            a_at,
            a_sic,
            V("monthlyCRSP.parquet", "exchcd"),
        ),
        "AnalystValue": O(
            "analyst_value_signal",
            ceq,
            V("m_aCompustat.parquet", "ibcom"),
            sale,
            meanest,
            m_prc,
            m_shrout,
            dvc,
            at,
        ),
        "PredictedFE": O("predicted_forecast_error", sale, ceq, V("IBES_EPS_Unadj.parquet", "meanest"), horizon_months=12),
        "ChForecastAccrual": O("chforecast_accrual_signal", meanest, act, che, lct, dlc, txp, at),
        "ChangeInRecommendation": O("change_in_recommendation_signal", rec),
        # rolling 60-obs CAPM slope vs EQUAL-weighted market, excess both sides
        "Beta": O("monthly_rolling_beta", x_dataset="monthlyMarket.parquet", x_column="ewretd", window=60, min_obs=20, y_excess=True, x_excess=True),
        # rolling 60-valid-obs OLS of excess ret on [ps_innov, mktrf, hml, smb] + const,
        # min 36 valid; keep the ps_innov slope. Published PS series for mechanics
        # validation; the point-in-time series swap follows (task #11).
        # USER DECISION: point-in-time liquidity series (Data/ps_innov_pit.parquet) —
        # the published series embeds full-sample estimation (look-ahead). The
        # regression mechanics were validated exactly against the published-series
        # oracle before the swap (registered deviation).
        "BetaLiquidityPS": O("monthly_rolling_multibeta", regressors=[["@pit_liquidity", "ps_innov_pit"], ["monthlyFF.parquet", "mktrf"], ["monthlyFF.parquet", "hml"], ["monthlyFF.parquet", "smb"]], keep=0, window=60, min_obs=36),
        "ResidualMomentum": O("residual_momentum", reg_window=36, mom_window=11),
        "PriceDelayRsq": O("price_delay", stat="rsq"),
        "PriceDelaySlope": O("price_delay", stat="slope"),
        "PriceDelayTstat": O("price_delay", stat="tstat"),
        # bucket-streamed numba kernel (no daily var leaf -> native-eligible)
        "RealizedVol": O("daily_monthly_stat", stat="std", min_obs=15),
        "High52": O("high52_signal"),
        # 12-month mean REQUIRING all 12 calendar months (n-ary ADD propagates NaN);
        # base = monthly mean of |ret|/(|prc|*vol) with inf->NaN, bucket-streamed
        "Illiquidity": DIV(ADD(*[LAG(O("daily_monthly_stat", stat="mean", derive="amihud", min_obs=1), k) if k else O("daily_monthly_stat", stat="mean", derive="amihud", min_obs=1) for k in range(12)]), C(12)),
        # zero-fill only on SMT rows (all 5 lags must be real rows — product NaN
        # otherwise); weights and output restricted to the SMT universe
        "IndMom": SMT_OUT(O("industry_weighted_momentum", m_ret_smt0, O("smt_only", m_mve_c), m_siccrsp)),
        "IndRetBig": O("industry_big_return", m_ret, m_mve_c, m_siccrsp),
        # raw m_ret: the kernel builds the SMT∩m_aCompustat domain itself and
        # zero-fills returns inside it only (a whole-grid FILLNA(m_ret,0) would
        # manufacture structural cumret=1 cells that poison the trim pool)
        "IntanBM": O("intangible_residual", LOG(DIV(ceq, m_mve)), m_ret),
        "IntanCFP": O("intangible_residual", DIV(ADD(ib, dp), m_mve), m_ret),
        "IntanEP": O("intangible_residual", DIV(ni, m_mve), m_ret),
        "IntanSP": O("intangible_residual", DIV(sale, m_mve), m_ret),
        # IntMom.py: calendar lags via self-merge on SMT rows — a lag landing on a
        # non-row month is missing and "missing lag -> missing IntMom"; no gap rows
        "IntMom": SMT_OUT(COMPOUND(m_ret_smt0, [7, 8, 9, 10, 11, 12])),
        "MaxRet": O("daily_monthly_stat", stat="max", min_obs=1),
        # first two of the nine previously tree-less factors (WS7): within-month
        # FF3 residual moments, shared regression pass
        "IdioVol3F": O("ff3_idio_stat", stat="std"),
        "ReturnSkew3F": O("ff3_idio_stat", stat="skew"),
        # third of the nine tree-less factors: Ang-Hodrick-Xing-Zhang idio vol
        "IdioVolAHT": O("rolling_market_rmse", window=252, min_obs=100),
        "Coskewness": O("coskew_signal", variant="monthly"),
        "CoskewACX": O("coskew_signal", variant="daily"),
        "BetaFP": O("betafp_signal"),
        "BetaTailRisk": O("betatailrisk_signal"),
        "TrendFactor": O("trendfactor_signal"),
        "AnnouncementReturn": O("announcement_return_signal"),
        "MeanRankRevGrowth": O("weighted_lagged_rank_growth", revt),
        "MS": O(
            "ms_signal",
            at,
            ceq,
            ni,
            oancf,
            fopt,
            V("m_aCompustat.parquet", "wcapch"),
            ib,
            dp,
            xrd,
            capx,
            xad,
            revt,
            V("monthlyCRSP.parquet", "mve_permco"),
            m_siccrsp,
            V("m_QCompustat.parquet", "niq"),
            V("m_QCompustat.parquet", "atq"),
            V("m_QCompustat.parquet", "saleq"),
            V("m_QCompustat.parquet", "oancfy"),
            V("m_QCompustat.parquet", "capxy"),
            V("m_QCompustat.parquet", "xrdq"),
            V("m_QCompustat.parquet", "fqtr"),
        ),
        # zero-fill/masking now inside the kernels (SMT rows + listing span from the store)
        "MomRev": O("momentum_reversal_signal", m_ret),
        "MomVol": O("momentum_volume_signal", m_ret, m_vol),
        "NumEarnIncrease": O("earnings_increase_streak", q_ib),
        "ReturnSkew": O("daily_monthly_stat", stat="skew", min_obs=15, min_obs_mode="rows"),
        "EarningsConsistency": O("earnings_consistency", epspx),
        "EarningsForecastDisparity": O("earnings_forecast_disparity_signal", meanest, fy0a),
        "EarningsSurprise": SMT_OUT(O("seasonal_surprise_zscore", O("smt_only", q_epspx))),
        # q_epspx masked to SMT rows BEFORE the zscore (as EarningsSurprise does):
        # the script computes GrTemp/Drift/SD on the SMT-gvkey INNER-JOIN panel,
        # so every lag must see only merged-panel rows — masking the OUTPUT alone
        # let off-universe lags shift big-firm zscores and thus industry means;
        # min_sd=1e-8 replicates np.where(SD==0|isna|<1e-8, NaN, ES/SD)
        "EarnSupBig": SMT_OUT(O(
            "industry_big_mean",
            O("seasonal_surprise_zscore", O("smt_only", q_epspx), min_sd=1e-8),
            O("smt_only", m_mve_c),
            O("smt_only", m_siccrsp),
        )),
        "EarningsStreak": O("earnings_streak_signal", V("IBES_EPS_Adj.parquet", "actual"), V("IBES_EPS_Adj.parquet", "meanest"), V("IBES_EPS_Adj.parquet", "price")),
        "EquityDuration": O("equity_duration_signal", a_ceq, a_ib, a_sale, a_prcc_f, a_csho),
        "Herf": O("rolling_industry_herfindahl", sale, m_siccrsp, m_shrcd, window_months=36, min_obs=12, min_year=1951),
        # require_compustat_row: their scripts inner-merge m_aCompustat, so the
        # universe is SMT ∩ Compustat-row presence (recipients include NaN-metric rows)
        "HerfAsset": O("rolling_industry_herfindahl", at, m_siccrsp, m_shrcd, window_months=36, min_obs=12, require_compustat_row=True),
        "HerfBE": O("rolling_industry_herfindahl", book_equity, m_siccrsp, m_shrcd, window_months=36, min_obs=12, require_compustat_row=True),
        # lag(at)==0 emits NaN, never inf (registered tree amendment)
        "InvestPPEInv": DIV(ADD(DELTA(ppegt, 12), DELTA(invt, 12)), ZERO_TO_NULL(LAG(at, 12))),
        "ExclExp": O("excluded_expenses_signal", int0a, V("m_QCompustat.parquet", "epspiq")),
        # SPAN_OUT: the scripts asrol over gap-filled listing spans, so gap months
        # inside the span are legitimate output rows
        "DivInit": SPAN_OUT(O("dividend_initiation_flag", divamt, distcd, lookback_months=12)),
        "DivOmit": SPAN_OUT(O("dividend_omission_flag", divamt, distcd, lookback_months=12)),
        "DivSeason": O("seasonal_dividend_indicator", divamt, distcd, lag_months=12),
        # SMT_OUT: the script never gap-fills, output rows are SMT rows only
        "DivYieldST": SMT_OUT(O("divyieldst_signal", divamt, distcd, m_prc)),
        # was a 2-month rolling mean; reference is log of month t-2 dollar volume.
        # vol==0 gives log(0) = -inf, which the reference KEEPS — matched here.
        # universe = RAW monthlyCRSP (the script applies no SMT filter). Residual
        # ~30k extras vs oracle = positional-shift deviation: the script's
        # groupby.shift(2) needs two prior ROWS, our calendar lag needs the real
        # t-2 month — ours is the defined horizon (registered).
        "DolVol": LOG(LAG(MUL(V("monthlyCRSP.parquet", "vol"), ABS(m_prc)), 2)),
        # bottom size tercile ONLY (min_tercile=None kills the implicit bottom-tercile
        # exclusion); xrd zeros confined to the m_aCompustat coverage span (script
        # fill_date_gaps+fillna(0), not grid-wide); terciles ranked over the
        # m_aCompustat-and-SMT-matched mve_c population; drop_missing_size: the
        # script NULLS RDcap where tempsizeq is NaN (off-SMT rows and SMT rows
        # without mve_c have no tercile), which is also its only universe mask
        "RDcap": O(
            "size_filtered",
            O(
                "year_filtered",
                DIV(
                    ADD(
                        O("compustat_span_fill", xrd, value=0.0),
                        MUL(C(0.8), LAG(O("compustat_span_fill", xrd, value=0.0), 12)),
                        MUL(C(0.6), LAG(O("compustat_span_fill", xrd, value=0.0), 24)),
                        MUL(C(0.4), LAG(O("compustat_span_fill", xrd, value=0.0), 36)),
                        MUL(C(0.2), LAG(O("compustat_span_fill", xrd, value=0.0), 48)),
                    ),
                    at,
                ),
                min_year=1980,
            ),
            O("compustat_rows_only", O("smt_only", m_mve_c)),
            min_tercile=None,
            max_tercile=1,
            drop_missing_size=True,
        ),
        "PS": O("piotroski_score", fopt, oancf, ib, at, dltt, act, lct, txt, xint, sale, ceq, m_mve, m_shrout),
        # revps masked to SMT-rows-with-gvkey BEFORE the op so every calendar
        # lag/drift/SD term sees the script's SMT-gvkey ∩ m_QCompustat row set;
        # min_sd=1e-8 = script L99's SD > 1e-8 keep-guard (the >= vs > distinction
        # matters only at sd == exactly 1e-8 — no such float cell exists)
        "RevenueSurprise": O("seasonal_surprise_zscore", O("smt_gvkey_only", DIV(q_revt, q_cshpr)), min_sd=1e-8),
        "SurpriseRD": O("surprise_rd_indicator", xrd, revt, at),
        "fgr5yrLag": O("fgr5yr_lag_signal", meanest),
        "sfe": O("sfe_signal", medest, numest, m_prc),
        # denominator is shrout*|prc| (share-level ME), numerator 12m mean dollar
        # volume with min 10 obs
        "VolMkt": DIV(O("rolling_mean", MUL(ABS(m_prc), m_vol), window_months=12, min_obs=10), MUL(m_shrout, ABS(m_prc))),
        # crsp_only INSIDE the trim: the script's frame is raw monthlyCRSP rows
        # (no SMT filter), so the full-sample 1%/99% trim pool must exclude the
        # off-row grid cells the wide panel manufactures — masking after the trim
        # let ~700k off-row members skew the quantile bounds; trim_global
        # deliberately reproduces the reference's full-sample trim (registered
        # look-ahead match — see register); slope kernel mirrors polars_ols
        # null_policy='drop' — null-vol rows are compacted before windowing, so
        # the regression spans the last 60 NON-NULL observations while the meanX
        # denominator stays calendar; VolSD keeps calendar windows (the oracle's
        # file-order windows are a registered script defect)
        "VolumeTrend": O("trim_global", O("crsp_only", O("rolling_trend_slope_scaled", m_vol, window_months=60, min_obs=30)), lower_q=0.01, upper_q=0.99),
        "VolSD": O("crsp_only", ROLLING_STD(m_vol, 36, min_obs=24)),
        # size = shrout*|prc| (mve_c, not mve_permco); bottom-3 quintiles only, no
        # implicit bottom-tercile exclusion (min_tercile=None); kernel uses
        # qcut-equivalent breakpoints that KEEP NaN-size rows ON-ROW; crsp_only:
        # the script's universe is raw monthlyCRSP rows, so post-delisting/in-gap
        # months whose trailing calendar window still has >=24 obs must not emit
        "std_turn": O("crsp_only", O("size_filtered", ROLLING_STD(DIV(V("monthlyCRSP.parquet", "vol"), V("monthlyCRSP.parquet", "shrout")), 36, min_obs=24), MUL(m_shrout, ABS(m_prc)), min_tercile=None, max_quintile=3)),
        "VarCF": SPAN_OUT(POW(ROLLING_STD(O("smt_only", DIV(ADD(ib, dp), m_mve)), 60, min_obs=24), 2)),
        # exchcd SMT-masked so the 12 calendar lags see the script's SMT rows;
        # SMT_OUT limits output support to SMT rows
        "ExchSwitch": SMT_OUT(O("exchange_switch_indicator", O("smt_only", m_exchcd), lookback_months=12)),
        "zerotrade1M": O("zerotrade_signal", window=1, deflator=480000.0),
        "zerotrade6M": O("zerotrade_signal", window=6, deflator=11000.0),
        "zerotrade12M": O("zerotrade_signal", window=12, deflator=11000.0),
    }
    return out


MANUAL_TREES = _manual_trees()


def _fallback_expr(factor: str, deps: tuple[str, ...]) -> Expr:
    """Build a generic parseable expression when no manual tree exists.

    The result is intentionally conservative: it preserves factor identity and
    dependency references without pretending we have a fully specified algebraic
    form for that factor yet.
    """

    args = [V(ds, var) for ds in deps for var in DEFAULT_DEP_VARS.get(ds, tuple())]
    return O(_snake(factor), *args)


def _collect_refs(expr: Expr) -> tuple[tuple[str, str], ...]:
    """Collect unique `(dataset, variable)` pairs referenced by an expression tree."""

    out, seen = [], set()
    def walk(node: Any) -> None:
        """Depth-first traversal over nested dict/list expression nodes."""

        if isinstance(node, dict):
            if node.get("kind") == "var":
                ref = (node["dataset"], node["name"])
                if ref not in seen:
                    seen.add(ref)
                    out.append(ref)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(expr)
    return tuple(out)


def build_p1_factor_dependencies() -> dict[str, tuple[str, ...]]:
    """Build a raw-dataset dependency map for each factor.

    The source manifest was created from the original repo, so some entries
    still mention convenience tables. This helper normalizes those names onto
    the raw dataset inventory used by the new expression registry.
    """

    deps_map: dict[str, tuple[str, ...]] = {}
    for row in _load_manifest():
        raw_deps = tuple(part.strip() for part in row["dependencies"].split(";") if part.strip())
        deps_map[row["factor"]] = _normalize_dependencies(raw_deps)
    return dict(sorted(deps_map.items(), key=lambda item: item[0].lower()))


def build_p1_factor_specs() -> dict[str, FactorSpec]:
    """Materialize the public factor registry.

    Each exported factor exposes only the two fields the downstream engine
    needs directly: a human-readable description and a parseable expression
    tree rooted in raw P1 datasets.
    """

    defs = _load_defs()
    specs: dict[str, FactorSpec] = {}
    for factor, deps in P1_FACTOR_DEPENDENCIES.items():
        expr = MANUAL_TREES.get(factor)
        if expr is None:
            expr = _fallback_expr(factor, deps)
        specs[factor] = FactorSpec(
            description=defs[factor],
            expression=expr,
        )
    return dict(sorted(specs.items(), key=lambda item: item[0].lower()))


P1_FACTOR_DEPENDENCIES = build_p1_factor_dependencies()
P1_FACTOR_SPECS = build_p1_factor_specs()
P1_FACTOR_VARIABLE_REFS = {factor: _collect_refs(spec.expression) for factor, spec in P1_FACTOR_SPECS.items()}


def get_factor_spec(factor: str) -> FactorSpec:
    """Fetch a single factor spec by acronym."""

    return P1_FACTOR_SPECS[factor]


if __name__ == "__main__":
    manual = sum(1 for factor in P1_FACTOR_SPECS if factor in MANUAL_TREES)
    print(f"P1 factor specs loaded: {len(P1_FACTOR_SPECS)}")
    print(f"Manual tree expressions: {manual}")
    print(f"Fallback operator trees: {len(P1_FACTOR_SPECS) - manual}")
    print(f"Datasets cataloged: {len(DATASET_VARIABLES)}")

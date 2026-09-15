from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from Factor_Construction.p1_factor_specs import MANUAL_TREES, P1_FACTOR_DEPENDENCIES, P1_FACTOR_SPECS
except ModuleNotFoundError:
    from p1_factor_specs import MANUAL_TREES, P1_FACTOR_DEPENDENCIES, P1_FACTOR_SPECS


ROOT = Path(__file__).resolve().parent
P1_MANIFEST = ROOT / "p1_core_available_factors.csv"
SIGNALS_ROOT = ROOT / "Open_Source_Asset_Pricing" / "Signals"
PYCODE_ROOT = SIGNALS_ROOT / "pyCode"
PREDICTORS_DIR = PYCODE_ROOT / "Predictors"
INTERMEDIATE_DIR = SIGNALS_ROOT / "pyData" / "Intermediate"
PREDICTOR_CSV_DIR = SIGNALS_ROOT / "pyData" / "Predictors"
DEFAULT_OUTPUT_DIR = ROOT / "Data" / "P1_Parquet"

MONTHLY_PANEL_DATASETS = {
    "monthlyCRSP.parquet",
    "m_aCompustat.parquet",
    "a_aCompustat.parquet",
    "m_QCompustat.parquet",
    "CompustatQuarterly.parquet",
}
MONTHLY_SERIES_DATASETS = {
    "monthlyFF.parquet",
    "monthlyMarket.parquet",
    "monthlyLiquidity.parquet",
}
SPECIAL_MONTHLY_EVENT_DATASETS = {
    "CRSPdistributions.parquet",
}
LINKED_TICKER_MONTHLY_DATASETS = {
    "IBES_EPS_Unadj.parquet",
    "IBES_EPS_Adj.parquet",
    "IBES_Recommendations.parquet",
    "IBES_UnadjustedActuals.parquet",
}
DAILY_PANEL_DATASETS = {
    "dailyCRSP.parquet",
    "dailyFF.parquet",
}

STAGE_ORDER = {
    "bootstrap": 0,
    "monthly_native": 1,
    "annual_native": 2,
    "monthly_annual_native": 3,
    "quarterly_native": 4,
    "ibes_native": 5,
    "distribution_native": 6,
    "daily_native": 7,
    "script_monthly": 20,
    "script_annual": 21,
    "script_monthly_annual": 22,
    "script_quarterly": 23,
    "script_ibes": 24,
    "script_distribution": 25,
    "script_daily": 26,
    "script_mixed": 27,
}

# Native execution is only enabled for factors whose raw-dataset expressions are
# explicit and whose operators are implemented below. The remaining factors are
# executed through script adapters so the full 171-factor library is still
# runnable end-to-end.
NATIVE_FACTORS = {
    "Accruals",
    "AssetGrowth",
    "BookLeverage",
    "CF",
    "cfp",
    "ChEQ",
    "CompEquIss",
    "CompositeDebtIssuance",
    "grcapx",
    "grcapx3y",
    "LRreversal",
    "Mom6m",
    "Mom12m",
    "Mom12mOffSeason",
    "MomSeason",
    "MomSeason06YrPlus",
    "MomSeason11YrPlus",
    "MomSeason16YrPlus",
    "MomSeasonShort",
    "MomOffSeason",
    "MomOffSeason06YrPlus",
    "MomOffSeason11YrPlus",
    "MomOffSeason16YrPlus",
    "MRreversal",
    "NetDebtFinance",
    "NetEquityFinance",
    "PctAcc",
    "PctTotAcc",
    "Price",
    "Size",
    "STreversal",
    "ShareIss1Y",
    "ShareIss5Y",
    "DolVol",
    "VolMkt",
    "VolumeTrend",
    "ExchSwitch",
}

# Verification-only switch: when True, estimation-based kernels reproduce the
# reference scripts' look-ahead behavior (full-sample trims, same-fiscal-year
# pooling) so their output can be diffed against the oracle to prove the kernel
# itself is correct. Production runs MUST keep this False — the reference
# behavior uses future information (see p1_deviation_register.md, section B).
REFERENCE_MODE = False

DELEGATED_SPECIAL_OPERATORS = {
    "analyst_optimism_ratio",
    "binary_recommendation_level",
    "bucket",
    "cash_flow_from_operations_fallback",
    "december_market_equity",
    "dividend_initiation_flag",
    "dividend_omission_flag",
    "earnings_forecast_disparity",
    "expected_dividend_yield_short_term",
    "filter",
    "market_equity_matched_to_datadate",
    "piecewise_tax_ratio",
    "predicted_forecast_error",
    "price_delay_rsq",
    "price_delay_slope",
    "price_delay_tstat",
    "realized_volatility",
    "recommendation_change_indicator",
    "rolling_beta",
    "rolling_residual_momentum",
    "rolling_trend_slope_scaled",
    "seasonal_dividend_indicator",
    "sum_scaled_revisions",
    "total_accruals",
    "zero_trade_measure",
}


@dataclass(frozen=True)
class FactorPlanStep:
    """A single execution step in the factor pipeline."""

    step_id: int
    name: str
    stage: str
    executor: str
    factors: tuple[str, ...]
    datasets: tuple[str, ...]
    script: str | None = None
    notes: str = ""


def _load_factor_manifest() -> list[dict[str, str]]:
    """Read the factor manifest with factor-to-script mappings."""

    with P1_MANIFEST.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


FACTOR_MANIFEST = {row["factor"]: row for row in _load_factor_manifest()}
FACTOR_TO_SCRIPT = {factor: row["script"] for factor, row in FACTOR_MANIFEST.items()}
SCRIPT_TO_FACTORS: dict[str, list[str]] = {}
for factor, script in FACTOR_TO_SCRIPT.items():
    SCRIPT_TO_FACTORS.setdefault(script, []).append(factor)


_FF48_FUNC: Callable[[Any], Any] | None = None


def _snake(name: str) -> str:
    s1 = pd.Series([name]).str.replace(r"(.)([A-Z][a-z]+)", r"\1_\2", regex=True).iat[0]
    return pd.Series([s1]).str.replace(r"([a-z0-9])([A-Z])", r"\1_\2", regex=True).iat[0].replace("-", "_").lower()


SNAKE_TO_FACTOR = {_snake(factor): factor for factor in FACTOR_MANIFEST}


def _get_ff48_func() -> Callable[[Any], Any]:
    """Load the original FF48 SIC classifier lazily from the source repo."""

    global _FF48_FUNC
    if _FF48_FUNC is None:
        sicff_path = PYCODE_ROOT / "utils" / "sicff.py"
        spec = importlib.util.spec_from_file_location("p1_sicff_runtime", sicff_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load sicff helper from {sicff_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FF48_FUNC = module.get_ff48
    return _FF48_FUNC


def _list_ops(expr: dict[str, Any]) -> set[str]:
    """Collect operator names used in an expression tree."""

    out: set[str] = set()
    stack: list[Any] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("kind") == "op":
                out.add(node["op"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return out


def _stage_for_datasets(datasets: tuple[str, ...], executor: str) -> str:
    """Assign a coarse scheduling stage from the dependency set."""

    deps = set(datasets)
    if executor == "native":
        if deps <= {"monthlyCRSP.parquet"}:
            return "monthly_native"
        if deps <= {"m_aCompustat.parquet"}:
            return "annual_native"
        if deps <= {"a_aCompustat.parquet"}:
            return "annual_native"
        if deps <= {"monthlyCRSP.parquet", "m_aCompustat.parquet"}:
            return "monthly_annual_native"
        if deps <= {"monthlyCRSP.parquet", "a_aCompustat.parquet"}:
            return "monthly_annual_native"
        if deps <= {"monthlyCRSP.parquet", "m_aCompustat.parquet", "m_QCompustat.parquet"}:
            return "quarterly_native"
        if deps <= {"m_QCompustat.parquet"}:
            return "quarterly_native"
        if any("IBES" in dep for dep in deps):
            return "ibes_native"
        if deps <= {"CRSPdistributions.parquet", "monthlyCRSP.parquet"}:
            return "distribution_native"
        if "dailyCRSP.parquet" in deps or "dailyFF.parquet" in deps:
            return "daily_native"
        return "script_mixed"

    if "dailyCRSP.parquet" in deps or "dailyFF.parquet" in deps:
        return "script_daily"
    if any("IBES" in dep for dep in deps):
        return "script_ibes"
    if "CRSPdistributions.parquet" in deps:
        return "script_distribution"
    if "m_QCompustat.parquet" in deps or "CompustatQuarterly.parquet" in deps:
        return "script_quarterly"
    if "m_aCompustat.parquet" in deps or "a_aCompustat.parquet" in deps:
        if "monthlyCRSP.parquet" in deps:
            return "script_monthly_annual"
        return "script_annual"
    if "monthlyCRSP.parquet" in deps:
        return "script_monthly"
    return "script_mixed"


# Factors whose reference logic is SOUND but genuinely cannot be expressed on the
# wide grid without bespoke ops that would themselves need per-quirk validation
# (see p1_principled_rulings.json). They run via the original script.
DEMOTED_FACTORS = {
    "ChNAnalyst",  # asymmetric </>= indicator NaN rules + 1987-07..09 exclusion + qcut-exact size quintiles
}


def _tree_var_datasets(expr: dict[str, Any]) -> set[str]:
    """Datasets referenced by var LEAVES of a tree (self-loading kernels invisible)."""

    out: set[str] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("kind") == "var":
                out.add(node["dataset"])
            stack.extend(a for a in node.get("args", []) if isinstance(a, (dict, list)))
        elif isinstance(node, list):
            stack.extend(node)
    return out


def _can_execute_natively(factor: str) -> bool:
    """Decide whether a factor should be executed from the expression tree."""

    if factor in DEMOTED_FACTORS:
        return False
    if factor not in MANUAL_TREES:
        return False
    expr = P1_FACTOR_SPECS[factor].expression
    # Daily-dependent factors are native ONLY when their tree has no daily var
    # leaf: a daily leaf would pull the day×permno wide pivot (~8 GB/variable)
    # through DatasetStore; the bucket-streamed self-loading kernels
    # (daily_monthly_stat & co.) load nothing through leaves.
    if any(dep in {"dailyCRSP.parquet", "dailyFF.parquet"} for dep in P1_FACTOR_DEPENDENCIES[factor]):
        if _tree_var_datasets(expr) & DAILY_PANEL_DATASETS:
            return False
    ops = _list_ops(expr)
    return all(op in OPERATOR_REGISTRY and OPERATOR_REGISTRY[op] is not _op_delegate_to_script for op in ops)


def _script_steps_for_factors(factors: list[str]) -> list[FactorPlanStep]:
    """Group script-backed factors by predictor script."""

    grouped: dict[str, list[str]] = {}
    for factor in factors:
        grouped.setdefault(FACTOR_TO_SCRIPT[factor], []).append(factor)

    steps: list[FactorPlanStep] = []
    for script, group in grouped.items():
        datasets = tuple(sorted({dep for factor in group for dep in P1_FACTOR_DEPENDENCIES[factor]}))
        stage = _stage_for_datasets(datasets, executor="script")
        steps.append(
            FactorPlanStep(
                step_id=-1,
                name=f"script::{script}",
                stage=stage,
                executor="script",
                factors=tuple(sorted(group)),
                datasets=datasets,
                script=script,
                notes="Run original predictor script once, then standardize all emitted factors.",
            )
        )
    return steps


def _native_steps_for_factors(factors: list[str]) -> list[FactorPlanStep]:
    """Group native factors by shared dependency bundles."""

    grouped: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for factor in factors:
        datasets = tuple(sorted(P1_FACTOR_DEPENDENCIES[factor]))
        stage = _stage_for_datasets(datasets, executor="native")
        grouped.setdefault((stage, datasets), []).append(factor)

    steps: list[FactorPlanStep] = []
    for (stage, datasets), group in grouped.items():
        steps.append(
            FactorPlanStep(
                step_id=-1,
                name=f"native::{stage}::{'+'.join(datasets)}",
                stage=stage,
                executor="native",
                factors=tuple(sorted(group)),
                datasets=datasets,
                notes="Load the shared raw datasets once and evaluate each factor tree into a wide panel.",
            )
        )
    return steps


def plan_factor_calculation_order(factors: list[str] | None = None, *, allow_script_backed: bool = True) -> list[FactorPlanStep]:
    """Create an execution plan for the full factor library.

    The planner is optimized for two things:
    1. reuse expensive dataset loads across multiple native factors
    2. avoid rerunning predictor scripts that emit multiple factors
    """

    selected = sorted(factors or list(P1_FACTOR_SPECS))
    native_factors = [factor for factor in selected if _can_execute_natively(factor)]
    script_factors = [factor for factor in selected if factor not in native_factors]
    if script_factors and not allow_script_backed:
        preview = ", ".join(script_factors[:25])
        suffix = "" if len(script_factors) <= 25 else f", ... ({len(script_factors)} total)"
        raise NotImplementedError(f"Factors still missing safe native implementations: {preview}{suffix}")

    steps: list[FactorPlanStep] = []
    if script_factors:
        steps.append(
            FactorPlanStep(
                step_id=-1,
                name="bootstrap::signal_master_compat",
                stage="bootstrap",
                executor="bootstrap",
                factors=tuple(),
                datasets=("monthlyCRSP.parquet", "m_aCompustat.parquet", "IBESCRSPLinkingTable.parquet"),
                script="SignalMasterTable.py",
                notes="Compatibility cache used only by script-backed factors from the original repo.",
            )
        )

    steps.extend(_native_steps_for_factors(native_factors))
    steps.extend(_script_steps_for_factors(script_factors))

    steps = sorted(
        steps,
        key=lambda step: (
            STAGE_ORDER.get(step.stage, 999),
            step.executor,
            len(step.datasets),
            step.name,
        ),
    )

    return [
        FactorPlanStep(
            step_id=idx + 1,
            name=step.name,
            stage=step.stage,
            executor=step.executor,
            factors=step.factors,
            datasets=step.datasets,
            script=step.script,
            notes=step.notes,
        )
        for idx, step in enumerate(steps)
    ]


class DatasetStore:
    """Load raw intermediate datasets and expose them as aligned panels."""

    def __init__(self, template_index: pd.DatetimeIndex, template_columns: pd.Index, data_dir: Path = INTERMEDIATE_DIR) -> None:
        self.template_index = template_index
        self.template_columns = template_columns
        self.data_dir = data_dir
        self._raw_cache: dict[str, pd.DataFrame] = {}
        self._panel_cache: dict[tuple[str, str], pd.DataFrame | pd.Series] = {}

    def clear(self) -> None:
        """Release dataset caches between plan stages."""

        self._raw_cache.clear()
        self._panel_cache.clear()

    def load_raw(self, dataset: str) -> pd.DataFrame:
        """Read a raw intermediate parquet file once and cache it."""

        if dataset not in self._raw_cache:
            path = self.data_dir / dataset
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            self._raw_cache[dataset] = pd.read_parquet(path)
        return self._raw_cache[dataset]

    def get(
        self,
        dataset: str,
        variable: str,
        row_filter: dict[str, Any] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Return an aligned panel or time series for a dataset variable.

        row_filter selects raw rows before pivoting: {column: value} keeps rows
        where column == value; the sentinel value "__notna__" keeps rows where
        the column is non-null. Needed for IBES summary files, where one
        (ticker, month) carries several forecast horizons (fpi) and pivoting
        without a filter silently picks whichever row sorts last. Currently
        implemented ONLY for the linked-ticker (IBES) family — other families
        raise so a new use is added deliberately, not by accident.
        """

        key = (dataset, variable, tuple(sorted(row_filter.items())) if row_filter else None)
        if key in self._panel_cache:
            return self._panel_cache[key]

        raw = self.load_raw(dataset)
        if row_filter and dataset not in LINKED_TICKER_MONTHLY_DATASETS:
            raise NotImplementedError(
                f"row_filter is only implemented for IBES linked-ticker datasets; "
                f"got {dataset} with filter {row_filter}"
            )
        if row_filter:
            for col, val in row_filter.items():
                raw = raw[raw[col].notna()] if val == "__notna__" else raw[raw[col] == val]
        if dataset in LINKED_TICKER_MONTHLY_DATASETS:
            link = (
                self.load_raw("IBESCRSPLinkingTable.parquet")
                .loc[:, ["tickerIBES", "permno", "time_avail_m"]]
                .dropna(subset=["permno"])
            )
            merged = raw.merge(link, on=["tickerIBES", "time_avail_m"], how="inner")
            panel = (
                merged.loc[:, ["time_avail_m", "permno", variable]]
                .dropna(subset=["permno"])
                .pivot_table(index="time_avail_m", columns="permno", values=variable, aggfunc="last")
                .reindex(index=self.template_index, columns=self.template_columns)
            )
            panel.columns = panel.columns.astype(self.template_columns.dtype)
            self._panel_cache[key] = panel
            return panel

        if dataset in MONTHLY_PANEL_DATASETS:
            if "permno" not in raw.columns:
                if "gvkey" not in raw.columns:
                    raise KeyError(f"Dataset {dataset} has neither permno nor gvkey for panel alignment")
                gvkey_map = (
                    self.load_raw("m_aCompustat.parquet")
                    .loc[:, ["gvkey", "permno", "time_avail_m"]]
                    .dropna(subset=["permno"])
                    .drop_duplicates(["gvkey", "time_avail_m"], keep="first")
                )
                raw = raw.merge(gvkey_map, on=["gvkey", "time_avail_m"], how="left")
            panel = (
                raw.loc[:, ["time_avail_m", "permno", variable]]
                .dropna(subset=["permno"])
                .pivot_table(index="time_avail_m", columns="permno", values=variable, aggfunc="last")
                .reindex(index=self.template_index, columns=self.template_columns)
            )
            panel.columns = panel.columns.astype(self.template_columns.dtype)
            self._panel_cache[key] = panel
            return panel

        if dataset in MONTHLY_SERIES_DATASETS:
            series = raw.set_index("time_avail_m")[variable].sort_index().reindex(self.template_index)
            self._panel_cache[key] = series
            return series

        if dataset in SPECIAL_MONTHLY_EVENT_DATASETS:
            events = raw.copy()
            events["time_avail_m"] = pd.to_datetime(pd.to_datetime(events["exdt"]).dt.to_period("M").dt.start_time)
            aggfunc = "sum" if variable == "divamt" else "last"
            panel = (
                events.loc[:, ["time_avail_m", "permno", variable]]
                .dropna(subset=["permno", "time_avail_m"])
                .pivot_table(index="time_avail_m", columns="permno", values=variable, aggfunc=aggfunc)
                .reindex(index=self.template_index, columns=self.template_columns)
            )
            panel.columns = panel.columns.astype(self.template_columns.dtype)
            self._panel_cache[key] = panel
            return panel

        if dataset in DAILY_PANEL_DATASETS:
            time_col = "time_d"
            if "permno" in raw.columns:
                panel = (
                    raw.loc[:, [time_col, "permno", variable]]
                    .dropna(subset=["permno"])
                    .pivot_table(index=time_col, columns="permno", values=variable, aggfunc="last")
                    .reindex(columns=self.template_columns)
                    .sort_index()
                )
                panel.columns = panel.columns.astype(self.template_columns.dtype)
                self._panel_cache[key] = panel
                return panel
            series = raw.set_index(time_col)[variable].sort_index()
            self._panel_cache[key] = series
            return series

        raise NotImplementedError(f"Native loading is not implemented for dataset {dataset}")


def _ensure_dataframe(value: pd.DataFrame | pd.Series | float | int, template_columns: pd.Index) -> pd.DataFrame:
    """Broadcast a scalar or time series to a monthly wide panel."""

    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return pd.DataFrame({col: value for col in template_columns})
    scalar = float(value)
    return pd.DataFrame(scalar, index=pd.Index([], dtype="datetime64[ns]"), columns=template_columns)


def _gvkey_permno_month_map(store: DatasetStore) -> pd.DataFrame:
    return (
        store.load_raw("m_aCompustat.parquet")
        .loc[:, ["gvkey", "permno", "time_avail_m"]]
        .dropna(subset=["permno"])
        .drop_duplicates(["gvkey", "time_avail_m"], keep="first")
    )


def _ibes_permno_month_map(store: DatasetStore) -> pd.DataFrame:
    return (
        store.load_raw("IBESCRSPLinkingTable.parquet")
        .loc[:, ["tickerIBES", "permno", "time_avail_m"]]
        .dropna(subset=["permno"])
        .drop_duplicates(["tickerIBES", "time_avail_m"], keep="first")
    )


def _smt_ticker_link(store: DatasetStore) -> pd.DataFrame:
    """SignalMasterTable (permno, time_avail_m, tickerIBES) rows with a ticker.

    The scripts' IBES link: joining IBES frames on [tickerIBES, time_avail_m]
    restricts output to the SMT universe AND fans one ticker out to every SMT
    permno sharing it in the month (NO keep-first dedup — the
    IBESCRSPLinkingTable keep-first map both leaked non-SMT rows and dropped
    multi-permno matches)."""

    return (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "time_avail_m", "tickerIBES"]]
        .dropna(subset=["tickerIBES"])
    )


def _to_wide_panel(store: DatasetStore, long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    panel = (
        long_df.loc[:, ["time_avail_m", "permno", value_col]]
        .dropna(subset=["permno"])
        .pivot_table(index="time_avail_m", columns="permno", values=value_col, aggfunc="last")
        .reindex(index=store.template_index, columns=store.template_columns)
    )
    panel.columns = panel.columns.astype(store.template_columns.dtype)
    return panel


def _expand_hold_months(long_df: pd.DataFrame, value_col: str, months: int) -> pd.DataFrame:
    pieces = []
    base = long_df.loc[:, ["permno", "time_avail_m", value_col]].copy()
    for offset in range(months):
        piece = base.copy()
        piece["time_avail_m"] = piece["time_avail_m"] + pd.DateOffset(months=offset)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _expand_annual_signal_ffill(store: DatasetStore, ann: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Reference-style monthly expansion of annual rows: fill_date_gaps_pl(
    end_padding='12mo') + UNLIMITED forward-fill within each permno.

    Semantics that _expand_hold_months cannot express: values carry forward
    without limit until the next non-null stamp (multi-year fiscal gaps stay
    filled); the last stamp covers 13 months (grid extends to max stamp + 12
    INCLUSIVE); null-value annual rows extend the span and are filled THROUGH
    (they carry the older value onward); the NEWEST stamp wins on overlaps.
    ``ann`` needs columns [permno, fyear, time_avail_m, value_col].
    """

    if ann.empty:
        raise ValueError(f"annual frame for {value_col} is empty — nothing to expand")
    # null permno cannot be pivoted onto the permno grid (same drop _to_wide_panel applies)
    ann = ann.loc[:, ["permno", "fyear", "time_avail_m", value_col]].dropna(subset=["permno", "time_avail_m"])
    # per-permno grid span from ALL annual rows: null-value rows extend it
    stamp_span = ann.groupby("permno")["time_avail_m"].agg(["min", "max"])
    span_first = stamp_span["min"]
    span_last = stamp_span["max"] + pd.DateOffset(months=12)  # end_padding='12mo': last stamp covers 13 months
    vals = ann.dropna(subset=[value_col])
    # newest fyear wins when two fiscal years share one stamp month
    vals = vals.sort_values(["permno", "fyear"], kind="stable").drop_duplicates(
        ["permno", "time_avail_m"], keep="last"
    )
    wide = vals.pivot(index="time_avail_m", columns="permno", values=value_col)
    grid_start = min(span_first.min(), store.template_index[0])
    grid_end = max(span_last.max(), store.template_index[-1])
    grid = pd.date_range(grid_start, grid_end, freq="MS")
    if not wide.index.isin(grid).all():
        raise ValueError(f"{value_col} stamps are not month-start aligned with the monthly grid")
    filled = wide.reindex(index=grid).ffill()  # unlimited forward-fill; span cap applied below
    filled = filled.reindex(index=store.template_index, columns=store.template_columns)
    filled.columns = filled.columns.astype(store.template_columns.dtype)
    months = store.template_index.to_numpy()
    first_np = span_first.reindex(filled.columns).to_numpy(dtype="datetime64[ns]")
    last_np = span_last.reindex(filled.columns).to_numpy(dtype="datetime64[ns]")
    in_span = (months[:, None] >= first_np[None, :]) & (months[:, None] <= last_np[None, :])
    return filled.where(pd.DataFrame(in_span, index=filled.index, columns=filled.columns))


def _binary_align(left: Any, right: Any, op: Callable[[Any, Any], Any]) -> Any:
    """Apply a binary operator while preserving pandas alignment semantics."""

    return op(left, right)


def _rowwise_max(a: Any, b: Any) -> Any:
    """Elementwise maximum for aligned pandas objects."""

    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return pd.concat([_to_panel_like(a), _to_panel_like(b)], axis=0, keys=["a", "b"]).groupby(level=1).max()
    return np.maximum(a, b)


def _to_panel_like(value: Any) -> pd.DataFrame:
    """Normalize a series to a one-column dataframe for reductions."""

    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame("_series")
    return pd.DataFrame({"_scalar": [value]})


class FactorScriptAdapter:
    """Run original predictor scripts and convert their CSV outputs to wide parquet."""

    def __init__(
        self,
        template_index: pd.DatetimeIndex,
        template_columns: pd.Index,
        output_dir: Path,
        python_executable: str | None = None,
    ) -> None:
        self.template_index = template_index
        self.template_columns = template_columns
        self.output_dir = output_dir
        self.python_executable = python_executable or sys.executable
        self._executed_scripts: set[str] = set()

    def ensure_signal_master(self) -> None:
        """Build the compatibility SignalMasterTable if it is missing."""

        target = INTERMEDIATE_DIR / "SignalMasterTable.parquet"
        if target.exists():
            return
        subprocess.run([self.python_executable, "SignalMasterTable.py"], cwd=PYCODE_ROOT, check=True)

    def run_script(self, script_name: str, force: bool = False) -> None:
        """Execute a predictor script once.

        Documented OSAP usage is ``python Predictors/<script>.py`` from pyCode/ —
        the scripts read data via paths like ``../pyData/Intermediate/...`` that
        only resolve from that working directory (running with cwd=Predictors/
        resolves them inside pyCode/, which has no data, and every script dies
        at its first read).
        """

        if script_name in self._executed_scripts and not force:
            return
        # PYTHONUTF8: several legacy scripts print emoji; on Windows the child
        # process otherwise inherits the ANSI codepage (e.g. GBK) and dies with
        # UnicodeEncodeError on a print statement.
        env = dict(os.environ, PYTHONUTF8="1")
        subprocess.run(
            [self.python_executable, str(Path("Predictors") / script_name)],
            cwd=PYCODE_ROOT,
            check=True,
            env=env,
        )
        self._executed_scripts.add(script_name)

    def script_to_parquet(self, factor: str, force_script: bool = False) -> Path:
        """Run the factor's predictor script and standardize the resulting CSV."""

        script = FACTOR_TO_SCRIPT[factor]
        self.run_script(script, force=force_script)
        csv_path = PREDICTOR_CSV_DIR / f"{factor}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Predictor CSV not found for {factor}: {csv_path}")
        output_path = self.output_dir / f"{factor}.parquet"
        standardize_factor_csv_to_parquet(
            csv_path=csv_path,
            factor=factor,
            output_path=output_path,
            template_index=self.template_index,
            template_columns=self.template_columns,
        )
        return output_path


def _yyyymm_to_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    """Convert integer yyyymm values to month-start timestamps."""

    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m")


def standardize_factor_csv_to_parquet(
    csv_path: str | Path,
    factor: str,
    output_path: str | Path,
    template_index: pd.DatetimeIndex,
    template_columns: pd.Index,
) -> Path:
    """Convert long-form predictor CSV output to the standard wide parquet layout."""

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    df = pd.read_csv(csv_path)
    if factor not in df.columns:
        raise KeyError(f"{factor} column not found in {csv_path}")
    df["month"] = _yyyymm_to_timestamp(df["yyyymm"])
    wide = (
        df.pivot_table(index="month", columns="permno", values=factor, aggfunc="last")
        .reindex(index=template_index, columns=template_columns)
        .astype("float32")
    )
    wide.index.name = "month"
    wide.columns.name = "permno"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output_path, compression="zstd")
    return output_path


def build_output_template(data_dir: Path = INTERMEDIATE_DIR) -> tuple[pd.DatetimeIndex, pd.Index]:
    """Create the common month index and permno columns used by every output file."""

    base = pd.read_parquet(data_dir / "monthlyCRSP.parquet", columns=["permno", "time_avail_m"])
    months = pd.date_range(base["time_avail_m"].min(), base["time_avail_m"].max(), freq="MS")
    permnos = pd.Index(sorted(base["permno"].dropna().astype(int).unique()), dtype="int64")
    return months, permnos


def _op_add(*args: Any, **_: Any) -> Any:
    return reduce(lambda left, right: left + right, args)


def _op_sub(a: Any, b: Any, **_: Any) -> Any:
    return a - b


def _op_mul(*args: Any, **_: Any) -> Any:
    return reduce(lambda left, right: left * right, args)


def _op_div(a: Any, b: Any, **_: Any) -> Any:
    return a / b


def _op_log(x: Any, **_: Any) -> Any:
    return np.log(x)


def _op_pow(x: Any, power: Any, **_: Any) -> Any:
    return x ** power


def _op_abs(x: Any, **_: Any) -> Any:
    return np.abs(x)


def _op_lag(x: Any, n: Any, unit: str = "months", **_: Any) -> Any:
    periods = int(n)
    if unit != "months":
        raise NotImplementedError(f"Only monthly lag is supported natively, got unit={unit}")
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.shift(periods)
    return x


def _op_delta(x: Any, n: Any, unit: str = "months", **_: Any) -> Any:
    return x - _op_lag(x, n, unit=unit)


def _op_mean_of_lags(x: Any, lags: list[int], fill_missing: Any | None = None, **_: Any) -> Any:
    shifted = [_op_lag(x, lag) for lag in lags]
    concat = pd.concat(shifted, axis=1)
    if fill_missing is not None:
        concat = concat.fillna(fill_missing)
    return concat.groupby(level=0, axis=1).mean() if isinstance(x, pd.DataFrame) else concat.mean(axis=1)


def _universe_masks(store: "DatasetStore") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(strict, span) SignalMasterTable masks, cached per store (see build_universe_masks)."""

    cache = _get_derived_cache(store)
    if "universe_masks" not in cache:
        cache["universe_masks"] = build_universe_masks(store)
    return cache["universe_masks"]


def _daily_derive_recipe(derive: str | None):
    """Named recipes turning a raw daily bucket frame into the kernel's series."""

    if derive is None:
        return None
    if derive == "amihud":
        def recipe(bucket: pd.DataFrame) -> np.ndarray:
            # |ret| / (|prc|*vol); the reference maps inf (vol==0) to NaN
            vol = bucket["vol"].to_numpy(np.float64)
            ret = bucket["ret"].to_numpy(np.float64)
            prc = bucket["prc"].to_numpy(np.float64)
            dollar = np.abs(prc) * vol
            out = np.abs(ret) / dollar
            out[~(dollar > 0)] = np.nan
            return out
        return recipe
    if derive == "zerovol":
        def recipe(bucket: pd.DataFrame) -> np.ndarray:
            # 1.0 on zero-volume days, else 0.0 (NaN volume counts as 0 — reference np.where)
            vol = bucket["vol"].to_numpy(np.float64)
            return (vol == 0).astype(np.float64)
        return recipe
    if derive == "turnover":
        def recipe(bucket: pd.DataFrame) -> np.ndarray:
            return bucket["vol"].to_numpy(np.float64) / bucket["shrout"].to_numpy(np.float64)
        return recipe
    if derive == "one":
        def recipe(bucket: pd.DataFrame) -> np.ndarray:
            return np.ones(len(bucket), dtype=np.float64)
        return recipe
    if derive == "absprc":
        def recipe(bucket: pd.DataFrame) -> np.ndarray:
            return np.abs(bucket["prc"].to_numpy(np.float64))
        return recipe
    raise KeyError(f"unknown daily derive recipe {derive!r} — add it explicitly")


def _op_high52_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """52-week-high ratio (George-Hwang): last unadjusted |prc| of month t over the
    max monthly-max-|prc| of calendar months t-1..t-12.

    Reference collapses to monthly rows then takes POSITIONAL lags; we use the
    calendar month grid per project policy (registered deviation family).
    """

    try:
        from Factor_Construction.p1_daily_kernels import monthly_stat_panel
    except ModuleNotFoundError:
        from p1_daily_kernels import monthly_stat_panel

    maxpr = _to_wide_panel(store, monthly_stat_panel("max", min_obs=1, value_expr=_daily_derive_recipe("absprc")), "value")
    lastpr = _to_wide_panel(store, monthly_stat_panel("last", min_obs=1, value_expr=_daily_derive_recipe("absprc")), "value")
    high = maxpr.rolling(12, min_periods=1).max().shift(1)
    return lastpr / high


def _op_zerotrade_signal(*_: Any, window: int, deflator: float, store: "DatasetStore", **__: Any) -> Any:
    """Liu (2006) turnover-adjusted zero-trading-days over `window` months.

    temp = (zero-days + (1/turnover)/deflator) * (21*window / trading-days),
    summed over the window (ALL window months required — the reference's
    shift-and-add propagates NaN), stamped one month later. Calendar months per
    project policy; 1/turnover keeps inf when window turnover is 0 (reference
    behavior). Deflators per Liu fn.4: 480,000 (1M), 11,000 (6M/12M).
    """

    try:
        from Factor_Construction.p1_daily_kernels import monthly_stat_panel
    except ModuleNotFoundError:
        from p1_daily_kernels import monthly_stat_panel

    # min_obs_mode='rows': a month whose turnover is NaN every day (missing
    # shrout) must still emit sum=0.0 (pandas skipna sum), giving 1/0 = inf like
    # the reference — dropping the month would silently lose oracle-inf cells
    cz = _to_wide_panel(store, monthly_stat_panel("sum", min_obs=1, min_obs_mode="rows", value_expr=_daily_derive_recipe("zerovol")), "value")
    tn = _to_wide_panel(store, monthly_stat_panel("sum", min_obs=1, min_obs_mode="rows", value_expr=_daily_derive_recipe("turnover")), "value")
    nd = _to_wide_panel(store, monthly_stat_panel("sum", min_obs=1, min_obs_mode="rows", value_expr=_daily_derive_recipe("one")), "value")
    observed = nd.notna()
    if window > 1:
        # shift-and-add, NOT pandas rolling().sum(): the rolling implementation
        # subtracts values sliding out of the window, so inf months (zero shrout
        # -> turnover inf) produce inf - inf = NaN and silently poison the tail.
        # Shift-add keeps inf exact and requires all window months like the
        # reference's shift chain.
        cz = sum(cz.shift(k) for k in range(window))
        tn = sum(tn.shift(k) for k in range(window))
        nd = sum(nd.shift(k) for k in range(window))
    temp = (cz + (1.0 / tn) / deflator) * (21.0 * window / nd)
    # emit only at months where the stock actually trades: the calendar shift
    # would otherwise stamp a ghost one month past delisting/gaps (P4)
    return temp.shift(1).where(observed)


def _op_rolling_market_rmse(*_: Any, window: int = 252, min_obs: int = 100, store: "DatasetStore", **__: Any) -> Any:
    """IdioVolAHT: RMSE of the market-model regression over the trailing
    `window` valid trading days (min `min_obs`), stamped at each month's last
    valid day. Bucket-streamed."""

    try:
        from Factor_Construction.p1_daily_kernels import rolling_market_rmse_panel
    except ModuleNotFoundError:
        from p1_daily_kernels import rolling_market_rmse_panel

    return _to_wide_panel(store, rolling_market_rmse_panel(window=window, min_obs=min_obs), "value")


def _op_trendfactor_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """Han-Yang-Zhou trend factor (bucket-streamed MAs + monthly cross-sectional
    regressions; look-ahead audited clean — coefficient window is t-12..t-1)."""

    try:
        from Factor_Construction.p1_kern_trend_announce import trendfactor_panel
    except ModuleNotFoundError:
        from p1_kern_trend_announce import trendfactor_panel
    return _to_wide_panel(store, trendfactor_panel(), "value")


def _op_announcement_return_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """CAR over earnings-announcement windows (CCM link-window validity, rdq)."""

    try:
        from Factor_Construction.p1_kern_trend_announce import announcement_return_panel
    except ModuleNotFoundError:
        from p1_kern_trend_announce import announcement_return_panel
    return _to_wide_panel(store, announcement_return_panel(), "value")


def _op_betafp_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """Frazzini-Pedersen beta (bucket-streamed daily kernel module)."""

    try:
        from Factor_Construction.p1_kern_betafp_tail import betafp_panel
    except ModuleNotFoundError:
        from p1_kern_betafp_tail import betafp_panel
    return _to_wide_panel(store, betafp_panel(), "value")


def _op_betatailrisk_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """Kelly-Jiang tail-risk beta; the market tail series caches under Data/."""

    try:
        from Factor_Construction.p1_kern_betafp_tail import betatailrisk_panel
    except ModuleNotFoundError:
        from p1_kern_betafp_tail import betatailrisk_panel
    cache = ROOT / "Data" / "tailrisk_series.parquet"
    return _to_wide_panel(store, betatailrisk_panel(tail_cache=str(cache)), "value")


def _op_coskew_signal(*_: Any, variant: str, store: "DatasetStore", **__: Any) -> Any:
    """Coskewness (Harvey-Siddique, monthly 60m windows) / CoskewACX (daily 12m)."""

    try:
        from Factor_Construction import p1_kern_coskew as pkc
    except ModuleNotFoundError:
        import p1_kern_coskew as pkc

    long = pkc.coskewness_panel() if variant == "monthly" else pkc.coskew_acx_panel()
    return _to_wide_panel(store, long, "value")


def _op_price_delay(*_: Any, stat: str, store: "DatasetStore", **__: Any) -> Any:
    """PriceDelaySlope/Rsq/Tstat: one shared daily-regression pass (bucket-
    streamed, July-June annual buckets, July stamps forward-filled), cached."""

    try:
        from Factor_Construction.p1_kern_pricedelay import price_delay_panels
    except ModuleNotFoundError:
        from p1_kern_pricedelay import price_delay_panels

    cache = _get_derived_cache(store)
    if "price_delay_long" not in cache:
        cache["price_delay_long"] = price_delay_panels()
    col = {"slope": "slope", "rsq": "rsq", "tstat": "tstat"}[stat]
    return _to_wide_panel(store, cache["price_delay_long"], col)


def _op_residual_momentum(*_: Any, reg_window: int = 36, mom_window: int = 11, store: "DatasetStore", **__: Any) -> Any:
    """Blitz et al residual momentum: mean/std (ddof=1) of the trailing
    `mom_window` lagged FF3 rolling residuals (36-valid-row regressions).

    Stage-2 windows are calendar months (reference: physical rows — the
    registered positional family); all `mom_window` lagged residuals required.
    Exact two-pass std via shifted frames: zero-variance windows give true
    inf/NaN where the reference's one-pass rolling emits dust (registered)."""

    try:
        from Factor_Construction.p1_daily_kernels import rolling_ff3_residual_kernel
    except ModuleNotFoundError:
        from p1_daily_kernels import rolling_ff3_residual_kernel

    y = store.get("monthlyCRSP.parquet", "ret").sub(store.get("monthlyFF.parquet", "rf"), axis=0)
    X = np.column_stack([store.get("monthlyFF.parquet", c).to_numpy(np.float64) for c in ("mktrf", "hml", "smb")])
    resid = pd.DataFrame(
        rolling_ff3_residual_kernel(y.to_numpy(np.float64), X, reg_window),
        index=y.index, columns=y.columns,
    )
    lags = [resid.shift(1 + k) for k in range(mom_window)]
    mean = sum(lags) / mom_window
    var = sum((f - mean) ** 2 for f in lags) / (mom_window - 1)
    out = mean / np.sqrt(var)
    return _op_crsp_only(out, store=store)


def _op_monthly_rolling_multibeta(
    *_: Any,
    regressors: list,
    keep: int,
    window: int,
    min_obs: int,
    store: "DatasetStore",
    **__: Any,
) -> Any:
    """Rolling multivariate OLS slope: monthly excess returns on common series.

    regressors: list of [dataset, column] pairs (order defines `keep` index).
    BetaLiquidityPS: [[monthlyLiquidity, ps_innov], [monthlyFF, mktrf],
    [monthlyFF, hml], [monthlyFF, smb]], keep=0, 60/36.
    """

    try:
        from Factor_Construction.p1_daily_kernels import rolling_multibeta_panel_kernel
    except ModuleNotFoundError:
        from p1_daily_kernels import rolling_multibeta_panel_kernel

    def _series(ds: str, col: str) -> np.ndarray:
        if ds == "@pit_liquidity":
            # user-mandated point-in-time series (Data/ps_innov_pit.parquet),
            # outside the Intermediate store — loaded explicitly
            pit = pd.read_parquet(ROOT / "Data" / "ps_innov_pit.parquet").set_index("time_avail_m")[col]
            return pit.reindex(store.template_index).to_numpy(np.float64)
        if ds.startswith("@"):
            raise KeyError(f"unknown special series token {ds!r}")
        return store.get(ds, col).to_numpy(np.float64)

    y = store.get("monthlyCRSP.parquet", "ret").sub(store.get("monthlyFF.parquet", "rf"), axis=0)
    X = np.column_stack([_series(ds, col) for ds, col in regressors])
    beta = rolling_multibeta_panel_kernel(y.to_numpy(np.float64), X, keep, window, min_obs)
    out = pd.DataFrame(beta, index=y.index, columns=y.columns)
    return _op_crsp_only(out, store=store)


def _op_monthly_rolling_beta(
    *_: Any,
    x_dataset: str,
    x_column: str,
    window: int,
    min_obs: int,
    y_excess: bool = False,
    x_excess: bool = False,
    store: "DatasetStore",
    **__: Any,
) -> Any:
    """Rolling OLS slope of each stock's monthly return on a common series.

    Beta: y = ret - rf on x = ewretd - rf, 60/20. BetaLiquidityPS: raw ret on
    ps_innov, 36/12. Output emitted only at months where the stock has a
    monthlyCRSP row (the reference emits at listed rows, including NaN-return
    rows whose window still has enough valid pairs).
    """

    try:
        from Factor_Construction.p1_daily_kernels import rolling_beta_panel_kernel
    except ModuleNotFoundError:
        from p1_daily_kernels import rolling_beta_panel_kernel

    y = store.get("monthlyCRSP.parquet", "ret")
    x = store.get(x_dataset, x_column)
    if y_excess or x_excess:
        rf = store.get("monthlyFF.parquet", "rf")
        if y_excess:
            y = y.sub(rf, axis=0)
        if x_excess:
            x = x - rf
    # raw post-join rows: a CRSP row exists AND the common regressor is defined
    strict_rows_cache = _get_derived_cache(store)
    if "crsp_rows_mask" not in strict_rows_cache:
        _op_crsp_only(y, store=store)  # populates the cache
    rows_mask = strict_rows_cache["crsp_rows_mask"].to_numpy() & (~x.isna()).to_numpy()[:, None]
    beta = rolling_beta_panel_kernel(
        y.to_numpy(np.float64), x.to_numpy(np.float64), rows_mask, window, min_obs
    )
    out = pd.DataFrame(beta, index=y.index, columns=y.columns)
    return _op_crsp_only(out, store=store)


def _op_ff3_idio_stat(*_: Any, stat: str, store: "DatasetStore", **__: Any) -> Any:
    """IdioVol3F / ReturnSkew3F: std or population skew of within-month FF3
    residuals (>=15 valid days, singular groups dropped). The two factors share
    one regression pass, cached on the store's derived cache."""

    try:
        from Factor_Construction.p1_daily_kernels import ff3_idio_panels
    except ModuleNotFoundError:
        from p1_daily_kernels import ff3_idio_panels

    cache = _get_derived_cache(store)
    if "ff3_idio_long" not in cache:
        cache["ff3_idio_long"] = ff3_idio_panels(min_obs=15)
    long = cache["ff3_idio_long"]
    col = {"std": "idiovol", "skew": "skew3f"}[stat]
    return _to_wide_panel(store, long, col)


def _op_daily_monthly_stat(
    *_: Any,
    stat: str,
    min_obs: int = 1,
    min_obs_mode: str = "valid",
    bias_adjusted: bool = False,
    derive: str | None = None,
    store: "DatasetStore",
    **__: Any,
) -> Any:
    """Within-calendar-month statistic of a daily series, bucket-streamed.

    Self-loading: reads Data/daily_buckets (never the day×permno wide pivot).
    ``derive`` selects a named daily series recipe ('amihud' = |ret|/(|prc|·vol),
    vol==0 days excluded like the reference); default is raw ret.
    """

    try:
        from Factor_Construction.p1_daily_kernels import monthly_stat_panel
    except ModuleNotFoundError:
        from p1_daily_kernels import monthly_stat_panel

    value_expr = _daily_derive_recipe(derive)

    long = monthly_stat_panel(
        stat,
        min_obs=min_obs,
        min_obs_mode=min_obs_mode,
        bias_adjusted=bias_adjusted,
        value_expr=value_expr,
    )
    return _to_wide_panel(store, long, "value")


def _op_cash_rdq_signal(*_: Any, store: "DatasetStore", **__: Any) -> Any:
    """Cash = cheq/atq with announcement-date (rdq) availability timing.

    Replicates Cash.py: one row per (gvkey, rdq) announcement, available from
    month(rdq) and carried +1/+2 months, newest announcement winning when
    windows overlap; inner-joined to SignalMasterTable on (gvkey, month);
    requires atq > 0.

    Registered deviation: the reference's Stata-style dedup keeps only rows
    where dup == 1, silently DISCARDING every (gvkey, rdq) group of size one —
    a translation artifact with no economic meaning. We keep one row per
    announcement (first with non-missing atq), singletons included.
    """

    # Deterministic tie-break: when two fiscal quarters share one announcement
    # date (858 (gvkey, rdq) keys), keep the LATEST fiscal quarter announced.
    # The previous keep='first' after sorting only on [gvkey, rdq] depended on
    # m_QCompustat's physical row order (written by polars group_by without
    # maintain_order), which changed across data refreshes -> 282 historical
    # Cash cells flipped for no economic reason (found by the 2026-08-17 refresh gate).
    q = store.load_raw("m_QCompustat.parquet").loc[:, ["gvkey", "rdq", "datadateq", "cheq", "atq"]]
    q = q.dropna(subset=["gvkey", "rdq", "atq"]).sort_values(
        ["gvkey", "rdq", "datadateq"], kind="mergesort"
    )
    q = q.drop_duplicates(["gvkey", "rdq"], keep="last").drop(columns=["datadateq"])
    q["time_avail_m"] = q["rdq"].to_numpy(dtype="datetime64[M]")

    expanded = pd.concat(
        [q.assign(time_avail_m=q["time_avail_m"] + pd.DateOffset(months=n)) for n in range(3)],
        ignore_index=True,
    )
    expanded = (
        expanded.sort_values(["gvkey", "time_avail_m", "rdq"], ascending=[True, True, False])
        .drop_duplicates(["gvkey", "time_avail_m"], keep="first")
    )

    smt = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "gvkey", "time_avail_m"]]
        .dropna(subset=["gvkey"])
    )
    merged = smt.merge(
        expanded.loc[:, ["gvkey", "time_avail_m", "cheq", "atq"]],
        on=["gvkey", "time_avail_m"],
        how="inner",
    )
    merged = merged[merged["atq"] > 0]
    merged["cash_value"] = merged["cheq"] / merged["atq"]
    return _to_wide_panel(store, merged, "cash_value")


def _op_span_fill(x: Any, value: float = 0.0, *, store: "DatasetStore", **_: Any) -> Any:
    """Fill within each permno's SignalMasterTable listing span; NaN outside.

    Replicates the legacy fill_date_gaps + fillna(value) sequence on the SMT
    panel: in-span months where the stock is outside the SMT universe (the
    legacy gap rows) deliberately take `value`, NOT the raw CRSP value, because
    the scripts never see those rows' data. Months outside [first, last] SMT
    month stay NaN so downstream lags/products propagate NaN (P4: no
    manufactured values outside a stock's history).
    """

    strict, span = _universe_masks(store)
    return x.where(strict).fillna(value).where(span)


def _op_smt_fill(x: Any, value: float = 0.0, *, store: "DatasetStore", **_: Any) -> Any:
    """fillna(value) on SignalMasterTable rows only; NaN everywhere else (no gap rows)."""

    strict, _span = _universe_masks(store)
    return x.where(strict).fillna(value).where(strict)


def _op_smt_only(x: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Restrict a panel to SignalMasterTable rows (shrcd 10/11/12 & exchcd 1/2/3)."""

    strict, _span = _universe_masks(store)
    return x.where(strict)


def _crsp_rows_mask(store: "DatasetStore") -> pd.DataFrame:
    """Boolean month × permno mask of monthlyCRSP ROW presence (no shrcd/exchcd
    filter), cached per store — shared by crsp_only and the PS merge universe."""

    cache = _get_derived_cache(store)
    if "crsp_rows_mask" not in cache:
        raw = store.load_raw("monthlyCRSP.parquet").loc[:, ["permno", "time_avail_m"]]
        marker = raw.assign(present=1.0).pivot_table(
            index="time_avail_m", columns="permno", values="present", aggfunc="last"
        )
        cache["crsp_rows_mask"] = (
            marker.reindex(index=store.template_index, columns=store.template_columns).notna()
        )
    return cache["crsp_rows_mask"]


def _op_crsp_only(x: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Restrict a panel to months where a monthlyCRSP ROW exists (no shrcd/exchcd
    filter — for scripts that inner-join CRSP without the SMT universe filter)."""

    return x.where(_crsp_rows_mask(store))


def _op_smt_gvkey_only(x: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Restrict a panel to cells where SignalMasterTable has a row with NON-NULL
    gvkey (plain smt_only is insufficient: ~1M SMT rows carry a null gvkey, and
    the gvkey-merge scripts drop those rows before any lag arithmetic)."""

    cache = _get_derived_cache(store)
    if "smt_gvkey_rows_mask" not in cache:
        raw = (
            store.load_raw("SignalMasterTable.parquet")
            .loc[:, ["permno", "gvkey", "time_avail_m"]]
            .dropna(subset=["gvkey"])
        )
        marker = raw.assign(present=1.0).pivot_table(
            index="time_avail_m", columns="permno", values="present", aggfunc="last"
        )
        cache["smt_gvkey_rows_mask"] = (
            marker.reindex(index=store.template_index, columns=store.template_columns).notna()
        )
    return x.where(cache["smt_gvkey_rows_mask"])


def _compustat_rows_mask(store: "DatasetStore") -> pd.DataFrame:
    """Boolean month × permno mask of m_aCompustat ROW presence.

    Row-support twin of the crsp_rows mask (the 'm_aCompustat_rows' candidate
    mask of p1_validation._candidate_masks) — for scripts whose universe is the
    Compustat monthly panel rather than SignalMasterTable."""

    cache = _get_derived_cache(store)
    if "m_aCompustat_rows_mask" not in cache:
        raw = store.load_raw("m_aCompustat.parquet").loc[:, ["permno", "time_avail_m"]].dropna(subset=["permno"])
        marker = raw.assign(present=1.0).pivot_table(
            index="time_avail_m", columns="permno", values="present", aggfunc="last"
        )
        cache["m_aCompustat_rows_mask"] = (
            marker.reindex(index=store.template_index, columns=store.template_columns).notna()
        )
    return cache["m_aCompustat_rows_mask"]


def _op_compustat_rows_only(x: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Restrict a panel to months where an m_aCompustat row exists."""

    return x.where(_compustat_rows_mask(store))


def _op_compustat_span_fill(x: Any, value: float = 0.0, *, store: "DatasetStore", **_: Any) -> Any:
    """fillna(value) confined to each permno's m_aCompustat [first, last]
    coverage span (fill_date_gaps + fillna on the Compustat panel); months
    outside the span stay NaN so lags never read manufactured zeros."""

    rows = _compustat_rows_mask(store)
    span = rows.cummax(axis=0) & rows.iloc[::-1].cummax(axis=0).iloc[::-1]
    return x.fillna(value).where(span)


def _op_abs_cap_to_null(x: Any, max_abs: float = 1.0, **_: Any) -> Any:
    """Null values with |x| > max_abs (script 'drop if abs(x) > c' semantics;
    |x| == max_abs is kept, NaN stays NaN)."""

    return x.where(x.abs() <= max_abs)


def _op_span_only(x: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Restrict a panel to each permno's [first, last] SignalMasterTable month.

    Output-domain twin of span_fill: legacy scripts that gap-fill emit rows for
    every month inside the listing span (including gap months) but never before
    listing or after delisting — lag windows that still reach real data after a
    stock dies must not produce post-delisting values.
    """

    _strict, span = _universe_masks(store)
    return x.where(span)


def _op_require_any_lag(x: Any, source: Any, lags: list[int], *, store: "DatasetStore", **_: Any) -> Any:
    """Null x where NONE of source's given lags carries a real SMT-row value.

    Used where a legacy script zero-fills lag windows but its row support still
    requires at least one genuine observation among the lags.
    """

    strict, _span = _universe_masks(store)
    src = source.where(strict)
    any_valid: pd.DataFrame | None = None
    for lag in lags:
        v = src.shift(int(lag)).notna()
        any_valid = v if any_valid is None else (any_valid | v)
    return x.where(any_valid)


def _op_compound_return(x: Any, lags: list[int], fill_missing: Any | None = None, **_: Any) -> Any:
    shifted = [_op_lag(x, lag) for lag in lags]
    if isinstance(x, pd.DataFrame):
        working = [(frame.fillna(fill_missing) if fill_missing is not None else frame) for frame in shifted]
        out = pd.DataFrame(1.0, index=x.index, columns=x.columns, dtype="float64")
        for frame in working:
            out = out * (1.0 + frame)
        return out - 1.0
    concat = pd.concat(shifted, axis=1)
    if fill_missing is not None:
        concat = concat.fillna(fill_missing)
    return (1.0 + concat).prod(axis=1) - 1.0


def _op_fillna(x: Any, value: Any = None, **_: Any) -> Any:
    return x.fillna(value)


def _op_ffill(x: Any, **_: Any) -> Any:
    """Per-permno forward-fill along time on the monthly wide grid (the
    scripts' present-row ffill; deliberately carries across coverage gaps)."""

    return x.ffill()


def _op_max(a: Any, b: Any, **_: Any) -> Any:
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        a_df = a if isinstance(a, pd.DataFrame) else pd.DataFrame(a, index=b.index, columns=b.columns)
        b_df = b if isinstance(b, pd.DataFrame) else pd.DataFrame(b, index=a.index, columns=a.columns)
        return np.maximum(a_df, b_df)
    return max(a, b)


def _op_time_lagged_value(x: Any, months: int, **_: Any) -> Any:
    return _op_lag(x, months)


def _op_market_equity_matched_to_datadate(x: Any, datadate: Any, lag_months: int = 6, **_: Any) -> Any:
    """Market equity frozen at the fiscal-year-end month, carried forward.

    Legacy BM.py:51-62: me = mve.shift(6) is kept ONLY where the month 6 back
    equals the firm's datadate month, then forward-filled per permno without
    limit — the denominator is the market equity of the latest fiscal-year-end,
    not a rolling 6-month lag (the previous implementation here, which ignored
    datadate entirely). NaT datadates match nothing and stay NaN.
    """

    lagged = _op_lag(x, lag_months)
    # month-resolution comparison: the datadate cell must equal the month
    # lag_months before the row's month
    dd_month = datadate.to_numpy(dtype="datetime64[M]")
    target = (
        (x.index.to_period("M") - lag_months).to_timestamp().to_numpy(dtype="datetime64[M]")
    )[:, None]
    frozen = lagged.where(pd.DataFrame(dd_month == target, index=x.index, columns=x.columns))
    return frozen.ffill()


def _op_exchange_switch_indicator(x: Any, lookback_months: int = 12, **_: Any) -> Any:
    """1 = moved onto NYSE (exchcd 1) from any AMEX/NASDAQ month, or onto AMEX
    (2) from any NASDAQ month, within the prior `lookback_months` calendar
    months (NaN lags compare False); 0/1 wherever current exchcd is defined."""

    lags = [_op_lag(x, i) for i in range(1, lookback_months + 1)]
    any_amex_nasdaq = lags[0].isin([2.0, 3.0])
    any_nasdaq = lags[0].eq(3.0)
    for lagged in lags[1:]:
        any_amex_nasdaq |= lagged.isin([2.0, 3.0])
        any_nasdaq |= lagged.eq(3.0)
    out = (((x == 1) & any_amex_nasdaq) | ((x == 2) & any_nasdaq)).astype("float64")
    return out.where(x.notna())


def _op_rolling_mean(x: Any, window_months: int, **_: Any) -> Any:
    min_obs = _.get("min_obs")
    return x.rolling(window_months, min_periods=min_obs or max(1, window_months // 2)).mean()


def _op_rolling_std(x: Any, window_months: int, min_obs: int | None = None, **_: Any) -> Any:
    return x.rolling(window_months, min_periods=min_obs or max(1, window_months // 2)).std()


def _op_positive_only(x: Any, **_: Any) -> Any:
    return x.where(x > 0)


def _op_negative_to_null(x: Any, **_: Any) -> Any:
    return x.where(x >= 0)


def _op_zero_to_null(x: Any, **_: Any) -> Any:
    return x.where(x != 0)


def _op_nonzero(x: Any, **_: Any) -> Any:
    return x.where(x != 0)


def _op_positive_indicator(x: Any, **_: Any) -> Any:
    return (x > 0).where(x.notna()).astype("float32")


def _op_nonfinancial_only(x: Any, sic: Any, **_: Any) -> Any:
    sic_num = sic.apply(pd.to_numeric, errors="coerce") if isinstance(sic, pd.DataFrame) else pd.to_numeric(sic, errors="coerce")
    mask = (sic_num < 6000) | (sic_num >= 7000)
    return x.where(mask)


def _op_manufacturing_only(x: Any, sic: Any, **_: Any) -> Any:
    sic_num = sic.apply(pd.to_numeric, errors="coerce") if isinstance(sic, pd.DataFrame) else pd.to_numeric(sic, errors="coerce")
    mask = (sic_num >= 2000) & (sic_num <= 3999)
    return x.where(mask)


def _op_financial_excluded(x: Any, sic: Any, **_: Any) -> Any:
    """Exclusion-form financial screen: null only where 6000<=sic<7000. NaN sic
    is KEPT — unlike nonfinancial_only, which drops NaN (scripts using
    ~((sic>=6000)&(sic<7000)) keep NaN via NaN-comparison-False)."""

    sic_num = sic.apply(pd.to_numeric, errors="coerce") if isinstance(sic, pd.DataFrame) else pd.to_numeric(sic, errors="coerce")
    return x.where(~((sic_num >= 6000) & (sic_num < 7000)))


def _op_shrcd_le11(x: Any, shrcd: Any, **_: Any) -> Any:
    """Null x where shrcd > 11; null shrcd KEPT (script NaN>11==False semantics)."""

    return x.where(~(shrcd > 11))


def _op_bm_defined_gate(x: Any, ceq: Any, mve: Any, **_: Any) -> Any:
    """Null x where the script's log(ceq/mve) BM screen is undefined (NaN).

    ceq==0 -> BM=-inf is DEFINED (the script keeps those rows; a 0*LOG gate
    would turn 0*-inf into NaN and wrongly drop them); negative ratios and
    missing inputs give NaN and drop the row."""

    with np.errstate(divide="ignore", invalid="ignore"):
        bm = np.log(ceq / mve)
    return x.where(bm.notna())


def _op_deldrc_sample_filter(x: Any, drc: Any, ceq: Any, sale: Any, sic: Any, **_: Any) -> Any:
    """DelDRC exclusions, each leg exclusion-form (x.where(~cond)) so NaN
    ceq/sale/sic never excludes (script NaN-comparison-False semantics):
    ceq<=0 | (drc==0 & signal==0) | sale<5 | financial SIC 6000-6999."""

    sic_num = sic.apply(pd.to_numeric, errors="coerce") if isinstance(sic, pd.DataFrame) else pd.to_numeric(sic, errors="coerce")
    cond = (ceq <= 0) | ((drc == 0) & (x == 0)) | (sale < 5) | ((sic_num >= 6000) & (sic_num < 7000))
    return x.where(~cond)


def _op_convertible_debt_indicator(dc: Any, cshrc: Any, **_: Any) -> Any:
    """1.0 if dc OR cshrc is present and nonzero (script ne(0), not gt(0));
    0.0 wherever at least one field is observed (the script's unconditional 0
    default on every m_aCompustat row); NaN only off the data grid."""

    has_convertible = (dc.notna() & dc.ne(0)) | (cshrc.notna() & cshrc.ne(0))
    return has_convertible.astype("float32").where(dc.notna() | cshrc.notna())


def _op_enterprise_multiple(mve: Any, dltt: Any, dlc: Any, dc: Any, che: Any, oibdp: Any, ceq: Any, **_: Any) -> Any:
    value = (mve + dltt + dlc + dc - che) / oibdp
    # exclusion-form screen: unobserved ceq PASSES the negative-book-equity
    # screen (script NaN-comparison-False), only observed negatives drop
    return value.where(~((ceq < 0) | (oibdp < 0)))


def _op_size_filtered(
    x: Any,
    size: Any,
    min_tercile: int | None = 2,
    max_tercile: int | None = None,
    max_quintile: int | None = None,
    drop_missing_size: bool = False,
    **_: Any,
) -> Any:
    """Size screens via per-month pd.qcut-equivalent VALUE breakpoints.

    Exclusion-form masks: NaN-size rows are KEPT (the scripts' drop conditions
    compare against the bin, and NaN comparisons are False).
    drop_missing_size=True instead NULLS NaN-bin rows (RDcap.py's explicit
    'tempsizeq.isna() -> NaN' rule — off-universe rows have no size and drop)."""

    mask = pd.DataFrame(True, index=x.index, columns=x.columns)
    bins: pd.DataFrame | None = None
    if min_tercile is not None or max_tercile is not None:
        bins = _qcut_frame(size, 3)
        if min_tercile is not None:
            mask &= ~(bins < min_tercile)
        if max_tercile is not None:
            mask &= ~(bins > max_tercile)
    if max_quintile is not None:
        bins = _qcut_frame(size, 5)
        mask &= ~(bins > max_quintile)
    if drop_missing_size:
        if bins is None:
            raise ValueError("drop_missing_size requires at least one tercile/quintile screen")
        mask &= bins.notna()
    return x.where(mask)


def _op_hire_growth(emp: Any, *, store: "DatasetStore", **_: Any) -> Any:
    lag = _op_lag(emp, 12)
    value = (emp - lag) / (0.5 * (emp + lag))
    # zero-fill confined to m_aCompustat row support: the script's frame only
    # ever contains Compustat rows, so off-row grid cells must stay NaN
    return value.where(emp.notna() & lag.notna(), 0).where(_compustat_rows_mask(store))


def _op_netpayout_yield_signal(
    dvc: Any, prstkc: Any, sstk: Any, mve: Any, sic: Any, ceq: Any, *, store: "DatasetStore", **_: Any
) -> Any:
    """NetPayoutYield per NetPayoutYield.py: universe = SMT ∩ m_aCompustat rows;
    calendar 6-month mve_permco lag sourced from those rows only (no grid
    fill); rows whose computed signal is exactly 0 (all three components zero
    with an existing lag) leave the SAMPLE, while 0.0 from offsetting nonzero
    flows is kept; non-financial SIC (NaN sic dropped) and (ceq>0 | ceq
    missing) screens; finally a >=24 surviving-row per-permno positional gate
    counted over the screened rows INCLUDING NaN-signal rows."""

    strict, _span = _universe_masks(store)
    rows = strict & _compustat_rows_mask(store)
    me_lag6 = mve.where(rows).shift(6)
    value = (dvc + prstkc - sstk) / me_lag6
    zero_dropped = value.eq(0) & dvc.eq(0) & prstkc.eq(0) & sstk.eq(0)
    sic_num = sic.apply(pd.to_numeric, errors="coerce")
    sample = rows & ~zero_dropped & ((sic_num < 6000) | (sic_num >= 7000)) & ~(ceq <= 0)
    obs_count = sample.cumsum(axis=0)
    return value.where(sample & (obs_count >= 24))


def _op_payout_yield_signal(
    dvc: Any, prstkc: Any, pstkrv: Any, mve: Any, sic: Any, ceq: Any, *, store: "DatasetStore", **_: Any
) -> Any:
    """PayoutYield per PayoutYield.py: universe = SMT ∩ m_aCompustat rows;
    calendar 6-month mve_permco lag from those rows only; yield<=0 -> NaN but
    the row STAYS in the sample count; non-financial SIC (NaN sic dropped) and
    (ceq>0 | ceq missing) screens; >=24 surviving-row per-permno positional
    gate counted over the screened rows INCLUDING NaN-signal rows."""

    strict, _span = _universe_masks(store)
    rows = strict & _compustat_rows_mask(store)
    me_lag6 = mve.where(rows).shift(6)
    value = (dvc + prstkc + pstkrv) / me_lag6
    value = value.where(value > 0)
    sic_num = sic.apply(pd.to_numeric, errors="coerce")
    sample = rows & ((sic_num < 6000) | (sic_num >= 7000)) & ~(ceq <= 0)
    obs_count = sample.cumsum(axis=0)
    return value.where(sample & (obs_count >= 24))


def _op_investment_to_sales_scaled(capx: Any, revt: Any, window_months: int = 36, min_obs: int = 24, **_: Any) -> Any:
    """capx/revt over its trailing historical mean, replicating the reference's
    polars inf-PROPAGATING rolling mean: an inf ratio (revt==0 month) inside the
    window makes the mean ±inf (mixed signs -> NaN), so downstream value =
    ratio/inf emits the oracle's exact 0.0 blocks. The obs gate counts inf as an
    observation (notna), which is why min_periods lives on the count, not the
    mean. Deliberately NOT replicated: polars' ~1e-17 float noise on all-zero
    windows (true math 0/0 stays NaN here — registered oracle artifact)."""

    ratio = capx / revt
    cnt = ratio.notna().astype("float64").rolling(window_months, min_periods=1).sum()
    pos = ratio.eq(np.inf).astype("float64").rolling(window_months, min_periods=1).sum() > 0
    neg = ratio.eq(-np.inf).astype("float64").rolling(window_months, min_periods=1).sum() > 0
    hist = ratio.replace([np.inf, -np.inf], np.nan).rolling(window_months, min_periods=1).mean()
    hist = hist.mask(pos & ~neg, np.inf).mask(neg & ~pos, -np.inf).mask(pos & neg, np.nan)
    hist = hist.where(cnt >= min_obs)
    value = ratio / hist
    return value.where(revt >= 10)


def _op_coalesce(*args: Any, **_: Any) -> Any:
    if not args:
        raise ValueError("coalesce requires at least one argument")
    out = args[0].copy()
    for arg in args[1:]:
        out = out.where(out.notna(), arg)
    return out


def _op_year_filtered(x: Any, min_year: int | None = None, max_year: int | None = None, **_: Any) -> Any:
    out = x.copy()
    if min_year is not None:
        out = out.where(pd.Series(pd.Index(out.index).year >= min_year, index=out.index), axis=0)
    if max_year is not None:
        out = out.where(pd.Series(pd.Index(out.index).year <= max_year, index=out.index), axis=0)
    return out


def _op_threshold_min(x: Any, gate: Any, min_value: float, **_: Any) -> Any:
    return x.where(gate >= min_value)


def _op_size_decile_filtered(x: Any, size: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """GrAdExp size screen: per-month qcut-equivalent decile breakpoints over
    the m_aCompustat-row population (the script's merged panel); null only
    where decile==1 — NaN size/decile PASSES (exclusion-form)."""

    decile = _qcut_frame(size.where(_compustat_rows_mask(store)), 10)
    return x.where(~(decile == 1))


def _op_industry_adjusted_mean(x: Any, sic: Any, count_basis: Any = None, min_industry_obs: int = 1, **_: Any) -> Any:
    """Demean x within 2-digit SIC industries per month. The >= min_industry_obs
    gate counts valid-x members by default; when count_basis is given it counts
    that panel's non-null cells instead (realestate's tempN = count of non-null
    'at' per (sic2, month), taken BEFORE any x-validity filtering — script L51),
    nulling ALL members of a failing group."""

    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype="float64")
    for date in x.index:
        vals = x.loc[date]
        sic_row = sic.loc[date].astype("string").str.slice(0, 2)
        # inf ratios (e.g. zero-base growth) must not poison the industry mean —
        # the reference drops non-finite values before grouping
        valid = vals.notna() & np.isfinite(vals) & sic_row.notna()
        if not valid.any():
            continue
        grouped_mean = vals[valid].groupby(sic_row[valid]).transform("mean")
        adjusted = vals[valid] - grouped_mean
        if count_basis is None:
            grouped_count = vals[valid].groupby(sic_row[valid]).transform("count")
            adjusted = adjusted.where(grouped_count >= min_industry_obs)
        else:
            basis_valid = count_basis.loc[date].notna() & sic_row.notna()
            group_n = basis_valid.astype("float64").groupby(sic_row).sum()
            adjusted = adjusted.where(sic_row[valid].map(group_n) >= min_industry_obs)
        out.loc[date, adjusted.index] = adjusted
    return out


def _monthly_group_index(x: pd.DataFrame | pd.Series) -> pd.Index:
    return pd.to_datetime(x.index).to_period("M").to_timestamp()


def _op_monthly_mean(x: Any, **_: Any) -> Any:
    return x.groupby(_monthly_group_index(x)).mean()


def _op_monthly_max(x: Any, **_: Any) -> Any:
    return x.groupby(_monthly_group_index(x)).max()


def _op_monthly_skew(x: Any, min_obs: int = 15, **_: Any) -> Any:
    grouped = x.groupby(_monthly_group_index(x))
    skew = grouped.skew()
    counts = grouped.count()
    return skew.where(counts >= min_obs)


def _op_high_52(prc: Any, **_: Any) -> Any:
    monthly_max = _op_monthly_max(np.abs(prc))
    monthly_last = np.abs(prc).groupby(_monthly_group_index(prc)).last()
    rolling_high = monthly_max.shift(1).rolling(12, min_periods=1).max()
    return monthly_last / rolling_high


def _op_firm_age(x: Any, require_compustat_row: bool = False, *, store: "DatasetStore", **_: Any) -> Any:
    """Cumulative SignalMasterTable row count per permno (== the script's
    per-permno cumcount+1; a row COUNTER has no positional/calendar issue),
    nulled where age equals tempcrsptime = round(days-since-1926-07/30.44)+1
    — the script's censor for firms alive since CRSP's start. The x argument
    is vestigial (dependency tracking only).

    require_compustat_row=True counts rows of the SMT ∩ m_aCompustat MERGED
    panel (the grcapx family's SMT-inner-merge frame) — counting SMT rows alone
    overstates age where Compustat coverage starts after listing."""

    _ = x
    strict, _span = _universe_masks(store)
    basis = (strict & _compustat_rows_mask(store)) if require_compustat_row else strict
    age = basis.cumsum(axis=0).astype("float64").where(basis)
    days = (basis.index - pd.Timestamp("1926-07-01")).days.to_numpy()
    crsp_time = pd.Series(np.round(days / 30.44).astype("int64") + 1, index=basis.index)
    return age.where(~age.eq(crsp_time, axis=0))


def _qcut_frame(frame: pd.DataFrame, q: int) -> pd.DataFrame:
    """Row-wise (per-month) pd.qcut bins 1..q (labels=False, duplicates='drop').

    The reference scripts split samples with VALUE breakpoints (pd.qcut /
    fastxtile), not percentile ranks — percentile-rank bucketing misplaces
    boundary firms. NaN inputs get NaN bins."""

    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype="float64")
    for date in frame.index:
        row = frame.loc[date]
        valid = row.notna()
        if not valid.any():
            continue
        bins = pd.qcut(row[valid], q=q, labels=False, duplicates="drop") + 1
        out.loc[date, row.index[valid]] = bins.astype("float64")
    return out


def _row_qcut(frame: pd.DataFrame, q: int) -> pd.DataFrame:
    """Per-month pd.qcut(labels=False, duplicates='drop') + 1, leaving the row
    NaN when the pool is empty or qcut fails on a degenerate all-equal pool
    (occurs only on grid months outside the source scripts' frames)."""

    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype="float64")
    for date in frame.index:
        row = frame.loc[date]
        valid = row.notna()
        if not valid.any():
            continue
        try:
            bins = pd.qcut(row[valid], q=q, labels=False, duplicates="drop") + 1
        except ValueError:
            continue
        out.loc[date, row.index[valid]] = bins.astype("float64")
    return out


def _fastxtile5(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-wise quintile bins replicating the reference fastxtile exactly:
    inf -> NaN; <5 valid obs -> all bin 1; exactly 2 unique values -> bins
    {1, 5}; else pandas linear-interpolation quantile cutpoints at i/5 with
    strictly-greater assignment (bin = 1 + #cutpoints strictly below value)."""

    clean = frame.replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(np.nan, index=frame.index, columns=frame.columns, dtype="float64")
    for date in clean.index:
        row = clean.loc[date]
        valid = row.notna()
        n = int(valid.sum())
        if n == 0:
            continue
        cols = row.index[valid]
        vals = row[valid].to_numpy(dtype="float64")
        if n < 5:
            out.loc[date, cols] = 1.0
            continue
        uniq = np.unique(vals)
        if uniq.size == 2:
            out.loc[date, cols] = np.where(vals == uniq[1], 5.0, 1.0)
            continue
        cuts = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])  # linear interpolation
        out.loc[date, cols] = 1.0 + (vals[:, None] > cuts[None, :]).sum(axis=1)
    return out


def _op_monthly_fastxtile_exclude(x: Any, key: Any, n: int = 5, max_excluded: int = 2, **_: Any) -> Any:
    """Null x where key's per-month Stata-fastxtile bucket is <= max_excluded.

    Bucket algorithm mirrors the reference utils/stata_fastxtile._fastxtile_core
    exactly: ±inf and |v| >= 1e100 -> NaN key; no valid keys -> all-NaN buckets;
    < n valid -> every valid cell bucket 1; one unique value -> bucket 1; two
    unique values -> min 1 / max n; else pandas linear-interpolation quantile
    cutpoints at i/n with strictly-greater promotion (ties stay in the lower
    bucket), clipped to [1, n]. Cells with a NaN bucket KEEP x — the script's
    'tempsort <= 2' test is False for NaN."""

    clean = key.replace([np.inf, -np.inf], np.nan)
    clean = clean.where(clean.abs() < 1e100)
    bucket = pd.DataFrame(np.nan, index=key.index, columns=key.columns, dtype="float64")
    for date in clean.index:
        row = clean.loc[date]
        valid = row.notna()
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        cols = row.index[valid]
        vals = row[valid].to_numpy(dtype="float64")
        if n_valid < n:
            bucket.loc[date, cols] = 1.0
            continue
        uniq = np.unique(vals)
        if uniq.size == 1:
            bucket.loc[date, cols] = 1.0
        elif uniq.size == 2:
            bucket.loc[date, cols] = np.where(vals == uniq[1], float(n), 1.0)
        else:
            cuts = pd.Series(vals).quantile([i / n for i in range(1, n)]).to_numpy()
            cats = 1.0 + (vals[:, None] > cuts[None, :]).sum(axis=1)
            bucket.loc[date, cols] = np.clip(cats, 1, n)
    return x.where(~(bucket <= max_excluded))


def _op_gr_ltnoa(rect: Any, invt: Any, ppent: Any, aco: Any, intan: Any, ao: Any, ap: Any, lco: Any, lo: Any, at: Any, dp: Any, **_: Any) -> Any:
    current_ltnoa = (rect + invt + ppent + aco + intan + ao - ap - lco - lo) / at
    lagged_ltnoa = (_op_lag(rect, 12) + _op_lag(invt, 12) + _op_lag(ppent, 12) + _op_lag(aco, 12) + _op_lag(intan, 12) + _op_lag(ao, 12) - _op_lag(ap, 12) - _op_lag(lco, 12) - _op_lag(lo, 12)) / _op_lag(at, 12)
    wc_adjustment = (
        (rect - _op_lag(rect, 12))
        + (invt - _op_lag(invt, 12))
        + (aco - _op_lag(aco, 12))
        - ((ap - _op_lag(ap, 12)) + (lco - _op_lag(lco, 12)))
        - dp
    ) / ((at + _op_lag(at, 12)) / 2.0)
    return current_ltnoa - lagged_ltnoa - wc_adjustment


def _op_firm_age_momentum(ret: Any, prc: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """FirmAgeMom per script semantics (FirmAgeMom.py):

    age = cumulative SignalMasterTable row count per permno (not monthlyCRSP
    prc presence); keep rows with (|prc|>=5 or prc missing) & age>=12; Mom6m
    compounds ret zero-filled on SMT rows, each calendar lag NULL when its
    month is not itself a kept row (the script lags over the FILTERED panel);
    age quintile 1 via per-month qcut-equivalent breakpoints over the kept
    cross-section."""

    strict, _span = _universe_masks(store)
    age = strict.cumsum(axis=0).where(strict)
    price_ok = prc.abs().ge(5) | prc.isna()
    keep = strict & price_ok & age.ge(12)
    ret_src = ret.where(strict).fillna(0.0).where(keep)
    mom6 = _op_compound_return(ret_src, [1, 2, 3, 4, 5])
    age_quintile = _qcut_frame(age.where(keep), 5)
    return mom6.where(keep & (age_quintile == 1))


def _op_industry_weighted_momentum(ret: Any, mve_c: Any, sic: Any, **_: Any) -> Any:
    mom6 = _op_compound_return(ret, [1, 2, 3, 4, 5])
    out = pd.DataFrame(np.nan, index=mom6.index, columns=mom6.columns, dtype="float64")
    sic2 = sic.astype("string").apply(lambda col: col.str.slice(0, 2))
    for date in mom6.index:
        vals = mom6.loc[date]
        weights = mve_c.loc[date]
        groups = sic2.loc[date]
        valid = vals.notna() & weights.notna() & (weights > 0) & groups.notna()
        if not valid.any():
            continue
        num = (vals[valid] * weights[valid]).groupby(groups[valid]).sum()
        den = weights[valid].groupby(groups[valid]).sum()
        means = num / den
        # the industry mean is assigned to EVERY stock in the industry — including
        # stocks whose own momentum is undefined (they receive, but don't contribute)
        recipients = groups.notna()
        out.loc[date, recipients.index[recipients]] = groups[recipients].map(means)
    return out


def _op_weighted_lagged_rank_growth(revt: Any, **_: Any) -> Any:
    valid = (revt > 0) & (_op_lag(revt, 12) > 0)
    growth = (np.log(revt.where(valid)) - np.log(_op_lag(revt, 12).where(valid)))
    ranks = growth.rank(axis=1, method="first", ascending=False)
    weighted = (
        5 * _op_lag(ranks, 12)
        + 4 * _op_lag(ranks, 24)
        + 3 * _op_lag(ranks, 36)
        + 2 * _op_lag(ranks, 48)
        + _op_lag(ranks, 60)
    ) / 15.0
    return weighted


def _op_momentum_reversal_signal(ret: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """1 if (Mom6m quintile 5, Mom36m quintile 1), 0 if the reverse: returns
    zero-filled on SMT rows, momenta on the listing span (asrol gap rows), and
    per-month pd.qcut VALUE breakpoints (inf kept — the script's *_clean
    columns are dead code)."""

    strict, _span = _universe_masks(store)
    r = ret.where(strict).fillna(0.0).where(strict)
    mom6 = _op_compound_return(r, [1, 2, 3, 4, 5])
    mom36 = _op_compound_return(r, list(range(13, 37)))
    span = strict.cummax(axis=0) & strict.iloc[::-1].cummax(axis=0).iloc[::-1]
    q6 = _row_qcut(mom6.where(span), 5)
    q36 = _row_qcut(mom36.where(span), 5)
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype="float64")
    out[(q6 == 5) & (q36 == 1)] = 1.0
    out[(q6 == 1) & (q36 == 5)] = 0.0
    return out


def _op_momentum_volume_signal(ret: Any, vol: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Mom6m decile kept where 6m-average volume is in tercile 3: SMT-row
    zero-filled returns, vol<0 nulled (0 kept), span-masked inputs to per-month
    pd.qcut breakpoints, and a >=24-calendar-month history gate counted from
    each permno's FIRST SMT month (the script's obs_num over gap-filled rows)."""

    strict, _span = _universe_masks(store)
    r = ret.where(strict).fillna(0.0).where(strict)
    v = vol.where(strict)
    v = v.where(v >= 0)  # script nulls vol<0, keeps 0
    mom6 = _op_compound_return(r, [1, 2, 3, 4, 5])
    temp = v.rolling(6, min_periods=5).mean()
    span = strict.cummax(axis=0) & strict.iloc[::-1].cummax(axis=0).iloc[::-1]
    cat_mom = _row_qcut(mom6.where(span), 10)
    cat_vol = _row_qcut(temp.where(span), 3)
    out = cat_mom.where(cat_vol == 3)
    # calendar months since the column's first SMT month; columns with no SMT
    # row get -inf so the >= 23 gate always fails
    strict_np = strict.to_numpy()
    has_any = strict_np.any(axis=0)
    first_pos = strict_np.argmax(axis=0).astype("float64")
    obs = np.arange(len(strict.index), dtype="float64")[:, None] - first_pos[None, :]
    obs = np.where(has_any[None, :], obs, -np.inf)
    obs_ok = pd.DataFrame(obs >= 23, index=strict.index, columns=strict.columns)
    return out.where(obs_ok)


def _op_share_volume_signal(vol: Any, shrout: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """ShareVol on SMT rows: 0 if 3m average turnover < 5, 1 if > 10 OR the
    turnover is missing (the script's null tempShareVol -> 1 branch); rows
    within 2 months of a share-count CHANGE are dropped — a permno's first SMT
    month never counts as a change (NaN lag compares False), and the exclusion
    skips each stock's first two SMT observations (obs_num < 2, counted over
    SMT rows, not months since first CRSP observation)."""

    strict, _span = _universe_masks(store)
    avg_turnover = ((vol + _op_lag(vol, 1) + _op_lag(vol, 2)) / (3.0 * shrout)) * 100.0
    l1_shrout = _op_lag(shrout, 1)
    dshrout = shrout.ne(l1_shrout) & l1_shrout.notna() & shrout.notna()
    d_f = dshrout.astype("float64")
    drop_obs = dshrout | _op_lag(d_f, 1).eq(1.0) | _op_lag(d_f, 2).eq(1.0)
    obs_num = strict.cumsum(axis=0) - 1
    keep = ~drop_obs | (obs_num < 2)
    out = pd.DataFrame(np.nan, index=vol.index, columns=vol.columns, dtype="float64")
    out[avg_turnover < 5] = 0.0
    out[avg_turnover > 10] = 1.0
    out[avg_turnover.isna()] = 1.0
    return out.where(keep & strict)


def _op_accruals_bm_signal(bm: Any, accruals: Any, ceq: Any, **_: Any) -> Any:
    # fastxtile-equivalent value breakpoints (see _fastxtile5) — percentile-rank
    # buckets misplace boundary firms vs the reference script's quintiles
    bm_q = _fastxtile5(bm)
    acc_q = _fastxtile5(accruals)
    out = pd.DataFrame(np.nan, index=bm.index, columns=bm.columns, dtype="float64")
    out[(bm_q == 5) & (acc_q == 1)] = 1.0
    out[(bm_q == 1) & (acc_q == 5)] = 0.0
    return out.where(ceq >= 0)


def _op_rolling_industry_herfindahl(
    metric: Any,
    sic: Any,
    shrcd: Any,
    window_months: int = 36,
    min_obs: int = 12,
    min_year: int | None = None,
    require_compustat_row: bool = False,
    *,
    store: "DatasetStore",
    **_: Any,
) -> Any:
    """Rolling mean of monthly 4-digit-sicCRSP sales(assets/BE)-share
    Herfindahls on the SignalMasterTable universe (optionally ∩ m_aCompustat
    rows), with asrol-style gap rows kept inside the span, industry sums NOT
    zero-guarded (x/0 -> inf like the script), zero-contributor industries
    receiving 0.0, and the regulated-industry masks fired on universe rows
    only; min_year applies ungated to every row (script computes year
    post-asrol)."""

    univ, _span = _universe_masks(store)
    if require_compustat_row:
        # inner-merge universe: the (permno, month) row must also exist in m_aCompustat
        univ = univ & _compustat_rows_mask(store)
    sic4 = sic.astype("string").apply(lambda col: col.str.slice(0, 4)).where(univ)
    met = metric.where(univ)

    temp_herf = pd.DataFrame(np.nan, index=met.index, columns=met.columns, dtype="float64")
    for date in met.index:
        met_row = met.loc[date]
        sic_row = sic4.loc[date]
        contrib = met_row.notna() & sic_row.notna()
        recipients = sic_row.notna()
        if not recipients.any():
            continue
        # NO replace(0, nan): the script divides by the raw transform sum, so a
        # zero industry sum must yield inf/NaN via plain division
        sums = met_row[contrib].groupby(sic_row[contrib]).sum()
        shares_sq = ((met_row[contrib] / sic_row[contrib].map(sums)) ** 2).groupby(sic_row[contrib]).sum()
        # every universe row with a SIC receives; zero-contributor industries get
        # 0.0 (pandas transform('sum') == 0.0 for empty groups in the script)
        temp_herf.loc[date, recipients.index[recipients]] = (
            sic_row[recipients].map(shares_sq).fillna(0.0).astype("float64")
        )
    out = temp_herf.rolling(window_months, min_periods=min_obs).mean()

    # asrol's fill_date_gaps emits gap rows inside the universe span
    span_univ = univ.cummax(axis=0) & univ.iloc[::-1].cummax(axis=0).iloc[::-1]
    out = out.where(span_univ)

    # exclusion masks gated on universe rows ONLY (gap rows keep values)
    out = out.mask(univ & shrcd.gt(11))
    years = out.index.year
    mask_1980 = years <= 1980
    out.loc[mask_1980] = out.loc[mask_1980].mask(
        univ.loc[mask_1980] & sic4.loc[mask_1980].isin(["4011", "4210", "4213"])
    )
    mask_1978 = years <= 1978
    out.loc[mask_1978] = out.loc[mask_1978].mask(
        univ.loc[mask_1978] & sic4.loc[mask_1978].eq("4512").fillna(False).astype(bool)
    )
    mask_1982 = years <= 1982
    out.loc[mask_1982] = out.loc[mask_1982].mask(
        univ.loc[mask_1982] & sic4.loc[mask_1982].isin(["4812", "4813"])
    )
    sic2_utilities = sic4.apply(lambda col: col.str.slice(0, 2)).eq("49").fillna(False).astype(bool)
    out = out.mask(univ & sic2_utilities)

    if min_year is not None:
        out = out.where(pd.Series(years >= min_year, index=out.index), axis=0)
    return out


def _op_earnings_consistency(epspx: Any, **_: Any) -> Any:
    l12 = _op_lag(epspx, 12)
    l24 = _op_lag(epspx, 24)
    egrowth = (epspx - l12) / (0.5 * (l12.abs() + l24.abs()))
    egrowth = egrowth.replace([np.inf, -np.inf], np.nan)
    growth_terms = [egrowth, _op_lag(egrowth, 12), _op_lag(egrowth, 24), _op_lag(egrowth, 36), _op_lag(egrowth, 48)]
    sum_terms = growth_terms[0].fillna(0)
    count_terms = growth_terms[0].notna().astype("float64")
    for term in growth_terms[1:]:
        sum_terms = sum_terms + term.fillna(0)
        count_terms = count_terms + term.notna().astype("float64")
    out = sum_terms / count_terms.where(count_terms > 0)
    l12_eg = _op_lag(egrowth, 12)
    exception = (
        epspx.isna()
        | l12.isna()
        | (epspx / l12).abs().gt(6)
        | ((egrowth > 0) & (l12_eg < 0) & egrowth.notna())
        | ((egrowth < 0) & ((l12_eg > 0) | l12_eg.isna()) & egrowth.notna())
    )
    return out.where(~exception)


def _op_seasonal_surprise_zscore(x: Any, min_sd: float | None = None, **_: Any) -> Any:
    growth = x - _op_lag(x, 12)
    drift_terms = [_op_lag(growth, lag) for lag in range(3, 25, 3)]
    drift_sum = drift_terms[0].fillna(0)
    drift_count = drift_terms[0].notna().astype("float64")
    for term in drift_terms[1:]:
        drift_sum = drift_sum + term.fillna(0)
        drift_count = drift_count + term.notna().astype("float64")
    drift = drift_sum / drift_count.where(drift_count > 0)
    surprise = growth - drift

    hist_terms = [_op_lag(surprise, lag) for lag in range(3, 25, 3)]
    hist_sum = hist_terms[0].fillna(0)
    hist_count = hist_terms[0].notna().astype("float64")
    for term in hist_terms[1:]:
        hist_sum = hist_sum + term.fillna(0)
        hist_count = hist_count + term.notna().astype("float64")
    # two-pass variance (pandas nanvar): the one-pass (sumsq - sum^2/n)/(n-1)
    # cancels catastrophically at near-constant histories (true SD ~1e-17 came
    # out ~1e-9), defeating the scripts' sd>threshold guard
    hist_mean = hist_sum / hist_count.where(hist_count > 0)
    sq_dev_sum = None
    for term in hist_terms:
        sq_dev = (term - hist_mean) ** 2
        # missing terms contribute 0; non-finite terms poison the sum (nanvar semantics)
        sq_dev = sq_dev.where(term.notna(), 0.0)
        sq_dev_sum = sq_dev if sq_dev_sum is None else sq_dev_sum + sq_dev
    variance = sq_dev_sum / (hist_count - 1).where(hist_count > 1)
    sd = np.sqrt(variance.where(variance >= 0))
    # min_sd=None keeps the sd>1e-10 guard (EarningsSurprise); EarnSupBig passes
    # min_sd=1e-8, replicating np.where(SD==0 | isna | |SD|<1e-8, NaN, ES/SD)
    valid = (sd > 1e-10) if min_sd is None else (sd >= min_sd)
    # scripts replace +/-inf z with NaN before emitting (e.g. RevenueSurprise revps=x/0)
    return (surprise / sd).where(valid).replace([np.inf, -np.inf], np.nan)


def _op_earnings_increase_streak(ibq: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Streak of consecutive quarterly earnings increases on the script's exact
    merged panel: SMT rows with a gvkey inner-joined to m_QCompustat ibq. The
    incoming ibq arg is vestigial (dependency tracking only) — chearn and every
    calendar lag are evaluated on the U-masked wide panel, so a (permno, t-12)
    pair absent from the merged panel lags to NaN exactly like the script's
    self-merge (rows present with NaN ibq stay IN the panel but carry NaN)."""

    _ = ibq
    smt = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "gvkey", "time_avail_m"]]
        .dropna(subset=["gvkey"])
    )
    q = store.load_raw("m_QCompustat.parquet").loc[:, ["gvkey", "time_avail_m", "ibq"]]
    merged = smt.merge(q, on=["gvkey", "time_avail_m"], how="inner")
    dedup = merged.drop_duplicates(["time_avail_m", "permno"], keep="first")
    # set_index + unstack, NOT pivot_table: NaN-ibq rows must stay in U while
    # remaining NaN in the value panel
    ibq_m = (
        dedup.set_index(["time_avail_m", "permno"])["ibq"]
        .unstack()
        .reindex(index=store.template_index, columns=store.template_columns)
    )
    marker = (
        dedup.assign(present=1.0)
        .set_index(["time_avail_m", "permno"])["present"]
        .unstack()
        .reindex(index=store.template_index, columns=store.template_columns)
    )
    u_mask = marker.notna()

    chearn = (ibq_m - ibq_m.shift(12)).where(u_mask)
    positive_or_missing = {lag: (_op_lag(chearn, lag).gt(0) | _op_lag(chearn, lag).isna()) for lag in range(0, 25, 3)}
    out = pd.DataFrame(0.0, index=chearn.index, columns=chearn.columns, dtype="float64")
    for streak in range(1, 9):
        required_lags = [3 * i for i in range(streak)]
        cond = positive_or_missing[required_lags[0]].copy()
        for lag in required_lags[1:]:
            cond &= positive_or_missing[lag]
        cond &= _op_lag(chearn, 3 * streak).le(0)
        out[cond] = float(streak)
    return out.where(u_mask)


def _op_surprise_rd_indicator(xrd: Any, revt: Any, at: Any, **_: Any) -> Any:
    xrd_l12 = _op_lag(xrd, 12)
    at_l12 = _op_lag(at, 12)
    observed = xrd.notna() & xrd_l12.notna()
    signal = (
        ((xrd / revt) > 0)
        & ((xrd / at) > 0)
        & ((xrd / xrd_l12) > 1.05)
        & (((xrd / at) / (xrd_l12 / at_l12)) > 1.05)
        & observed
    )
    out = pd.DataFrame(np.nan, index=xrd.index, columns=xrd.columns, dtype="float64")
    out[signal] = 1.0
    out[observed & ~signal] = 0.0
    return out


def _op_industry_big_mean(x: Any, mve_c: Any, sic: Any, **_: Any) -> Any:
    """Mean value of each FF48 industry's big firms (size rank >= 0.7 within
    industry, NaN-skipping mean) mapped to ALL industry-classified firms —
    recipients need neither size nor an own value, and only big firms are
    nulled (NaN-rank firms still receive), per the script's relrank/df_big
    semantics."""

    ff48_func = _get_ff48_func()
    ff48 = sic.apply(lambda col: col.map(ff48_func))
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype="float64")
    for date in x.index:
        values = x.loc[date]
        size = mve_c.loc[date]
        industry = ff48.loc[date]
        rank_valid = size.notna() & industry.notna()
        if not rank_valid.any():
            continue
        group_rank = size[rank_valid].groupby(industry[rank_valid]).rank(pct=True, method="average")
        big_idx = group_rank.index[group_rank >= 0.7]
        industry_mean = values[big_idx].groupby(industry[big_idx]).mean()
        recipients = industry.notna()
        mapped = industry[recipients].map(industry_mean)
        mapped[mapped.index.isin(big_idx)] = np.nan
        out.loc[date, mapped.index] = mapped
    return out


def _trim_by_month(frame: pd.DataFrame, lower_q: float, upper_q: float) -> pd.DataFrame:
    """Null values outside per-MONTH cross-sectional quantiles.

    Point-in-time replacement for full-sample trimming (registered deviation):
    the reference computes trim cutoffs over the entire panel, so whether a 1985
    value survives depends on 2020 data. Cutoffs here use only the value's own
    month — same outlier-removal intent, no future information.
    """

    lo = frame.quantile(lower_q, axis=1)
    hi = frame.quantile(upper_q, axis=1)
    return frame.where(frame.ge(lo, axis=0) & frame.le(hi, axis=0))


def _trim_global(frame: pd.DataFrame, lower_q: float, upper_q: float, method: str = "linear") -> pd.DataFrame:
    values = frame.to_numpy(dtype="float64", copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        return frame.copy()
    lo = np.quantile(values[finite], lower_q, method=method)
    hi = np.quantile(values[finite], upper_q, method=method)
    trimmed = frame.copy()
    trimmed[(trimmed < lo) | (trimmed > hi)] = np.nan
    return trimmed


def _op_trim_global(x: Any, lower_q: float = 0.01, upper_q: float = 0.99, **_: Any) -> Any:
    """Null cells outside FULL-SAMPLE quantile bounds (reference winsor2 with
    by=None and trim; interpolation='nearest' matches the polars quantile
    default). Deliberately reproduces the reference's look-ahead trim — see the
    VolumeTrend entries in p1_deviation_register.md."""

    return _trim_global(x, lower_q, upper_q, method="nearest")


def _cross_sectional_residual(
    y: pd.DataFrame,
    regressors: list[pd.DataFrame],
    *,
    min_obs: int,
) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns, dtype="float64")
    for date in y.index:
        y_row = y.loc[date]
        x_rows = [reg.loc[date] for reg in regressors]
        valid = y_row.notna()
        for x_row in x_rows:
            valid &= x_row.notna()
        if valid.sum() < min_obs:
            continue
        yv = y_row[valid].to_numpy(dtype="float64")
        X = np.column_stack([np.ones(valid.sum())] + [x_row[valid].to_numpy(dtype="float64") for x_row in x_rows])
        try:
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            resid = yv - X @ beta
        except np.linalg.LinAlgError:
            continue
        out.loc[date, valid.index[valid]] = resid
    return out


def _get_derived_cache(store: DatasetStore) -> dict[str, Any]:
    """Return a per-store cache for expensive derived long-form tables."""

    cache = getattr(store, "_derived_cache", None)
    if cache is None:
        cache = {}
        setattr(store, "_derived_cache", cache)
    return cache


def _polars_nearest_quantile(values: pd.Series | np.ndarray, q: float) -> float:
    """polars Series.quantile(q, interpolation='nearest') for float members.

    Callers pass the polars-NON-NULL members only; NaN entries are the 0/0
    division results, which polars keeps as members ranking ABOVE +inf
    (np.sort places NaN last, matching polars' total order). The picked index
    is round-half-AWAY-from-zero of q*(n-1) — Rust f64::round, not numpy's
    half-to-even."""

    arr = np.sort(np.asarray(values, dtype="float64"))
    if arr.size == 0:
        return float("nan")
    return float(arr[int(np.floor(q * (arr.size - 1) + 0.5))])


def _trim_by_group(
    df: pd.DataFrame,
    value_cols: list[str],
    by_col: str,
    low_pct: float,
    high_pct: float,
    interpolation: str = "linear",
) -> pd.DataFrame:
    """Trim extreme values within groups by setting them to missing.

    interpolation='polars_nearest' replicates the reference winsor2's polars
    quantile (round-half-AWAY-from-zero pick index); pandas 'nearest' rounds
    half-to-EVEN, keeping/dropping a different boundary observation whenever
    q*(n-1) lands exactly on x.5. 'linear' keeps legacy callers byte-identical."""

    out = df.copy()
    for col in value_cols:
        if interpolation == "polars_nearest":
            q_low = out.groupby(by_col)[col].transform(
                lambda x: _polars_nearest_quantile(x.dropna(), low_pct)
            )
            q_high = out.groupby(by_col)[col].transform(
                lambda x: _polars_nearest_quantile(x.dropna(), high_pct)
            )
        else:
            q_low = out.groupby(by_col)[col].transform(lambda x: x.quantile(low_pct, interpolation=interpolation))
            q_high = out.groupby(by_col)[col].transform(lambda x: x.quantile(high_pct, interpolation=interpolation))
        out[col] = out[col].where((out[col] >= q_low) & (out[col] <= q_high))
    return out


def _ols_residuals_by_groups(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    group_cols: str | list[str],
    min_obs: int,
    dropna_groups: bool = True,
) -> pd.Series:
    """Run grouped OLS and return residuals aligned to the original dataframe.

    dropna_groups=False keeps NaN group keys as their OWN group (polars .over
    semantics — e.g. null sic2 is a real regression group in the reference)."""

    residuals = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, group in df.groupby(group_cols, sort=False, dropna=dropna_groups):
        fit = group[[y_col] + x_cols].dropna()
        if len(fit) < min_obs:
            continue
        X = np.column_stack(
            [np.ones(len(fit), dtype="float64")]
            + [fit[col].to_numpy(dtype="float64") for col in x_cols]
        )
        y = fit[y_col].to_numpy(dtype="float64")
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals.loc[fit.index] = y - (X @ beta)
    return residuals


def _ols_predictions_by_group(
    df: pd.DataFrame,
    y_col: str,
    fit_x_cols: list[str],
    pred_x_cols: list[str],
    group_col: str,
    min_obs: int,
) -> pd.Series:
    """Fit grouped OLS models and score each group on a second regressor set.

    min_obs=0 fits every group like polars_ols null_policy='drop': an empty or
    rank-deficient sample takes the lstsq minimum-norm solution (zeros when
    empty) instead of skipping the group."""

    predictions = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, group in df.groupby(group_col, sort=False):
        fit = group[[y_col] + fit_x_cols].dropna()
        if len(fit) < min_obs:
            continue
        X = np.column_stack(
            [np.ones(len(fit), dtype="float64")]
            + [fit[col].to_numpy(dtype="float64") for col in fit_x_cols]
        )
        y = fit[y_col].to_numpy(dtype="float64")
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = group[pred_x_cols].dropna()
        if pred.empty:
            continue
        X_pred = np.column_stack(
            [np.ones(len(pred), dtype="float64")]
            + [pred[col].to_numpy(dtype="float64") for col in pred_x_cols]
        )
        predictions.loc[pred.index] = X_pred @ beta
    return predictions


def _prepare_analyst_value_signals(store: DatasetStore) -> pd.DataFrame:
    """Build the shared analyst-valuation panel used by three factors."""

    cache = _get_derived_cache(store)
    if "analyst_value_signals" in cache:
        return cache["analyst_value_signals"]

    ibes = store.load_raw("IBES_EPS_Unadj.parquet").copy()
    froe1 = ibes[
        (ibes["fpi"] == "1")
        & (ibes["statpers"].dt.month == 5)
        & ibes["fpedats"].notna()
        & (ibes["fpedats"] > ibes["statpers"] + pd.Timedelta(days=30))
    ][["tickerIBES", "time_avail_m", "meanest"]].rename(columns={"meanest": "feps1"})
    froe1["time_avail_m"] = froe1["time_avail_m"] + pd.DateOffset(months=1)

    froe2 = ibes[(ibes["fpi"] == "2") & (ibes["statpers"].dt.month == 5)][
        ["tickerIBES", "time_avail_m", "meanest"]
    ].rename(columns={"meanest": "feps2"})
    froe2["time_avail_m"] = froe2["time_avail_m"] + pd.DateOffset(months=1)

    ltg = ibes[ibes["fpi"] == "0"][["tickerIBES", "time_avail_m", "meanest"]].rename(
        columns={"meanest": "LTG"}
    )

    # Base panel = the FULL SignalMasterTable (script L91-92), with LEFT joins for
    # shrout and Compustat. The previous inner-join through the IBES link table
    # truncated each firm's June history to its IBES-coverage span, so the
    # positional prior-June lag (l12_ceq) missed the true prior June and
    # ceq_ave collapsed to ceq (proven cell-exact for permno 10161 June-1976).
    smt = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "tickerIBES", "time_avail_m", "prc"]]
    )
    crsp = (
        store.load_raw("monthlyCRSP.parquet")
        .loc[:, ["permno", "time_avail_m", "shrout"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )
    comp = (
        store.load_raw("m_aCompustat.parquet")
        .loc[:, ["permno", "time_avail_m", "ceq", "ib", "ibcom", "ni", "sale", "datadate", "dvc", "at"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )

    df = smt.merge(crsp, on=["permno", "time_avail_m"], how="left").merge(
        comp, on=["permno", "time_avail_m"], how="left"
    )
    df = df.sort_values(["permno", "time_avail_m"]).copy()
    df["SG"] = df["sale"] / df.groupby("permno")["sale"].shift(60)
    df = df[df["time_avail_m"].dt.month == 6].copy()
    df = df.merge(froe1, on=["tickerIBES", "time_avail_m"], how="left")
    df = df.merge(froe2, on=["tickerIBES", "time_avail_m"], how="left")
    df = df.merge(ltg, on=["tickerIBES", "time_avail_m"], how="left")

    df = df.sort_values(["permno", "time_avail_m"]).copy()
    df["l12_ceq"] = df.groupby("permno")["ceq"].shift(1)
    df["ceq_ave"] = np.where(df["l12_ceq"].isna(), df["ceq"], (df["ceq"] + df["l12_ceq"]) / 2.0)
    df["mve_permco"] = df["shrout"] * df["prc"].abs()
    df["BM"] = df["ceq"] / df["mve_permco"]
    df["k"] = np.where(df["ibcom"] < 0, df["dvc"] / (0.06 * df["at"]), df["dvc"] / df["ibcom"])
    df["ROE"] = df["ibcom"] / df["ceq_ave"]
    df["FROE1"] = df["feps1"] * df["shrout"] / df["ceq_ave"]
    df["ceq1"] = df["ceq"] * (1 + df["FROE1"] * (1 - df["k"]))
    df["ceq1h"] = df["ceq"] * (1 + df["ROE"] * (1 - df["k"]))
    df["FROE2"] = df["feps2"] * df["shrout"] / ((df["ceq1"] + df["ceq"]) / 2.0)
    df["ceq2"] = df["ceq1"] * (1 + df["FROE1"] * (1 - df["k"]))
    df["ceq2h"] = df["ceq1h"] * (1 + df["ROE"] * (1 - df["k"]))
    df["FROE3"] = np.where(
        df["LTG"].isna(),
        df["FROE2"],
        df["feps2"] * (1 + df["LTG"] / 100.0) * df["shrout"] / ((df["ceq1"] + df["ceq2"]) / 2.0),
    )
    df["ceq3"] = df["ceq2"] * (1 + df["FROE2"] * (1 - df["k"]))

    # polars null-vs-NaN screen semantics: in the reference a 0/0 division (e.g.
    # k = dvc/ibcom with dvc=ibcom=0) yields NaN, NOT null, and FAILS the
    # '<=1 or is_null' screen — only missing INPUTS make the ratio null. pandas
    # .isna() cannot tell the two apart, so null-ness is derived from the inputs.
    roe_isnull = df["ibcom"].isna() | df["ceq"].isna()
    froe1_isnull = df["feps1"].isna() | df["shrout"].isna() | df["ceq"].isna()
    k_isnull = np.where(
        df["ibcom"] < 0,
        df["dvc"].isna() | df["at"].isna(),
        df["dvc"].isna() | df["ibcom"].isna(),
    )
    mask = (
        (df["ceq"] > 0)
        & df["ceq"].notna()
        & ((df["ROE"].abs() <= 1) | roe_isnull)
        & ((df["FROE1"].abs() <= 1) | froe1_isnull)
        & ((df["k"] <= 1) | k_isnull)
        & (pd.to_datetime(df["datadate"]).dt.month >= 6)
        & df["feps1"].notna()
        & df["feps2"].notna()
    )
    df = df[mask].copy()

    r = 0.12
    df["AnalystValue"] = (
        df["ceq1"]
        + ((df["FROE1"] - r) / (1 + r) * df["ceq1"])
        + ((df["FROE2"] - r) / ((1 + r) ** 2) * df["ceq2"])
        + ((df["FROE3"] - r) / ((1 + r) ** 2) / r * df["ceq3"])
    ) / df["mve_permco"]
    df["IntrinsicValue"] = (
        df["ceq1h"]
        + ((df["ROE"] - r) / (1 + r) * df["ceq1h"])
        + ((df["ROE"] - r) / (1 + r) / r * df["ceq2h"])
    ) / df["mve_permco"]
    df["AOP"] = (df["AnalystValue"] - df["IntrinsicValue"]) / df["IntrinsicValue"].abs()

    lag = df[["permno", "time_avail_m", "FROE1"]].copy()
    lag["time_avail_m"] = lag["time_avail_m"] + pd.DateOffset(months=12)
    lag = lag.rename(columns={"FROE1": "FROE1_lag12"})
    df = df.merge(lag, on=["permno", "time_avail_m"], how="left")
    df["FErr"] = df["FROE1_lag12"] - df["ROE"]
    # polars_nearest: winsor2 runs polars quantile 'nearest', which rounds the
    # pick index half-AWAY-from-zero (pandas 'nearest' rounds half-to-even)
    df = _trim_by_group(df, ["FErr"], "time_avail_m", 0.01, 0.99, interpolation="polars_nearest")

    for var in ["SG", "BM", "AOP", "LTG"]:
        df[f"rank{var}"] = df.groupby("time_avail_m")[var].rank(method="average", pct=True)
        # calendar +12-month self-merge (same form as the FROE1_lag12 merge above):
        # the prior-June rank exists only when the panel has that exact June row,
        # unlike the positional groupby.shift(1) it replaces
        rank_lag = df[["permno", "time_avail_m", f"rank{var}"]].copy()
        rank_lag["time_avail_m"] = rank_lag["time_avail_m"] + pd.DateOffset(months=12)
        rank_lag = rank_lag.rename(columns={f"rank{var}": f"lag{var}"})
        df = df.merge(rank_lag, on=["permno", "time_avail_m"], how="left")

    # min_obs=0: the reference polars_ols ols(null_policy='drop') fits EVERY June
    # cohort — an empty fit sample yields the zero minimum-norm solution, emitting
    # the all-0.0 first cohort (1982-06: 0 fit rows, 1,043 pred rows) the min_obs=5
    # gate wrongly suppressed
    df["PredictedFE"] = _ols_predictions_by_group(
        df,
        y_col="FErr",
        fit_x_cols=["lagSG", "lagBM", "lagAOP", "lagLTG"],
        pred_x_cols=["rankSG", "rankBM", "rankAOP", "rankLTG"],
        group_col="time_avail_m",
        min_obs=0,
    )

    base = df[["permno", "time_avail_m", "AnalystValue", "AOP", "PredictedFE"]].copy()
    pieces = []
    for offset in range(12):
        piece = base.copy()
        piece["time_avail_m"] = piece["time_avail_m"] + pd.DateOffset(months=offset)
        pieces.append(piece)
    expanded = pd.concat(pieces, ignore_index=True)
    cache["analyst_value_signals"] = expanded
    return expanded


def _prepare_recommendation_monthly(store: DatasetStore) -> pd.DataFrame:
    """Build firm-month mean IBES recommendations before permno expansion."""

    cache = _get_derived_cache(store)
    if "recommendation_monthly" in cache:
        return cache["recommendation_monthly"]

    rec = store.load_raw("IBES_Recommendations.parquet").loc[
        :, ["tickerIBES", "amaskcd", "time_avail_m", "anndats", "ireccd"]
    ].copy()
    rec = rec.sort_values(["tickerIBES", "amaskcd", "time_avail_m", "anndats"])
    rec = rec.groupby(["tickerIBES", "amaskcd", "time_avail_m"], as_index=False)["ireccd"].last()
    rec = rec.groupby(["tickerIBES", "time_avail_m"], as_index=False)["ireccd"].mean()
    cache["recommendation_monthly"] = rec
    return rec


def _prepare_monthly_stock_base(store: DatasetStore) -> pd.DataFrame:
    """Load the monthly CRSP stock panel once for long-form signal operators."""

    cache = _get_derived_cache(store)
    if "monthly_stock_base" in cache:
        return cache["monthly_stock_base"]

    base = (
        store.load_raw("monthlyCRSP.parquet")
        .loc[:, ["permno", "time_avail_m", "exchcd", "shrcd", "prc", "ret", "retx"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
        .sort_values(["permno", "time_avail_m"])
    )
    cache["monthly_stock_base"] = base
    return base


def _prepare_cash_dividend_monthly(store: DatasetStore) -> pd.DataFrame:
    """Merge monthly stock rows with cash-dividend amounts from CRSP distributions."""

    cache = _get_derived_cache(store)
    if "cash_dividend_monthly" in cache:
        return cache["cash_dividend_monthly"]

    dist = store.load_raw("CRSPdistributions.parquet").copy()
    dist = dist[dist["cd2"].isin([2, 3])].copy()
    dist["time_avail_m"] = pd.to_datetime(pd.to_datetime(dist["exdt"]).dt.to_period("M").dt.start_time)
    dist = dist.dropna(subset=["time_avail_m", "divamt"])
    tempdivamt = dist.groupby(["permno", "time_avail_m"], as_index=False)["divamt"].sum()
    df = _prepare_monthly_stock_base(store).merge(tempdivamt, on=["permno", "time_avail_m"], how="left")
    cache["cash_dividend_monthly"] = df
    return df


def _prepare_regular_dividend_monthly(store: DatasetStore, *, yearly_only: bool = False) -> pd.DataFrame:
    """Merge monthly stock rows with regular dividend frequency and amount data."""

    key = "regular_dividend_monthly_yearly" if yearly_only else "regular_dividend_monthly"
    cache = _get_derived_cache(store)
    if key in cache:
        return cache[key]

    dist = store.load_raw("CRSPdistributions.parquet").copy()
    dist = dist[(dist["cd1"] == 1) & (dist["cd2"] == 2)].copy()
    if yearly_only:
        dist = dist[dist["cd3"].isin([3, 4, 5])].copy()
    dist["time_avail_m"] = pd.to_datetime(pd.to_datetime(dist["exdt"]).dt.to_period("M").dt.start_time)
    dist = dist.dropna(subset=["time_avail_m", "divamt"])
    tempdivamt = dist.groupby(["permno", "cd3", "time_avail_m"], as_index=False)["divamt"].sum()
    tempdivamt = tempdivamt.sort_values(["permno", "time_avail_m", "cd3"])
    tempdivamt = tempdivamt.groupby(["permno", "time_avail_m"], as_index=False).first()
    df = _prepare_monthly_stock_base(store).merge(
        tempdivamt[["permno", "time_avail_m", "cd3", "divamt"]],
        on=["permno", "time_avail_m"],
        how="left",
    )
    cache[key] = df
    return df


def _op_intangible_residual(account_measure: Any, ret: Any, *, store: DatasetStore, **_: Any) -> Any:
    """Daniel-Titman intangible residual (IntanBM/IntanCFP/IntanEP/IntanSP).

    Everything lives on the script's merged-panel domain U = SignalMasterTable
    row AND m_aCompustat row presence — a whole-grid zero-fill manufactures
    structural cumret=1 cells that poison the trim pool and the regressions.
    """

    # U: SMT row (shrcd 10/11/12 & exchcd 1/2/3) AND m_aCompustat row presence
    strict, _span = _universe_masks(store)
    universe = strict & _compustat_rows_mask(store)
    account_measure = account_measure.replace([np.inf, -np.inf], np.nan).where(universe)
    # merged-panel compounding: U cells contribute log1p(ret NaN->0); non-U months add exactly 0.0
    cumret = np.exp(np.log1p(ret.where(universe).fillna(0.0)).cumsum())
    lag60 = _op_lag(cumret, 60)
    # ret60 requires a merged row at t AND at exactly t-60 (script's calendar lag-merge presence rule)
    ret60 = ((cumret - lag60) / lag60).where(universe & universe.shift(60, fill_value=False))
    ret60 = ret60.replace([np.inf, -np.inf], np.nan)
    if REFERENCE_MODE:
        # verification-only look-ahead: full-sample 1%/99% bounds, pandas-linear
        # quantiles (script winsor2 on a pandas frame) over the U-masked pool only
        ret60 = _trim_global(ret60, 0.01, 0.99, method="linear")
    else:
        # look-ahead correction (registered deviation): per-month cutoffs on the same masked domain
        ret60 = _trim_by_month(ret60, 0.01, 0.99)
    measure_lag60 = _op_lag(account_measure, 60)  # U-masked measure: lag is NaN unless a merged row exists at t-60
    measure_ret = (account_measure - measure_lag60 + ret60).replace([np.inf, -np.inf], np.nan)
    # per-month OLS with intercept; residuals land only on complete-case cells (a subset of U)
    return _cross_sectional_residual(ret60, [measure_lag60, measure_ret], min_obs=2)


def _op_brand_invest_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    df = store.load_raw("a_aCompustat.parquet").loc[
        :, ["gvkey", "permno", "time_avail_m", "fyear", "datadate", "xad", "xad0", "at", "sic"]
    ].copy()
    df = df.sort_values(["gvkey", "fyear"]).copy()
    df["sic_num"] = pd.to_numeric(df["sic"], errors="coerce")

    frames = []
    for _, group in df.groupby("gvkey", sort=False):
        group = group.copy()
        ok = group["xad"].notna()
        if not ok.any():
            continue
        first_pos = int(np.flatnonzero(ok.to_numpy())[0])
        # script-exact seed semantics (Stata sequential replace, deliberately
        # matched oracle quirk): the xad/0.6 seed survives ONLY when the gvkey's
        # very first row has non-missing xad — the unconditional i-from-1
        # recursion over zero-initialized values overwrites it otherwise
        brand_capital = np.zeros(len(group), dtype="float64")
        tempxad = group["xad"].fillna(0.0).to_numpy(dtype="float64")
        if first_pos == 0:
            brand_capital[0] = float(group["xad"].iloc[0]) / 0.6
        for i in range(1, len(group)):
            brand_capital[i] = 0.5 * brand_capital[i - 1] + tempxad[i]
        brand_capital[:first_pos] = np.nan  # fyear < FirstNMyear
        at_values = group["at"].to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            # plain division: at==0 & BC!=0 -> inf, kept (downstream xad0/inf
            # -> 0.0 rows must survive, matching the reference)
            brand_capital = brand_capital / at_values
        brand_capital[~ok.to_numpy()] = np.nan
        group["BrandCapital"] = brand_capital
        group["BrandInvest"] = group["xad0"] / pd.Series(group["BrandCapital"], index=group.index).shift(1)
        frames.append(group)

    if not frames:
        return pd.DataFrame(index=store.template_index, columns=store.template_columns, dtype="float32")

    out = pd.concat(frames, ignore_index=True)
    out = out[
        ~(((out["sic_num"] >= 4900) & (out["sic_num"] <= 4999)) | ((out["sic_num"] >= 6000) & (out["sic_num"] <= 6999)))
    ].copy()
    out = out[pd.to_datetime(out["datadate"]).dt.month == 12].copy()

    base = out[["gvkey", "permno", "time_avail_m", "datadate", "BrandInvest"]].copy()
    pieces = []
    for offset in range(12):
        piece = base.copy()
        piece["time_avail_m"] = piece["time_avail_m"] + pd.DateOffset(months=offset)
        pieces.append(piece)
    expanded = pd.concat(pieces, ignore_index=True)
    expanded = expanded.sort_values(["gvkey", "time_avail_m", "datadate"]).drop_duplicates(
        ["gvkey", "time_avail_m"], keep="last"
    )
    expanded = expanded.drop_duplicates(["permno", "time_avail_m"], keep="first")
    expanded = expanded.dropna(subset=["BrandInvest"])
    return _to_wide_panel(store, expanded, "BrandInvest")


def _op_equity_duration_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    df = store.load_raw("a_aCompustat.parquet").loc[
        :, ["gvkey", "permno", "time_avail_m", "fyear", "datadate", "ceq", "ib", "sale", "prcc_f", "csho"]
    ].copy()
    df = df.sort_values(["gvkey", "fyear"]).copy()

    autocorr_roe = 0.57
    cost_equity = 0.12
    autocorr_growth = 0.24
    longrun_growth = 0.06

    ceq_lag = df.groupby("gvkey")["ceq"].shift(1)
    sale_lag = df.groupby("gvkey")["sale"].shift(1)
    df["tempRoE"] = df["ib"] / ceq_lag
    df["temp_g_eq"] = (df["sale"] / sale_lag) - 1
    df["temp_g_eq"] = df["temp_g_eq"].replace([np.inf, -np.inf], np.nan)

    df["tempRoE1"] = autocorr_roe * df["tempRoE"] + cost_equity * (1 - autocorr_roe)
    df["temp_g_eq1"] = autocorr_growth * df["temp_g_eq"].fillna(0) + longrun_growth * (1 - autocorr_growth)
    df["tempBV1"] = df["ceq"] * (1 + df["temp_g_eq1"])
    df["tempCD1"] = df["ceq"] - df["tempBV1"] + df["ceq"] * df["tempRoE1"]

    for t in range(2, 11):
        j = t - 1
        df[f"tempRoE{t}"] = autocorr_roe * df[f"tempRoE{j}"] + cost_equity * (1 - autocorr_roe)
        df[f"temp_g_eq{t}"] = autocorr_growth * df[f"temp_g_eq{j}"].fillna(0) + longrun_growth * (1 - autocorr_growth)
        df[f"tempBV{t}"] = df[f"tempBV{j}"] * (1 + df[f"temp_g_eq{t}"])
        df[f"tempCD{t}"] = df[f"tempBV{j}"] - df[f"tempBV{t}"] + df[f"tempBV{j}"] * df[f"tempRoE{t}"]

    discount_rate = 1 + cost_equity
    md_part1 = 0.0
    pv_part1 = 0.0
    for t in range(1, 11):
        md_part1 = md_part1 + (t * df[f"tempCD{t}"] / (discount_rate**t))
        pv_part1 = pv_part1 + (df[f"tempCD{t}"] / (discount_rate**t))

    df["tempME"] = df["prcc_f"] * df["csho"]
    df["EquityDuration"] = md_part1 / df["tempME"] + (10 + (1 + cost_equity) / cost_equity) * (1 - pv_part1 / df["tempME"])

    base = df[["gvkey", "permno", "time_avail_m", "datadate", "EquityDuration"]].copy()
    pieces = []
    for offset in range(12):
        piece = base.copy()
        piece["time_avail_m"] = piece["time_avail_m"] + pd.DateOffset(months=offset)
        pieces.append(piece)
    expanded = pd.concat(pieces, ignore_index=True)
    expanded = expanded.sort_values(["gvkey", "time_avail_m", "datadate"]).drop_duplicates(
        ["gvkey", "time_avail_m"], keep="last"
    )
    expanded["_is_na"] = expanded["EquityDuration"].isna()
    expanded = expanded.sort_values(["permno", "time_avail_m", "_is_na", "gvkey", "datadate"]).drop_duplicates(
        ["permno", "time_avail_m"], keep="first"
    )
    expanded = expanded.drop(columns="_is_na")
    expanded = expanded.dropna(subset=["EquityDuration"])
    return _to_wide_panel(store, expanded, "EquityDuration")


def _op_abnormal_accruals_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    df = store.load_raw("a_aCompustat.parquet").loc[
        :,
        ["gvkey", "permno", "time_avail_m", "fyear", "datadate", "at", "oancf", "fopt", "act", "che", "lct", "dlc", "ib", "sale", "ppegt", "ni", "sic"],
    ].copy()
    # exchcd source is SignalMasterTable (the reference's join; monthlyCRSP exchcd
    # differs on ~1k pre-1982 keys) — the spec tree's exchcd arg is dependency-tracking only
    exch = store.load_raw("SignalMasterTable.parquet").loc[:, ["permno", "time_avail_m", "exchcd"]]
    df = df.merge(exch, on=["permno", "time_avail_m"], how="left")
    df = df.sort_values(["gvkey", "fyear"], kind="stable").copy()
    prev_year = df.groupby("gvkey")["fyear"].shift(1)
    consecutive = (df["fyear"] - prev_year) == 1
    for col in ["act", "che", "lct", "dlc", "at", "sale"]:
        df[f"l1_{col}"] = df.groupby("gvkey")[col].shift(1).where(consecutive)

    fallback_cfo = (
        df["fopt"]
        - (df["act"] - df["l1_act"])
        + (df["che"] - df["l1_che"])
        + (df["lct"] - df["l1_lct"])
        - (df["dlc"] - df["l1_dlc"])
    )
    df["tempCFO"] = df["oancf"].where(df["oancf"].notna(), fallback_cfo)
    df["tempInvTA"] = 1 / df["l1_at"]
    df["tempAccruals"] = (df["ib"] - df["tempCFO"]) / df["l1_at"]
    df["tempDelRev"] = (df["sale"] - df["l1_sale"]) / df["l1_at"]
    df["tempPPE"] = df["ppegt"] / df["l1_at"]
    # look-ahead correction (registered deviation): the reference trims and
    # estimates the Jones model within the SAME fiscal year, so a firm's
    # residual depends on peers' statements published up to ~11 months after
    # its own stamp. The PIT branch takes trim bounds (per-fyear pools below)
    # AND regression coefficients from the PREVIOUS fiscal year, which is fully
    # published before the first fyear-y statement is stamped (y-1 completes
    # ~Nov-y; the earliest y stamp is Dec-y). A firm's residual then uses only
    # its own statement and last year's model.
    df["sic2"] = np.floor(pd.to_numeric(df["sic"], errors="coerce") / 100.0)
    model_cols = ["tempAccruals", "tempInvTA", "tempDelRev", "tempPPE"]
    # polars-membership masks: the reference computes ratios in polars, where a
    # 0/0 division yields NaN (NOT null) — those cells COUNT as winsor-pool
    # members ranking above +inf, while pandas .quantile/.count would drop them
    member = {
        "tempAccruals": df["ib"].notna() & df["tempCFO"].notna() & df["l1_at"].notna(),
        "tempInvTA": df["l1_at"].notna(),
        "tempDelRev": df["sale"].notna() & df["l1_sale"].notna() & df["l1_at"].notna(),
        "tempPPE": df["ppegt"].notna() & df["l1_at"].notna(),
    }
    # phantom gap rows: the reference's fiscal-year gap-fill inserts grid rows whose
    # l1_at is the prior real year's at and whose forward-filled fyear tag is the
    # STALE previous fyear, so tempInvTA=1/at(y) joins fyear y's winsor pool
    # (pool only — phantoms are never scored and never reach the output)
    next_fyear = df.groupby("gvkey")["fyear"].shift(-1)
    phantom_mask = (next_fyear.notna() & (next_fyear > df["fyear"] + 1)) & df["at"].notna()
    phantom_pool = pd.DataFrame(
        {"fyear": df.loc[phantom_mask, "fyear"], "val": 1.0 / df.loc[phantom_mask, "at"]}
    )
    # per-fyear 0.1%/99.9% trim bounds over the phantom-inclusive membership pools,
    # polars-'nearest' quantiles (reference winsor2 runs on a polars frame)
    pools = {}
    for col in model_cols:
        vals = pd.DataFrame(
            {"fyear": df.loc[member[col], "fyear"], "val": df.loc[member[col], col].astype("float64")}
        )
        if col == "tempInvTA":
            vals = pd.concat([vals, phantom_pool], ignore_index=True)
        pools[col] = vals
    lo_bounds = pd.DataFrame(
        {col: pools[col].groupby("fyear")["val"].agg(_polars_nearest_quantile, q=0.001) for col in model_cols}
    )
    hi_bounds = pd.DataFrame(
        {col: pools[col].groupby("fyear")["val"].agg(_polars_nearest_quantile, q=0.999) for col in model_cols}
    )
    if REFERENCE_MODE:
        # verification-only: the reference's same-fiscal-year pooling (look-ahead);
        # trim removes only values STRICTLY outside bounds — a NaN bound (empty
        # pool) or NaN value compares False and trims nothing, like polars
        trimmed_out = {}
        for col in model_cols:
            col_lo = df["fyear"].map(lo_bounds[col])
            col_hi = df["fyear"].map(hi_bounds[col])
            trimmed_out[col] = (df[col] < col_lo) | (df[col] > col_hi)
            df[col] = df[col].where(~trimmed_out[col])
        # null sic2 is its OWN group; NO complete-case minimum (polars_ols
        # null_policy='drop' fits every group, minimum-norm when underdetermined)
        df["AbnormalAccruals"] = _ols_residuals_by_groups(
            df, y_col="tempAccruals", x_cols=["tempInvTA", "tempDelRev", "tempPPE"],
            group_cols=["fyear", "sic2"], min_obs=1, dropna_groups=False,
        )
        # _Nobs = post-trim polars-NON-NULL tempAccruals per (fyear, sic2) — NaN
        # members count; failing groups are DROPPED (rows removed — nulling them
        # would still shape the ffill span)
        nobs_flag = member["tempAccruals"] & ~trimmed_out["tempAccruals"]
        nobs = nobs_flag.groupby([df["fyear"], df["sic2"]], dropna=False).transform("sum")
        df = df.loc[nobs >= 6]
        # polars null propagation: pre-1982 rows with NULL exchcd drop along with exchcd==3
        df = df.loc[~((df["fyear"] < 1982) & ((df["exchcd"] == 3) | df["exchcd"].isna()))]
        # keep-first per (permno, fyear); null-residual rows RETAINED (they extend
        # the monthly grid span and are forward-filled through). The reference's
        # polars sort is UNSTABLE, so its kept row at the 8 duplicate keys is
        # arbitrary — stable gvkey-ascending order is the deterministic choice
        # (2 tie keys / <=37 grid cells can differ from a given oracle run).
        df = df.sort_values(["permno", "fyear"], kind="stable").drop_duplicates(["permno", "fyear"], keep="first")
        return _expand_annual_signal_ffill(
            store, df[["permno", "fyear", "time_avail_m", "AbnormalAccruals"]], "AbnormalAccruals"
        )
    df["AbnormalAccruals"] = np.nan
    betas: dict[tuple, np.ndarray] = {}
    for (fyear, sic2), grp in df.groupby(["fyear", "sic2"], sort=False):
        lo_row, hi_row = lo_bounds.loc[fyear], hi_bounds.loc[fyear]
        fit = grp[model_cols].dropna()
        for c in model_cols:  # own-year pool bounds trim the estimation sample
            fit = fit[~((fit[c] < lo_row[c]) | (fit[c] > hi_row[c]))]
        if len(fit) < 6:
            continue
        X = np.column_stack([np.ones(len(fit))] + [fit[c].to_numpy(dtype="float64") for c in model_cols[1:]])
        y = fit["tempAccruals"].to_numpy(dtype="float64")
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        betas[(fyear, sic2)] = beta
    for (fyear, sic2), grp in df.groupby(["fyear", "sic2"], sort=False):
        prior = (fyear - 1, sic2)
        if prior not in betas:
            continue
        beta = betas[prior]
        lo_row, hi_row = lo_bounds.loc[fyear - 1], hi_bounds.loc[fyear - 1]
        score = grp[model_cols].dropna()
        for c in model_cols:  # prior-year pool bounds applied to this year's inputs
            score = score[~((score[c] < lo_row[c]) | (score[c] > hi_row[c]))]
        if score.empty:
            continue
        X = np.column_stack([np.ones(len(score))] + [score[c].to_numpy(dtype="float64") for c in model_cols[1:]])
        df.loc[score.index, "AbnormalAccruals"] = score["tempAccruals"].to_numpy(dtype="float64") - X @ beta
    # PIT branch keeps pandas ==3 semantics: unknown (null) exchcd rows are kept, not silently dropped
    df = df[~((df["exchcd"] == 3) & (df["fyear"] < 1982))].copy()
    df = df.sort_values(["permno", "fyear"]).drop_duplicates(["permno", "fyear"], keep="first")
    expanded = _expand_hold_months(df[["permno", "time_avail_m", "AbnormalAccruals"]], "AbnormalAccruals", 12)
    expanded = expanded.dropna(subset=["AbnormalAccruals"])
    return _to_wide_panel(store, expanded, "AbnormalAccruals")


def _op_analyst_value_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    return _to_wide_panel(store, _prepare_analyst_value_signals(store), "AnalystValue")


def _op_analyst_optimism_ratio(*_: Any, store: DatasetStore, **__: Any) -> Any:
    return _to_wide_panel(store, _prepare_analyst_value_signals(store), "AOP")


def _op_predicted_forecast_error(*_: Any, store: DatasetStore, **__: Any) -> Any:
    return _to_wide_panel(store, _prepare_analyst_value_signals(store), "PredictedFE")


def _op_frontier_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    # base = SignalMasterTable (the script's universe), which carries both
    # mve_permco and sicCRSP; dedup is a no-op safeguard
    df = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "time_avail_m", "mve_permco", "sicCRSP"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )
    comp = (
        store.load_raw("m_aCompustat.parquet")
        .loc[:, ["permno", "time_avail_m", "at", "ceq", "dltt", "capx", "sale", "xrd", "xad", "ppent", "ebitda"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )
    df = df.merge(comp, on=["permno", "time_avail_m"], how="inner").copy()
    df["time_avail"] = (df["time_avail_m"].dt.year - 1960) * 12 + (df["time_avail_m"].dt.month - 1)
    df["xad"] = df["xad"].fillna(0.0)
    df["YtempBM"] = np.nan
    mask_me = df["mve_permco"] > 0
    df.loc[mask_me, "YtempBM"] = np.log(df.loc[mask_me, "mve_permco"])
    df["tempBook"] = np.nan
    mask_be = df["ceq"] > 0
    df.loc[mask_be, "tempBook"] = np.log(df.loc[mask_be, "ceq"])
    df["tempLTDebt"] = np.where((df["at"] == 0) | df["at"].isna(), np.nan, df["dltt"] / df["at"])
    df["tempCapx"] = np.where((df["sale"] == 0) | df["sale"].isna(), np.nan, df["capx"] / df["sale"])
    df["tempRD"] = np.where((df["sale"] == 0) | df["sale"].isna(), np.nan, df["xrd"] / df["sale"])
    df["tempAdv"] = np.where((df["sale"] == 0) | df["sale"].isna(), np.nan, df["xad"] / df["sale"])
    df["tempPPE"] = np.where((df["at"] == 0) | df["at"].isna(), np.nan, df["ppent"] / df["at"])
    df["tempEBIT"] = np.where((df["at"] == 0) | df["at"].isna(), np.nan, df["ebitda"] / df["at"])

    ff48_func = _get_ff48_func()
    df["tempFF48"] = pd.to_numeric(df["sicCRSP"], errors="coerce").map(ff48_func)
    df = df.dropna(subset=["tempFF48"]).sort_values(["permno", "time_avail_m"]).copy()

    reg_vars = ["tempBook", "tempLTDebt", "tempCapx", "tempRD", "tempAdv", "tempPPE", "tempEBIT"]
    df["logmefit_NS"] = np.nan
    unique_dates = sorted(df["time_avail_m"].unique())

    for current_date in unique_dates:
        current_time = (current_date.year - 1960) * 12 + (current_date.month - 1)
        train = df[(df["time_avail"] <= current_time) & (df["time_avail"] > current_time - 60)].copy()
        train = train.dropna(subset=["YtempBM"] + reg_vars)
        if len(train) < 3:  # script L165's minimal-training gate
            continue

        current = df[df["time_avail_m"] == current_date].copy()
        current = current.dropna(subset=reg_vars)
        if current.empty:
            continue

        train_dummies = pd.get_dummies(train["tempFF48"], prefix="ff48")
        current_dummies = pd.get_dummies(current["tempFF48"], prefix="ff48")
        current_dummies = current_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        X_train = pd.concat([train[reg_vars], train_dummies], axis=1).astype("float64")
        y_train = train["YtempBM"].to_numpy(dtype="float64")
        X_pred = pd.concat([current[reg_vars], current_dummies], axis=1).astype("float64")
        try:
            beta, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(X_train), dtype="float64"), X_train.to_numpy(dtype="float64")]),
                y_train,
                rcond=None,
            )
        except np.linalg.LinAlgError:
            continue
        pred = np.column_stack([np.ones(len(X_pred), dtype="float64"), X_pred.to_numpy(dtype="float64")]) @ beta
        df.loc[current.index, "logmefit_NS"] = pred

    df["Frontier"] = -(df["YtempBM"] - df["logmefit_NS"])
    df = df[(df["ceq"] > 0) & df["Frontier"].notna()].copy()
    return _to_wide_panel(store, df[["permno", "time_avail_m", "Frontier"]], "Frontier")


def _op_rd_ability_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    df = store.load_raw("a_aCompustat.parquet").loc[
        :, ["gvkey", "permno", "time_avail_m", "fyear", "datadate", "xrd", "sale"]
    ].copy()
    df = df.sort_values(["gvkey", "fyear"]).copy()
    df["tempXRD"] = df["xrd"].where(~((df["xrd"] < 0) & df["xrd"].notna()))
    df["tempSale"] = df["sale"].where(~((df["sale"] < 0) & df["sale"].notna()))
    df["l_tempSale"] = df.groupby("gvkey")["tempSale"].shift(1)
    df["tempY"] = np.nan
    mask_y = df["tempSale"].notna() & df["l_tempSale"].notna() & (df["tempSale"] > 0) & (df["l_tempSale"] > 0)
    df.loc[mask_y, "tempY"] = np.log(df.loc[mask_y, "tempSale"] / df.loc[mask_y, "l_tempSale"])
    df["tempX"] = np.nan
    mask_x = df["tempXRD"].notna() & df["tempSale"].notna() & (df["tempSale"] > 0)
    df.loc[mask_x, "tempX"] = np.log1p(df.loc[mask_x, "tempXRD"] / df.loc[mask_x, "tempSale"])

    for n in range(1, 6):
        gamma = pd.Series(np.nan, index=df.index, dtype="float64")
        temp_mean = pd.Series(np.nan, index=df.index, dtype="float64")
        for _, group in df.groupby("gvkey", sort=False):
            group = group.copy()
            years = group["fyear"].to_numpy()
            # calendar fyear lag (registered deviation): the script's positional
            # shift(n).over('gvkey') bridges multi-year fyear gaps; the true lag
            # is fyear-n, NaN when that fiscal year is absent (raises on
            # duplicate (gvkey, fyear) rows — surface, don't guess)
            x = (
                pd.Series(group["tempX"].to_numpy(dtype="float64"), index=years)
                .reindex(years - n)
                .to_numpy(dtype="float64")
            )
            y = group["tempY"].to_numpy(dtype="float64")
            gamma_vals = np.full(len(group), np.nan, dtype="float64")
            mean_vals = np.full(len(group), np.nan, dtype="float64")
            finite_pair = np.isfinite(y) & np.isfinite(x)
            seg_start = 0
            for i in range(len(group)):
                # calendar window (registered deviation): the regression sample is
                # the valid pairs inside the trailing 8 FISCAL YEARS, not the last
                # 8 valid pairs compacted across gap-bridged rows
                start = int(np.searchsorted(years, years[i] - 7, side="left"))
                valid_idx = start + np.flatnonzero(finite_pair[start : i + 1])
                if valid_idx.size >= 6:
                    xw = x[valid_idx]
                    # zero-variance regressor -> singular design: polars_ols
                    # rolling_ols returns a null coefficient, not the lstsq
                    # minimum-norm slope of 0.0
                    if xw.max() != xw.min():
                        X = np.column_stack([np.ones(len(valid_idx), dtype="float64"), xw])
                        try:
                            beta, *_ = np.linalg.lstsq(X, y[valid_idx], rcond=None)
                            gamma_vals[i] = beta[1]
                        except np.linalg.LinAlgError:
                            pass

                if i > 0 and (years[i] - years[i - 1] > 1):
                    seg_start = i
                mean_start = max(seg_start, i - 7)
                seg_vals = x[mean_start : i + 1]
                nonzero = (np.isfinite(seg_vals) & (seg_vals > 0)).astype("float64")
                if len(seg_vals) >= 6:
                    mean_vals[i] = nonzero.mean()

            gamma.loc[group.index] = gamma_vals
            temp_mean.loc[group.index] = mean_vals

        # mask(< 0.5), not where(>= 0.5): gamma must SURVIVE a NaN temp_mean
        # (NaN comparisons are False), matching the script's null-passes gate
        df[f"gammaAbility{n}"] = gamma.mask(temp_mean < 0.5)

    gamma_cols = [f"gammaAbility{n}" for n in range(1, 6)]
    df["RDAbility"] = df[gamma_cols].mean(axis=1, skipna=True)
    df["tempRD"] = np.where(
        df["xrd"].notna() & df["sale"].notna() & (df["sale"] > 0) & (df["xrd"] > 0),
        df["xrd"] / df["sale"],
        np.nan,
    )
    df["tempRDQuant"] = df.groupby("time_avail_m")["tempRD"].transform(
        lambda x: pd.qcut(x, q=3, labels=False, duplicates="drop") + 1
    )
    df["RDAbility"] = df["RDAbility"].where(df["tempRDQuant"] == 3)
    df["RDAbility"] = df["RDAbility"].where(~((df["xrd"] <= 0) & df["xrd"].notna()))

    base = df[["gvkey", "permno", "time_avail_m", "datadate", "RDAbility"]].copy()
    pieces = []
    for offset in range(12):
        piece = base.copy()
        piece["time_avail_m"] = piece["time_avail_m"] + pd.DateOffset(months=offset)
        pieces.append(piece)
    expanded = pd.concat(pieces, ignore_index=True)
    # stable sorts: at gvkey-switch overlap months one permno carries two gvkeys'
    # rows; the reference's polars sort is UNSTABLE so its kept row is arbitrary —
    # gvkey-ascending is the deterministic choice (registered tie residual, same
    # class as AbnormalAccruals' dedup note)
    expanded = expanded.sort_values(["gvkey", "time_avail_m", "datadate"], kind="stable").drop_duplicates(
        ["gvkey", "time_avail_m"], keep="last"
    )
    expanded = expanded.sort_values(["permno", "time_avail_m"], kind="stable").drop_duplicates(
        ["permno", "time_avail_m"], keep="first"
    )
    expanded = expanded.dropna(subset=["RDAbility"])
    return _to_wide_panel(store, expanded, "RDAbility")


def _op_ms_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    def _rolling_with_month_gaps(
        frame: pd.DataFrame,
        value_col: str,
        window: int,
        min_obs: int,
        stat: str,
    ) -> pd.Series:
        result = pd.Series(np.nan, index=frame.index, dtype="float64")
        for _, group in frame.groupby("permno", sort=False):
            group = group.sort_values("time_avail_m")
            monthly_index = pd.date_range(group["time_avail_m"].min(), group["time_avail_m"].max(), freq="MS")
            series = group.set_index("time_avail_m")[value_col].reindex(monthly_index)
            if stat == "mean":
                rolled = series.rolling(window=window, min_periods=min_obs).mean()
            elif stat == "std":
                rolled = series.rolling(window=window, min_periods=min_obs).std()
            else:
                raise ValueError(f"Unsupported rolling stat: {stat}")
            result.loc[group.index] = rolled.reindex(group["time_avail_m"]).to_numpy(dtype="float64")
        return result

    def _median_binary(
        frame: pd.DataFrame,
        value_col: str,
        industry_col: str,
        greater: bool,
    ) -> pd.Series:
        # Stata missing = +inf on BOTH sides of the comparison (script semantics):
        # every row scores 0/1, no notna gates
        med = frame.groupby([industry_col, "time_avail_m"])[value_col].transform("median")
        v = frame[value_col].fillna(np.inf)
        m = med.fillna(np.inf)
        return ((v > m) if greater else (v < m)).astype("float64")

    comp = (
        store.load_raw("m_aCompustat.parquet")
        .loc[:, ["permno", "gvkey", "time_avail_m", "datadate", "at", "ceq", "ni", "oancf", "fopt", "wcapch", "ib", "dp", "xrd", "capx", "xad", "revt"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )
    # SignalMasterTable, not monthlyCRSP: the script's inner merge is against the
    # SMT universe (shrcd/exchcd-filtered), which SMT carries mve_permco/sicCRSP for
    crsp = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "time_avail_m", "mve_permco", "sicCRSP"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
    )
    qcomp = (
        store.load_raw("m_QCompustat.parquet")
        .loc[:, ["gvkey", "time_avail_m", "niq", "atq", "saleq", "oancfy", "capxy", "xrdq", "fyearq", "fqtr", "datafqtr", "datadateq"]]
        .drop_duplicates(["gvkey", "time_avail_m"], keep="first")
    )

    df = comp.merge(crsp, on=["permno", "time_avail_m"], how="inner").merge(
        qcomp, on=["gvkey", "time_avail_m"], how="left"
    )
    df = df.sort_values(["permno", "time_avail_m"]).copy()

    bm_mask = (df["ceq"] > 0) & (df["mve_permco"] > 0)
    df = df[df["ceq"] > 0].copy()
    df["BM"] = np.nan
    df.loc[bm_mask.loc[df.index], "BM"] = np.log(df.loc[bm_mask.loc[df.index], "ceq"] / df.loc[bm_mask.loc[df.index], "mve_permco"])
    df["BM_quintile"] = df.groupby("time_avail_m")["BM"].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop") + 1
    )
    df = df[df["BM_quintile"] == 1].copy()
    df["sic2D"] = df["sicCRSP"].astype("string").str.slice(0, 2)
    sic_count = df.groupby(["sic2D", "time_avail_m"])["permno"].transform("size")
    df = df[sic_count >= 3].copy()

    df["xad"] = df["xad"].fillna(0.0)
    df["xrdq"] = df["xrdq"].fillna(0.0)
    df["capxq"] = np.where(
        df["fqtr"] == 1,
        df["capxy"],
        np.where(df["fqtr"].notna() & (df["fqtr"] > 1), df["capxy"] - df.groupby("permno")["capxy"].shift(3), np.nan),
    )
    df["oancfq"] = np.where(
        df["fqtr"] == 1,
        df["oancfy"],
        np.where(df["fqtr"].notna() & (df["fqtr"] > 1), df["oancfy"] - df.groupby("permno")["oancfy"].shift(3), np.nan),
    )

    # all rolling stats month-gap-aware (calendar windows), not row-positional
    df["niqsum"] = _rolling_with_month_gaps(df, "niq", 12, 12, "mean") * 4.0
    df["xrdqsum"] = _rolling_with_month_gaps(df, "xrdq", 12, 12, "mean") * 4.0
    df["oancfqsum"] = _rolling_with_month_gaps(df, "oancfq", 12, 12, "mean") * 4.0
    df["capxqsum"] = _rolling_with_month_gaps(df, "capxq", 12, 12, "mean") * 4.0
    early_mask = pd.to_datetime(df["datadate"]).dt.year <= 1988
    df.loc[early_mask, "oancfqsum"] = df.loc[early_mask, "fopt"] - df.loc[early_mask, "wcapch"]

    df["atdenom"] = (df["atq"] + df.groupby("permno")["atq"].shift(3)) / 2.0
    df["roa"] = df["niqsum"] / df["atdenom"]
    df["cfroa"] = df["oancfqsum"] / df["atdenom"]
    df["m1"] = _median_binary(df, "roa", "sic2D", greater=True)
    df["m2"] = _median_binary(df, "cfroa", "sic2D", greater=True)
    # Stata missing = +inf on both sides (inf > inf is False when both missing)
    df["m3"] = (df["oancfqsum"].fillna(np.inf) > df["niqsum"].fillna(np.inf)).astype("float64")

    df["roaq"] = df["niq"] / df["atq"]
    df["sg"] = df["saleq"] / df.groupby("permno")["saleq"].shift(3)
    df["niVol"] = _rolling_with_month_gaps(df, "roaq", 48, 18, "std")
    df["revVol"] = _rolling_with_month_gaps(df, "sg", 48, 18, "std")
    df["m4"] = _median_binary(df, "niVol", "sic2D", greater=False)
    df["m5"] = _median_binary(df, "revVol", "sic2D", greater=False)

    df["atdenom2"] = df.groupby("permno")["atq"].shift(3)
    df["xrdint"] = df["xrdqsum"] / df["atdenom2"]
    df["capxint"] = df["capxqsum"] / df["atdenom2"]
    df["xadint"] = df["xad"] / df["atdenom2"]
    df["m6"] = _median_binary(df, "xrdint", "sic2D", greater=True)
    df["m7"] = _median_binary(df, "capxint", "sic2D", greater=True)
    df["m8"] = _median_binary(df, "xadint", "sic2D", greater=True)

    # plain sum: every m is now unconditionally 0/1 (Stata inf semantics above)
    df["tempMS"] = df[["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]].sum(axis=1)

    # NO +1 correction: June FYE maps to 0 and never matches any calendar month,
    # replicating the reference bug; null datadate keeps the row (polars when()
    # treats a null condition as False)
    timing_month = (pd.to_datetime(df["datadate"]).dt.month + 6) % 12
    df.loc[df["datadate"].notna() & (df["time_avail_m"].dt.month != timing_month), "tempMS"] = np.nan
    df["tempMS"] = df.groupby("permno")["tempMS"].ffill()
    df["MS"] = df["tempMS"].copy()
    df.loc[df["tempMS"] >= 6, "MS"] = 6.0
    df.loc[df["tempMS"] <= 1, "MS"] = 1.0
    df = df.dropna(subset=["MS"])

    return _to_wide_panel(store, df[["permno", "time_avail_m", "MS"]], "MS")


def _op_bmdec_signal(
    ceq: Any,
    txditc: Any,
    seq: Any,
    at: Any,
    lt: Any,
    pstk: Any,
    pstkrv: Any,
    pstkl: Any,
    prc: Any,
    shrout: Any,
    **_: Any,
) -> Any:
    month = pd.Index(ceq.index).month
    month_series = pd.Series(month, index=ceq.index)
    year = pd.Index(ceq.index).year

    temp_me = (prc.abs() * shrout).where(month_series == 12, axis=0)
    temp_dec_me = temp_me.groupby(year).transform("min")

    temp_ps = pstk.where(pstk.notna(), pstkrv)
    temp_ps = temp_ps.where(temp_ps.notna(), pstkl)
    temp_se = seq.where(seq.notna(), ceq + temp_ps)
    temp_se = temp_se.where(temp_se.notna(), at - lt)
    temp_be = temp_se + txditc.fillna(0) - temp_ps

    bm_l12 = temp_be / temp_dec_me.shift(12)
    bm_l17 = temp_be / temp_dec_me.shift(17)
    out = bm_l12.where(month_series >= 6, bm_l17, axis=0)
    return out.where(np.isfinite(out))


def _op_feps_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet")
    df = ibes[ibes["fpi"] == "1"][["tickerIBES", "time_avail_m", "meanest"]].copy()
    # the script's left-merge onto SMT via SMT's own tickerIBES: output domain
    # is the SMT universe, one ticker fans out to all SMT permnos sharing it
    df = _smt_ticker_link(store).merge(df, on=["tickerIBES", "time_avail_m"], how="left")
    return _to_wide_panel(store, df.rename(columns={"meanest": "FEPS"}), "FEPS")


def _op_binary_recommendation_level(
    *_: Any,
    low_threshold: float,
    high_threshold: float,
    store: DatasetStore,
    **__: Any,
) -> Any:
    rec = _prepare_recommendation_monthly(store).copy()
    rec["ConsRecomm"] = np.nan
    rec.loc[(rec["ireccd"] > high_threshold) & rec["ireccd"].notna(), "ConsRecomm"] = 1.0
    rec.loc[rec["ireccd"] <= low_threshold, "ConsRecomm"] = 0.0
    # SMT ticker link with FULL fan-out: restricts output to the SMT universe
    # and expands one ticker to every SMT permno sharing it in the month
    rec = rec.merge(_smt_ticker_link(store), on=["tickerIBES", "time_avail_m"], how="inner")
    return _to_wide_panel(store, rec, "ConsRecomm")


def _op_recommendation_change_indicator(
    *_: Any,
    direction: str,
    lag_months: int = 1,
    store: DatasetStore,
    **__: Any,
) -> Any:
    rec = _prepare_recommendation_monthly(store).copy()
    rec = rec.sort_values(["tickerIBES", "time_avail_m"])
    rec["ireccd_lag"] = rec.groupby("tickerIBES")["ireccd"].shift(lag_months)
    if direction == "up":
        value_col = "UpRecomm"
        rec[value_col] = ((rec["ireccd"] < rec["ireccd_lag"]) & rec["ireccd_lag"].notna()).astype("float64")
    elif direction == "down":
        value_col = "DownRecomm"
        rec[value_col] = ((rec["ireccd"] > rec["ireccd_lag"]) & rec["ireccd_lag"].notna()).astype("float64")
    else:
        raise ValueError(f"Unsupported recommendation direction: {direction}")
    # SMT ticker triples with NO dedup: restricts to the SMT universe and fans a
    # ticker out to every SMT permno sharing it (the IBES link-table map both
    # leaked non-SMT rows and dropped multi-permno matches)
    rec = rec.merge(_smt_ticker_link(store), on=["tickerIBES", "time_avail_m"], how="inner")
    return _to_wide_panel(store, rec[["permno", "time_avail_m", value_col]], value_col)


def _op_dividend_initiation_flag(*_: Any, store: DatasetStore, **__: Any) -> Any:
    """First cash dividend in 24 months, wide-grid calendar rebuild: dividends
    zero-filled on SMT rows only (fillna BEFORE gap-fill — interior gap months
    NaN), NaN-skipping 24m rolling sum, init fires where div>0 and the lagged
    sum is ~0 (NaN comparisons False, so a span's first row never fires), 6m
    rolling max carries the flag; tree SPAN_OUT masks to the listing span."""

    strict, _span = _universe_masks(store)
    divamt_wide = _to_wide_panel(store, _prepare_cash_dividend_monthly(store), "divamt")
    div0 = divamt_wide.where(strict).fillna(0.0).where(strict)
    divamt_sum = div0.rolling(24, min_periods=1).sum()
    divsum_lag1 = divamt_sum.shift(1)
    first_month = ((div0 > 0) & (divsum_lag1 < 1e-10)).astype("float64")
    return first_month.rolling(6, min_periods=1).max()


def _op_dividend_omission_flag(*_: Any, store: DatasetStore, **__: Any) -> Any:
    """Dividend omission after an established payment pattern, wide-grid
    calendar rebuild of the script's asrol cascade over the gap-filled SMT
    span: ==1/==0 comparisons turn in-span gap-month NaN stats into 0 exactly
    like the script's astype(int), while .where(span) keeps pre-span months
    NaN so each rolling window is start-truncated like asrol; all shifts are
    plain calendar shifts on the month grid; tree SPAN_OUT masks support."""

    strict, span = _universe_masks(store)
    divamt_wide = _to_wide_panel(store, _prepare_cash_dividend_monthly(store), "divamt")
    div0 = divamt_wide.where(strict).fillna(0.0).where(strict)
    divind = (div0 > 0).astype("float64").where(strict)

    omit_any: pd.DataFrame | None = None
    for window, mean_window, payer_lag in [(3, 18, 3), (6, 18, 6), (12, 24, 12)]:
        sum_w = divind.rolling(window, min_periods=1).sum()
        temppaid = (sum_w == 1).astype("float64").where(span)
        mean_paid = temppaid.rolling(mean_window, min_periods=1).mean()
        temppayer = (mean_paid == 1).astype("float64").where(span)
        omit_w = (sum_w == 0) & (sum_w.shift(1) > 0) & (temppayer.shift(payer_lag) == 1)
        omit_any = omit_w if omit_any is None else (omit_any | omit_w)
    omitnow = omit_any.astype("float64").where(span)
    omit_recent = omitnow.rolling(2, min_periods=1).sum()
    return (omit_recent == 1).astype("float64")


def _op_seasonal_dividend_indicator(*_: Any, store: DatasetStore, **__: Any) -> Any:
    """Expected-dividend-month indicator, wide-grid calendar rebuild: ffilled
    cd3 payment codes on SMT rows, calendar payment-schedule lags with Stata
    missing-passes semantics (lag > 0 OR lag missing), and output support =
    the span of the cd3-FILTERED kept rows (the script gap-fills the filtered
    frame, which is narrower than the SMT span — span_only cannot be used)."""

    strict, _span = _universe_masks(store)
    long_df = _prepare_regular_dividend_monthly(store)
    divamt = _to_wide_panel(store, long_df, "divamt").where(strict)
    cd3 = _to_wide_panel(store, long_df, "cd3").where(strict)

    pre_ok = cd3.notna().cummax(axis=0)  # from each permno's first coded dividend month
    cd3_f = cd3.ffill().where(strict)  # ffill down the grid, re-masked to SMT rows
    base = strict & pre_ok
    div0 = divamt.fillna(0.0).where(base)
    divpaid = (div0 > 0).astype("float64").where(base)

    keep = base & cd3_f.notna() & (cd3_f != 2) & (cd3_f < 6)
    divpaid_k = divpaid.where(keep)
    kept_span = keep.cummax(axis=0) & keep.iloc[::-1].cummax(axis=0).iloc[::-1]

    divpaid_sum = divpaid_k.rolling(12, min_periods=1).sum()
    paid_or_missing: dict[int, pd.DataFrame] = {}
    for lag in [2, 5, 8, 11]:
        lagged = divpaid_k.shift(lag)
        paid_or_missing[lag] = (lagged > 0) | lagged.isna()  # Stata: missing passes >0
    temp3 = cd3_f.isin([0, 1, 3]) & (
        paid_or_missing[2] | paid_or_missing[5] | paid_or_missing[8] | paid_or_missing[11]
    )
    temp4 = (cd3_f == 4) & (paid_or_missing[5] | paid_or_missing[11])
    temp5 = (cd3_f == 5) & paid_or_missing[11]

    out = pd.DataFrame(np.nan, index=strict.index, columns=strict.columns, dtype="float64")
    out[divpaid_sum > 0] = 0.0
    out[temp3 | temp4 | temp5] = 1.0
    return out.where(kept_span)


def _op_divyieldst_signal(divamt: Any, distcd: Any, prc: Any, *_: Any, store: DatasetStore, **__: Any) -> Any:
    """Expected near-term dividend-yield terciles, wide-grid rebuild: yearly-
    coded regular dividends zero-filled on SMT rows, calendar 12m payment sum
    and calendar payment-schedule lags on the month grid (registered deviation:
    the script's rolling/shift are positional over div12>0-FILTERED SMT rows),
    per-month qcut terciles of positive expected yield, 0 bucket where the
    expected dividend is exactly 0; tree SMT_OUT masks support."""

    _ = (divamt, distcd)  # dependency-tracking args; dividends rebuilt from the store
    strict, _span = _universe_masks(store)
    long_df = _prepare_regular_dividend_monthly(store, yearly_only=True)
    divamt_wide = _to_wide_panel(store, long_df, "divamt")
    cd3 = _to_wide_panel(store, long_df, "cd3")
    prc_w = prc.where(strict)
    cd3_f = cd3.where(strict).ffill().where(strict)
    div0 = divamt_wide.where(strict).fillna(0.0).where(strict)
    div12 = div0.rolling(12, min_periods=1).sum()
    eligible = div12 > 0
    ediv1 = pd.DataFrame(np.nan, index=strict.index, columns=strict.columns, dtype="float64")
    quarterly_mask = cd3_f.isin([3, 0, 1]) | cd3_f.isna()
    ediv1[quarterly_mask] = div0.shift(2)
    ediv1[cd3_f == 4] = div0.shift(5)
    ediv1[cd3_f == 5] = div0.shift(11)
    edy1 = (ediv1 / prc_w.abs()).where(eligible)
    edy1_pos = edy1.where(edy1 > 0)
    out = _qcut_frame(edy1_pos, 3)
    out[edy1 == 0] = 0.0
    return out


def _op_piecewise_tax_ratio(
    txfo: Any,
    txfed: Any,
    txt: Any,
    txdi: Any,
    ib: Any,
    **_: Any,
) -> Any:
    year = pd.Index(ib.index).year
    tr = pd.Series(0.48, index=ib.index, dtype="float64")
    tr[(year >= 1979) & (year <= 1986)] = 0.46
    tr[year == 1987] = 0.40
    tr[(year >= 1988) & (year <= 1992)] = 0.34
    tr[year >= 1993] = 0.35

    tax = (txfo + txfed).div(tr, axis=0).div(ib)
    alt = (txt - txdi).div(tr, axis=0).div(ib)
    tax = tax.where(~(txfo.isna() | txfed.isna()), alt)

    div_by_zero = ib.eq(0) & (
        (txfo.notna() & txfo.ne(0))
        | (txfed.notna() & txfed.ne(0))
        | (txt.notna() & txt.ne(0))
        | (txdi.notna() & txdi.ne(0))
    )
    tax = tax.mask(div_by_zero, 1.0)

    cond_step3_simple = tax.isna() & ib.le(0).fillna(False)
    tax = tax.mask(cond_step3_simple, 1.0)

    cond_txfo_txfed_fixed = (txfed.isna() & txfo.gt(0).fillna(False)) | (
        ~txfed.isna() & (txfo + txfed).gt(0).fillna(False)
    )
    cond_txt_txdi = txt.gt(txdi).fillna(False)
    cond_both_missing = (
        txfo.isna()
        & txfed.isna()
        & ((txt.notna() & txt.ne(0)) | (txdi.notna() & txdi.ne(0)))
        & ib.le(0).fillna(False)
    )
    cond_standard = (cond_txfo_txfed_fixed | cond_txt_txdi | cond_both_missing) & ib.le(0).fillna(False)
    tax = tax.mask(cond_standard, 1.0)
    return tax.where(np.isfinite(tax))


def _op_total_accruals_signal(
    ivao: Any,
    ivst: Any,
    dltt: Any,
    dlc: Any,
    pstk: Any,
    sstk: Any,
    prstkc: Any,
    dv: Any,
    act: Any,
    che: Any,
    lct: Any,
    at: Any,
    lt: Any,
    ni: Any,
    oancf: Any,
    ivncf: Any,
    fincf: Any,
    **_: Any,
) -> Any:
    tempivao = ivao.fillna(0)
    tempivst = ivst.fillna(0)
    tempdltt = dltt.fillna(0)
    tempdlc = dlc.fillna(0)
    temppstk = pstk.fillna(0)
    tempsstk = sstk.fillna(0)
    tempprstkc = prstkc.fillna(0)
    tempdv = dv.fillna(0)

    temp_wc = (act - che) - (lct - tempdlc)
    temp_nc = (at - act - tempivao) - (lt - tempdlc - tempdltt)
    temp_fi = (tempivst + tempivao) - (tempdltt + tempdlc + temppstk)

    at_lag12 = at.shift(12)
    early = (
        (temp_wc - temp_wc.shift(12))
        + (temp_nc - temp_nc.shift(12))
        + (temp_fi - temp_fi.shift(12))
    )
    late = ni - (oancf + ivncf + fincf) + (tempsstk - tempprstkc - tempdv)
    year_mask = pd.Series(pd.Index(at.index).year <= 1989, index=at.index)
    total = early.where(year_mask, late, axis=0)
    total = total / at_lag12
    return total.where(np.isfinite(total))


def _op_sum_scaled_revisions(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet").copy()
    ibes = ibes[ibes["fpi"] == "1"].copy()
    ibes["tmp"] = np.where(
        ibes["fpedats"].notna() & (ibes["fpedats"] > ibes["statpers"] + pd.Timedelta(days=30)),
        1.0,
        np.nan,
    )
    ibes = ibes.sort_values(["tickerIBES", "time_avail_m"]).copy()
    ibes["meanest_lag1"] = ibes.groupby("tickerIBES")["meanest"].shift(1)
    ibes["fpedats_lag1"] = ibes.groupby("tickerIBES")["fpedats"].shift(1)
    fill_condition = ibes["tmp"].isna() & (ibes["fpedats"] == ibes["fpedats_lag1"]) & ibes["meanest_lag1"].notna()
    ibes.loc[fill_condition, "meanest"] = ibes.loc[fill_condition, "meanest_lag1"]

    # base = the FULL SignalMasterTable with SMT's own prc (the script's panel):
    # a multi-permno ticker fans out to every SMT permno sharing it (NO dedup),
    # and output support is inherently the SMT universe
    base = store.load_raw("SignalMasterTable.parquet").loc[:, ["permno", "tickerIBES", "time_avail_m", "prc"]]
    df = base.merge(
        ibes[["tickerIBES", "time_avail_m", "meanest"]],
        on=["tickerIBES", "time_avail_m"],
        how="left",
    )
    # all lags CALENDAR month-grid lags per permno (wide pivot + shift), not
    # positional shifts over SMT row order (registered deviation at SMT gaps)
    meanest_w = _to_wide_panel(store, df, "meanest")
    prc_w = _to_wide_panel(store, df, "prc")
    temp_rev = (meanest_w - meanest_w.shift(1)) / prc_w.shift(1).abs()
    terms = [temp_rev.shift(lag) for lag in range(0, 7)]
    count = terms[0].notna().astype("float64")
    total = terms[0].fillna(0.0)
    for term in terms[1:]:
        count = count + term.notna().astype("float64")
        total = total + term.fillna(0.0)
    return total.where(count == 7)  # min_count=7: all seven terms required


def _op_change_in_recommendation_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    rec = _prepare_recommendation_monthly(store).copy()
    rec["opscore"] = 6 - rec["ireccd"]
    rec = rec.sort_values(["tickerIBES", "time_avail_m"])
    rec["ChangeInRecommendation"] = rec.groupby("tickerIBES")["opscore"].diff()
    # SMT ticker triples with NO dedup: SMT universe plus multi-permno fan-out
    # (replaces the IBES link-table map — see _smt_ticker_link)
    linked = rec.merge(_smt_ticker_link(store), on=["tickerIBES", "time_avail_m"], how="inner")
    linked = linked.dropna(subset=["ChangeInRecommendation"])
    return _to_wide_panel(store, linked, "ChangeInRecommendation")


def _op_earnings_forecast_disparity_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet")
    short = ibes[
        (ibes["fpi"] == "1")
        & ibes["fpedats"].notna()
        & (ibes["fpedats"] > ibes["statpers"] + pd.Timedelta(days=30))
    ][["tickerIBES", "time_avail_m", "meanest"]].copy()
    long = ibes[ibes["fpi"] == "0"][["tickerIBES", "time_avail_m", "meanest"]].rename(
        columns={"meanest": "fgr5yr"}
    )
    actuals = store.load_raw("IBES_UnadjustedActuals.parquet").loc[
        :, ["tickerIBES", "time_avail_m", "fy0a"]
    ]
    # SMT (permno, month, ticker) base — no keep-first dedup: SMT domain plus
    # multi-permno ticker fan-out
    df = _smt_ticker_link(store)
    df = df.merge(short, on=["tickerIBES", "time_avail_m"], how="left")
    df = df.merge(long, on=["tickerIBES", "time_avail_m"], how="left")
    df = df.merge(actuals, on=["tickerIBES", "time_avail_m"], how="left")
    df["tempShort"] = np.where(
        df["fy0a"] == 0,
        np.nan,
        100 * (df["meanest"] - df["fy0a"]) / df["fy0a"].abs(),
    )
    df["EarningsForecastDisparity"] = df["fgr5yr"] - df["tempShort"]
    df = df.dropna(subset=["EarningsForecastDisparity"])
    return _to_wide_panel(store, df, "EarningsForecastDisparity")


def _op_excluded_expenses_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    # script base: SMT rows with a gvkey inner-joined to quarterly epspiq, with
    # SMT's own tickerIBES linking the IBES actuals (replaces the two link-table
    # maps, whose universe both leaked non-SMT rows and dropped SMT rows)
    smt = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "gvkey", "time_avail_m", "tickerIBES"]]
        .dropna(subset=["gvkey"])
    )
    qcomp = store.load_raw("m_QCompustat.parquet").loc[:, ["gvkey", "time_avail_m", "epspiq"]]
    actuals = store.load_raw("IBES_UnadjustedActuals.parquet").loc[
        :, ["tickerIBES", "time_avail_m", "int0a"]
    ]
    df = smt.merge(qcomp, on=["gvkey", "time_avail_m"], how="inner")
    df = df.merge(actuals, on=["tickerIBES", "time_avail_m"], how="left")
    df["ExclExp"] = df["int0a"] - df["epspiq"]
    # look-ahead correction (registered deviation): reference clips at FULL-SAMPLE
    # 1%/99% quantiles — month-t bounds there depend on future data. Clip within
    # each month's cross-section instead.
    grouped = df.groupby("time_avail_m")["ExclExp"]
    lo = grouped.transform(lambda s: s.quantile(0.01))
    hi = grouped.transform(lambda s: s.quantile(0.99))
    df["ExclExp"] = df["ExclExp"].clip(lower=lo, upper=hi)
    df = df.dropna(subset=["ExclExp"])
    return _to_wide_panel(store, df, "ExclExp")


def _op_chforecast_accrual_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet")
    ibes = ibes[ibes["fpi"] == "1"][["tickerIBES", "time_avail_m", "meanest"]].copy()

    comp = (
        store.load_raw("m_aCompustat.parquet")
        .loc[:, ["permno", "time_avail_m", "act", "che", "lct", "dlc", "txp", "at"]]
        .drop_duplicates(["permno", "time_avail_m"], keep="first")
        .sort_values(["permno", "time_avail_m"])
    )
    # permno->tickerIBES from SignalMasterTable (not the IBES link table): the
    # output universe becomes m_aCompustat ∩ SMT ∩ IBES automatically
    df = comp.merge(_smt_ticker_link(store), on=["permno", "time_avail_m"], how="left")
    df = df.merge(ibes, on=["tickerIBES", "time_avail_m"], how="left")

    # calendar t-12 self-merge lags (script create_calendar_lag), not shift(12)
    lag_cols = ["act", "che", "lct", "dlc", "txp", "at"]
    lag = df[["permno", "time_avail_m"] + lag_cols].copy()
    lag["time_avail_m"] = lag["time_avail_m"] + pd.DateOffset(months=12)
    df = df.merge(
        lag.rename(columns={col: f"l12_{col}" for col in lag_cols}),
        on=["permno", "time_avail_m"],
        how="left",
    )
    numerator = (
        (df["act"] - df["l12_act"])
        - (df["che"] - df["l12_che"])
        - (
            (df["lct"] - df["l12_lct"])
            - (df["dlc"] - df["l12_dlc"])
            - (df["txp"] - df["l12_txp"])
        )
    )
    denominator = (df["at"] + df["l12_at"]) / 2.0
    df["tempAccruals"] = numerator / denominator.replace(0, np.nan)
    df["tempsort"] = df.groupby("time_avail_m")["tempAccruals"].transform(
        lambda x: pd.qcut(x, q=2, labels=False, duplicates="drop") + 1
    )
    df["meanest_l"] = df.groupby("permno", sort=False)["meanest"].shift(1)
    df["ChForecastAccrual"] = np.nan
    increase = df["meanest"].gt(df["meanest_l"]) & df["meanest"].notna() & df["meanest_l"].notna()
    decrease = df["meanest"].lt(df["meanest_l"]) & df["meanest"].notna() & df["meanest_l"].notna()
    df.loc[increase, "ChForecastAccrual"] = 1.0
    df.loc[decrease, "ChForecastAccrual"] = 0.0
    df.loc[df["tempsort"] == 1, "ChForecastAccrual"] = np.nan
    df = df.dropna(subset=["ChForecastAccrual"])
    return _to_wide_panel(store, df, "ChForecastAccrual")


def _op_fgr5yr_lag_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet")
    ibes = ibes[ibes["fpi"] == "0"][["tickerIBES", "time_avail_m", "meanest"]].rename(
        columns={"meanest": "fgr5yr"}
    )
    comp = store.load_raw("m_aCompustat.parquet").loc[
        :, ["permno", "time_avail_m", "ceq", "ib", "txdi", "dv", "sale", "ni", "dp"]
    ].drop_duplicates(["permno", "time_avail_m"], keep="first")
    # SMT ticker link (the script's merge): the SMT universe and SMT ticker
    # gate both the June observation and the calendar t-6 lag source
    df = comp.merge(_smt_ticker_link(store), on=["permno", "time_avail_m"], how="inner")
    df = df.merge(ibes, on=["tickerIBES", "time_avail_m"], how="inner")
    df = df.dropna(subset=["ceq", "ib", "txdi", "dv", "sale", "ni", "dp", "fgr5yr"])
    lag_lookup = df[["permno", "time_avail_m", "fgr5yr"]].rename(
        columns={"time_avail_m": "lag6_date", "fgr5yr": "fgr5yrLag"}
    )
    df["lag6_date"] = df["time_avail_m"] - pd.DateOffset(months=6)
    df = df.merge(lag_lookup, on=["permno", "lag6_date"], how="left")
    df = df[df["time_avail_m"].dt.month == 6][["permno", "time_avail_m", "fgr5yrLag"]]
    df = df.dropna(subset=["fgr5yrLag"])
    return _to_wide_panel(store, _expand_hold_months(df, "fgr5yrLag", 12), "fgr5yrLag")


def _op_sfe_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    ibes = store.load_raw("IBES_EPS_Unadj.parquet")
    ibes = ibes[
        (ibes["fpi"] == "1")
        & (ibes["statpers"].dt.month == 3)
        & ibes["fpedats"].notna()
        & (ibes["fpedats"] > ibes["statpers"] + pd.Timedelta(days=90))
    ][["tickerIBES", "time_avail_m", "medest", "numest"]].copy()
    ibes["prc_time"] = ibes["time_avail_m"] - pd.DateOffset(months=3)

    # single inner merge with SignalMasterTable (permno, ticker, prc) at the
    # price month — the qcut coverage split then runs on the script's sample
    smt = (
        store.load_raw("SignalMasterTable.parquet")
        .loc[:, ["permno", "time_avail_m", "tickerIBES", "prc"]]
        .dropna(subset=["tickerIBES"])
        .rename(columns={"time_avail_m": "prc_time"})
    )
    comp = store.load_raw("m_aCompustat.parquet").loc[:, ["permno", "time_avail_m", "datadate"]]

    df = ibes.merge(smt, on=["tickerIBES", "prc_time"], how="inner")
    df = df.merge(comp, on=["permno", "time_avail_m"], how="inner")
    df = df[pd.to_datetime(df["datadate"]).dt.month == 12].copy()
    df["tempcoverage"] = df.groupby("time_avail_m")["numest"].transform(
        lambda x: pd.qcut(x, q=2, labels=False, duplicates="drop") + 1
    )
    df = df[df["tempcoverage"] == 1].copy()
    df["sfe"] = df["medest"] / df["prc"].abs()
    df = df.dropna(subset=["sfe"])
    return _to_wide_panel(store, _expand_hold_months(df[["permno", "time_avail_m", "sfe"]], "sfe", 12), "sfe")


def _op_earnings_streak_signal(*_: Any, store: DatasetStore, **__: Any) -> Any:
    df = store.load_raw("IBES_EPS_Adj.parquet").copy()
    df = df[(df["fpi"] == "6")].dropna(subset=["actual", "meanest", "price"]).copy()
    df["time_avail_m"] = pd.to_datetime(df["anndats_act"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["tickerIBES", "time_avail_m", "anndats_act", "statpers"])
    df = df.groupby(["tickerIBES", "time_avail_m"]).last().reset_index()
    df["surp"] = (df["actual"] - df["meanest"]) / df["price"]
    df = df.sort_values(["tickerIBES", "anndats_act"])
    df["surp_sign"] = np.sign(df["surp"])
    df["surp_sign_lag"] = df.groupby("tickerIBES")["surp_sign"].shift(1)
    df = df[df["surp_sign"] == df["surp_sign_lag"]].copy()
    temp = df[["tickerIBES", "time_avail_m", "anndats_act", "surp"]].copy()

    # SMT (permno, month, ticker) base — the script's row set, so the positional
    # ffill below runs over exactly the same rows (no positional/calendar residual)
    smt = store.load_raw("SignalMasterTable.parquet").loc[:, ["permno", "time_avail_m", "tickerIBES"]]
    panel = smt.merge(temp, on=["tickerIBES", "time_avail_m"], how="left").drop(columns=["tickerIBES"])
    panel = panel.sort_values(["permno", "time_avail_m"])
    panel["anndats_act"] = panel.groupby("permno")["anndats_act"].ffill()
    panel["surp"] = panel.groupby("permno")["surp"].ffill()
    panel = panel.dropna(subset=["anndats_act"])
    ann_month = pd.to_datetime(panel["anndats_act"]).dt.to_period("M")
    cur_month = panel["time_avail_m"].dt.to_period("M")
    panel["month_diff"] = (cur_month - ann_month).apply(lambda x: x.n)
    panel = panel[panel["month_diff"] <= 6].copy()
    panel = panel.dropna(subset=["surp"]).rename(columns={"surp": "EarningsStreak"})
    return _to_wide_panel(store, panel, "EarningsStreak")


def _op_industry_big_return(ret: Any, mve_c: Any, sic: Any, *, store: "DatasetStore", **_: Any) -> Any:
    """Mean return of each FF48 industry's big firms (size rank > 0.7) mapped to
    ALL SMT rows with a valid FF48 — the pool never requires ret or mve to be
    non-null (relrank gives NaN-size rows a NaN rank and they still RECEIVE the
    mean; NaN-ret big firms drop out of the skipna mean), and only rank >= 0.7
    firms are nulled. Inputs SMT-masked internally, so values are only ever
    assigned on SMT rows (the script's universe and output support)."""

    strict, _span = _universe_masks(store)
    ret = ret.where(strict)
    mve_c = mve_c.where(strict)
    sic = sic.where(strict)
    ff48_func = _get_ff48_func()
    ff48 = sic.apply(lambda col: col.map(ff48_func))
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype="float64")
    for date in ret.index:
        r = ret.loc[date]
        size = mve_c.loc[date]
        industry = ff48.loc[date]
        pool = industry.notna()
        if not pool.any():
            continue
        group_rank = size[pool].groupby(industry[pool]).rank(pct=True, method="average")
        big_mask = group_rank > 0.7
        industry_mean = r[pool][big_mask].groupby(industry[pool][big_mask]).mean()
        mapped = industry[pool].map(industry_mean)
        mapped[group_rank >= 0.7] = np.nan  # NaN-rank rows KEEP the industry mean
        out.loc[date, mapped.index] = mapped
    return out


def _stata_edge_metric(metric: pd.DataFrame) -> pd.DataFrame:
    metric = metric.replace([np.inf, -np.inf], np.nan)
    return metric.fillna(np.inf)


def _op_piotroski_score(
    fopt: Any,
    oancf: Any,
    ib: Any,
    at: Any,
    dltt: Any,
    act: Any,
    lct: Any,
    txt: Any,
    xint: Any,
    sale: Any,
    ceq: Any,
    mve: Any,
    shrout: Any,
    *,
    store: "DatasetStore",
    **_: Any,
) -> Any:
    """Piotroski F-score on the script's merged-panel domain: U = rows present
    in m_aCompustat AND SignalMasterTable AND monthlyCRSP (the chain of inner
    merges). EVERY input is U-masked BEFORE the 12-month calendar lags, so a
    lag lands NaN when (permno, t-12) is not in U or the variable was NaN there
    — reproducing fill_date_gaps + stata_multi_lag semantics including the p9
    l12_shrout -> +inf quirk. BM quintiles via per-month pd.qcut breakpoints
    (duplicates='drop': a collapsed top label < 5 NaNs the whole month)."""

    strict, _span = _universe_masks(store)
    u_mask = _compustat_rows_mask(store) & strict & _crsp_rows_mask(store)
    fopt, oancf, ib, at, dltt, act, lct, txt, xint, sale, ceq, mve, shrout = (
        panel.where(u_mask)
        for panel in (fopt, oancf, ib, at, dltt, act, lct, txt, xint, sale, ceq, mve, shrout)
    )
    fopt_eff = _op_fillna(fopt, oancf)
    tempebit = ib + txt + xint
    l12_ib = _op_lag(ib, 12)
    l12_at = _op_lag(at, 12)
    l12_dltt = _op_lag(dltt, 12)
    l12_act = _op_lag(act, 12)
    l12_lct = _op_lag(lct, 12)
    l12_sale = _op_lag(sale, 12)
    l12_shrout = _op_lag(shrout, 12)

    p1 = (_stata_edge_metric(ib) > 0).astype("float64")
    p2 = (_stata_edge_metric(fopt_eff) > 0).astype("float64")
    p3 = (_stata_edge_metric((ib / at) - (l12_ib / l12_at)) > 0).astype("float64")
    p4 = (_stata_edge_metric(fopt_eff - ib) > 0).astype("float64")
    p5 = (_stata_edge_metric((dltt / at) - (l12_dltt / l12_at)) < 0).astype("float64")
    p6 = (_stata_edge_metric((act / lct) - (l12_act / l12_lct)) > 0).astype("float64")
    p7 = (_stata_edge_metric((tempebit / sale) - (tempebit / l12_sale)) > 0).astype("float64")
    p8 = (_stata_edge_metric((sale / at) - (l12_sale / l12_at)) > 0).astype("float64")
    p9 = (_stata_edge_metric(l12_shrout - shrout) >= 0).astype("float64")

    out = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    required = (
        fopt_eff.isna()
        | ib.isna()
        | at.isna()
        | dltt.isna()
        | sale.isna()
        | act.isna()
        | tempebit.isna()
        | shrout.isna()
    )
    out = out.where(~required)

    bm = np.log(ceq.where(ceq > 0) / mve).replace([np.inf, -np.inf], np.nan)
    bm_q = _qcut_frame(bm, 5)  # per-month qcut value breakpoints (script fastxtile)
    return out.where(bm_q == 5).where(u_mask)


def _op_rolling_trend_slope_scaled(x: Any, window_months: int, min_obs: int = 30, **_: Any) -> Any:
    """OLS slope of x on calendar position over the trailing `window_months`
    NON-NULL observations, scaled by the trailing calendar-window mean.

    Slope replicates polars_ols rolling_ols(null_policy='drop'): null rows are
    COMPACTED before windowing, so the regression window is the last
    window_months non-null observations (spanning more calendar months when
    nulls are present) with min_obs of them required; the regressor stays the
    true calendar position; null rows inherit the last valid row's coefficient
    (coefficient forward-fill). The denominator stays the asrol CALENDAR-window
    mean (window_months months, min_obs samples). Grid position is
    affine-equivalent to the script's time_numeric — slopes are shift-invariant.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError("rolling_trend_slope_scaled expects a monthly wide DataFrame")
    values = x.to_numpy(dtype="float64")
    beta = np.full(values.shape, np.nan, dtype="float64")
    for j in range(values.shape[1]):
        col = values[:, j]
        valid_pos = np.flatnonzero(np.isfinite(col))
        m = valid_pos.size
        if m < min_obs:
            continue
        t_all = valid_pos.astype("float64")
        v_all = col[valid_pos]
        slopes = np.full(m, np.nan, dtype="float64")
        # truncated leading windows: fewer than window_months obs available
        for i in range(min_obs - 1, min(window_months - 1, m - 1) + 1):
            tw = t_all[: i + 1]
            vw = v_all[: i + 1]
            td = tw - tw.mean()
            slopes[i] = (td * (vw - vw.mean())).sum() / (td**2).sum()
        if m >= window_months:
            tv = np.lib.stride_tricks.sliding_window_view(t_all, window_months)
            vv = np.lib.stride_tricks.sliding_window_view(v_all, window_months)
            td = tv - tv.mean(axis=1, keepdims=True)
            full = (td * (vv - vv.mean(axis=1, keepdims=True))).sum(axis=1) / (td**2).sum(axis=1)
            slopes[window_months - 1 :] = full
        beta[valid_pos, j] = slopes
    # coefficient forward-fill onto null-x months (polars_ols scatter-back)
    beta_df = pd.DataFrame(beta, index=x.index, columns=x.columns).ffill()
    mean_x = x.rolling(window_months, min_periods=min_obs).mean()
    return beta_df / mean_x.where(mean_x != 0.0)  # zero mean -> NaN as before


def _op_delegate_to_script(current_factor: str | None, **_: Any) -> Any:
    if current_factor is None:
        raise NotImplementedError("Script-delegated operator needs the current factor context")
    raise NotImplementedError(f"{current_factor} is planned for script-backed execution, not native tree execution")


OPERATOR_REGISTRY: dict[str, Callable[..., Any]] = {
    "add": _op_add,
    "sub": _op_sub,
    "mul": _op_mul,
    "div": _op_div,
    "log": _op_log,
    "pow": _op_pow,
    "abs": _op_abs,
    "lag": _op_lag,
    "delta": _op_delta,
    "mean_of_lags": _op_mean_of_lags,
    "compound_return": _op_compound_return,
    "span_fill": _op_span_fill,
    "smt_fill": _op_smt_fill,
    "smt_only": _op_smt_only,
    "smt_gvkey_only": _op_smt_gvkey_only,
    "span_only": _op_span_only,
    "crsp_only": _op_crsp_only,
    "compustat_rows_only": _op_compustat_rows_only,
    "compustat_span_fill": _op_compustat_span_fill,
    "abs_cap_to_null": _op_abs_cap_to_null,
    "require_any_lag": _op_require_any_lag,
    "cash_rdq_signal": _op_cash_rdq_signal,
    "daily_monthly_stat": _op_daily_monthly_stat,
    "high52_signal": _op_high52_signal,
    "ff3_idio_stat": _op_ff3_idio_stat,
    "rolling_market_rmse": _op_rolling_market_rmse,
    "monthly_rolling_beta": _op_monthly_rolling_beta,
    "monthly_rolling_multibeta": _op_monthly_rolling_multibeta,
    "residual_momentum": _op_residual_momentum,
    "price_delay": _op_price_delay,
    "coskew_signal": _op_coskew_signal,
    "betafp_signal": _op_betafp_signal,
    "betatailrisk_signal": _op_betatailrisk_signal,
    "trendfactor_signal": _op_trendfactor_signal,
    "announcement_return_signal": _op_announcement_return_signal,
    "zerotrade_signal": _op_zerotrade_signal,
    "fillna": _op_fillna,
    "ffill": _op_ffill,
    "max": _op_max,
    "time_lagged_value": _op_time_lagged_value,
    "market_equity_matched_to_datadate": _op_market_equity_matched_to_datadate,
    "exchange_switch_indicator": _op_exchange_switch_indicator,
    "rolling_mean": _op_rolling_mean,
    "rolling_std": _op_rolling_std,
    "rolling_trend_slope_scaled": _op_rolling_trend_slope_scaled,
    "trim_global": _op_trim_global,
    "monthly_fastxtile_exclude": _op_monthly_fastxtile_exclude,
    "positive_only": _op_positive_only,
    "negative_to_null": _op_negative_to_null,
    "zero_to_null": _op_zero_to_null,
    "nonzero": _op_nonzero,
    "positive_indicator": _op_positive_indicator,
    "nonfinancial_only": _op_nonfinancial_only,
    "manufacturing_only": _op_manufacturing_only,
    "financial_excluded": _op_financial_excluded,
    "shrcd_le11": _op_shrcd_le11,
    "bm_defined_gate": _op_bm_defined_gate,
    "deldrc_sample_filter": _op_deldrc_sample_filter,
    "convertible_debt_indicator": _op_convertible_debt_indicator,
    "enterprise_multiple": _op_enterprise_multiple,
    "size_filtered": _op_size_filtered,
    "hire_growth": _op_hire_growth,
    "netpayout_yield_signal": _op_netpayout_yield_signal,
    "payout_yield_signal": _op_payout_yield_signal,
    "investment_to_sales_scaled": _op_investment_to_sales_scaled,
    "coalesce": _op_coalesce,
    "year_filtered": _op_year_filtered,
    "threshold_min": _op_threshold_min,
    "size_decile_filtered": _op_size_decile_filtered,
    "industry_adjusted_mean": _op_industry_adjusted_mean,
    "monthly_mean": _op_monthly_mean,
    "monthly_max": _op_monthly_max,
    "monthly_skew": _op_monthly_skew,
    "high_52": _op_high_52,
    "firm_age": _op_firm_age,
    "gr_ltnoa": _op_gr_ltnoa,
    "firm_age_momentum": _op_firm_age_momentum,
    "industry_weighted_momentum": _op_industry_weighted_momentum,
    "weighted_lagged_rank_growth": _op_weighted_lagged_rank_growth,
    "momentum_reversal_signal": _op_momentum_reversal_signal,
    "momentum_volume_signal": _op_momentum_volume_signal,
    "share_volume_signal": _op_share_volume_signal,
    "accruals_bm_signal": _op_accruals_bm_signal,
    "rolling_industry_herfindahl": _op_rolling_industry_herfindahl,
    "earnings_consistency": _op_earnings_consistency,
    "seasonal_surprise_zscore": _op_seasonal_surprise_zscore,
    "earnings_increase_streak": _op_earnings_increase_streak,
    "industry_big_mean": _op_industry_big_mean,
    "intangible_residual": _op_intangible_residual,
    "brand_invest_signal": _op_brand_invest_signal,
    "equity_duration_signal": _op_equity_duration_signal,
    "abnormal_accruals_signal": _op_abnormal_accruals_signal,
    "analyst_value_signal": _op_analyst_value_signal,
    "analyst_optimism_ratio": _op_analyst_optimism_ratio,
    "binary_recommendation_level": _op_binary_recommendation_level,
    "bmdec_signal": _op_bmdec_signal,
    "dividend_initiation_flag": _op_dividend_initiation_flag,
    "dividend_omission_flag": _op_dividend_omission_flag,
    "divyieldst_signal": _op_divyieldst_signal,
    "predicted_forecast_error": _op_predicted_forecast_error,
    "feps_signal": _op_feps_signal,
    "frontier_signal": _op_frontier_signal,
    "piecewise_tax_ratio": _op_piecewise_tax_ratio,
    "rd_ability_signal": _op_rd_ability_signal,
    "recommendation_change_indicator": _op_recommendation_change_indicator,
    "ms_signal": _op_ms_signal,
    "seasonal_dividend_indicator": _op_seasonal_dividend_indicator,
    "change_in_recommendation_signal": _op_change_in_recommendation_signal,
    "earnings_forecast_disparity_signal": _op_earnings_forecast_disparity_signal,
    "excluded_expenses_signal": _op_excluded_expenses_signal,
    "chforecast_accrual_signal": _op_chforecast_accrual_signal,
    "fgr5yr_lag_signal": _op_fgr5yr_lag_signal,
    "sfe_signal": _op_sfe_signal,
    "sum_scaled_revisions": _op_sum_scaled_revisions,
    "earnings_streak_signal": _op_earnings_streak_signal,
    "total_accruals_signal": _op_total_accruals_signal,
    "industry_big_return": _op_industry_big_return,
    "piotroski_score": _op_piotroski_score,
    "surprise_rd_indicator": _op_surprise_rd_indicator,
}

for op_name in DELEGATED_SPECIAL_OPERATORS:
    OPERATOR_REGISTRY.setdefault(op_name, _op_delegate_to_script)
for op_name in SNAKE_TO_FACTOR:
    OPERATOR_REGISTRY.setdefault(op_name, _op_delegate_to_script)


class ExpressionEvaluator:
    """Evaluate native factor expression trees against raw monthly panels."""

    def __init__(self, store: DatasetStore, script_adapter: FactorScriptAdapter | None = None) -> None:
        self.store = store
        self.script_adapter = script_adapter

    def evaluate(self, expr: dict[str, Any], current_factor: str) -> Any:
        """Recursively evaluate an expression tree."""

        kind = expr.get("kind")
        if kind == "var":
            return self.store.get(expr["dataset"], expr["name"], row_filter=expr.get("filter"))
        if kind == "const":
            return expr["value"]
        if kind != "op":
            raise ValueError(f"Unsupported expression node: {expr}")

        op = expr["op"]
        fn = OPERATOR_REGISTRY.get(op)
        if fn is None:
            raise KeyError(f"Operator not implemented: {op}")
        if fn is _op_delegate_to_script:
            if self.script_adapter is None:
                raise NotImplementedError(f"{current_factor} requires script-backed execution for operator {op}")
            path = self.script_adapter.script_to_parquet(current_factor)
            return pd.read_parquet(path)
        args = [self.evaluate(arg, current_factor) for arg in expr.get("args", [])]
        params = dict(expr.get("params", {}))
        params["current_factor"] = current_factor
        return fn(*args, **params, store=self.store)


def build_universe_masks(store: DatasetStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boolean month×permno masks replicating the SignalMasterTable universe.

    strict: the (permno, month) row exists in monthlyCRSP with shrcd ∈ {10,11,12}
    and exchcd ∈ {1,2,3} — the SignalMasterTable membership filter.
    span:   True from a permno's first strict-True month through its last —
    the domain legacy scripts operate on after ``fill_date_gaps`` (gap months
    inside the listing span are legitimate output rows for those scripts).

    Native factor outputs must be masked with one of these (chosen per source
    script) — the engine's wide grid otherwise manufactures values at
    (permno, month) cells the legacy universe never contains.
    """

    raw = store.load_raw("monthlyCRSP.parquet")
    sub = raw.loc[:, ["permno", "time_avail_m", "shrcd", "exchcd"]]
    keep = sub[sub["shrcd"].isin([10, 11, 12]) & sub["exchcd"].isin([1, 2, 3])]
    marker = keep.assign(present=1.0).pivot_table(
        index="time_avail_m", columns="permno", values="present", aggfunc="last"
    )
    strict = (
        marker.reindex(index=store.template_index, columns=store.template_columns)
        .notna()
    )
    alive_from_first = strict.cummax(axis=0)
    alive_until_last = strict.iloc[::-1].cummax(axis=0).iloc[::-1]
    span = alive_from_first & alive_until_last
    return strict, span


def _factor_to_wide_output(
    value: Any,
    template_index: pd.DatetimeIndex,
    template_columns: pd.Index,
    universe_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align an evaluated factor result to the common wide output template.

    ``universe_mask`` (from build_universe_masks) nulls cells outside the legacy
    universe; None preserves the raw grid (pre-fix behavior, kept for A/B diffs).
    """

    if isinstance(value, pd.DataFrame):
        wide = value.reindex(index=template_index, columns=template_columns)
    elif isinstance(value, pd.Series):
        wide = pd.DataFrame({col: value for col in template_columns}).reindex(index=template_index, columns=template_columns)
    else:
        wide = pd.DataFrame(value, index=template_index, columns=template_columns)
    if universe_mask is not None:
        wide = wide.where(universe_mask)
    wide.index.name = "month"
    wide.columns.name = "permno"
    return wide.astype("float32")


def _write_factor_parquet(factor: str, wide: pd.DataFrame, output_dir: Path) -> Path:
    """Persist a factor panel in the standardized wide parquet layout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{factor}.parquet"
    wide.to_parquet(path, compression="zstd")
    return path


def calculate_factors_from_plan(
    plan: list[FactorPlanStep],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    force_script: bool = False,
    clear_native_cache_between_steps: bool = True,
    python_executable: str | None = None,
) -> dict[str, Path]:
    """Execute a factor plan and write one wide parquet per factor."""

    output_dir = Path(output_dir)
    template_index, template_columns = build_output_template()
    store = DatasetStore(template_index, template_columns)
    adapter = FactorScriptAdapter(template_index, template_columns, output_dir, python_executable=python_executable)
    evaluator = ExpressionEvaluator(store, script_adapter=adapter)
    written: dict[str, Path] = {}

    for step in plan:
        if step.executor == "bootstrap":
            adapter.ensure_signal_master()
            continue

        if step.executor == "native":
            for factor in step.factors:
                value = evaluator.evaluate(P1_FACTOR_SPECS[factor].expression, current_factor=factor)
                wide = _factor_to_wide_output(value, template_index, template_columns)
                written[factor] = _write_factor_parquet(factor, wide, output_dir)
            if clear_native_cache_between_steps:
                store.clear()
            continue

        if step.executor == "script":
            for factor in step.factors:
                written[factor] = adapter.script_to_parquet(factor, force_script=force_script)
            continue

        raise ValueError(f"Unknown executor: {step.executor}")

    return written


def plan_and_calculate_all_factors(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    force_script: bool = False,
    python_executable: str | None = None,
    allow_script_backed: bool = True,
) -> tuple[list[FactorPlanStep], dict[str, Path]]:
    """Convenience wrapper for planning and executing the full P1 library."""

    plan = plan_factor_calculation_order(allow_script_backed=allow_script_backed)
    outputs = calculate_factors_from_plan(
        plan,
        output_dir=output_dir,
        force_script=force_script,
        python_executable=python_executable,
    )
    return plan, outputs


def native_conversion_status() -> pd.DataFrame:
    """Return a factor-level status table for native conversion progress."""

    rows: list[dict[str, Any]] = []
    for factor in sorted(P1_FACTOR_SPECS):
        deps = P1_FACTOR_DEPENDENCIES[factor]
        has_manual_tree = factor in MANUAL_TREES
        daily_dependent = any(dep in {"dailyCRSP.parquet", "dailyFF.parquet"} for dep in deps)
        ops = sorted(_list_ops(P1_FACTOR_SPECS[factor].expression))
        missing_ops = [op for op in ops if op not in OPERATOR_REGISTRY]
        delegated_ops = [op for op in ops if op in OPERATOR_REGISTRY and OPERATOR_REGISTRY[op] is _op_delegate_to_script]
        native_ready = _can_execute_natively(factor)
        if native_ready:
            reason = "native_ready"
        elif not has_manual_tree:
            reason = "missing_explicit_tree"
        elif daily_dependent:
            reason = "daily_kernel_not_streaming_safe"
        elif missing_ops:
            reason = "missing_operator"
        elif delegated_ops:
            reason = "delegated_operator"
        else:
            reason = "not_native"
        rows.append(
            {
                "factor": factor,
                "native_ready": native_ready,
                "reason": reason,
                "dependencies": ";".join(deps),
                "operators": ";".join(ops),
                "missing_operators": ";".join(missing_ops),
                "delegated_operators": ";".join(delegated_ops),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    plan = plan_factor_calculation_order()
    native = sum(1 for step in plan if step.executor == "native" for _ in step.factors)
    script = sum(1 for step in plan if step.executor == "script" for _ in step.factors)
    print(f"Planned steps: {len(plan)}")
    print(f"Native factors: {native}")
    print(f"Script-backed factors: {script}")

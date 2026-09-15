"""Golden validation harness for the P1 factor engine rebuild (WS0).

Two jobs:

1. Freeze golden baselines from the CURRENT engine:
   - ``freeze_golden_native``  -> Data/golden/native/{factor}.parquet   (148 native factors)
   - ``freeze_golden_oracle``  -> Data/golden/oracle/{factor}.parquet   (23 script-backed factors,
     produced by the original Open Source Asset Pricing predictor scripts)

2. Diff candidate outputs against goldens:
   - ``diff_panels`` / ``diff_factor_files``: NaN-mask equality + value comparison with
     configurable tolerance + worst-offender samples
   - ``validate_factors``: per-factor PASS/FAIL table written as a CSV report

Every workstream of the v2 rebuild (see p1_plan_engine_v2_design.md) gates through this
module: orchestration-only changes must diff exactly (atol=rtol=0); new numba kernels diff
against the oracle outputs with an explicit tolerance.

CLI:
    python p1_validation.py freeze-native [--force]
    python p1_validation.py freeze-oracle [--force]
    python p1_validation.py compare <candidate_dir> --golden-dir <dir> [--atol X --rtol Y]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Factor_Construction.p1_factor_engine import (
        FACTOR_TO_SCRIPT,
        DatasetStore,
        ExpressionEvaluator,
        FactorScriptAdapter,
        build_output_template,
        build_universe_masks,
        plan_factor_calculation_order,
        _factor_to_wide_output,
        _write_factor_parquet,
    )
    from Factor_Construction.p1_factor_specs import P1_FACTOR_SPECS
except ModuleNotFoundError:
    from p1_factor_engine import (
        FACTOR_TO_SCRIPT,
        DatasetStore,
        ExpressionEvaluator,
        FactorScriptAdapter,
        build_output_template,
        build_universe_masks,
        plan_factor_calculation_order,
        _factor_to_wide_output,
        _write_factor_parquet,
    )
    from p1_factor_specs import P1_FACTOR_SPECS

ROOT = Path(__file__).resolve().parent
GOLDEN_ROOT = ROOT / "Data" / "golden"
GOLDEN_NATIVE_DIR = GOLDEN_ROOT / "native"
GOLDEN_ORACLE_DIR = GOLDEN_ROOT / "oracle"
REPORT_DIR = GOLDEN_ROOT / "reports"


# ---------------------------------------------------------------------------
# Diff core
# ---------------------------------------------------------------------------


@dataclass
class DiffReport:
    """Result of comparing one candidate panel against one golden panel."""

    factor: str
    passed: bool
    # shape / alignment
    shape_equal: bool
    candidate_shape: tuple[int, int]
    golden_shape: tuple[int, int]
    index_mismatches: int
    column_mismatches: int
    # NaN structure
    nan_only_in_candidate: int
    nan_only_in_golden: int
    both_valued_cells: int
    # inf cells are legitimate factor values (log/div-by-zero artifacts, e.g. CompEquIss
    # has 160). They are compared by exact equality; counted here so drift in inf
    # structure is visible, never silent.
    inf_in_candidate: int
    inf_in_golden: int
    # values (on jointly non-NaN cells)
    max_abs_diff: float
    cells_beyond_tol: int
    atol: float
    rtol: float
    # diagnostics
    worst_offenders: list[dict] = field(default_factory=list)
    error: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("worst_offenders")
        return d


def diff_panels(
    candidate: pd.DataFrame,
    golden: pd.DataFrame,
    *,
    factor: str = "",
    atol: float = 0.0,
    rtol: float = 0.0,
    n_offenders: int = 5,
) -> DiffReport:
    """Compare two wide (month x permno) factor panels.

    PASS requires: identical NaN masks AND all jointly-valued cells within tolerance
    on the intersection of index/columns, with no index/column mismatches.
    """

    idx_mism = len(candidate.index.symmetric_difference(golden.index))
    col_mism = len(candidate.columns.symmetric_difference(golden.columns))
    shape_equal = candidate.shape == golden.shape and idx_mism == 0 and col_mism == 0

    common_idx = candidate.index.intersection(golden.index)
    common_col = candidate.columns.intersection(golden.columns)
    c = candidate.loc[common_idx, common_col].to_numpy(dtype="float64")
    g = golden.loc[common_idx, common_col].to_numpy(dtype="float64")

    c_nan = np.isnan(c)
    g_nan = np.isnan(g)
    nan_only_c = int((c_nan & ~g_nan).sum())
    nan_only_g = int((~c_nan & g_nan).sum())

    both = ~c_nan & ~g_nan
    n_both = int(both.sum())
    n_inf_c = int(np.isinf(c).sum())
    n_inf_g = int(np.isinf(g).sum())
    if n_both:
        cb, gb = c[both], g[both]
        # Exact-equal cells (finite==finite or same-signed inf==inf) have diff 0 by
        # definition. Subtracting only unequal cells makes inf semantics explicit:
        # (inf, finite) -> inf, (+inf, -inf) -> inf — both correctly beyond any
        # tolerance — and inf-inf (the NaN-producing case) cannot occur.
        eq = cb == gb
        ne = ~eq
        diff = np.zeros_like(cb)
        diff[ne] = np.abs(cb[ne] - gb[ne])
        assert not np.isnan(diff).any(), "diff produced NaN — unhandled value combination, investigate"
        max_abs = float(diff.max())
        # Tolerance applies to finite-vs-finite disagreements only. A cell whose inf
        # structure differs (inf vs finite, +inf vs -inf) is a mismatch at ANY
        # tolerance — rtol * |inf| would otherwise yield an infinite tolerance (and
        # rtol=0 would yield 0*inf=NaN) and silently absorb the disagreement.
        finite_pair = np.isfinite(cb) & np.isfinite(gb)
        check = ne & finite_pair
        n_beyond = int((diff[check] > atol + rtol * np.abs(gb[check])).sum())
        n_beyond += int((ne & ~finite_pair).sum())
    else:
        max_abs, n_beyond = 0.0, 0

    worst: list[dict] = []
    if n_beyond:
        rows, cols = np.where(both)
        bad_pos = np.argsort(-diff)[:n_offenders]
        flat_rows, flat_cols = rows[bad_pos], cols[bad_pos]
        for r, cc in zip(flat_rows, flat_cols):
            worst.append(
                {
                    "month": str(common_idx[r].date()) if hasattr(common_idx[r], "date") else str(common_idx[r]),
                    "permno": int(common_col[cc]),
                    "candidate": float(c[r, cc]),
                    "golden": float(g[r, cc]),
                    "abs_diff": float(abs(c[r, cc] - g[r, cc])),
                }
            )

    passed = shape_equal and nan_only_c == 0 and nan_only_g == 0 and n_beyond == 0
    return DiffReport(
        factor=factor,
        passed=passed,
        shape_equal=shape_equal,
        candidate_shape=candidate.shape,
        golden_shape=golden.shape,
        index_mismatches=idx_mism,
        column_mismatches=col_mism,
        nan_only_in_candidate=nan_only_c,
        nan_only_in_golden=nan_only_g,
        both_valued_cells=n_both,
        inf_in_candidate=n_inf_c,
        inf_in_golden=n_inf_g,
        max_abs_diff=max_abs,
        cells_beyond_tol=n_beyond,
        atol=atol,
        rtol=rtol,
        worst_offenders=worst,
    )


def diff_factor_files(
    candidate_path: str | Path,
    golden_path: str | Path,
    *,
    factor: str = "",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> DiffReport:
    """Load two factor parquets and diff them."""

    factor = factor or Path(candidate_path).stem
    candidate = pd.read_parquet(candidate_path)
    golden = pd.read_parquet(golden_path)
    return diff_panels(candidate, golden, factor=factor, atol=atol, rtol=rtol)


def validate_factors(
    candidate_dir: str | Path,
    golden_dir: str | Path,
    *,
    factors: list[str] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    report_name: str | None = None,
) -> pd.DataFrame:
    """Diff every factor present in golden_dir (or the given subset) against candidate_dir.

    Returns a one-row-per-factor report DataFrame; also writes it to Data/golden/reports/.
    Missing candidate files are reported as failures, not skipped.
    """

    candidate_dir, golden_dir = Path(candidate_dir), Path(golden_dir)
    names = sorted(factors or [p.stem for p in golden_dir.glob("*.parquet")])
    rows: list[dict] = []
    offender_log: dict[str, list[dict]] = {}
    for name in names:
        gpath = golden_dir / f"{name}.parquet"
        cpath = candidate_dir / f"{name}.parquet"
        if not cpath.exists():
            rows.append(
                DiffReport(
                    factor=name, passed=False, shape_equal=False,
                    candidate_shape=(0, 0), golden_shape=(0, 0),
                    index_mismatches=0, column_mismatches=0,
                    nan_only_in_candidate=0, nan_only_in_golden=0, both_valued_cells=0,
                    inf_in_candidate=0, inf_in_golden=0,
                    max_abs_diff=float("nan"), cells_beyond_tol=0, atol=atol, rtol=rtol,
                    error="candidate file missing",
                ).to_row()
            )
            continue
        try:
            rep = diff_factor_files(cpath, gpath, factor=name, atol=atol, rtol=rtol)
        except Exception as exc:  # comparison itself must never kill the sweep
            rows.append(
                DiffReport(
                    factor=name, passed=False, shape_equal=False,
                    candidate_shape=(0, 0), golden_shape=(0, 0),
                    index_mismatches=0, column_mismatches=0,
                    nan_only_in_candidate=0, nan_only_in_golden=0, both_valued_cells=0,
                    inf_in_candidate=0, inf_in_golden=0,
                    max_abs_diff=float("nan"), cells_beyond_tol=0, atol=atol, rtol=rtol,
                    error=f"{type(exc).__name__}: {exc}",
                ).to_row()
            )
            continue
        rows.append(rep.to_row())
        if rep.worst_offenders:
            offender_log[name] = rep.worst_offenders

    report = pd.DataFrame(rows).set_index("factor").sort_index()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = report_name or f"validate_{stamp}"
    report.to_csv(REPORT_DIR / f"{name}.csv")
    if offender_log:
        (REPORT_DIR / f"{name}_offenders.json").write_text(json.dumps(offender_log, indent=2))
    n_pass = int(report["passed"].sum())
    print(f"[validate] {n_pass}/{len(report)} passed  (atol={atol}, rtol={rtol})  -> {REPORT_DIR / (name + '.csv')}")
    if n_pass < len(report):
        print(report.loc[~report["passed"], ["max_abs_diff", "cells_beyond_tol", "nan_only_in_candidate", "nan_only_in_golden", "error"]].to_string())
    return report


# ---------------------------------------------------------------------------
# Golden freezing (runs the CURRENT engine, with per-factor error isolation)
# ---------------------------------------------------------------------------


def _mask_policy(verbose: bool = False) -> dict[str, str]:
    """Per-factor universe-mask kind from the script-semantics audit.

    Kinds:
      'strict' — script outputs only SignalMasterTable rows (shrcd/exchcd filtered)
      'span'   — SMT-universe script that fill_date_gaps: gap months inside the
                 listing span are legitimate output rows
      'none'   — script's row support is Compustat-driven (or otherwise not SMT):
                 masking to SMT removes cells the oracle has (measured: ~450-500k
                 over-cut cells per factor on Tier A iteration 1). The tree's own
                 NaN propagation defines the domain; residual extras are handled
                 per-factor in Tier B.

    Bucketing keys off the audited free-text universe_source. Factors whose script
    is absent from the semantics file raise — classify, don't default.
    """

    semantics = json.loads((ROOT / "p1_osap_script_semantics.json").read_text(encoding="utf-8"))
    by_script = {s["script"]: s for s in semantics["scripts"]}
    policy: dict[str, str] = {}
    for factor in _native_factor_names():
        script = FACTOR_TO_SCRIPT[factor]
        if script not in by_script:
            raise KeyError(
                f"{factor}: script {script!r} missing from p1_osap_script_semantics.json — "
                f"classify it before freezing with a universe mask"
            )
        s = by_script[script]
        u = s["universe_source"].lower()
        if "signalmastertable" in u:
            policy[factor] = "span" if s["uses_fill_date_gaps"] else "strict"
        else:
            # Compustat-driven / monthlyCRSP / complex: the script never applies
            # the SMT filter, so neither do we.
            policy[factor] = "none"
    if verbose:
        from collections import Counter
        print("[mask-policy]", dict(Counter(policy.values())))
    return policy


def _native_factor_names() -> list[str]:
    plan = plan_factor_calculation_order()
    return sorted(f for step in plan if step.executor == "native" for f in step.factors)


def _candidate_masks(store: "DatasetStore") -> dict[str, pd.DataFrame | None]:
    """All candidate universe masks, keyed by policy name.

    'smt_strict'/'smt_span' from build_universe_masks; input-support masks are
    the row support of each primary input dataset (a Compustat-driven script
    outputs wherever its input has rows); 'none' leaves the grid untouched.
    """

    strict, span = build_universe_masks(store)
    masks: dict[str, pd.DataFrame | None] = {"none": None, "smt_strict": strict, "smt_span": span}
    for name, dataset, time_col in [
        ("crsp_rows", "monthlyCRSP.parquet", "time_avail_m"),
        ("m_aCompustat_rows", "m_aCompustat.parquet", "time_avail_m"),
        ("m_QCompustat_rows", "m_QCompustat.parquet", "time_avail_m"),
    ]:
        raw = store.load_raw(dataset)
        if "permno" in raw.columns:
            sub = raw.loc[:, ["permno", time_col]].assign(present=1.0)
        else:
            gmap = store.load_raw("m_aCompustat.parquet").loc[:, ["gvkey", "permno", "time_avail_m"]]
            gmap = gmap.dropna(subset=["permno"]).drop_duplicates(["gvkey", "time_avail_m"], keep="first")
            sub = raw.merge(gmap, on=["gvkey", "time_avail_m"], how="left").dropna(subset=["permno"])
            sub = sub.loc[:, ["permno", time_col]].assign(present=1.0)
        marker = sub.pivot_table(index=time_col, columns="permno", values="present", aggfunc="last")
        masks[name] = (
            marker.reindex(index=store.template_index, columns=store.template_columns).notna()
        )
    return masks


def choose_universe_policy(
    *,
    factors: list[str] | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    golden_native_dir: str | Path = GOLDEN_NATIVE_DIR,
    golden_oracle_dir: str | Path = GOLDEN_ORACLE_DIR,
) -> pd.DataFrame:
    """Empirically pick each factor's universe mask by testing against its oracle.

    For every factor (UNMASKED native golden required): apply each candidate mask
    at diff time and record the outcome. A mask 'wins' when the masked native
    passes exactly (no value diffs, no NaN-structure diffs). Results — including
    factors where NO mask wins (the Tier B/C work list) — are persisted to
    p1_universe_policy.json and returned as a DataFrame.
    """

    template_index, template_columns = build_output_template()
    store = DatasetStore(template_index, template_columns)
    masks = _candidate_masks(store)
    native_dir, oracle_dir = Path(golden_native_dir), Path(golden_oracle_dir)
    names = factors or [f for f in _native_factor_names() if (oracle_dir / f"{f}.parquet").exists()]

    rows = []
    for factor in names:
        native = pd.read_parquet(native_dir / f"{factor}.parquet")
        oracle = pd.read_parquet(oracle_dir / f"{factor}.parquet")
        outcome: dict[str, Any] = {"factor": factor, "winner": None}
        best = None
        for mask_name, mask in masks.items():
            cand = native.where(mask) if mask is not None else native
            rep = diff_panels(cand, oracle, factor=factor, atol=atol, rtol=rtol)
            score = (rep.cells_beyond_tol, rep.nan_only_in_candidate + rep.nan_only_in_golden)
            outcome[f"{mask_name}_beyond"] = rep.cells_beyond_tol
            outcome[f"{mask_name}_nanmis"] = rep.nan_only_in_candidate + rep.nan_only_in_golden
            if rep.passed and outcome["winner"] is None:
                outcome["winner"] = mask_name
            if best is None or score < best[0]:
                best = (score, mask_name)
        outcome["closest"] = best[1]
        rows.append(outcome)
        print(f"[universe-policy] {factor}: winner={outcome['winner'] or '-'} closest={best[1]}")

    table = pd.DataFrame(rows).set_index("factor")
    policy = {
        f: (row["winner"] or "UNRESOLVED") for f, row in table.iterrows()
    }
    (ROOT / "p1_universe_policy.json").write_text(json.dumps(
        {"_meta": {"criterion": f"exact oracle parity at atol={atol}, rtol={rtol}",
                   "candidates": list(masks.keys())},
         "policy": policy}, indent=2), encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "universe_policy_selection.csv")
    n_win = sum(1 for v in policy.values() if v != "UNRESOLVED")
    print(f"[universe-policy] resolved {n_win}/{len(policy)} factors; rest -> Tier B/C")
    return table


def _script_factor_names() -> list[str]:
    plan = plan_factor_calculation_order()
    return sorted(f for step in plan if step.executor == "script" for f in step.factors)


def freeze_golden_native(
    output_dir: str | Path = GOLDEN_NATIVE_DIR,
    *,
    factors: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Run the current native engine per plan step and freeze one parquet per factor.

    Mirrors calculate_factors_from_plan's native path but adds per-factor error
    isolation and skip-if-exists, so a single bad factor cannot kill the freeze.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = factors or _native_factor_names()
    plan = plan_factor_calculation_order(selected)

    template_index, template_columns = build_output_template()
    store = DatasetStore(template_index, template_columns)
    evaluator = ExpressionEvaluator(store)  # no script adapter: natives must not fall back
    # Universe masking is applied at diff/emit time via the empirically chosen
    # per-factor policy (choose_universe_policy) — golden natives stay UNMASKED
    # so every candidate mask can be evaluated against the oracle without
    # re-running the engine.
    masks = {"none": None}
    policy = {f: "none" for f in selected}

    results: list[dict] = []
    t0 = time.time()
    for step in plan:
        if step.executor != "native":
            continue
        for factor in step.factors:
            path = output_dir / f"{factor}.parquet"
            if path.exists() and not force:
                results.append({"factor": factor, "status": "skipped_exists", "seconds": 0.0})
                continue
            t = time.time()
            try:
                value = evaluator.evaluate(P1_FACTOR_SPECS[factor].expression, current_factor=factor)
                wide = _factor_to_wide_output(
                    value, template_index, template_columns,
                    universe_mask=masks[policy[factor]],
                )
                _write_factor_parquet(factor, wide, output_dir)
                results.append({"factor": factor, "status": "ok", "seconds": round(time.time() - t, 1)})
                print(f"[freeze-native] ok      {factor}  ({time.time() - t:.1f}s)")
            except Exception as exc:
                results.append({"factor": factor, "status": f"FAILED: {type(exc).__name__}: {exc}", "seconds": round(time.time() - t, 1)})
                print(f"[freeze-native] FAILED  {factor}: {exc}")
                traceback.print_exc()
        store.clear()

    summary = pd.DataFrame(results).set_index("factor")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / "freeze_native_summary.csv")
    n_ok = int((summary["status"] == "ok").sum()) + int((summary["status"] == "skipped_exists").sum())
    print(f"[freeze-native] {n_ok}/{len(summary)} frozen or already present in {time.time() - t0:.0f}s -> {output_dir}")
    return summary


def freeze_golden_oracle(
    output_dir: str | Path = GOLDEN_ORACLE_DIR,
    *,
    factors: list[str] | None = None,
    force: bool = False,
    summary_name: str = "freeze_oracle_summary",
) -> pd.DataFrame:
    """Run the original predictor scripts for the script-backed factors and freeze outputs.

    These are the fidelity oracles for the numba daily kernels (WS8) and the
    missing-tree factors (WS7). Slow: the daily Polars scripts scan 107.7M rows.
    Per-factor error isolation; scripts emitting multiple factors run once.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = factors or _script_factor_names()

    template_index, template_columns = build_output_template()
    adapter = FactorScriptAdapter(template_index, template_columns, output_dir)
    adapter.ensure_signal_master()

    results: list[dict] = []
    t0 = time.time()
    for factor in selected:
        path = output_dir / f"{factor}.parquet"
        if path.exists() and not force:
            results.append({"factor": factor, "status": "skipped_exists", "seconds": 0.0})
            continue
        t = time.time()
        try:
            adapter.script_to_parquet(factor, force_script=False)
            results.append({"factor": factor, "status": "ok", "seconds": round(time.time() - t, 1)})
            print(f"[freeze-oracle] ok      {factor}  ({time.time() - t:.1f}s)")
        except Exception as exc:
            results.append({"factor": factor, "status": f"FAILED: {type(exc).__name__}: {exc}", "seconds": round(time.time() - t, 1)})
            print(f"[freeze-oracle] FAILED  {factor}: {exc}")
            traceback.print_exc()

    summary = pd.DataFrame(results).set_index("factor")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_DIR / f"{summary_name}.csv")
    n_ok = int((summary["status"] == "ok").sum()) + int((summary["status"] == "skipped_exists").sum())
    print(f"[freeze-oracle] {n_ok}/{len(summary)} frozen or already present in {time.time() - t0:.0f}s -> {output_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_native = sub.add_parser("freeze-native", help="freeze golden panels for all native factors")
    p_native.add_argument("--force", action="store_true")
    p_native.add_argument("--factors", nargs="*", default=None)

    p_oracle = sub.add_parser("freeze-oracle", help="freeze oracle outputs for script-backed factors")
    p_oracle.add_argument("--force", action="store_true")
    p_oracle.add_argument("--factors", nargs="*", default=None)

    p_cmp = sub.add_parser("compare", help="diff a candidate output dir against a golden dir")
    p_cmp.add_argument("candidate_dir")
    p_cmp.add_argument("--golden-dir", default=str(GOLDEN_NATIVE_DIR))
    p_cmp.add_argument("--atol", type=float, default=0.0)
    p_cmp.add_argument("--rtol", type=float, default=0.0)
    p_cmp.add_argument("--factors", nargs="*", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "freeze-native":
        summary = freeze_golden_native(force=args.force, factors=args.factors)
        return 0 if summary["status"].str.startswith(("ok", "skipped")).all() else 1
    if args.cmd == "freeze-oracle":
        summary = freeze_golden_oracle(force=args.force, factors=args.factors)
        return 0 if summary["status"].str.startswith(("ok", "skipped")).all() else 1
    if args.cmd == "compare":
        report = validate_factors(
            args.candidate_dir, args.golden_dir,
            factors=args.factors, atol=args.atol, rtol=args.rtol,
        )
        return 0 if report["passed"].all() else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

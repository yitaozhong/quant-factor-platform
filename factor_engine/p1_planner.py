"""Plan generator for the P1 factor library (WS5).

Turns the canonical hash-consed DAG (p1_factor_dag) into an explicit,
JSON-serializable execution plan the v2 executor (p1_executor) replays:

    ComputeStep(node)        evaluate one unique DAG node into the value cache
    FinalizeStep(factor)     align/downcast/write one factor + manifest entry
    EvictDatasetStep(ds)     drop a dataset's raw frame + leaf panel cache

Scheduling model (design doc L2, deliberately un-gold-plated):

  * Factor-major DFS with global memoization. Factors are ordered by greedy
    dataset-affinity clustering so raw files and leaf panels have short
    lifetimes; each factor's unscheduled subgraph is emitted children-first,
    then the factor finalizes immediately — its exclusive nodes free at the
    finalize, shared nodes free at their last consumer via refcounts.
  * Column-pruned loads: the plan carries, per dataset, the exact column list
    the executor's store passes to pyarrow (var leaves + audited opaque-kernel
    footprints + family join keys), resolved against on-disk schemas at plan
    time. A planned column missing on disk is a plan-time hard error.
  * Static memory simulation over the schedule estimates live bytes per step
    and records the predicted peak. Exceeding the budget raises PlanMemoryError
    (spill/recompute repair is added only when a real workload triggers it).
  * Incremental builds: with incremental=True, factors whose manifest key
    (p1_manifests) is unchanged and whose output exists are dropped from the
    plan (dead-code elimination); their subgraphs vanish with them.

Fidelity invariants the planner must never break:
  - every op node executes the SAME operator function with the SAME params and
    child order as the legacy recursive evaluator;
  - no algebraic rewrites; node identity is structural (p1_factor_dag);
  - column pruning only removes columns proven unread (audited footprints);
    an unaudited access fails loudly at run time via KeyError, never silently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from Factor_Construction import p1_manifests
    from Factor_Construction.p1_factor_dag import (
        DAILY_KERNEL_OPS,
        FactorDAG,
        PANEL,
        SERIES,
        build_dag,
        column_plan,
    )
    from Factor_Construction.p1_factor_engine import (
        DEMOTED_FACTORS,
        INTERMEDIATE_DIR,
        LINKED_TICKER_MONTHLY_DATASETS,
        MONTHLY_PANEL_DATASETS,
        ROOT,
        _can_execute_natively,
        build_output_template,
    )
    from Factor_Construction.p1_factor_specs import P1_FACTOR_SPECS
except ModuleNotFoundError:
    import p1_manifests
    from p1_factor_dag import (
        DAILY_KERNEL_OPS,
        FactorDAG,
        PANEL,
        SERIES,
        build_dag,
        column_plan,
    )
    from p1_factor_engine import (
        DEMOTED_FACTORS,
        INTERMEDIATE_DIR,
        LINKED_TICKER_MONTHLY_DATASETS,
        MONTHLY_PANEL_DATASETS,
        ROOT,
        _can_execute_natively,
        build_output_template,
    )
    from p1_factor_specs import P1_FACTOR_SPECS

PLAN_DIR = ROOT / "Data" / "plans"
FOOTPRINTS_BASE = ROOT / "p1_kernel_footprints.json"
FOOTPRINTS_PATCH = ROOT / "p1_kernel_footprints_patch.json"

# Universe-mask ops read monthlyCRSP (and the compustat variants read the
# compustat files) through DatasetStore helpers. These static fallbacks keep
# eviction timing sane even before/without an audited footprint entry;
# an op touching an already-evicted dataset just reloads lazily (slower,
# never wrong).
MASK_OP_DATASETS: dict[str, set[str]] = {
    "span_fill": {"monthlyCRSP.parquet"},
    "smt_fill": {"monthlyCRSP.parquet"},
    "smt_only": {"monthlyCRSP.parquet"},
    "span_only": {"monthlyCRSP.parquet"},
    "crsp_only": {"monthlyCRSP.parquet"},
    "smt_gvkey_only": {"monthlyCRSP.parquet", "m_aCompustat.parquet"},
    "compustat_rows_only": {"m_aCompustat.parquet"},
    "compustat_span_fill": {"m_aCompustat.parquet"},
    "require_any_lag": {"monthlyCRSP.parquet"},
}
DAILY_KERNEL_STORE_DATASETS = {
    "monthlyCRSP.parquet",
    "monthlyFF.parquet",
    "monthlyMarket.parquet",
}

# Per-op transient memory beyond (inputs + result), in units of one PANEL.
# Annotated offenders from the design doc L1; everything else gets the default.
_TRANSIENT_PANELS_DEFAULT = 1.0
_TRANSIENT_PANELS: dict[str, Any] = {
    "mean_of_lags": lambda p: len(p.get("lags", [])) + 1.0,
    "compound_return": lambda p: len(p.get("lags", [])) + 1.0,
    "require_any_lag": lambda p: len(p.get("lags", [])) + 1.0,
    "industry_adjusted_mean": lambda p: 8.0,   # object-dtype SIC string panels
    "industry_big_mean": lambda p: 8.0,
    "industry_weighted_momentum": lambda p: 8.0,
    "industry_big_return": lambda p: 8.0,
    "rolling_industry_herfindahl": lambda p: 10.0,
}
_DAILY_KERNEL_TRANSIENT_BYTES = 2.5e9   # bucket streaming + assembly panels
_ANALYST_PREP_BYTES = 2.5e9             # _prepare_analyst_value_signals cache
_ANALYST_OPS = {"analyst_value_signal", "analyst_optimism_ratio", "predicted_forecast_error"}


class PlanMemoryError(RuntimeError):
    pass


@dataclass
class Plan:
    """Serializable execution plan. steps entries:
    {"t": "compute", "node": id} | {"t": "finalize", "factor": f, "root": id}
    | {"t": "evict_dataset", "dataset": file}
    """

    factors: list[str]
    steps: list[dict[str, Any]]
    read_columns: dict[str, list[str] | None]
    refcounts: dict[str, int]
    manifest_keys: dict[str, str]
    skipped_current: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        path = path or PLAN_DIR / f"plan_{self.meta['plan_id']}.json"
        payload = {
            "factors": self.factors,
            "steps": self.steps,
            "read_columns": self.read_columns,
            "refcounts": self.refcounts,
            "manifest_keys": self.manifest_keys,
            "skipped_current": self.skipped_current,
            "meta": self.meta,
        }
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return Path(path)

    @staticmethod
    def load(path: Path) -> "Plan":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return Plan(**d)


def native_factor_names() -> list[str]:
    return sorted(
        f for f in P1_FACTOR_SPECS
        if f not in DEMOTED_FACTORS and _can_execute_natively(f)
    )


def _merged_footprints() -> dict[str, list[dict]]:
    """Base audit + patch audit, keyed by op/function name. Patch wins."""

    try:
        from Factor_Construction.p1_factor_dag import _load_footprints
    except ModuleNotFoundError:
        from p1_factor_dag import _load_footprints
    return _load_footprints()


def _node_store_datasets(node, footprints: dict[str, list[dict]]) -> set[str]:
    """Datasets THIS node's own evaluation reads through the store."""

    if node.kind == "var":
        ds = {node.expr["dataset"]}
        if node.expr["dataset"] in LINKED_TICKER_MONTHLY_DATASETS:
            ds.add("IBESCRSPLinkingTable.parquet")
        return ds
    if node.kind != "op":
        return set()
    ds: set[str] = set()
    fp = footprints.get(f"_op_{node.op}", footprints.get(node.op, []))
    for entry in fp:
        ds.add(entry["dataset"].split(" ")[0])
    ds |= MASK_OP_DATASETS.get(node.op, set())
    if node.op in DAILY_KERNEL_OPS:
        ds |= DAILY_KERNEL_STORE_DATASETS
    return ds


def _order_factors(dag: FactorDAG, factors: list[str]) -> list[str]:
    """Greedy dataset-affinity ordering: next factor shares the most datasets
    with the running working set. Deterministic (alphabetical tie-break)."""

    fsets = {f: dag.nodes[dag.factor_roots[f]].datasets for f in factors}
    remaining = sorted(factors)
    ordered: list[str] = []
    working: frozenset[str] = frozenset()
    while remaining:
        best = max(remaining, key=lambda f: (len(fsets[f] & working), -len(fsets[f]), _neg_alpha(f)))
        ordered.append(best)
        remaining.remove(best)
        working = fsets[best]
    return ordered


def _neg_alpha(name: str) -> tuple[int, ...]:
    # max() tie-break helper: lexicographically SMALLEST name wins
    return tuple(-ord(c) for c in name)


def _resolve_read_columns(plan_cols: dict[str, list[str]]) -> dict[str, list[str] | None]:
    """Resolve the column_plan union against on-disk schemas.

    Handles the gvkey path (panel file without permno on disk reads gvkey and
    joins the m_aCompustat map — the map's own key columns are forced into
    m_aCompustat's union). Any other planned-but-missing column is a hard error.
    """

    import pyarrow.parquet as pq

    resolved: dict[str, list[str] | None] = {}
    needs_gvkey_map = False
    schemas: dict[str, set[str]] = {}
    for ds in plan_cols:
        path = INTERMEDIATE_DIR / ds
        if not path.exists():
            raise FileNotFoundError(f"planned dataset missing on disk: {path}")
        schemas[ds] = set(pq.ParquetFile(path).schema_arrow.names)

    try:
        from Factor_Construction.p1_factor_engine import MONTHLY_SERIES_DATASETS
    except ModuleNotFoundError:
        from p1_factor_engine import MONTHLY_SERIES_DATASETS

    for ds, cols in plan_cols.items():
        if ds in MONTHLY_SERIES_DATASETS:
            # KB-sized time-series files: full load costs nothing and removes
            # any risk of a kernel requesting an unplanned series column
            resolved[ds] = None
            continue
        want = set(cols)
        if ds in MONTHLY_PANEL_DATASETS and "permno" in want and "permno" not in schemas[ds]:
            want = (want - {"permno"}) | {"gvkey"}
            needs_gvkey_map = True
        missing = want - schemas[ds]
        if missing:
            raise KeyError(
                f"column plan for {ds} wants columns absent on disk: {sorted(missing)} "
                f"— audit or spec error, refusing to prune silently"
            )
        resolved[ds] = sorted(want)

    if needs_gvkey_map:
        m_a = set(resolved.get("m_aCompustat.parquet") or [])
        m_a |= {"gvkey", "permno", "time_avail_m"}
        missing = m_a - schemas.get("m_aCompustat.parquet", m_a)
        if missing:
            raise KeyError(f"gvkey map needs columns missing from m_aCompustat: {sorted(missing)}")
        resolved["m_aCompustat.parquet"] = sorted(m_a)
    return resolved


def _factor_manifest_key(dag: FactorDAG, factor: str) -> str:
    tree_cols = column_plan(dag, [factor])
    dataset_files = [INTERMEDIATE_DIR / ds for ds in tree_cols]
    ops: list[str] = []
    daily = False

    def walk(nid: str, seen: set[str]) -> None:
        nonlocal daily
        if nid in seen:
            return
        seen.add(nid)
        node = dag.nodes[nid]
        if node.kind == "op":
            ops.append(node.op)
            if node.op in DAILY_KERNEL_OPS:
                daily = True
        for c in node.children:
            walk(c, seen)

    walk(dag.factor_roots[factor], set())

    try:
        from Factor_Construction.p1_factor_dag import ROOT as FC_ROOT, _load_direct_files
    except ModuleNotFoundError:
        from p1_factor_dag import ROOT as FC_ROOT, _load_direct_files

    direct: list[Path] = list(p1_manifests.DAILY_ARTIFACT_PATHS) if daily else []
    audited_direct = _load_direct_files()
    for op in set(ops):
        for entry in audited_direct.get(f"_op_{op}", []):
            rel = entry["path"]
            if rel in p1_manifests.EXCLUDED_DIRECT_FILES:
                continue
            direct.append(FC_ROOT / rel)
    return p1_manifests.factor_manifest_key(
        factor,
        merkle_root=dag.factor_roots[factor],
        tree_ops=ops,
        dataset_files=dataset_files,
        direct_files=direct,
    )


def generate_plan(
    factors: list[str] | None = None,
    *,
    output_dir: Path | None = None,
    memory_budget_bytes: float = 48e9,
    incremental: bool = False,
) -> Plan:
    t0 = time.time()
    factors = factors or native_factor_names()
    dag = build_dag(factors)
    footprints = _merged_footprints()

    # -- WS3 dead-code elimination ------------------------------------------
    manifest_keys = {f: _factor_manifest_key(dag, f) for f in factors}
    skipped: list[str] = []
    if incremental:
        if output_dir is None:
            raise ValueError("incremental planning needs output_dir to check manifests")
        manifest = p1_manifests.load_manifest(output_dir)
        skipped = [
            f for f in factors
            if p1_manifests.factor_is_current(f, manifest_keys[f], output_dir, manifest)
        ]
        factors = [f for f in factors if f not in set(skipped)]
        if factors:
            dag = build_dag(factors)  # shrink the graph to live roots only

    # -- schedule ------------------------------------------------------------
    ordered = _order_factors(dag, factors)
    steps: list[dict[str, Any]] = []
    scheduled: set[str] = set()
    refcounts: dict[str, int] = {}

    def emit_subtree(root: str) -> None:
        # iterative postorder, children in declared order
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            nid, expanded = stack.pop()
            if nid in scheduled:
                continue
            node = dag.nodes[nid]
            if node.kind == "const":
                scheduled.add(nid)
                continue
            if expanded:
                scheduled.add(nid)
                steps.append({"t": "compute", "node": nid})
                for c in node.children:
                    if dag.nodes[c].kind != "const":
                        refcounts[c] = refcounts.get(c, 0) + 1
            else:
                stack.append((nid, True))
                for c in reversed(node.children):
                    stack.append((c, False))

    for f in ordered:
        root = dag.factor_roots[f]
        emit_subtree(root)
        steps.append({"t": "finalize", "factor": f, "root": root})
        refcounts[root] = refcounts.get(root, 0) + 1

    # -- column-pruned loads (WS2) -------------------------------------------
    read_columns = _resolve_read_columns(column_plan(dag, factors))

    # -- dataset eviction at last use ----------------------------------------
    last_touch: dict[str, int] = {}
    for i, st in enumerate(steps):
        if st["t"] != "compute":
            continue
        for ds in _node_store_datasets(dag.nodes[st["node"]], footprints):
            last_touch[ds] = i
    with_evictions: list[dict[str, Any]] = []
    evict_after: dict[int, list[str]] = {}
    for ds, i in last_touch.items():
        evict_after.setdefault(i, []).append(ds)
    for i, st in enumerate(steps):
        with_evictions.append(st)
        for ds in sorted(evict_after.get(i, [])):
            with_evictions.append({"t": "evict_dataset", "dataset": ds})
    steps = with_evictions

    # -- static memory simulation ---------------------------------------------
    sim = _simulate_memory(dag, steps, refcounts, read_columns)
    if sim["predicted_peak_bytes"] > memory_budget_bytes:
        raise PlanMemoryError(
            f"predicted peak {sim['predicted_peak_bytes'] / 1e9:.1f} GB exceeds budget "
            f"{memory_budget_bytes / 1e9:.1f} GB at step {sim['peak_step']}: "
            f"{json.dumps(sim['peak_breakdown'])} — add spill/recompute repair or raise budget"
        )

    node_summary = {
        nid: (f"var:{n.expr['dataset']}:{n.expr['name']}" if n.kind == "var" else f"op:{n.op}")
        for nid, n in dag.nodes.items()
        if nid in refcounts or nid in scheduled and n.kind != "const"
    }
    plan_id = f"{len(factors)}f_{int(t0)}"
    meta = {
        "plan_id": plan_id,
        "created_unix": t0,
        "planning_seconds": round(time.time() - t0, 2),
        "n_factors": len(factors),
        "n_compute_steps": sum(1 for s in steps if s["t"] == "compute"),
        "n_evictions": sum(1 for s in steps if s["t"] == "evict_dataset"),
        "n_skipped_current": len(skipped),
        "memory_budget_bytes": memory_budget_bytes,
        **sim,
        "node_summary": node_summary,
        "engine_schema": p1_manifests.ENGINE_SCHEMA_VERSION,
    }
    return Plan(
        factors=ordered,
        steps=steps,
        read_columns=read_columns,
        refcounts=refcounts,
        manifest_keys=manifest_keys,
        skipped_current=skipped,
        meta=meta,
    )


def _simulate_memory(
    dag: FactorDAG,
    steps: list[dict[str, Any]],
    refcounts: dict[str, int],
    read_columns: dict[str, list[str] | None],
) -> dict[str, Any]:
    """Walk the schedule tracking estimated live bytes; return peak stats."""

    import pyarrow.parquet as pq

    idx, cols = build_output_template()
    panel = len(idx) * len(cols) * 8.0
    series = len(idx) * 8.0

    def value_bytes(nid: str) -> float:
        n = dag.nodes[nid]
        if n.shape == PANEL:
            return panel
        if n.shape == SERIES:
            return series
        return 0.0

    raw_bytes: dict[str, float] = {}
    for ds, rc in read_columns.items():
        meta_pq = pq.ParquetFile(INTERMEDIATE_DIR / ds)
        ncols = len(rc) if rc is not None else meta_pq.metadata.num_columns
        raw_bytes[ds] = meta_pq.metadata.num_rows * ncols * 8.0 * 1.6

    footprints = _merged_footprints()
    live_nodes: dict[str, float] = {}
    live_raw: dict[str, float] = {}
    rc = dict(refcounts)
    derived_added = 0.0
    peak, peak_step, peak_breakdown = 0.0, -1, {}

    for i, st in enumerate(steps):
        transient = 0.0
        if st["t"] == "compute":
            node = dag.nodes[st["node"]]
            for ds in _node_store_datasets(node, footprints):
                if ds in raw_bytes:
                    live_raw.setdefault(ds, raw_bytes[ds])
            if node.kind == "op":
                if node.op in DAILY_KERNEL_OPS:
                    transient += _DAILY_KERNEL_TRANSIENT_BYTES
                else:
                    f = _TRANSIENT_PANELS.get(node.op)
                    transient += (f(node.params) if f else _TRANSIENT_PANELS_DEFAULT) * panel
                if node.op in _ANALYST_OPS and derived_added == 0.0:
                    derived_added = _ANALYST_PREP_BYTES
                for c in node.children:
                    if dag.nodes[c].kind == "const":
                        continue
                    if rc.get(c, 0) >= 2:
                        transient += value_bytes(c)  # defensive copy handed to the op
            live_nodes[st["node"]] = value_bytes(st["node"])
            consumed = [c for c in node.children if dag.nodes[c].kind != "const"]
        elif st["t"] == "finalize":
            transient += panel * 0.5  # float32 output copy
            consumed = [st["root"]]
        else:
            live_raw.pop(st["dataset"], None)
            consumed = []

        live = sum(live_nodes.values()) + sum(live_raw.values()) + derived_added + transient
        if live > peak:
            peak, peak_step = live, i
            peak_breakdown = {
                "nodes_gb": round(sum(live_nodes.values()) / 1e9, 2),
                "raw_gb": round(sum(live_raw.values()) / 1e9, 2),
                "derived_gb": round(derived_added / 1e9, 2),
                "transient_gb": round(transient / 1e9, 2),
                "step": st,
            }
        for c in consumed:
            rc[c] = rc.get(c, 0) - 1
            if rc[c] == 0:
                live_nodes.pop(c, None)

    return {
        "predicted_peak_bytes": peak,
        "predicted_peak_gb": round(peak / 1e9, 2),
        "peak_step": peak_step,
        "peak_breakdown": peak_breakdown,
        "panel_bytes": panel,
    }


if __name__ == "__main__":
    plan = generate_plan()
    path = plan.save()
    m = plan.meta
    print(f"plan: {path}")
    print(f"factors={m['n_factors']} compute_steps={m['n_compute_steps']} "
          f"evictions={m['n_evictions']} predicted_peak={m['predicted_peak_gb']} GB "
          f"(budget {m['memory_budget_bytes']/1e9:.0f} GB) planned in {m['planning_seconds']}s")
    print("peak:", json.dumps(m["peak_breakdown"], indent=1))

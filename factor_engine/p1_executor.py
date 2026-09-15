"""Plan executor for the P1 factor library (WS4).

Replays a p1_planner.Plan:

  * value cache keyed by DAG node id, freed by plan refcounts (a node's
    memory is released the moment its last consumer has run);
  * SELECTIVE aliasing safety: cached OP results handed to a non-final
    consumer are defensive copies (the legacy engine recomputed shared
    subtrees per consumer, so each consumer historically saw a fresh
    object). VAR leaf panels are handed out UNCOPIED — exactly the legacy
    DatasetStore.get cache-sharing semantics. Both choices reproduce the
    frozen-golden execution model; the golden gate is the arbiter.
  * column-pruned loads: the store reads only the plan's per-dataset column
    unions (resolved at plan time). An unaudited column access surfaces as a
    loud KeyError — never a silently wrong value;
  * per-factor error isolation: a failing node poisons exactly the factors
    that consume it; refcounts still decrement so nothing stays pinned;
  * incremental resume: each finalized factor updates the content-addressed
    manifest (p1_manifests) immediately, so a crashed build re-plans with
    incremental=True and skips completed roots.

The executor calls the SAME operator functions as the legacy recursive
evaluator, with identical params and child order — output must be
bit-identical to the frozen goldens (validated by run_golden_gate()).
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from Factor_Construction import p1_manifests
    from Factor_Construction.p1_factor_dag import build_dag
    from Factor_Construction.p1_factor_engine import (
        DEFAULT_OUTPUT_DIR,
        INTERMEDIATE_DIR,
        OPERATOR_REGISTRY,
        DatasetStore,
        _factor_to_wide_output,
        _write_factor_parquet,
        build_output_template,
    )
    from Factor_Construction.p1_planner import Plan, generate_plan
except ModuleNotFoundError:
    import p1_manifests
    from p1_factor_dag import build_dag
    from p1_factor_engine import (
        DEFAULT_OUTPUT_DIR,
        INTERMEDIATE_DIR,
        OPERATOR_REGISTRY,
        DatasetStore,
        _factor_to_wide_output,
        _write_factor_parquet,
        build_output_template,
    )
    from p1_planner import Plan, generate_plan


def _rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        return pmc.WorkingSetSize


class PrunedDatasetStore(DatasetStore):
    """DatasetStore v2 (WS2): pyarrow column-pruned loads + dataset eviction.

    Same reader (pd.read_parquet / pyarrow), same row order and dtypes for the
    columns read — pruning cannot change values, only omit unread columns.
    """

    def __init__(self, template_index, template_columns, read_columns: dict[str, list[str] | None],
                 data_dir: Path = INTERMEDIATE_DIR) -> None:
        super().__init__(template_index, template_columns, data_dir=data_dir)
        self._read_columns = read_columns
        self.load_events: list[str] = []

    def load_raw(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._raw_cache:
            path = self.data_dir / dataset
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            cols = self._read_columns.get(dataset)
            self._raw_cache[dataset] = pd.read_parquet(path, columns=cols)
            self.load_events.append(dataset)
        return self._raw_cache[dataset]

    def evict_dataset(self, dataset: str) -> None:
        self._raw_cache.pop(dataset, None)
        for key in [k for k in self._panel_cache if k[0] == dataset]:
            del self._panel_cache[key]


def execute_plan(
    plan: Plan,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    update_manifest: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dag = build_dag(plan.factors)
    for st in plan.steps:
        if st["t"] == "compute" and st["node"] not in dag.nodes:
            raise RuntimeError(
                f"plan/spec drift: plan node {st['node']} "
                f"({plan.meta['node_summary'].get(st['node'], '?')}) is not in the "
                f"current DAG — regenerate the plan"
            )

    template_index, template_columns = build_output_template()
    store = PrunedDatasetStore(template_index, template_columns, plan.read_columns)
    values: dict[str, Any] = {}
    rc = dict(plan.refcounts)
    poisoned: dict[str, str] = {}
    manifest = p1_manifests.load_manifest(output_dir) if update_manifest else {}

    written: dict[str, str] = {}
    failures: dict[str, str] = {}
    factor_seconds: dict[str, float] = {}
    step_seconds = {"compute": 0.0, "finalize": 0.0, "evict_dataset": 0.0}
    peak_rss = 0
    t_start = time.time()
    t_factor = t_start

    def peek(nid: str) -> Any:
        """Fetch a child value under the aliasing policy WITHOUT decrementing.

        Cached OP results still awaited by another consumer (rc >= 2 counts
        this consumption plus at least one more) are handed out as copies —
        the legacy evaluator recomputed shared subtrees per consumer, so every
        consumer historically received a fresh object. VAR leaves are shared
        uncopied, exactly like the legacy DatasetStore cache.
        """

        node = dag.nodes[nid]
        if node.kind == "const":
            return node.expr["value"]
        v = values[nid]
        if node.kind == "op" and rc[nid] >= 2 and isinstance(v, (pd.DataFrame, pd.Series)):
            return v.copy()
        return v

    def release_children(node) -> None:
        """Decrement each child once per declared use; free at zero."""

        for c in node.children:
            if dag.nodes[c].kind == "const":
                continue
            rc[c] -= 1
            if rc[c] == 0:
                values.pop(c, None)

    n_finalized = 0
    for st in plan.steps:
        t0 = time.time()
        kind = st["t"]

        if kind == "compute":
            nid = st["node"]
            node = dag.nodes[nid]
            bad_child = next((c for c in node.children if c in poisoned), None)
            if bad_child is not None:
                poisoned[nid] = poisoned[bad_child]
                release_children(node)
            else:
                try:
                    if node.kind == "var":
                        values[nid] = store.get(
                            node.expr["dataset"], node.expr["name"],
                            row_filter=node.expr.get("filter"),
                        )
                    else:
                        args = [peek(c) for c in node.children]
                        params = dict(node.params)
                        params["current_factor"] = None
                        fn = OPERATOR_REGISTRY[node.op]
                        values[nid] = fn(*args, **params, store=store)
                except Exception:
                    poisoned[nid] = (
                        f"{plan.meta['node_summary'].get(nid, nid)}: "
                        f"{traceback.format_exc(limit=8)}"
                    )
                finally:
                    if node.kind == "op":
                        release_children(node)

        elif kind == "finalize":
            factor, root = st["factor"], st["root"]
            if root in poisoned:
                failures[factor] = poisoned[root]
                rc[root] -= 1
                if rc[root] == 0:
                    values.pop(root, None)
            else:
                try:
                    value = values[root]
                    wide = _factor_to_wide_output(value, template_index, template_columns, None)
                    path = _write_factor_parquet(factor, wide, output_dir)
                    written[factor] = str(path)
                    if update_manifest:
                        manifest[factor] = {
                            "key": plan.manifest_keys[factor],
                            "written_unix": time.time(),
                            "output": str(path),
                        }
                        p1_manifests.save_manifest(output_dir, manifest)
                except Exception:
                    failures[factor] = traceback.format_exc(limit=8)
                finally:
                    rc[root] -= 1
                    if rc[root] == 0:
                        values.pop(root, None)
            now = time.time()
            factor_seconds[factor] = round(now - t_factor, 2)
            t_factor = now
            n_finalized += 1
            if n_finalized % 20 == 0:
                gc.collect()
                print(f"  [{n_finalized}/{len(plan.factors)}] {factor}  "
                      f"rss={_rss_bytes()/1e9:.1f}GB  t={now - t_start:.0f}s", flush=True)

        elif kind == "evict_dataset":
            store.evict_dataset(st["dataset"])

        else:
            raise ValueError(f"unknown plan step type: {st}")

        step_seconds[kind] += time.time() - t0
        peak_rss = max(peak_rss, _rss_bytes())

    report = {
        "factors_planned": len(plan.factors),
        "written": len(written),
        "failed": len(failures),
        "skipped_current": plan.skipped_current,
        "failures": failures,
        "total_seconds": round(time.time() - t_start, 1),
        "step_seconds": {k: round(v, 1) for k, v in step_seconds.items()},
        "peak_rss_gb": round(peak_rss / 1e9, 2),
        "predicted_peak_gb": plan.meta.get("predicted_peak_gb"),
        "slowest_factors": dict(sorted(factor_seconds.items(), key=lambda kv: -kv[1])[:10]),
        "raw_loads": store.load_events,
        "output_dir": str(output_dir),
    }
    (output_dir / "_build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def run_build(
    factors: list[str] | None = None,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    incremental: bool = False,
    memory_budget_bytes: float = 48e9,
) -> dict[str, Any]:
    """Plan + execute in one call (the Phase 3 production entry point)."""

    plan = generate_plan(
        factors,
        output_dir=Path(output_dir),
        memory_budget_bytes=memory_budget_bytes,
        incremental=incremental,
    )
    plan_path = plan.save()
    print(f"plan {plan.meta['plan_id']}: {plan.meta['n_factors']} factors, "
          f"{plan.meta['n_compute_steps']} compute steps, predicted peak "
          f"{plan.meta['predicted_peak_gb']} GB → {plan_path}", flush=True)
    report = execute_plan(plan, output_dir=output_dir)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("raw_loads", "failures", "slowest_factors")}, indent=1))
    if report["failures"]:
        print("FAILURES:")
        for f, tb in report["failures"].items():
            print(f"--- {f} ---\n{tb}")
    return report


if __name__ == "__main__":
    run_build()

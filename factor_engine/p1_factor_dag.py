"""Canonical hash-consed DAG over the P1 factor expression trees (WS1).

This module is the planning substrate described in p1_plan_engine_v2_design.md:
every expression tree from ``p1_factor_specs`` is interned bottom-up into a single
node table keyed by a structural Merkle hash, so identical subexpressions across
factors collapse to ONE node regardless of which factor declared them.

Hashing is STRICTLY structural — no commutative reordering, no reassociation —
because any algebraic rewrite changes float op order and would break the
match-the-golden-outputs constraint.

Node identity:
    op    : blake2b(kind, op, canonical(params), child_ids)
    var   : blake2b(kind, dataset, name)
    const : blake2b(kind, repr(value))

Each node carries a back-pointer to one original ``Expr`` dict (the first
occurrence interned), its consumers, and annotations the planner needs:
shape class, datasets touched by the subtree, and — for opaque store-reading
kernels — the audited footprint from ``p1_kernel_footprints.json``.

Public API:
    build_dag(factors=None)         -> FactorDAG
    FactorDAG.nodes / .factor_roots / .consumers_of / .refcount / .topo_order()
    FactorDAG.factor_merkle_roots() -> per-factor content hash (feeds WS3 manifests)
    FactorDAG.stats()               -> dedup/coverage numbers

Run as a script to print build stats and self-checks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

try:
    from Factor_Construction.p1_factor_specs import P1_FACTOR_SPECS
    from Factor_Construction.p1_factor_engine import (
        DAILY_PANEL_DATASETS,
        MONTHLY_SERIES_DATASETS,
        DELEGATED_SPECIAL_OPERATORS,
        OPERATOR_REGISTRY,
        _op_delegate_to_script,
    )
except ModuleNotFoundError:
    from p1_factor_specs import P1_FACTOR_SPECS
    from p1_factor_engine import (
        DAILY_PANEL_DATASETS,
        MONTHLY_SERIES_DATASETS,
        DELEGATED_SPECIAL_OPERATORS,
        OPERATOR_REGISTRY,
        _op_delegate_to_script,
    )

ROOT = Path(__file__).resolve().parent
FOOTPRINTS_PATH = ROOT / "p1_kernel_footprints.json"
FOOTPRINTS_PATCH_PATH = ROOT / "p1_kernel_footprints_patch.json"
FOOTPRINTS_PATCH2_PATH = ROOT / "p1_kernel_footprints_patch2.json"

# Operators that consume DAILY-shaped children and emit a MONTHLY panel.
# This is the reduction boundary the planner uses to carve out DailyKernelSteps.
# Extend this set as WS8 registers new numba daily kernels.
DAILY_REDUCING_OPS = {
    "monthly_mean",
    "monthly_max",
    "monthly_skew",
    "high_52",
    "realized_volatility",
    "zero_trade_measure",
    "price_delay_slope",
    "price_delay_rsq",
    "price_delay_tstat",
}

# Zero-child opaque kernels that stream Data/daily_buckets/ through numba
# kernels and emit a monthly panel — the WS8 daily->monthly boundary as built.
DAILY_KERNEL_OPS = {
    "daily_monthly_stat",
    "ff3_idio_stat",
    "rolling_market_rmse",
    "monthly_rolling_beta",
    "monthly_rolling_multibeta",
    "residual_momentum",
    "price_delay",
    "coskew_signal",
    "betafp_signal",
    "betatailrisk_signal",
    "trendfactor_signal",
    "announcement_return_signal",
    "high52_signal",
    "zerotrade_signal",
}

# Shape classes (planner cost model keys — see design doc L1).
PANEL = "PANEL"            # month x permno wide frame (~368 MB float64)
SERIES = "SERIES"          # month-indexed series (KB)
SCALAR = "SCALAR"          # python literal
DAILY = "DAILY"            # day-indexed data; must never persist past a reduction boundary
DELEGATED = "DELEGATED"    # script-backed op — output is a finished factor panel


def canonical_params(params: dict[str, Any]) -> str:
    """Deterministic serialization of an op node's params.

    json with sort_keys covers the current param vocabulary (ints, floats, str,
    bool, None, lists of ints). Anything json cannot serialize is a new param
    type the hash must be taught about — fail loudly rather than repr-fallback,
    which could silently collide or drift between python versions.
    """

    return json.dumps(params, sort_keys=True, separators=(",", ":"))


@dataclass
class DagNode:
    """One unique computation in the library-wide DAG."""

    node_id: str
    kind: str                          # 'op' | 'var' | 'const'
    op: str | None                     # op name for op nodes
    params: dict[str, Any]
    children: tuple[str, ...]          # child node_ids, order-preserving
    expr: dict[str, Any]               # back-pointer to one original Expr dict
    shape: str                         # PANEL | SERIES | SCALAR | DAILY | DELEGATED
    datasets: frozenset[str]           # datasets touched by this node's subtree
    footprint: list[dict] = field(default_factory=list)  # audited opaque-kernel footprint
    is_daily_boundary: bool = False    # daily-reducing op: DAILY in, PANEL out


class FactorDAG:
    """Interned node table plus factor roots and reverse edges."""

    def __init__(self) -> None:
        self.nodes: dict[str, DagNode] = {}
        self.factor_roots: dict[str, str] = {}
        self._consumers: dict[str, set[str]] = {}
        self._interned_instances = 0  # every visit, incl. cache hits — dedup stat

    # -- construction ------------------------------------------------------

    def intern(self, expr: dict[str, Any], footprints: dict[str, list[dict]]) -> str:
        """Bottom-up hash-cons one expression tree; returns the root node_id."""

        self._interned_instances += 1
        kind = expr["kind"]
        h = hashlib.blake2b(digest_size=16)

        if kind == "var":
            # a filtered leaf (e.g. IBES fpi=='1') is a DIFFERENT computation
            # than the unfiltered column — the filter must be part of identity
            h.update(f"var|{expr['dataset']}|{expr['name']}|{canonical_params(expr.get('filter', {}))}".encode())
            node_id = h.hexdigest()
            if node_id not in self.nodes:
                dataset = expr["dataset"]
                if dataset in DAILY_PANEL_DATASETS:
                    shape = DAILY
                elif dataset in MONTHLY_SERIES_DATASETS:
                    shape = SERIES
                else:
                    shape = PANEL
                self.nodes[node_id] = DagNode(
                    node_id=node_id, kind="var", op=None, params={},
                    children=(), expr=expr, shape=shape,
                    datasets=frozenset({dataset}),
                )
            return node_id

        if kind == "const":
            h.update(f"const|{expr['value']!r}".encode())
            node_id = h.hexdigest()
            if node_id not in self.nodes:
                self.nodes[node_id] = DagNode(
                    node_id=node_id, kind="const", op=None, params={},
                    children=(), expr=expr, shape=SCALAR, datasets=frozenset(),
                )
            return node_id

        if kind != "op":
            raise ValueError(f"Unknown expression node kind: {kind!r} in {expr}")

        child_ids = tuple(self.intern(arg, footprints) for arg in expr.get("args", []))
        params = expr.get("params", {})
        op = expr["op"]
        h.update(f"op|{op}|{canonical_params(params)}|".encode())
        for cid in child_ids:
            h.update(cid.encode())
        node_id = h.hexdigest()

        if node_id not in self.nodes:
            child_nodes = [self.nodes[c] for c in child_ids]
            datasets = frozenset().union(*(c.datasets for c in child_nodes)) if child_nodes else frozenset()
            fp = footprints.get(f"_op_{op}", footprints.get(op, []))
            # opaque kernels touch datasets invisible to the tree — merge them in
            if fp:
                datasets = datasets | frozenset(e["dataset"].split(" ")[0] for e in fp if "dataset" in e)
            shape, boundary = self._infer_shape(op, child_nodes)
            self.nodes[node_id] = DagNode(
                node_id=node_id, kind="op", op=op, params=params,
                children=child_ids, expr=expr, shape=shape, datasets=datasets,
                footprint=fp, is_daily_boundary=boundary,
            )
            for cid in child_ids:
                self._consumers.setdefault(cid, set()).add(node_id)
        return node_id

    @staticmethod
    def _infer_shape(op: str, children: list[DagNode]) -> tuple[str, bool]:
        """Shape of an op node's output, and whether it is a daily→monthly boundary."""

        registered = OPERATOR_REGISTRY.get(op)
        if registered is _op_delegate_to_script or registered is None:
            # Script-backed (or not-yet-registered numba) op: today its output is a
            # finished factor panel. DELEGATED nodes with daily children are exactly
            # the subgraphs WS8 converts to real DAILY_REDUCING_OPS.
            return DELEGATED, False
        if op in DAILY_REDUCING_OPS:
            return PANEL, True
        if not children:
            # Zero-child registered op = opaque kernel (signature (*_, store, **__)):
            # it self-loads from the store / bucket artifacts and emits a full
            # month x permno panel. Without this rule the empty child-shape set
            # would fall through to SCALAR and the planner would cost it at 0 bytes.
            return PANEL, op in DAILY_KERNEL_OPS
        child_shapes = {c.shape for c in children}
        if DAILY in child_shapes:
            # Elementwise/rolling op over daily data stays daily (e.g. the DIV/ABS
            # inside Illiquidity before monthly_mean reduces it).
            return DAILY, False
        if child_shapes <= {SCALAR}:
            return SCALAR, False
        if child_shapes <= {SERIES, SCALAR}:
            return SERIES, False
        return PANEL, False

    # -- queries ------------------------------------------------------------

    def consumers_of(self, node_id: str) -> frozenset[str]:
        return frozenset(self._consumers.get(node_id, set()))

    def refcount(self, node_id: str) -> int:
        """Number of distinct consuming op nodes, plus 1 per factor rooted here."""

        roots = sum(1 for r in self.factor_roots.values() if r == node_id)
        return len(self._consumers.get(node_id, ())) + roots

    def topo_order(self) -> Iterator[DagNode]:
        """Children-before-parents order (deterministic: sorted by node_id at ties)."""

        seen: set[str] = set()
        out: list[str] = []

        def visit(nid: str) -> None:
            if nid in seen:
                return
            seen.add(nid)
            for c in self.nodes[nid].children:
                visit(c)
            out.append(nid)

        for root in sorted(set(self.factor_roots.values())):
            visit(root)
        return iter(self.nodes[n] for n in out)

    def factor_merkle_roots(self) -> dict[str, str]:
        """Per-factor content hash — the tree-structure component of WS3 manifests."""

        return dict(sorted(self.factor_roots.items()))

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for n in self.nodes.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        op_nodes = [n for n in self.nodes.values() if n.kind == "op"]
        shared = [n for n in op_nodes if self.refcount(n.node_id) >= 2]
        return {
            "factors": len(self.factor_roots),
            "unique_nodes": len(self.nodes),
            "interned_instances": self._interned_instances,
            "nodes_by_kind": by_kind,
            "op_nodes_shared_by_2plus": len(shared),
            "delegated_op_nodes": sum(1 for n in op_nodes if n.shape == DELEGATED),
            "daily_boundary_nodes": sum(1 for n in op_nodes if n.is_daily_boundary),
            "daily_shaped_nodes": sum(1 for n in self.nodes.values() if n.shape == DAILY),
        }


def column_plan(dag: "FactorDAG", factors: list[str] | None = None) -> dict[str, list[str]]:
    """Per-dataset union of columns the native evaluation of ``factors`` will read.

    Sources, in order:
      1. var leaves reachable from the requested roots — WITHOUT descending into
         DELEGATED nodes (the evaluator's script fallback preempts arg recursion,
         so a delegated node's children are never evaluated; including their
         leaves would over-fetch);
      2. audited opaque-kernel footprints attached to reachable op nodes;
      3. join/derivation keys DatasetStore.get needs per dataset family
         (audited at engine:407-489), including the IBES link table whenever an
         IBES leaf is present and the gvkey→permno map whenever a panel file
         lacks a permno column on disk.

    This function does NOT silently drop anything: datasets it cannot classify
    raise, so a new dataset family must be taught here explicitly.
    """

    try:
        from Factor_Construction.p1_factor_engine import (
            LINKED_TICKER_MONTHLY_DATASETS,
            MONTHLY_PANEL_DATASETS,
            SPECIAL_MONTHLY_EVENT_DATASETS,
        )
    except ModuleNotFoundError:
        from p1_factor_engine import (
            LINKED_TICKER_MONTHLY_DATASETS,
            MONTHLY_PANEL_DATASETS,
            SPECIAL_MONTHLY_EVENT_DATASETS,
        )

    roots = [dag.factor_roots[f] for f in (factors or dag.factor_roots)]
    needed: dict[str, set[str]] = {}
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        node = dag.nodes[nid]
        if node.kind == "var":
            needed.setdefault(node.expr["dataset"], set()).add(node.expr["name"])
            return
        for entry in node.footprint:
            ds = entry["dataset"].split(" ")[0]  # audit entries are plain filenames
            needed.setdefault(ds, set()).update(entry.get("columns", []))
        if node.shape == DELEGATED:
            return  # children never evaluated on the native path
        for c in node.children:
            visit(c)

    for r in roots:
        visit(r)

    # family join keys (engine:407-489); template source always needs its keys
    needed.setdefault("monthlyCRSP.parquet", set())
    # snapshot: the IBES branch adds the link table to `needed` mid-loop; the
    # added entry needs no classification of its own (keys set at insertion)
    for ds, cols in list(needed.items()):
        if ds in LINKED_TICKER_MONTHLY_DATASETS:
            cols.update({"tickerIBES", "time_avail_m"})
            needed.setdefault("IBESCRSPLinkingTable.parquet", set()).update(
                {"tickerIBES", "permno", "time_avail_m"}
            )
        elif ds in MONTHLY_PANEL_DATASETS:
            cols.update({"time_avail_m", "permno"})
        elif ds in SPECIAL_MONTHLY_EVENT_DATASETS:
            cols.update({"exdt", "permno"})  # time_avail_m is DERIVED from exdt
        elif ds in DAILY_PANEL_DATASETS:
            cols.add("time_d")
            if ds == "dailyCRSP.parquet":
                cols.add("permno")
        elif ds in MONTHLY_SERIES_DATASETS:
            cols.add("time_avail_m")
        elif ds == "IBESCRSPLinkingTable.parquet":
            pass  # keys added above
        elif ds == "SignalMasterTable.parquet":
            pass  # load_raw-only dataset (mask/cash kernels); no pivot join keys
        else:
            raise ValueError(
                f"column_plan cannot classify dataset {ds!r} — new dataset family "
                f"must be added here explicitly (columns requested: {sorted(cols)})"
            )
    return {ds: sorted(cols) for ds, cols in sorted(needed.items())}


def verify_column_plan(plan: dict[str, list[str]], data_dir: Path | None = None) -> dict[str, dict]:
    """Check every planned column against the actual parquet schemas on disk.

    Returns per-dataset {planned, on_disk, missing, prune_ratio}. Missing columns
    mean an audit or spec error — surface them, never drop them silently. A panel
    file lacking 'permno' on disk additionally requires the gvkey→permno map
    (engine:433-442); that requirement is validated here too.
    """

    import pyarrow.parquet as pq

    try:
        from Factor_Construction.p1_factor_engine import INTERMEDIATE_DIR
    except ModuleNotFoundError:
        from p1_factor_engine import INTERMEDIATE_DIR

    data_dir = data_dir or INTERMEDIATE_DIR
    report: dict[str, dict] = {}
    for ds, cols in plan.items():
        path = data_dir / ds
        if not path.exists():
            report[ds] = {"error": "file missing on disk", "planned": cols}
            continue
        on_disk = set(pq.ParquetFile(path).schema_arrow.names)
        planned = set(cols)
        gvkey_path_needed = "permno" in planned and "permno" not in on_disk
        if gvkey_path_needed:
            planned = (planned - {"permno"}) | {"gvkey"}
        report[ds] = {
            "planned": len(planned),
            "on_disk": len(on_disk),
            "missing": sorted(planned - on_disk),
            "gvkey_path": gvkey_path_needed,
            "prune_ratio": round(1 - len(planned) / len(on_disk), 2),
        }
    return report


def _load_footprints(path: Path = FOOTPRINTS_PATH, patch: Path = FOOTPRINTS_PATCH_PATH,
                     patch2: Path = FOOTPRINTS_PATCH2_PATH) -> dict[str, list[dict]]:
    """Audited opaque-kernel footprints, keyed by function/op name.

    Merge order (later wins on collision):
      1. base audit  (2026-07: the 21 legacy opaque kernels)
      2. patch       (2026-08: the 38 store-consuming ops added since — mask
                      family, daily-track kernels, cash_rdq etc.)
      3. patch2      (2026-08: re-audit of the 16 ops whose base entries went
                      stale after the SignalMasterTable refactor; closures
                      proven by column-restricted execution of all 18 affected
                      factors against goldens at zero tolerance)
    """

    merged: dict[str, list[dict]] = {}
    for p in (path, patch, patch2):
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        for k in raw["kernels"]:
            merged[k["op_name"]] = k.get("footprint", [])
    return merged


def _load_direct_files(patch: Path = FOOTPRINTS_PATCH_PATH) -> dict[str, list[dict]]:
    """Per-op NON-store file dependencies (bucket dirs, cached artifacts) from
    the patch audit — exempt from column pruning, required for manifests."""

    if not patch.exists():
        return {}
    raw = json.loads(patch.read_text(encoding="utf-8"))
    return {k["op_name"]: k.get("direct_files", []) for k in raw["kernels"]}


def build_dag(factors: list[str] | None = None) -> FactorDAG:
    """Intern the expression trees of the given factors (default: all 171)."""

    footprints = _load_footprints()
    dag = FactorDAG()
    for name in sorted(factors or P1_FACTOR_SPECS):
        dag.factor_roots[name] = dag.intern(P1_FACTOR_SPECS[name].expression, footprints)
    return dag


if __name__ == "__main__":
    dag = build_dag()
    stats = dag.stats()
    print(json.dumps(stats, indent=2))

    # determinism: rebuilding must produce identical hashes
    again = build_dag()
    assert dag.factor_merkle_roots() == again.factor_merkle_roots(), "non-deterministic hashing"

    # interning must not mutate the shared spec trees
    rebuilt = {f: P1_FACTOR_SPECS[f].expression for f in dag.factor_roots}
    assert all(isinstance(e, dict) for e in rebuilt.values())

    # spot-check: the most-shared op subtrees
    op_nodes = [n for n in dag.nodes.values() if n.kind == "op"]
    top = sorted(op_nodes, key=lambda n: -dag.refcount(n.node_id))[:5]
    print("\nmost shared op subtrees:")
    for n in top:
        print(f"  refcount={dag.refcount(n.node_id):>3}  op={n.op}  params={canonical_params(n.params)}  datasets={sorted(n.datasets)}")

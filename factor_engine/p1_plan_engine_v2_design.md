# P1 Plan Engine v2 — Plan Generation & Execution Design

*Synthesized 2026-07-23 from a 3-lens design panel (compiler / dataflow / pragmatic), each design
adversarially critiqued against the actual code. Supersedes the planning sections of
`p1_factor_engine_design.md`; the family/kernel taxonomy there remains valid.*

## 0. Problem statement

Replace the current planner (`plan_factor_calculation_order`) and executor
(`calculate_factors_from_plan`) with a system that:

- (a) builds a **canonical hash-merged DAG** from the 171 expression trees (this upgrades
  `p1_factor_forest.py` from drawing aid to planning substrate; rendering becomes a view),
- (b) **generates a plan** from that DAG balancing execution speed against memory under an
  explicit RAM budget,
- (c) **executes the plan** with column-pruned loads, incremental skip-if-exists, per-factor
  error isolation, and streaming daily kernels that never materialize day×permno wide frames.

**Hard constraint:** the 148 already-native factors must produce identical output panels
(NaN-aware `allclose`, gated by a golden-diff harness) after every change.

Measured baseline being fixed: monthlyCRSP re-read 12×, 72/199 redundant pivots, ~30 GB peak
(19.5 GB panel cache + 4.7 GB unpruned raw + 8 GB float64 recursion transients), zero
memoization (LAG(at,12) recomputed 18×), no incrementality, no error isolation. Machine: 64 GB.

## 1. Architecture (five layers)

### L0 — Canonical DAG (`p1_factor_dag.py`, new)

Bottom-up **hash-consing / Merkle interning** of every tree:

```
node_id = blake2(kind, op, canonical(params), child_ids)     # op nodes
        = blake2(dataset, variable)                          # var leaves
        = blake2(repr(value))                                # const leaves
```

- **No algebraic rewrites** — no commutative reordering, no reassociation. Float op order
  changes outputs; structural identity alone already merges 530 op nodes → 378 unique (29%).
- Node table: `nodes: dict[id, Node]`, `children`, `consumers` (reverse edges),
  `factor_roots: dict[name, id]`. Node annotations: shape class
  (`PANEL` 1188×38,872 ≈ 368 MB f64 | `SERIES` | `SCALAR` | `DAILY_REDUCE` | `OPAQUE_KERNEL`),
  est. bytes, est. compute cost, **declared footprint** (datasets+columns).
- `p1_factor_forest.py` re-targets to render this table (shared subtrees now shown merged —
  what the original per-factor `op::{factor}::{path}` IDs prevented).

**Prerequisite audit (critical, from adversarial review):** 21 of ~87 operators
(engine:1512-2439, signature `(*_, store, **__)`) are *opaque kernels* — they ignore tree args
and self-load raw long frames from the store. Each must get a hand-audited declared footprint
`{(dataset, [columns])}`; a store shim that logs every `get()`/`load_raw()` during a golden run
cross-checks the declarations. Three of them additionally share a module-level global cache
(`_prepare_analyst_value_signals`, engine:1301-1421, feeding AnalystValue/AOP/PredictedFE) —
that cache must be brought inside the executor's memory model or explicitly costed.

### L1 — Cost & size model

- Leaf sizes: exact from parquet metadata (rows/cols/dtypes) → panel bytes analytical.
- Op cost: O(cells) vectorized default, **plus per-op transient annotations** where reality
  diverges: `_op_mean_of_lags`/`_op_compound_return` materialize `len(lags)` shifted panels +
  a concat (36 lags ⇒ ~13 GB transient inside ONE call); SIC-consuming industry ops build
  object-dtype string panels (multi-GB, non-vectorized). A flat "2× headroom" model is
  falsified by these — annotate the ~6 known offenders, default the rest.
- Everything interior stays **float64** (downcast only at final write, as today) — computing
  on float32 leaves would drift vs golden outputs on ill-conditioned ops (rolling_std, OLS).

### L2 — Plan generator

| Sub-problem | Algorithm | Complexity | Rejected alternatives |
|---|---|---|---|
| Subexpression merging | Hash-consing (above) | O(N), N≈900 | e-graphs/rewrites — find equivalences we must not exploit |
| Killing redundant loads | **Leaf-load clustering**: all pivots of one dataset scheduled consecutively; raw file read once with `columns=` union, dropped after last pivot. Liveness on the raw-file node | O(L log L) | Jaccard/ILP factor clustering — solves a problem that no longer exists once leaves are refcounted |
| Node ordering | **Greedy list scheduling** over topo order; priority = (net bytes freed, fewest remaining uses, depth) — DAG-generalized Sethi–Ullman. Underlying problem (min-register pebbling) is NP-complete (Sethi 1975); greedy captures >90% here | O(N log N) | ILP over ~900 nodes — unjustified; keep the simulator so evidence for stronger methods arrives automatically if the library grows |
| Budget enforcement | **Static memory simulation** of the schedule (refcount frees, exact next-use known). On overrun, insert directives: RECOMPUTE cheap nodes (lag/arith, <0.1 s), SPILL expensive ones (pivots, daily boundaries) to scratch parquet. Victim = cost-weighted furthest-next-use | O(N·live-set) | Calling this "optimal Belady" — with variable sizes/costs offline weighted caching is NP-hard; furthest-next-use is a good heuristic, not optimality |
| Incremental skip | **Content-addressed manifests** (Bazel/Nix style): per-factor key = Merkle root ⊕ input-file fingerprints ⊕ per-op `KERNEL_VERSIONS[op]` salt ⊕ engine schema version. Up-to-date factors are dead-code-eliminated at plan time | O(N) | mtime-only (misses tree/kernel edits); hashing `fn.__code__.co_code` (misses `co_consts` — a changed numeric literal is invisible in bytecode). Hand-bumped integer salts per op are dumb and correct |

**Fingerprint closure (from adversarial review):** a leaf's value depends on up to three files —
its own parquet, its link table (`IBESCRSPLinkingTable` for IBES leaves, `m_aCompustat` for the
gvkey→permno map), and `monthlyCRSP` (source of the global template index/columns). The
manifest key must cover this closure, else refreshing a link table leaves stale factors
"up to date".

Plan artifact: JSON-serializable list of
`LoadStep | ComputeStep | DailyKernelStep | EvictDirective | FinalizeFactorStep`, each carrying
node hashes — inspectable, diffable, replayable.

### L3 — Executor

- Value cache `dict[node_id, DataFrame]` + plan-derived refcounts; free at zero.
- **Selective memoization**: only nodes with global consumer count ≥ 2 enter the cache;
  single-consumer chains stay on the recursion stack (memoizing everything would pin a
  30-node factor's ~11 GB of intermediates until factor completion — worse than today).
- **Aliasing safety**: pandas CoW is OFF by default in 2.x and ops were written as
  legacy scripts — assume mutation. Shared cache entries are handed out as defensive
  copies unless the op is on an audited no-mutation allowlist. (Flipping global CoW on is
  itself a semantics change that must pass the golden gate before adoption.)
- `FinalizeFactorStep` wraps align/downcast/write/manifest in try/except: a failure poisons
  only that factor's exclusive subgraph; refcounts on shared nodes still decrement so nothing
  pins. Progress log + failure report; `written` manifest updated per factor (resume = re-plan,
  DCE skips completed roots).
- Loads via `pd.read_parquet(path, columns=...)` — same pyarrow reader as today, identical
  row order/dtypes. **Not** DuckDB at this boundary (different reader ⇒ fidelity risk;
  `aggfunc='last'` is row-order-sensitive). DuckDB remains an option inside daily kernels only.

### L4 — Validation harness (build FIRST)

Freeze golden panels for the 148 native factors from the current engine (pinned environment),
then per-change diff: NaN-mask equality + `allclose` + per-factor max-abs-diff report.
Every workstream's definition of done. All three design lenses independently arrived at
"this is workstream zero."

## 2. Daily kernel contract (streaming, 23 factors)

A maximal daily subgraph collapses into one `DAILY_REDUCE` node implementing:

```python
class DailyKernel(Protocol):
    # plan time
    def footprint(self) -> list[tuple[str, list[str]]]         # datasets+columns → pruned reads, costing
    def daily_inputs(self) -> list[str]                        # day-indexed aux series (dailyFF mktrf/rf, VIX) joined inside fold
    def monthly_inputs(self) -> list[NodeId]                   # monthly panels/series broadcast into folds
    def cost(self, rows, permnos) -> CostEst
    # run time — executor streams permno-hash buckets
    def fold(self, bucket: pd.DataFrame, ctx: Ctx) -> pd.DataFrame   # bucket: long, sorted (permno, time_d); MUST return month-indexed partial
    def merge(self, partials) -> pd.DataFrame                  # disjoint permnos → concat, reindex to template
    def finalize(self, merged) -> pd.DataFrame                 # wide month×permno
```

- **Preprocessing artifact:** `dailyCRSP.parquet` is date-ordered (CRSP appends by date), so
  permno-contiguous streaming requires a **one-time permno-bucketed, (permno, time_d)-sorted
  repartitioned copy** (~64 hash buckets, ≈1.25 GB rewrite, out-of-core sort via DuckDB).
  Bucket ≈ 600 permnos / ~1.7 M rows / ~20 MB — rolling windows are permno-local by
  construction. This artifact gets its own fingerprint in the manifest scheme.
- `daily_inputs` exists because Beta/IdioVol3F/BetaFP/PriceDelay regress against day-indexed
  market/FF series *inside* fold — delivering those only as monthly panels was a verified gap.
- Executor asserts fold returns month-indexed output (day-indexed ⇒ contract violation, fail
  fast rather than silent all-NaN reindex).
- Parallelism: a **process pool over buckets** is legitimate here (small ctx, ~20 MB buckets)
  even though the monthly pipeline stays single-process (RAM-bound, cache-shared).

**⚠ Fidelity target correction (verified):** the legacy daily/rolling predictor scripts are
written in **Polars** (30 files in `pyCode/Predictors/` import polars — Beta.py,
ZZ0_RealizedVol…, ZZ2_BetaFP.py, ZZ2_PriceDelay….py, ZZ1_ResidualMomentum…, etc.).

**Compute substrate — DECIDED (2026-07-23): numba + numpy.** All daily kernels are
implemented as `@njit` per-permno loops over sorted contiguous arrays and/or vectorized
numpy sufficient-statistics (cumsum-diff rolling sums + batched `np.linalg.solve` on stacked
small normal-equation matrices — exploiting that regressors (mkt/FF factors) are common
across stocks). Pandas/Polars are NOT used inside kernels — too slow for daily rolling
windows. The Polars legacy scripts serve **exclusively as validation oracles**: each numba
kernel must diff clean against its legacy script output via the L4 harness before shipping
(the drift budget must account for polars↔numpy semantic differences: `min_periods` analogs,
`ddof`, NaN propagation, sort stability). Environment verified: numba 0.63.1 JIT-compiles on
Python 3.14; numpy 2.3.5 (`sliding_window_view` available); duckdb 1.5.2 for bucketed reads.
Fallback if a kernel's diff resists diagnosis: embed the legacy Polars logic per-bucket for
that kernel only.

The main engineering risk is not speed but **silent numerical divergence** (a plausible but
wrong beta panel) — which is why every kernel gates through the golden harness, and why
`fold()` outputs are asserted month-indexed rather than silently reindexed.

## 3. Buy-vs-build verdict (from the dataflow lens, upheld under critique)

Do **not** lower the monthly operator zoo to Polars/DuckDB/Dask — 87 operators with
pandas-specific semantics (`pivot_table(aggfunc='last')`, template reindex NaN propagation,
cross-sectional ranks) make bit-matching under a foreign engine a losing game. Take engine
benefits only at boundaries: projection pushdown at the read boundary (pyarrow `columns=`),
DuckDB for the one-time daily repartition, optional DuckDB inside daily kernels.

## 4. Workstreams, difficulty, order

| # | Workstream | Difficulty | Depends on |
|---|---|---|---|
| WS0 | Golden validation harness | 2/5 | — (build first) |
| WS1 | Canonical DAG + interning + **opaque-kernel footprint audit** (21 kernels, ~35 hidden `load_raw` sites) | 2/5 | — |
| WS2 | DatasetStore v2: `columns=` pruned loads, raw-frame release after last pivot | 1/5 | WS1 footprints |
| WS3 | Merkle manifests + skip-if-exists + `KERNEL_VERSIONS` registry | 2/5 | WS1 |
| WS4 | Executor: selective memoization, refcount eviction, error isolation, progress/resume | 3/5 | WS1, WS2 |
| WS5 | **Plan generator**: leaf clustering, list scheduling, memory simulation, spill/recompute repair | 3/5 | WS1, WS2, WS4 |
| WS6 | Daily preprocessing artifact (permno-bucketed sorted copy) | 1/5 | — |
| WS7 | Missing MANUAL_TREES for 8 factors (reverse-engineer legacy scripts into DSL) | 4/5 | WS0 |
| WS8 | **Streaming daily kernels** — 23 factors with numerical fidelity to Polars legacy scripts | 5/5 | WS4, WS6; per-kernel validation via WS0 |

Sequence: WS0 → (WS1 ∥ WS6) → WS2 → (WS3 ∥ WS4) → WS5 → (WS7 ∥ WS8 per-factor).
Each of WS1–WS6 lands independently behind the golden gate.

## 5. Is plan generation the hardest part? — No.

All three lenses independently ranked it **medium (3/5)**: the graph is small (~378 unique op
nodes), every algorithm is a well-trodden heuristic with a simulator as a safety net, and
planning bugs are loud (wrong peak-memory prediction, visible schedule) rather than silent.

The genuinely hard work is **WS8, the daily kernels (5/5)** — 23 separate numerical-fidelity
negotiations against Polars-implemented legacy scripts, where bugs are *silent* (a plausible
but slightly-wrong beta panel). Second: **WS7 (4/5)**, per-factor reverse-engineering of legacy
semantics into trees. Everything else (harness, DAG, pruning, manifests, executor) is 1–3/5
disciplined engineering.

Corollary: the plan generator earns its keep mostly through what it *enables* (bounded memory
for daily work, incrementality, one-read loads) rather than through algorithmic sophistication.
Resist gold-plating it.

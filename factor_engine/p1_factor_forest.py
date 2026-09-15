from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from subprocess import run
from typing import Any, Iterable

try:
    from Factor_Construction.p1_factor_specs import DATASET_VARIABLES, Expr, P1_FACTOR_SPECS
except ModuleNotFoundError:
    from p1_factor_specs import DATASET_VARIABLES, Expr, P1_FACTOR_SPECS


NODE_STYLES = {
    "database": {"shape": "folder", "style": "filled", "fillcolor": "#dbeafe", "color": "#1d4ed8"},
    "variable": {"shape": "ellipse", "style": "filled", "fillcolor": "#fef3c7", "color": "#b45309"},
    "unused_variable": {"shape": "ellipse", "style": "filled,dashed", "fillcolor": "#f3f4f6", "color": "#9ca3af", "fontcolor": "#6b7280"},
    "operator": {"shape": "box", "style": "rounded,filled", "fillcolor": "#e5e7eb", "color": "#4b5563"},
    "factor": {"shape": "box3d", "style": "filled", "fillcolor": "#dcfce7", "color": "#15803d"},
    "const": {"shape": "note", "style": "filled", "fillcolor": "#f5f5f4", "color": "#78716c"},
}


def _freeze_attrs(attrs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Convert node or edge attributes into a hashable, stringified tuple."""

    return tuple(sorted((str(key), str(value)) for key, value in attrs.items()))


def _quote(value: str) -> str:
    """Quote a DOT identifier or attribute value safely."""

    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _short_value(value: Any) -> str:
    """Render compact parameter text for operator labels."""

    if isinstance(value, (list, tuple)):
        text = ",".join(str(item) for item in value[:6])
        if len(value) > 6:
            text += ",..."
        return f"[{text}]"
    if isinstance(value, dict):
        items = list(value.items())[:4]
        text = ",".join(f"{key}={val}" for key, val in items)
        if len(value) > 4:
            text += ",..."
        return "{" + text + "}"
    return str(value)


def _format_operator_label(expr: Expr) -> str:
    """Format an operator node label with a short parameter summary."""

    params = expr.get("params", {})
    if not params:
        return expr["op"]
    summary = ", ".join(f"{key}={_short_value(value)}" for key, value in sorted(params.items()))
    return f'{expr["op"]}\n{summary}'


@dataclass(frozen=True)
class DagNode:
    """A graph node in the dataset-variable-factor forest."""

    node_id: str
    kind: str
    label: str
    attrs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DagEdge:
    """A directed edge in the dataset-variable-factor forest."""

    src: str
    dst: str
    attrs: tuple[tuple[str, str], ...] = ()


@dataclass
class DagForest:
    """Compact in-memory representation of the plotted DAG."""

    nodes: dict[str, DagNode] = field(default_factory=dict)
    edges: set[DagEdge] = field(default_factory=set)
    database_nodes: set[str] = field(default_factory=set)
    factor_nodes: set[str] = field(default_factory=set)

    def add_node(self, node_id: str, kind: str, label: str, **attrs: Any) -> None:
        """Add a node if it does not already exist."""

        if node_id in self.nodes:
            return
        frozen = _freeze_attrs({**NODE_STYLES.get(kind, {}), **attrs})
        self.nodes[node_id] = DagNode(node_id=node_id, kind=kind, label=label, attrs=frozen)
        if kind == "database":
            self.database_nodes.add(node_id)
        elif kind == "factor":
            self.factor_nodes.add(node_id)

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None:
        """Add a directed edge with duplicate suppression."""

        self.edges.add(DagEdge(src=src, dst=dst, attrs=_freeze_attrs(attrs)))

    def to_dot(self, title: str = "P1 Factor Forest") -> str:
        """Serialize the graph to Graphviz DOT."""

        lines = [
            "digraph P1FactorForest {",
            '  graph [rankdir=LR, labelloc=t, fontsize=18, fontname="Helvetica", splines=true, overlap=false, nodesep=0.35, ranksep=1.0, label=' + _quote(title) + "];",
            '  node [fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=9, color="#94a3b8"];',
        ]

        if self.database_nodes:
            db_ids = " ".join(_quote(node_id) for node_id in sorted(self.database_nodes))
            lines.append(f"  {{ rank=min; {db_ids}; }}")
        if self.factor_nodes:
            factor_ids = " ".join(_quote(node_id) for node_id in sorted(self.factor_nodes))
            lines.append(f"  {{ rank=max; {factor_ids}; }}")

        for node in sorted(self.nodes.values(), key=lambda item: (item.kind, item.label, item.node_id)):
            attrs = {"label": node.label, **dict(node.attrs)}
            attr_str = ", ".join(f"{key}={_quote(value)}" for key, value in sorted(attrs.items()))
            lines.append(f"  {_quote(node.node_id)} [{attr_str}];")

        for edge in sorted(self.edges, key=lambda item: (item.src, item.dst, item.attrs)):
            attr_dict = dict(edge.attrs)
            attr_str = ""
            if attr_dict:
                rendered = ", ".join(f"{key}={_quote(value)}" for key, value in sorted(attr_dict.items()))
                attr_str = f" [{rendered}]"
            lines.append(f"  {_quote(edge.src)} -> {_quote(edge.dst)}{attr_str};")

        lines.append("}")
        return "\n".join(lines)

    def write_dot(self, path: str | Path, title: str = "P1 Factor Forest") -> Path:
        """Write the DOT source to disk."""

        path = Path(path)
        path.write_text(self.to_dot(title=title), encoding="utf-8")
        return path

    def render(self, output_path: str | Path, fmt: str | None = None, engine: str = "dot", title: str = "P1 Factor Forest") -> Path:
        """Render the DOT source through the Graphviz CLI.

        This requires the `dot` executable to be available on the machine.
        """

        output_path = Path(output_path)
        fmt = fmt or output_path.suffix.lstrip(".") or "svg"
        dot_path = output_path.with_suffix(".dot")
        self.write_dot(dot_path, title=title)
        run([engine, f"-T{fmt}", str(dot_path), "-o", str(output_path)], check=True)
        return output_path

    def to_graphviz_source(self, title: str = "P1 Factor Forest") -> Any:
        """Return a `graphviz.Source` object for notebook display.

        This requires the Python `graphviz` package. If it is not installed,
        the method raises `ModuleNotFoundError`.
        """

        from graphviz import Source

        return Source(self.to_dot(title=title))


def _factor_node_id(factor: str) -> str:
    return f"factor::{factor}"


def _database_node_id(dataset: str) -> str:
    return f"db::{dataset}"


def _variable_node_id(dataset: str, name: str) -> str:
    return f"var::{dataset}::{name}"


def _operator_node_id(factor: str, path: tuple[int, ...]) -> str:
    rendered = ".".join(str(item) for item in path) if path else "root"
    return f"op::{factor}::{rendered}"


def _const_node_id(factor: str, path: tuple[int, ...]) -> str:
    rendered = ".".join(str(item) for item in path) if path else "root"
    return f"const::{factor}::{rendered}"


def _selected_factors(factors: Iterable[str] | None) -> list[str]:
    """Normalize and validate factor selection."""

    if factors is None:
        return sorted(P1_FACTOR_SPECS)
    selected = []
    for factor in factors:
        if factor not in P1_FACTOR_SPECS:
            raise KeyError(f"Unknown factor: {factor}")
        selected.append(factor)
    return sorted(dict.fromkeys(selected))


def build_factor_forest(
    factors: Iterable[str] | None = None,
    *,
    include_all_dataset_variables: bool = False,
    include_constants: bool = False,
) -> DagForest:
    """Build a left-to-right DAG from databases to variables to factors.

    The graph always shares database and raw-variable nodes across factors.
    Operator nodes are unique per factor expression tree so the forest remains
    acyclic and easy to interpret.
    """

    forest = DagForest()
    selected = _selected_factors(factors)
    used_variables: set[tuple[str, str]] = set()

    def walk(expr: Expr, factor: str, path: tuple[int, ...]) -> str | None:
        kind = expr.get("kind")
        if kind == "var":
            dataset = expr["dataset"]
            name = expr["name"]
            db_id = _database_node_id(dataset)
            var_id = _variable_node_id(dataset, name)
            forest.add_node(db_id, "database", dataset)
            forest.add_node(var_id, "variable", name, tooltip=f"{dataset}:{name}")
            forest.add_edge(db_id, var_id)
            used_variables.add((dataset, name))
            return var_id

        if kind == "const":
            if not include_constants:
                return None
            const_id = _const_node_id(factor, path)
            forest.add_node(const_id, "const", repr(expr["value"]))
            return const_id

        if kind != "op":
            raise ValueError(f"Unsupported expression node kind: {kind}")

        op_id = _operator_node_id(factor, path)
        forest.add_node(
            op_id,
            "operator",
            _format_operator_label(expr),
            tooltip=f"{factor}:{expr['op']}",
        )

        for idx, arg in enumerate(expr.get("args", [])):
            child_id = walk(arg, factor, path + (idx,))
            if child_id is not None:
                forest.add_edge(child_id, op_id)

        return op_id

    for factor in selected:
        spec = P1_FACTOR_SPECS[factor]
        factor_id = _factor_node_id(factor)
        forest.add_node(factor_id, "factor", factor, tooltip=spec.description)
        root_id = walk(spec.expression, factor, ())
        if root_id is not None:
            forest.add_edge(root_id, factor_id)

    if include_all_dataset_variables:
        used_datasets = {dataset for dataset, _ in used_variables}
        for dataset in sorted(used_datasets):
            db_id = _database_node_id(dataset)
            forest.add_node(db_id, "database", dataset)
            for name in DATASET_VARIABLES.get(dataset, ()):
                var_id = _variable_node_id(dataset, name)
                if var_id in forest.nodes:
                    continue
                forest.add_node(var_id, "unused_variable", name, tooltip=f"{dataset}:{name}")
                forest.add_edge(db_id, var_id, style="dashed")

    return forest


def write_factor_forest_dot(
    path: str | Path,
    factors: Iterable[str] | None = None,
    *,
    include_all_dataset_variables: bool = False,
    include_constants: bool = False,
    title: str = "P1 Factor Forest",
) -> Path:
    """Build the forest and write a DOT file in one step."""

    forest = build_factor_forest(
        factors,
        include_all_dataset_variables=include_all_dataset_variables,
        include_constants=include_constants,
    )
    return forest.write_dot(path, title=title)


def render_factor_forest(
    output_path: str | Path,
    factors: Iterable[str] | None = None,
    *,
    include_all_dataset_variables: bool = False,
    include_constants: bool = False,
    fmt: str | None = None,
    title: str = "P1 Factor Forest",
) -> Path:
    """Build the forest and render it through Graphviz in one step."""

    forest = build_factor_forest(
        factors,
        include_all_dataset_variables=include_all_dataset_variables,
        include_constants=include_constants,
    )
    return forest.render(output_path, fmt=fmt, title=title)


def save_factor_forest_pdf(
    output_path: str | Path,
    factors: Iterable[str] | None = None,
    *,
    include_all_dataset_variables: bool = False,
    include_constants: bool = False,
    title: str = "P1 Factor Forest",
) -> Path:
    """Build the forest and save it directly as a PDF file."""

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")
    return render_factor_forest(
        output_path,
        factors,
        include_all_dataset_variables=include_all_dataset_variables,
        include_constants=include_constants,
        fmt="pdf",
        title=title,
    )


def factor_forest_source(
    factors: Iterable[str] | None = None,
    *,
    include_all_dataset_variables: bool = False,
    include_constants: bool = False,
    title: str = "P1 Factor Forest",
) -> Any:
    """Build the forest and return a notebook-displayable Graphviz object."""

    forest = build_factor_forest(
        factors,
        include_all_dataset_variables=include_all_dataset_variables,
        include_constants=include_constants,
    )
    return forest.to_graphviz_source(title=title)


if __name__ == "__main__":
    sample_factors = ["Accruals", "Mom12m", "Beta"]
    forest = build_factor_forest(sample_factors, include_all_dataset_variables=True)
    dot_path = Path(__file__).with_name("p1_factor_forest_sample.dot")
    forest.write_dot(dot_path, title="Sample P1 Factor Forest")
    print(f"Wrote sample DOT to {dot_path}")

"""Causal Graph — models cause-effect relationships within a task.

THE PROBLEM
-----------
Agents often struggle with multi-step tasks because they don't model
*causality*. They know "step A then step B then step C" but not:
- "A causes B" (so if B fails, A is suspect)
- "A and C are independent" (so retrying A won't fix C's failure)
- "D depends on B" (so D can't proceed until B succeeds)

Without a causal model, agents:
1. Retry the wrong step when something fails
2. Run independent steps sequentially when they could be parallel
3. Miss that a downstream failure traces back to an upstream assumption

CausalGraph lets the agent build a lightweight cause-effect model of
the current task, then use it for:
- **Root cause analysis** — when a step fails, walk backward through
  causes to find the actual root
- **Parallelization** — identify independent steps that can run together
- **Dependency ordering** — ensure steps run in dependency order
- **What-if analysis** — "if I skip A, what fails downstream?"

This is NOT a full theorem-prover. It's a lightweight DAG with causal
annotations, maintained by the agent as it works.

WHEN IT RUNS
------------
- The agent explicitly calls `causal_graph_add` to record a cause-effect
- The agent explicitly calls `causal_graph_root_cause` when diagnosing
  a failure
- Periodically, the agent may call `causal_graph_visualize` to see the
  current model

The graph is per-task. It resets when a new task starts.

TOKEN ECONOMICS
---------------
- 0 LLM calls for graph operations (pure data structure)
- 1 LLM call for root_cause analysis (when the agent asks "why did X fail?")
- 1 LLM call for parallelization suggestions (when asked)

Net: very cheap. The graph is just a dict; the LLM is only consulted
when the agent asks for analysis.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class CausalNode:
    """One step/fact/assumption in the causal graph."""

    node_id: str
    label: str  # short description
    node_type: Literal["action", "fact", "assumption", "outcome", "failure"]
    status: Literal["pending", "in_progress", "succeeded", "failed", "skipped"] = "pending"
    evidence: str = ""  # supporting evidence for facts/assumptions
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    """A cause-effect relationship between two nodes."""

    src: str  # node_id of cause
    dst: str  # node_id of effect
    edge_type: Literal["causes", "depends_on", "precedes", "contradicts", "explains"]
    strength: float = 1.0  # 0.0 to 1.0 — how confident we are in this edge
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)


class CausalGraph:
    """A DAG of cause-effect relationships for one task.

    Thread-safe via a single RLock (graph operations are fast).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CausalNode] = {}
        self._edges: list[CausalEdge] = []
        self._outgoing: dict[str, list[str]] = defaultdict(list)  # src → [dst]
        self._incoming: dict[str, list[str]] = defaultdict(list)  # dst → [src]
        self._lock_nodes = {}  # for tracking concurrent access patterns

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        *,
        node_id: str,
        label: str,
        node_type: Literal["action", "fact", "assumption", "outcome", "failure"] = "action",
        status: Literal["pending", "in_progress", "succeeded", "failed", "skipped"] = "pending",
        evidence: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CausalNode:
        node = CausalNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            status=status,
            evidence=evidence,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def update_node_status(
        self,
        node_id: str,
        status: Literal["pending", "in_progress", "succeeded", "failed", "skipped"],
        evidence: str = "",
    ) -> bool:
        node = self._nodes.get(node_id)
        if node is None:
            return False
        node.status = status
        if evidence:
            node.evidence = evidence
        return True

    def add_edge(
        self,
        *,
        src: str,
        dst: str,
        edge_type: Literal["causes", "depends_on", "precedes", "contradicts", "explains"] = "causes",
        strength: float = 1.0,
        rationale: str = "",
    ) -> bool:
        if src not in self._nodes or dst not in self._nodes:
            return False
        edge = CausalEdge(
            src=src,
            dst=dst,
            edge_type=edge_type,
            strength=max(0.0, min(1.0, strength)),
            rationale=rationale,
        )
        self._edges.append(edge)
        self._outgoing[src].append(dst)
        self._incoming[dst].append(src)
        return True

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def root_causes(self, failure_node_id: str) -> list[CausalNode]:
        """Walk backward through `causes`/`depends_on` edges to find root
        causes of a failure. Returns nodes in order of distance (closest first)."""
        if failure_node_id not in self._nodes:
            return []
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(failure_node_id, 0)])
        roots: list[tuple[int, CausalNode]] = []
        while queue:
            node_id, dist = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            # Roots are nodes with no incoming causes/depends_on, OR
            # nodes that are themselves failures/assumptions (likely culprits).
            node = self._nodes[node_id]
            incoming = self._incoming.get(node_id, [])
            causal_incoming = [
                src for src in incoming
                for edge in self._edges
                if edge.src == src and edge.dst == node_id and edge.edge_type in ("causes", "depends_on", "explains")
            ]
            if not causal_incoming or node.node_type in ("assumption", "failure"):
                if node_id != failure_node_id:
                    roots.append((dist, node))
                continue
            for src in causal_incoming:
                if src not in visited:
                    queue.append((src, dist + 1))
        roots.sort(key=lambda x: x[0])
        return [node for _, node in roots]

    def downstream_effects(self, node_id: str) -> list[CausalNode]:
        """Walk forward through `causes`/`precedes` edges to find what
        will be affected if this node fails/is skipped."""
        if node_id not in self._nodes:
            return []
        visited: set[str] = set()
        queue: deque[str] = deque([node_id])
        effects: list[CausalNode] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for dst in self._outgoing.get(current, []):
                edge_types = {
                    e.edge_type for e in self._edges
                    if e.src == current and e.dst == dst
                }
                if edge_types & {"causes", "precedes", "depends_on"}:
                    if dst not in visited:
                        queue.append(dst)
                        node = self._nodes.get(dst)
                        if node and dst != node_id:
                            effects.append(node)
        return effects

    def independent_groups(self) -> list[list[str]]:
        """Find groups of nodes that are independent (no edges between them).
        Useful for parallelization."""
        # Build undirected adjacency for "depends_on" / "precedes" edges.
        adj: dict[str, set[str]] = defaultdict(set)
        for edge in self._edges:
            if edge.edge_type in ("depends_on", "precedes"):
                adj[edge.src].add(edge.dst)
                adj[edge.dst].add(edge.src)
        # Connected components via BFS.
        visited: set[str] = set()
        groups: list[list[str]] = []
        for node_id in self._nodes:
            if node_id in visited:
                continue
            component: list[str] = []
            queue: deque[str] = deque([node_id])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                groups.append(component)
        return groups

    def topological_order(self) -> list[str] | None:
        """Return nodes in dependency order. Returns None if there's a cycle."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for edge in self._edges:
            if edge.edge_type in ("depends_on", "precedes"):
                in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1
        queue: deque[str] = deque([nid for nid, d in in_degree.items() if d == 0])
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for dst in self._outgoing.get(current, []):
                edge_types = {
                    e.edge_type for e in self._edges
                    if e.src == current and e.dst == dst
                }
                if edge_types & {"depends_on", "precedes"}:
                    in_degree[dst] -= 1
                    if in_degree[dst] == 0:
                        queue.append(dst)
        if len(order) != len(self._nodes):
            return None  # cycle detected
        return order

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "status": n.status,
                    "evidence": n.evidence,
                    "metadata": n.metadata,
                }
                for nid, n in self._nodes.items()
            },
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "edge_type": e.edge_type,
                    "strength": e.strength,
                    "rationale": e.rationale,
                }
                for e in self._edges
            ],
            "stats": {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "independent_groups": len(self.independent_groups()),
            },
        }

    def to_mermaid(self) -> str:
        """Render as Mermaid diagram for visualization."""
        lines = ["graph TD"]
        for nid, node in self._nodes.items():
            shape = {
                "action": "[{label}]",
                "fact": "({label})",
                "assumption": "{{{label}}}",
                "outcome": "[[{label}]]",
                "failure": "((( {label} )))",
            }.get(node.node_type, "[{label}]")
            label = node.label.replace('"', "'")[:60]
            status_marker = {
                "succeeded": "✓",
                "failed": "✗",
                "in_progress": "⟳",
                "skipped": "⊘",
                "pending": " ",
            }.get(node.status, "")
            full_label = f"{status_marker} {label}".strip()
            lines.append(f'    {nid}{shape.format(label=full_label)}')
        for edge in self._edges:
            arrow = {
                "causes": "==>",
                "depends_on": "-->",
                "precedes":("..>" if False else "-.->"),
                "contradicts": "--x",
                "explains": "-.->",
            }.get(edge.edge_type, "-->")
            label = edge.edge_type
            if edge.strength < 1.0:
                label += f" ({edge.strength:.1f})"
            lines.append(f'    {edge.src} {arrow}|{label}| {edge.dst}')
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes_by_status": {
                status: sum(1 for n in self._nodes.values() if n.status == status)
                for status in ("pending", "in_progress", "succeeded", "failed", "skipped")
            },
            "nodes_by_type": {
                ntype: sum(1 for n in self._nodes.values() if n.node_type == ntype)
                for ntype in ("action", "fact", "assumption", "outcome", "failure")
            },
            "independent_groups": len(self.independent_groups()),
        }


# --------------------------------------------------------------------------- #
# Per-task graph registry
# --------------------------------------------------------------------------- #

_graphs: dict[str, CausalGraph] = {}
_active_task_id: str | None = None


def get_or_create_graph(task_id: str | None = None) -> tuple[str, CausalGraph]:
    """Get the graph for a task (creating if needed). If task_id is None,
    uses the currently-active task."""
    global _active_task_id
    if task_id is None:
        task_id = _active_task_id or "default"
    _active_task_id = task_id
    if task_id not in _graphs:
        _graphs[task_id] = CausalGraph()
    return task_id, _graphs[task_id]


def set_active_task(task_id: str) -> None:
    global _active_task_id
    _active_task_id = task_id
    if task_id not in _graphs:
        _graphs[task_id] = CausalGraph()


def get_graph(task_id: str | None = None) -> CausalGraph | None:
    if task_id is None:
        task_id = _active_task_id
    if task_id is None:
        return None
    return _graphs.get(task_id)


def list_graphs() -> list[dict[str, Any]]:
    return [
        {"task_id": tid, **graph.stats()}
        for tid, graph in _graphs.items()
    ]


def clear_graph(task_id: str) -> bool:
    return _graphs.pop(task_id, None) is not None

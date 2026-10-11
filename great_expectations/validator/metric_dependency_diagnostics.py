from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from great_expectations.compatibility.typing_extensions import override
from great_expectations.exceptions import GreatExpectationsError
from great_expectations.util import convert_to_json_serializable  # noqa: TID251 # FIXME CoP

if TYPE_CHECKING:
    from great_expectations.validator.metric_configuration import MetricConfigurationID

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ON_CYCLE_SKIP = "skip"
ON_CYCLE_RAISE = "raise"
_ON_CYCLE_POLICIES = (ON_CYCLE_SKIP, ON_CYCLE_RAISE)

# Runtime configuration keys understood by expansion and resolution in
# addition to the pre-existing "catch_exceptions" key.
RUNTIME_KEY_MAX_DEPENDENCY_DEPTH = "max_metric_dependency_depth"
RUNTIME_KEY_ON_CYCLE = "on_cycle"
RUNTIME_KEY_CONSOLIDATE_EXCEPTIONS = "consolidate_exceptions"

# Failure categories and non-convergence reasons.
CATEGORY_CYCLE = "cycle"
CATEGORY_UNSATISFIABLE_DEPENDENCY = "unsatisfiable_dependency"
CATEGORY_PROVIDER_ERROR = "provider_error"
CATEGORY_DEPTH_TRUNCATION = "depth_truncation"
CATEGORY_BLOCKED = "blocked"
CATEGORY_EXPECTATION_ERROR = "expectation_error"

REASON_CYCLE = "cycle"
REASON_UNSATISFIABLE_DEPENDENCY = "unsatisfiable_dependency"
REASON_PROVIDER_ERROR = "provider_error"
REASON_DEPTH_TRUNCATION = "depth_truncation"
REASON_RESOLUTION_ABORTED = "resolution_aborted"

STAGE_EXPANSION = "expansion"
STAGE_RESOLUTION = "resolution"

# Safety bound for elementary-cycle enumeration within a single strongly
# connected component; metric dependency graphs are always very small.
_MAX_CYCLES_PER_SCC = 100

# An SCC needs at least two nodes to harbor a directed cycle.
_MIN_CYCLIC_SCC_SIZE = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MetricDependencyDiagnosticsConfig:
    """Read-only diagnostics and expansion/resolution tuning options.

    Args:
        max_dependency_depth: Optional recursion depth limit while expanding
            metric dependencies. ``None`` (the default) means unlimited. A
            value of ``0`` only keeps the requested root metrics.
        on_cycle: Policy when a cycle with more than one participating metric
            is detected: ``"skip"`` (default; record the cycle and continue
            resolving the rest of the graph) or ``"raise"`` (fail fast).
            Self-dependencies are always skipped (legacy behavior).
        catch_exceptions: Exception isolation switch. When ``True`` (the
            default for diagnostics) every failure is collected into the
            report; when ``False`` failures are raised using the legacy
            semantics.
        consolidate_exceptions: When ``True`` and ``catch_exceptions`` is
            ``False``, all failures are merged into a single
            ``MetricDependencyDiagnosticsError`` whose entries map back to the
            report. When ``False`` the first native error is raised.
        resolve_metrics: When ``True`` the graph is also solved (requires data
            on the execution engine); when ``False`` (default) only the graph
            structure and structural satisfiability are reported.
    """

    max_dependency_depth: Optional[int] = None
    on_cycle: str = ON_CYCLE_SKIP
    catch_exceptions: bool = True
    consolidate_exceptions: bool = False
    resolve_metrics: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        depth = self.max_dependency_depth
        if depth is not None:
            if isinstance(depth, bool) or not isinstance(depth, int):
                raise ValueError(  # noqa: TRY003 # FIXME CoP
                    f'"{RUNTIME_KEY_MAX_DEPENDENCY_DEPTH}" must be an int or None, '  # FIXME CoP
                    f"got {type(depth).__name__}: {depth!r}."
                )
            if depth < 0:
                raise ValueError(  # noqa: TRY003 # FIXME CoP
                    f'"{RUNTIME_KEY_MAX_DEPENDENCY_DEPTH}" must be >= 0, got {depth}.'  # FIXME CoP
                )

        if self.on_cycle not in _ON_CYCLE_POLICIES:
            raise ValueError(  # noqa: TRY003 # FIXME CoP
                f'"{RUNTIME_KEY_ON_CYCLE}" must be one of {_ON_CYCLE_POLICIES}, '  # FIXME CoP
                f"got {self.on_cycle!r}."
            )

    @classmethod
    def from_runtime_configuration(
        cls,
        runtime_configuration: Optional[dict] = None,
        resolve_metrics: bool = False,
    ) -> MetricDependencyDiagnosticsConfig:
        """Builds a config from a validation ``runtime_configuration`` mapping.

        Only known keys are consumed; unknown keys are ignored. ``catch_exceptions``
        defaults to ``True`` here (isolation), matching graph resolution's default.
        """  # FIXME CoP
        runtime_configuration = runtime_configuration or {}
        return cls(
            max_dependency_depth=runtime_configuration.get(RUNTIME_KEY_MAX_DEPENDENCY_DEPTH),
            on_cycle=runtime_configuration.get(RUNTIME_KEY_ON_CYCLE, ON_CYCLE_SKIP),
            catch_exceptions=runtime_configuration.get("catch_exceptions", True),
            consolidate_exceptions=runtime_configuration.get(
                RUNTIME_KEY_CONSOLIDATE_EXCEPTIONS, False
            ),
            resolve_metrics=resolve_metrics,
        )

    @classmethod
    def if_runtime_keys_present(
        cls, runtime_configuration: Optional[dict]
    ) -> Optional[MetricDependencyDiagnosticsConfig]:
        """Returns a config only when new diagnostics keys are present, else None.

        Lets the normal validation path preserve its exact legacy behavior when
        no new options are configured.
        """  # FIXME CoP
        if not runtime_configuration:
            return None
        new_keys = {
            RUNTIME_KEY_MAX_DEPENDENCY_DEPTH,
            RUNTIME_KEY_ON_CYCLE,
            RUNTIME_KEY_CONSOLIDATE_EXCEPTIONS,
        }
        if not new_keys.intersection(runtime_configuration):
            return None
        return cls.from_runtime_configuration(
            runtime_configuration=runtime_configuration,
        )


# ---------------------------------------------------------------------------
# Report entries
# ---------------------------------------------------------------------------


@dataclass
class CycleInfo:
    """A detected dependency cycle.

    ``path`` is the ordered list of participating metric ids, closing with the
    first node again so the full cycle path is visible (e.g. [A, B, A]).
    """

    path: List[MetricConfigurationID]
    is_self_loop: bool = False

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "path": list(self.path),
                "is_self_loop": self.is_self_loop,
            }
        )


@dataclass
class UnsatisfiableDependencyInfo:
    """A metric whose dependency can never be satisfied because no provider exists."""  # FIXME CoP

    metric_id: MetricConfigurationID
    metric_name: str
    reason: str
    referenced_by: List[MetricConfigurationID] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "metric_id": self.metric_id,
                "metric_name": self.metric_name,
                "reason": self.reason,
                "referenced_by": list(self.referenced_by),
            }
        )


@dataclass
class ProviderErrorInfo:
    """A metric whose provider raised during expansion or resolution."""

    metric_id: Optional[MetricConfigurationID]
    error_type: str
    error_message: str
    stage: str = STAGE_RESOLUTION
    num_failures: int = 1

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "metric_id": self.metric_id,
                "error_type": self.error_type,
                "error_message": self.error_message,
                "stage": self.stage,
                "num_failures": self.num_failures,
            }
        )


@dataclass
class TruncatedMetricInfo:
    """A metric whose expansion was stopped by the configured depth limit."""

    metric_id: MetricConfigurationID
    depth: int
    max_dependency_depth: int

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "metric_id": self.metric_id,
                "depth": self.depth,
                "max_dependency_depth": self.max_dependency_depth,
            }
        )


@dataclass
class BlockedMetricInfo:
    """A metric not computed because a dependency never resolved."""

    metric_id: MetricConfigurationID
    reason: str
    blocking_metric_ids: List[MetricConfigurationID] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "metric_id": self.metric_id,
                "reason": self.reason,
                "blocking_metric_ids": list(self.blocking_metric_ids),
            }
        )


@dataclass
class ExpectationErrorInfo:
    """An error encountered while deriving metric dependencies from an expectation."""  # FIXME CoP

    expectation_type: Optional[str]
    error_type: str
    error_message: str

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "expectation_type": self.expectation_type,
                "error_type": self.error_type,
                "error_message": self.error_message,
            }
        )


@dataclass
class DiagnosticFailureEntry:
    """One structured failure entry carried by a consolidated diagnostic exception."""  # FIXME CoP

    category: str
    message: str
    metric_id: Optional[MetricConfigurationID] = None
    cycle_path: Optional[List[MetricConfigurationID]] = None

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "category": self.category,
                "metric_id": self.metric_id,
                "cycle_path": list(self.cycle_path) if self.cycle_path else None,
                "message": self.message,
            }
        )


# ---------------------------------------------------------------------------
# Resolution diagnostics (internal bookkeeping shared with ValidationGraph)
# ---------------------------------------------------------------------------


@dataclass
class ResolutionDiagnostics:
    converged: bool = True
    non_convergence_reasons: List[str] = field(default_factory=list)
    blocked_metrics: Dict[Any, BlockedMetricInfo] = field(default_factory=dict)
    resolution_aborted: Optional[ProviderErrorInfo] = None

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "converged": self.converged,
                "non_convergence_reasons": list(self.non_convergence_reasons),
                "blocked_metrics": {
                    str(metric_id): info.to_json_dict()
                    for metric_id, info in self.blocked_metrics.items()
                },
                "resolution_aborted": (
                    self.resolution_aborted.to_json_dict() if self.resolution_aborted else None
                ),
            }
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class MetricDependencyReport:
    """Read-only report of the metric dependency graph behind one validation."""

    roots: List[MetricConfigurationID] = field(default_factory=list)
    nodes: List[MetricConfigurationID] = field(default_factory=list)
    edges: List[Tuple[MetricConfigurationID, Optional[MetricConfigurationID]]] = field(
        default_factory=list
    )
    cycles: List[CycleInfo] = field(default_factory=list)
    unsatisfiable_dependencies: List[UnsatisfiableDependencyInfo] = field(default_factory=list)
    provider_errors: List[ProviderErrorInfo] = field(default_factory=list)
    truncated_nodes: List[TruncatedMetricInfo] = field(default_factory=list)
    blocked_nodes: List[BlockedMetricInfo] = field(default_factory=list)
    expectation_errors: List[ExpectationErrorInfo] = field(default_factory=list)
    resolved_metric_ids: List[MetricConfigurationID] = field(default_factory=list)
    metrics_resolved: bool = False
    converged: Optional[bool] = None
    structure_complete: bool = True
    non_convergence_reasons: List[str] = field(default_factory=list)
    cycles_enumeration_truncated: bool = False

    @property
    def self_dependencies(self) -> List[MetricConfigurationID]:
        return [cycle.path[0] for cycle in self.cycles if cycle.is_self_loop]

    def is_satisfiable(self) -> Optional[bool]:
        """Structural satisfiability conclusion without computing any metric.

        Returns ``False`` when a cycle or a missing provider makes the graph
        unsatisfiable, ``None`` when the answer is inconclusive (e.g. the depth
        limit truncated expansion), and ``True`` when no structural blocker was
        found. After resolution, ``converged`` is the authoritative answer.
        """  # FIXME CoP
        if self.cycles or self.unsatisfiable_dependencies:
            return False
        if not self.structure_complete:
            return None
        return True

    def to_json_dict(self) -> dict:
        return convert_to_json_serializable(
            {
                "roots": list(self.roots),
                "nodes": list(self.nodes),
                "edges": [list(edge) for edge in self.edges],
                "cycles": [cycle.to_json_dict() for cycle in self.cycles],
                "unsatisfiable_dependencies": [
                    item.to_json_dict() for item in self.unsatisfiable_dependencies
                ],
                "provider_errors": [item.to_json_dict() for item in self.provider_errors],
                "truncated_nodes": [item.to_json_dict() for item in self.truncated_nodes],
                "blocked_nodes": [item.to_json_dict() for item in self.blocked_nodes],
                "expectation_errors": [item.to_json_dict() for item in self.expectation_errors],
                "resolved_metric_ids": list(self.resolved_metric_ids),
                "metrics_resolved": self.metrics_resolved,
                "converged": self.converged,
                "structure_complete": self.structure_complete,
                "satisfiable": self.is_satisfiable(),
                "non_convergence_reasons": list(self.non_convergence_reasons),
                "cycles_enumeration_truncated": self.cycles_enumeration_truncated,
            }
        )

    @override
    def __repr__(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2)


# ---------------------------------------------------------------------------
# Aggregated exception
# ---------------------------------------------------------------------------


class MetricDependencyDiagnosticsError(GreatExpectationsError):
    """Raised (optionally) when diagnostics is configured not to isolate failures.

    Every item in ``failures`` corresponds to an entry in the report that would
    have been returned with exception isolation enabled.
    """

    def __init__(
        self,
        message: str,
        failures: List[DiagnosticFailureEntry],
    ) -> None:
        super().__init__(message)
        self.failures = failures


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _canonical_cycle(path: List[MetricConfigurationID]) -> Tuple[MetricConfigurationID, ...]:
    """Rotates a closed-free cycle path to its lexicographically smallest rotation."""  # FIXME CoP
    rotations = [tuple(path[i:] + path[:i]) for i in range(len(path))]
    return min(rotations)


def find_cycles(
    adjacency: Dict[MetricConfigurationID, Set[MetricConfigurationID]],
    self_dependency_ids: Optional[Set[MetricConfigurationID]] = None,
) -> Tuple[List[CycleInfo], bool]:
    """Finds elementary cycles in a directed dependency graph.

    Self-dependencies (which are not added as graph edges) must be supplied
    explicitly via ``self_dependency_ids``.

    Returns the cycles (self-loops first) and a flag indicating whether the
    per-strongly-connected-component enumeration safety cap was reached.
    """  # FIXME CoP
    self_dependency_ids = self_dependency_ids or set()
    cycles: List[CycleInfo] = [
        CycleInfo(path=[metric_id, metric_id], is_self_loop=True)
        for metric_id in sorted(self_dependency_ids)
    ]
    enumeration_truncated = False

    for scc_nodes in _strongly_connected_components(adjacency):
        if len(scc_nodes) < _MIN_CYCLIC_SCC_SIZE:
            continue
        scc_cycles, truncated = _elementary_cycles_in_scc(scc_nodes=scc_nodes, adjacency=adjacency)
        for path in scc_cycles:
            cycles.append(CycleInfo(path=path, is_self_loop=False))
        enumeration_truncated = enumeration_truncated or truncated

    return cycles, enumeration_truncated


def _strongly_connected_components(  # noqa: C901 # FIXME CoP
    adjacency: Dict[MetricConfigurationID, Set[MetricConfigurationID]],
) -> List[Set[MetricConfigurationID]]:
    """Iterative Tarjan SCC; avoids relying on the Python recursion limit."""
    index_counter = 0
    stack: List[MetricConfigurationID] = []
    lowlinks: Dict[MetricConfigurationID, int] = {}
    index: Dict[MetricConfigurationID, int] = {}
    on_stack: Dict[MetricConfigurationID, bool] = {}
    result: List[Set[MetricConfigurationID]] = []

    nodes = set(adjacency)
    for neighbors in adjacency.values():
        nodes.update(neighbors)

    for root in sorted(nodes):
        if root in index:
            continue

        work_stack: List[Tuple[MetricConfigurationID, Any]] = [
            (root, iter(sorted(adjacency.get(root, set()))))
        ]
        index[root] = index_counter
        lowlinks[root] = index_counter
        index_counter += 1
        stack.append(root)
        on_stack[root] = True

        while work_stack:
            node, neighbors_iter = work_stack[-1]
            advanced = False
            for neighbor in neighbors_iter:
                if neighbor not in index:
                    index[neighbor] = index_counter
                    lowlinks[neighbor] = index_counter
                    index_counter += 1
                    stack.append(neighbor)
                    on_stack[neighbor] = True
                    work_stack.append((neighbor, iter(sorted(adjacency.get(neighbor, set())))))
                    advanced = True
                    break
                if on_stack.get(neighbor):
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            if advanced:
                continue

            work_stack.pop()
            if work_stack:
                parent = work_stack[-1][0]
                lowlinks[parent] = min(lowlinks[parent], lowlinks[node])

            if lowlinks[node] == index[node]:
                component: Set[MetricConfigurationID] = set()
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.add(member)
                    if member == node:
                        break
                result.append(component)

    return result


def _elementary_cycles_in_scc(  # noqa: C901 # FIXME CoP
    scc_nodes: Set[MetricConfigurationID],
    adjacency: Dict[MetricConfigurationID, Set[MetricConfigurationID]],
) -> Tuple[List[List[MetricConfigurationID]], bool]:
    """Enumerates elementary cycles within one SCC via bounded DFS.

    Each cycle is only explored from its lexicographically smallest member,
    which keeps the search free of duplicate starts.
    """  # FIXME CoP
    ordered_nodes = sorted(scc_nodes)
    node_index = {metric_id: i for i, metric_id in enumerate(ordered_nodes)}
    seen: Set[Tuple[MetricConfigurationID, ...]] = set()
    cycles: List[List[MetricConfigurationID]] = []
    truncated = False

    for start in ordered_nodes:
        # (current node, path so far, visited set on this DFS branch)
        dfs_stack: List[Tuple[MetricConfigurationID, list, set]] = [(start, [start], {start})]
        while dfs_stack:
            node, path, visited = dfs_stack.pop()
            for neighbor in sorted(adjacency.get(node, set())):
                if neighbor not in node_index:
                    continue
                if neighbor == start and len(path) > 1:
                    canonical = _canonical_cycle(path)
                    if canonical not in seen:
                        seen.add(canonical)
                        cycles.append(list(canonical) + [canonical[0]])
                        if len(cycles) >= _MAX_CYCLES_PER_SCC:
                            return cycles, True
                elif neighbor not in visited and node_index[neighbor] >= node_index[start]:
                    dfs_stack.append((neighbor, path + [neighbor], visited | {neighbor}))

    return cycles, truncated

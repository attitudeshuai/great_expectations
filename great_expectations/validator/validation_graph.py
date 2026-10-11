from __future__ import annotations

import logging
import traceback
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from tqdm.auto import tqdm

import great_expectations.exceptions as gx_exceptions
from great_expectations.compatibility.typing_extensions import override
from great_expectations.expectations.registry import get_metric_provider
from great_expectations.validator.exception_info import ExceptionInfo
from great_expectations.validator.metric_configuration import (
    MetricConfiguration,
    MetricConfigurationID,
)
from great_expectations.validator.metric_dependency_diagnostics import (
    ON_CYCLE_RAISE,
    REASON_CYCLE,
    REASON_DEPTH_TRUNCATION,
    REASON_PROVIDER_ERROR,
    REASON_RESOLUTION_ABORTED,
    REASON_UNSATISFIABLE_DEPENDENCY,
    STAGE_RESOLUTION,
    BlockedMetricInfo,
    CycleInfo,
    MetricDependencyDiagnosticsConfig,
    ProviderErrorInfo,
    ResolutionDiagnostics,
    TruncatedMetricInfo,
    find_cycles,
)

if TYPE_CHECKING:
    from great_expectations.core import IDDict
    from great_expectations.execution_engine import ExecutionEngine
    from great_expectations.expectations.expectation_configuration import (
        ExpectationConfiguration,
    )
    from great_expectations.expectations.metrics.metric_provider import MetricProvider
    from great_expectations.validator.computed_metric import MetricValue
    from great_expectations.validator.metrics_calculator import (
        _AbortedMetricsInfoDict,
    )


class MetricsCalculatorErrorResultValue(TypedDict):
    metric_configuration: MetricConfiguration
    exception_info: ExceptionInfo
    num_failures: int


class MetricExpansionErrorResultValue(TypedDict):
    metric_configuration: MetricConfiguration
    exception_info: ExceptionInfo
    error_type: str


__all__ = [
    "ExpectationValidationGraph",
    "MetricConfiguration",
    "MetricEdge",
    "MetricExpansionErrorResultValue",
    "MetricValue",
    "ValidationGraph",
]

logger = logging.getLogger(__name__)
logging.captureWarnings(True)

MAX_METRIC_COMPUTATION_RETRIES: int = 3


class MetricEdge:
    def __init__(
        self, left: MetricConfiguration, right: Optional[MetricConfiguration] = None
    ) -> None:
        self._left = left
        self._right = right

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def id(self):
        if self.right:
            return self.left.id, self.right.id

        return self.left.id, None

    def __repr__(self):  # type: ignore[explicit-override] # FIXME
        return f"<{self._left.__repr__()}|{self._right.__repr__()}>"


class ValidationGraph:
    def __init__(
        self,
        execution_engine: ExecutionEngine,
        edges: Optional[List[MetricEdge]] = None,
    ) -> None:
        self._execution_engine = execution_engine

        if edges:
            self._edges = edges
        else:
            self._edges = []

        self._edge_ids = {edge.id for edge in self._edges}

        # Diagnostics bookkeeping, populated while expanding dependencies.
        self._expanded_metric_ids: Set[MetricConfigurationID] = set()
        self._self_dependency_ids: Set[MetricConfigurationID] = set()
        self._expansion_errors: Dict[MetricConfigurationID, MetricExpansionErrorResultValue] = {}
        self._truncated_metric_infos: Dict[MetricConfigurationID, TruncatedMetricInfo] = {}
        self.last_resolution_diagnostics: Optional[ResolutionDiagnostics] = None

    @override
    def __eq__(self, other) -> bool:
        """Supports comparing two "ValidationGraph" objects."""
        return self.edge_ids == other.edge_ids

    @override
    def __hash__(self) -> int:
        return hash(frozenset(self.edge_ids))

    @property
    def edges(self) -> List[MetricEdge]:
        """Returns "MetricEdge" objects, contained within this "ValidationGraph" object (as list)."""  # noqa: E501 # FIXME CoP
        return self._edges

    @property
    def edge_ids(self) -> Set[Tuple[str, str]]:
        """Returns "MetricEdge" objects, contained within this "ValidationGraph" object (as set of two-tuples)."""  # noqa: E501 # FIXME CoP
        return {edge.id for edge in self._edges}

    def add(self, edge: MetricEdge) -> None:
        """Adds supplied "MetricEdge" object to this "ValidationGraph" object (if not already present)."""  # noqa: E501 # FIXME CoP
        if edge.id not in self._edge_ids:
            self._edges.append(edge)
            self._edge_ids.add(edge.id)

    def build_metric_dependency_graph(  # noqa: C901, PLR0912 # FIXME CoP
        self,
        metric_configuration: MetricConfiguration,
        runtime_configuration: Optional[dict] = None,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
        _depth: int = 0,
        _active_path: Optional[List[MetricConfigurationID]] = None,
    ) -> None:
        """
        Obtain domain and value keys for metrics and proceeds to add these metrics to the validation graph
        until all metrics have been added.

        Args:
            metric_configuration: Desired MetricConfiguration object to be resolved.
            runtime_configuration: Additional run-time settings (see "Validator.DEFAULT_RUNTIME_CONFIGURATION").
            diagnostics_config: Optional expansion policy (depth limit, cycle handling, exception isolation).
                When omitted, expansion preserves its legacy behavior (missing providers raise immediately).

        Multi-node dependency cycles are detected via the active DFS path instead of relying on the Python
        recursion limit: the edge that closes a cycle is still added, but traversal does not descend into it.
        """  # noqa: E501 # FIXME CoP
        if _active_path is None:
            _active_path = []

        metric_id: MetricConfigurationID = metric_configuration.id

        # A metric previously truncated may be reachable at a shallower depth from
        # another root, so give it another chance instead of considering it done.
        if metric_id in self._truncated_metric_infos:
            del self._truncated_metric_infos[metric_id]

        # Diamond / shared sub-graph: re-expansion only re-adds identical edges.
        if metric_id in self._expanded_metric_ids:
            return

        max_depth = (
            diagnostics_config.max_dependency_depth if diagnostics_config is not None else None
        )
        if max_depth is not None and _depth >= max_depth:
            self._truncated_metric_infos[metric_id] = TruncatedMetricInfo(
                metric_id=metric_id,
                depth=_depth,
                max_dependency_depth=max_depth,
            )
            # Do not add a leaf edge: with its dependencies unknown the node must
            # never look "ready" to the solver. Any incoming edge keeps it present
            # as a dangling vertex; roots are added to the report explicitly.
            return

        metric_impl_klass: MetricProvider
        try:
            (
                metric_impl_klass,
                _metric_provider,
            ) = self.set_metric_configuration_default_kwargs_if_absent(
                metric_configuration=metric_configuration
            )

            metric_dependencies = metric_impl_klass.get_evaluation_dependencies(
                metric=metric_configuration,
                execution_engine=self._execution_engine,
                runtime_configuration=runtime_configuration,
            )
        except Exception as err:
            if diagnostics_config is not None and diagnostics_config.catch_exceptions:
                self._record_expansion_error(
                    metric_configuration=metric_configuration,
                    error=err,
                )
                # Do not add a leaf edge: a node whose provider could not be
                # resolved is structurally unsatisfiable and must never look
                # "ready" to the solver. Its incoming edge (if any) keeps the node
                # visible as a dangling vertex; roots are reported explicitly.
                return
            raise

        self._expanded_metric_ids.add(metric_id)

        if len(metric_dependencies) == 0:
            self.add(
                MetricEdge(
                    left=metric_configuration,
                )
            )
        else:
            metric_configuration.metric_dependencies = metric_dependencies
            next_active_path = _active_path + [metric_id]
            for metric_dependency in metric_dependencies.values():
                if metric_dependency.id == metric_configuration.id:
                    # Self-dependencies are always skipped (legacy behavior) and
                    # surfaced separately in diagnostics as self-loops.
                    logger.warning(
                        f"Metric {metric_configuration.id!s} has created a circular dependency"
                    )
                    self._self_dependency_ids.add(metric_id)
                    continue

                self.add(
                    MetricEdge(
                        left=metric_configuration,
                        right=metric_dependency,
                    )
                )

                if metric_dependency.id in next_active_path:
                    # A -> B -> ... -> A: a cycle with more than one participant.
                    cycle_start_index = next_active_path.index(metric_dependency.id)
                    cycle_path: List[MetricConfigurationID] = next_active_path[
                        cycle_start_index:
                    ] + [metric_dependency.id]
                    if (
                        diagnostics_config is not None
                        and diagnostics_config.on_cycle == ON_CYCLE_RAISE
                        and not diagnostics_config.catch_exceptions
                    ):
                        raise gx_exceptions.MetricDependencyCycleError(
                            message=(
                                "Metric dependencies form a cycle: "
                                + " -> ".join(str(node) for node in cycle_path)
                            ),
                            cycle_paths=[cycle_path],
                        )
                    # Do not re-descend; the rest of the graph remains solvable.
                    continue

                if metric_dependency.id in self._expanded_metric_ids:
                    continue

                self.build_metric_dependency_graph(
                    metric_configuration=metric_dependency,
                    runtime_configuration=runtime_configuration,
                    diagnostics_config=diagnostics_config,
                    _depth=_depth + 1,
                    _active_path=next_active_path,
                )

    def _record_expansion_error(
        self,
        metric_configuration: MetricConfiguration,
        error: Exception,
    ) -> None:
        exception_traceback = traceback.format_exc()
        exception_message = f"{type(error).__name__}: {error!s}"
        exception_info = ExceptionInfo(
            exception_traceback=exception_traceback,
            exception_message=exception_message,
        )
        self._expansion_errors[metric_configuration.id] = MetricExpansionErrorResultValue(
            metric_configuration=metric_configuration,
            exception_info=exception_info,
            error_type=type(error).__name__,
        )

    def set_metric_configuration_default_kwargs_if_absent(
        self, metric_configuration: MetricConfiguration
    ) -> Tuple[MetricProvider, Callable]:
        """
        Updates "metric_domain_kwargs" and/or "metric_value_kwargs" of "MetricConfiguration" with defualts (if needed).
        """  # noqa: E501 # FIXME CoP
        metric_impl_klass: MetricProvider
        metric_provider: Callable
        metric_impl_klass, metric_provider = get_metric_provider(
            metric_name=metric_configuration.metric_name,
            execution_engine=self._execution_engine,
        )
        self._set_default_metric_kwargs_if_absent(
            default_kwarg_values=metric_impl_klass.default_kwarg_values,
            metric_kwargs=metric_configuration.metric_domain_kwargs,
            keys=metric_impl_klass.domain_keys,
        )
        self._set_default_metric_kwargs_if_absent(
            default_kwarg_values=metric_impl_klass.default_kwarg_values,
            metric_kwargs=metric_configuration.metric_value_kwargs,
            keys=metric_impl_klass.value_keys,
        )
        return metric_impl_klass, metric_provider

    def resolve(
        self,
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,
        # Set to low number (e.g., 3) to suppress progress bar for small graphs.
        show_progress_bars: bool = True,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
    ) -> Tuple[
        Dict[MetricConfigurationID, MetricValue],
        _AbortedMetricsInfoDict,
    ]:
        resolved_metrics: Dict[MetricConfigurationID, MetricValue] = {}

        # updates graph with aborted metrics
        aborted_metrics_info: _AbortedMetricsInfoDict
        resolution_diagnostics: ResolutionDiagnostics
        (
            aborted_metrics_info,
            resolution_diagnostics,
        ) = self._resolve_core(
            metrics=resolved_metrics,
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
            show_progress_bars=show_progress_bars,
            diagnostics_config=diagnostics_config,
        )
        self.last_resolution_diagnostics = resolution_diagnostics

        return resolved_metrics, aborted_metrics_info

    def resolve_with_diagnostics(
        self,
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,
        # Set to low number (e.g., 3) to suppress progress bar for small graphs.
        show_progress_bars: bool = True,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
    ) -> Tuple[
        Dict[MetricConfigurationID, MetricValue],
        _AbortedMetricsInfoDict,
        ResolutionDiagnostics,
    ]:
        """Like ``resolve()`` but also returns a structured explanation of why
        resolution stopped (cycles, provider failures, missing providers, ...)."""  # FIXME CoP
        resolved_metrics: Dict[MetricConfigurationID, MetricValue] = {}

        aborted_metrics_info: _AbortedMetricsInfoDict
        resolution_diagnostics: ResolutionDiagnostics
        (
            aborted_metrics_info,
            resolution_diagnostics,
        ) = self._resolve_core(
            metrics=resolved_metrics,
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
            show_progress_bars=show_progress_bars,
            diagnostics_config=diagnostics_config,
        )
        self.last_resolution_diagnostics = resolution_diagnostics

        return resolved_metrics, aborted_metrics_info, resolution_diagnostics

    def _resolve(  # FIXME CoP
        self,
        metrics: Optional[Dict[MetricConfigurationID, MetricValue]] = None,
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,  # Set to low number (e.g., 3) to suppress progress bar for small graphs.  # noqa: E501 # FIXME CoP
        show_progress_bars: bool = True,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
    ) -> _AbortedMetricsInfoDict:
        """Backward-compatible wrapper: mutates ``metrics``, returns aborted info."""  # FIXME CoP
        if metrics is None:
            metrics = {}

        aborted_metrics_info: _AbortedMetricsInfoDict
        resolution_diagnostics: ResolutionDiagnostics
        (
            aborted_metrics_info,
            resolution_diagnostics,
        ) = self._resolve_core(
            metrics=metrics,
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
            show_progress_bars=show_progress_bars,
            diagnostics_config=diagnostics_config,
        )
        self.last_resolution_diagnostics = resolution_diagnostics

        return aborted_metrics_info

    def _resolve_core(  # noqa: C901, PLR0912, PLR0915 # FIXME CoP
        self,
        metrics: Dict[MetricConfigurationID, MetricValue],
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,  # Set to low number (e.g., 3) to suppress progress bar for small graphs.  # noqa: E501 # FIXME CoP
        show_progress_bars: bool = True,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
    ) -> Tuple[_AbortedMetricsInfoDict, ResolutionDiagnostics]:
        if metrics is None:
            metrics = {}

        if runtime_configuration is None:
            runtime_configuration = {}

        if diagnostics_config is None:
            catch_exceptions = runtime_configuration.get("catch_exceptions", True)
        else:
            catch_exceptions = diagnostics_config.catch_exceptions

        failed_metric_info: _AbortedMetricsInfoDict = {}
        aborted_metrics_info: _AbortedMetricsInfoDict = {}

        ready_metrics: Set[MetricConfiguration]
        needed_metrics: Set[MetricConfiguration]

        exception_info: ExceptionInfo

        progress_bar: Optional[tqdm] = None

        converged: bool = True
        resolution_aborted: Optional[ProviderErrorInfo] = None

        while True:
            ready_metrics, needed_metrics = self._parse(metrics=metrics)

            # Check to see if the user has disabled progress bars
            disable = not show_progress_bars
            if len(self.edges) < min_graph_edges_pbar_enable:
                disable = True

            if progress_bar is None:
                # noinspection PyProtectedMember,SpellCheckingInspection
                progress_bar = tqdm(
                    total=len(ready_metrics) + len(needed_metrics),
                    desc="Calculating Metrics",
                    disable=disable,
                )
            progress_bar.update(0)
            progress_bar.refresh()

            computable_metrics = set()

            for metric in ready_metrics:
                if (
                    metric.id in failed_metric_info
                    and failed_metric_info[metric.id]["num_failures"]
                    >= MAX_METRIC_COMPUTATION_RETRIES
                ):
                    aborted_metrics_info[metric.id] = failed_metric_info[metric.id]
                else:
                    computable_metrics.add(metric)

            if computable_metrics:
                try:
                    # Access "ExecutionEngine.resolve_metrics()" method, to resolve missing "MetricConfiguration" objects.  # noqa: E501 # FIXME CoP
                    metrics.update(
                        self._execution_engine.resolve_metrics(
                            metrics_to_resolve=computable_metrics,
                            metrics=metrics,
                            runtime_configuration=runtime_configuration,
                        )
                    )
                    progress_bar.update(len(computable_metrics))
                    progress_bar.refresh()
                except gx_exceptions.MetricResolutionError as err:
                    if catch_exceptions:
                        exception_traceback = traceback.format_exc()
                        exception_message = str(err)
                        exception_info = ExceptionInfo(
                            exception_traceback=exception_traceback,
                            exception_message=exception_message,
                        )
                        for failed_metric in err.failed_metrics:
                            if failed_metric.id in failed_metric_info:
                                failed_metric_info[failed_metric.id]["num_failures"] += 1
                                failed_metric_info[failed_metric.id]["exception_info"] = (
                                    exception_info
                                )
                            else:
                                failed_metric_info[failed_metric.id] = (
                                    MetricsCalculatorErrorResultValue(
                                        metric_configuration=failed_metric,
                                        exception_info=exception_info,
                                        num_failures=1,
                                    )
                                )

                    else:
                        raise err  # noqa: TRY201 # FIXME CoP
                except Exception as e:
                    if catch_exceptions:
                        logger.error(  # noqa: TRY400 # FIXME CoP
                            f"""Caught exception {e!s} while trying to resolve a set of {len(ready_metrics)} metrics; aborting graph resolution."""  # noqa: E501 # FIXME CoP
                        )
                        resolution_aborted = ProviderErrorInfo(
                            metric_id=None,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            stage=STAGE_RESOLUTION,
                            num_failures=1,
                        )
                        converged = False
                        break
                    else:
                        raise e  # noqa: TRY201 # FIXME CoP

            if not ready_metrics and not needed_metrics:
                converged = True
                break

            if not computable_metrics:
                # Every still-wanted metric is only waiting on others and no
                # metric can be attempted this round: the graph cannot converge.
                converged = not needed_metrics
                break

        progress_bar.close()  # type: ignore[union-attr]  # Incorrect flagging of 'Item "None" of "Optional[Any]" has no attribute "close"' in external package.

        resolution_diagnostics = self._build_resolution_diagnostics(
            metrics=metrics,
            aborted_metrics_info=aborted_metrics_info,
            failed_metric_info=failed_metric_info,
            converged=converged,
            resolution_aborted=resolution_aborted,
        )

        return aborted_metrics_info, resolution_diagnostics

    def _build_resolution_diagnostics(  # noqa: C901 # FIXME CoP
        self,
        metrics: Dict[MetricConfigurationID, MetricValue],
        aborted_metrics_info: _AbortedMetricsInfoDict,
        failed_metric_info: _AbortedMetricsInfoDict,
        converged: bool,
        resolution_aborted: Optional[ProviderErrorInfo],
    ) -> ResolutionDiagnostics:
        # A fixpoint with aborted metrics is not a successful convergence: the
        # requested metric(s) were not delivered, even if nothing is still
        # waiting. Distinguishing the two prevents reporting an empty "success".
        if aborted_metrics_info or resolution_aborted is not None:
            converged = False

        diagnostics = ResolutionDiagnostics(
            converged=converged,
            resolution_aborted=resolution_aborted,
        )
        if converged:
            return diagnostics

        _ready_metrics, needed_metrics = self._parse(metrics=metrics)

        provider_failed_dependency_ids = set(aborted_metrics_info) | set(failed_metric_info)
        unsatisfiable_dependency_ids = {
            metric_id
            for metric_id, error_info in self._expansion_errors.items()
            if error_info["error_type"] == "MetricProviderError"
        }
        expansion_provider_error_ids = {
            metric_id
            for metric_id, error_info in self._expansion_errors.items()
            if error_info["error_type"] != "MetricProviderError"
        }
        truncated_dependency_ids = set(self._truncated_metric_infos)
        cycle_node_ids = self.cycle_node_ids()

        non_convergence_reasons: Set[str] = set()

        metric: MetricConfiguration
        for metric in needed_metrics:
            dependency_ids: List[MetricConfigurationID] = [
                edge.right.id
                for edge in self.edges
                if edge.left.id == metric.id and edge.right is not None
            ]
            blocking_metric_ids = [
                dependency_id for dependency_id in dependency_ids if dependency_id not in metrics
            ]

            if metric.id in cycle_node_ids:
                reason = REASON_CYCLE
                cycle_blockers = [
                    dependency_id
                    for dependency_id in blocking_metric_ids
                    if dependency_id in cycle_node_ids
                ]
                blocking_metric_ids = cycle_blockers or blocking_metric_ids
            elif any(
                dependency_id in provider_failed_dependency_ids for dependency_id in dependency_ids
            ):
                reason = REASON_PROVIDER_ERROR
                blocking_metric_ids = [
                    dependency_id
                    for dependency_id in dependency_ids
                    if dependency_id in provider_failed_dependency_ids
                ]
            elif any(
                dependency_id in unsatisfiable_dependency_ids for dependency_id in dependency_ids
            ):
                reason = REASON_UNSATISFIABLE_DEPENDENCY
                blocking_metric_ids = [
                    dependency_id
                    for dependency_id in dependency_ids
                    if dependency_id in unsatisfiable_dependency_ids
                ]
            elif any(
                dependency_id in expansion_provider_error_ids for dependency_id in dependency_ids
            ):
                reason = REASON_PROVIDER_ERROR
                blocking_metric_ids = [
                    dependency_id
                    for dependency_id in dependency_ids
                    if dependency_id in expansion_provider_error_ids
                ]
            elif any(dependency_id in cycle_node_ids for dependency_id in blocking_metric_ids):
                # Outside a cycle itself but blocked by a cyclic dependency chain.
                reason = REASON_CYCLE
                blocking_metric_ids = [
                    dependency_id
                    for dependency_id in blocking_metric_ids
                    if dependency_id in cycle_node_ids
                ]
            elif any(dependency_id in truncated_dependency_ids for dependency_id in dependency_ids):
                reason = REASON_DEPTH_TRUNCATION
                blocking_metric_ids = [
                    dependency_id
                    for dependency_id in dependency_ids
                    if dependency_id in truncated_dependency_ids
                ]
            else:
                reason = REASON_RESOLUTION_ABORTED

            diagnostics.blocked_metrics[metric.id] = BlockedMetricInfo(
                metric_id=metric.id,
                reason=reason,
                blocking_metric_ids=blocking_metric_ids,
            )
            non_convergence_reasons.add(reason)

        if resolution_aborted is not None:
            non_convergence_reasons.add(REASON_RESOLUTION_ABORTED)
        if aborted_metrics_info:
            non_convergence_reasons.add(REASON_PROVIDER_ERROR)

        diagnostics.non_convergence_reasons = sorted(non_convergence_reasons)
        return diagnostics

    # ------------------------------------------------------------------
    # Read-only graph analysis, used by the diagnostics entry point.
    # ------------------------------------------------------------------

    @property
    def self_dependency_ids(self) -> Set[MetricConfigurationID]:
        return set(self._self_dependency_ids)

    @property
    def expansion_errors(
        self,
    ) -> Dict[MetricConfigurationID, MetricExpansionErrorResultValue]:
        return self._expansion_errors

    @property
    def truncated_metric_infos(self) -> Dict[MetricConfigurationID, TruncatedMetricInfo]:
        return self._truncated_metric_infos

    @property
    def expanded_metric_ids(self) -> Set[MetricConfigurationID]:
        return set(self._expanded_metric_ids)

    def metric_node_ids(self) -> List[MetricConfigurationID]:
        """All metric ids participating in the graph, including dangling vertices."""  # FIXME CoP
        ordered_node_ids: List[MetricConfigurationID] = []
        seen_node_ids: Set[MetricConfigurationID] = set()

        def _add(metric_id: MetricConfigurationID) -> None:
            if metric_id not in seen_node_ids:
                seen_node_ids.add(metric_id)
                ordered_node_ids.append(metric_id)

        edge: MetricEdge
        for edge in self.edges:
            _add(edge.left.id)
            if edge.right is not None:
                _add(edge.right.id)
        for metric_id in self._expansion_errors:
            _add(metric_id)
        for metric_id in self._truncated_metric_infos:
            _add(metric_id)

        return ordered_node_ids

    def adjacency_map(
        self,
    ) -> Dict[MetricConfigurationID, Set[MetricConfigurationID]]:
        adjacency: Dict[MetricConfigurationID, Set[MetricConfigurationID]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.left.id, set())
            if edge.right is not None:
                adjacency.setdefault(edge.right.id, set())
                adjacency[edge.left.id].add(edge.right.id)
        for metric_id in self._expansion_errors:
            adjacency.setdefault(metric_id, set())
        for metric_id in self._truncated_metric_infos:
            adjacency.setdefault(metric_id, set())
        return adjacency

    def find_dependency_cycles(self) -> Tuple[List[CycleInfo], bool]:
        return find_cycles(
            adjacency=self.adjacency_map(),
            self_dependency_ids=self._self_dependency_ids,
        )

    def cycle_node_ids(self) -> Set[MetricConfigurationID]:
        cycles, _ = self.find_dependency_cycles()
        cycle_node_ids: Set[MetricConfigurationID] = set()
        for cycle in cycles:
            if cycle.is_self_loop:
                continue
            cycle_node_ids.update(cycle.path)
        return cycle_node_ids

    def _parse(
        self,
        metrics: Dict[MetricConfigurationID, MetricValue],
    ) -> Tuple[Set[MetricConfiguration], Set[MetricConfiguration]]:
        """Given validation graph, returns the ready and needed metrics necessary for validation using a traversal of
        validation graph (a graph structure of metric ids) edges"""  # noqa: E501 # FIXME CoP
        unmet_dependency_ids = set()
        unmet_dependency = set()
        maybe_ready_ids = set()
        maybe_ready = set()

        for edge in self.edges:
            if edge.left.id not in metrics:
                if edge.right is None or edge.right.id in metrics:
                    if edge.left.id not in maybe_ready_ids:
                        maybe_ready_ids.add(edge.left.id)
                        maybe_ready.add(edge.left)
                else:  # noqa: PLR5501 # FIXME CoP
                    if edge.left.id not in unmet_dependency_ids:
                        unmet_dependency_ids.add(edge.left.id)
                        unmet_dependency.add(edge.left)

        return maybe_ready - unmet_dependency, unmet_dependency

    @staticmethod
    def _set_default_metric_kwargs_if_absent(
        default_kwarg_values: dict,
        metric_kwargs: IDDict,
        keys: Tuple[str, ...],
    ) -> None:
        key: str
        for key in keys:
            if (
                key not in metric_kwargs
                and key in default_kwarg_values
                and default_kwarg_values[key] is not None
            ):
                metric_kwargs[key] = default_kwarg_values[key]

    def __repr__(self):  # type: ignore[explicit-override] # FIXME
        edge: MetricEdge
        return ", ".join([edge.__repr__() for edge in self._edges])


class ExpectationValidationGraph:
    def __init__(
        self,
        configuration: ExpectationConfiguration,
        graph: ValidationGraph,
    ) -> None:
        if configuration is None:
            raise ValueError(  # noqa: TRY003 # FIXME CoP
                """Instantiation of "ExpectationValidationGraph" requires valid "ExpectationConfiguration" object."""  # noqa: E501 # FIXME CoP
            )

        if graph is None:
            raise ValueError(  # noqa: TRY003 # FIXME CoP
                """Instantiation of "ExpectationValidationGraph" requires valid "ValidationGraph" object."""  # noqa: E501 # FIXME CoP
            )

        self._configuration = configuration
        self._graph = graph

    @property
    def configuration(self) -> ExpectationConfiguration:
        return self._configuration

    @property
    def graph(self) -> ValidationGraph:
        return self._graph

    def update(self, graph: ValidationGraph) -> None:
        edge: MetricEdge
        for edge in graph.edges:
            self.graph.add(edge=edge)

    def get_exception_info(
        self,
        metric_info: _AbortedMetricsInfoDict,
    ) -> Dict[str, Union[MetricConfiguration, ExceptionInfo, int]]:
        metric_info = self._filter_metric_info_in_graph(metric_info=metric_info)
        metric_exception_info: Dict[str, Union[MetricConfiguration, ExceptionInfo, int]] = {}
        metric_id: MetricConfigurationID
        metric_info_item: MetricsCalculatorErrorResultValue
        for metric_id, metric_info_item in metric_info.items():
            metric_exception_info[str(metric_id)] = metric_info_item["exception_info"]

        return metric_exception_info

    def _filter_metric_info_in_graph(
        self,
        metric_info: _AbortedMetricsInfoDict,
    ) -> _AbortedMetricsInfoDict:
        graph_metric_ids: List[MetricConfigurationID] = []
        edge: MetricEdge
        vertex: MetricConfiguration
        for edge in self.graph.edges:
            for vertex in [edge.left, edge.right]:
                if vertex is not None:
                    graph_metric_ids.append(vertex.id)

        metric_id: MetricConfigurationID
        metric_info_item: Dict[str, Union[MetricConfiguration, Set[ExceptionInfo], int]]
        return {
            metric_id: metric_info_item
            for metric_id, metric_info_item in metric_info.items()
            if metric_id in graph_metric_ids
        }

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpectationValidationGraph):
            return False
        return self.configuration == other.configuration and self.graph == other.graph

    @override
    def __hash__(self) -> int:
        return hash((self.configuration, self.graph))

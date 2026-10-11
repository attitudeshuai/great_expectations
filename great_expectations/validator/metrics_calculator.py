from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import great_expectations.exceptions as gx_exceptions
from great_expectations.validator.computed_metric import MetricValue
from great_expectations.validator.metric_configuration import (
    MetricConfiguration,
    MetricConfigurationID,
)
from great_expectations.validator.metric_dependency_diagnostics import (
    CATEGORY_BLOCKED,
    CATEGORY_CYCLE,
    CATEGORY_PROVIDER_ERROR,
    CATEGORY_UNSATISFIABLE_DEPENDENCY,
    ON_CYCLE_SKIP,
    REASON_CYCLE,
    REASON_DEPTH_TRUNCATION,
    REASON_PROVIDER_ERROR,
    REASON_RESOLUTION_ABORTED,
    REASON_UNSATISFIABLE_DEPENDENCY,
    STAGE_EXPANSION,
    STAGE_RESOLUTION,
    DiagnosticFailureEntry,
    MetricDependencyDiagnosticsConfig,
    MetricDependencyDiagnosticsError,
    MetricDependencyReport,
    ProviderErrorInfo,
    UnsatisfiableDependencyInfo,
)
from great_expectations.validator.validation_graph import (
    MetricExpansionErrorResultValue,
    MetricsCalculatorErrorResultValue,
    ValidationGraph,
)

if TYPE_CHECKING:
    import pandas as pd
    from typing_extensions import TypeAlias

    from great_expectations.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)
logging.captureWarnings(True)


_MetricsDict: TypeAlias = Dict[MetricConfigurationID, MetricValue]
_AbortedMetricsInfoDict: TypeAlias = Dict[
    MetricConfigurationID,
    MetricsCalculatorErrorResultValue,
]


class MetricsCalculator:
    def __init__(
        self,
        execution_engine: ExecutionEngine,
        show_progress_bars: bool = False,
    ) -> None:
        """
        MetricsCalculator accepts and processes metrics calculation requests.

        Args:
            execution_engine: ExecutionEngine to perform metrics computation.
            show_progress_bars: Directive for whether or not to show progress bars.
        """
        self._execution_engine: ExecutionEngine = execution_engine
        self._show_progress_bars: bool = show_progress_bars

    @property
    def show_progress_bars(self) -> bool:
        return self._show_progress_bars

    @show_progress_bars.setter
    def show_progress_bars(self, enable: bool) -> None:
        self._show_progress_bars = enable

    def columns(self, domain_kwargs: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Convenience method to run "table.columns" metric.

        Arguments:
            domain_kwargs: Optional dictionary of domain kwargs (e.g., containing "batch_id").

        Returns:
            The list of Batch columns.
        """
        if domain_kwargs is None:
            domain_kwargs = {}

        if domain_kwargs.get("batch_id") is None:
            domain_kwargs["batch_id"] = self._execution_engine.batch_manager.active_batch_id

        columns: List[str] = self.get_metric(
            metric=MetricConfiguration(
                metric_name="table.columns",
                metric_domain_kwargs=domain_kwargs,
            )
        )

        return columns

    def head(
        self,
        n_rows: int = 5,
        domain_kwargs: Optional[Dict[str, Any]] = None,
        fetch_all: bool = False,
    ) -> pd.DataFrame:
        """Convenience method to return the first several rows or records from a Batch of data.

        Args:
            n_rows: The number of rows to return.
            domain_kwargs: If provided, the domain for which to return records.
            fetch_all: If True, ignore n_rows and return the entire batch.

        Returns:
            A Pandas DataFrame containing the records' data.
        """
        if domain_kwargs is None:
            domain_kwargs = {}

        if domain_kwargs.get("batch_id") is None:
            domain_kwargs["batch_id"] = self._execution_engine.batch_manager.active_batch_id

        df: pd.DataFrame = self.get_metric(
            metric=MetricConfiguration(
                metric_name="table.head",
                metric_domain_kwargs=domain_kwargs,
                metric_value_kwargs={
                    "n_rows": n_rows,
                    "fetch_all": fetch_all,
                },
            )
        )

        return df.reset_index(drop=True, inplace=False)

    def get_metric(
        self,
        metric: MetricConfiguration,
    ) -> Any:
        """return the value of the requested metric."""
        return self.get_metrics(
            metrics={metric.metric_name: metric},
        )[metric.metric_name]

    def get_metrics(
        self,
        metrics: Dict[str, MetricConfiguration],
    ) -> Dict[str, Any]:
        """
        Args:
            metrics: Dictionary of desired metrics to be resolved; metric_name is key and MetricConfiguration is value.

        Returns:
            Return Dictionary with requested metrics resolved, with metric_name as key and computed metric as value.
        """  # noqa: E501 # FIXME CoP
        resolved_metrics: _MetricsDict
        resolved_metrics, _ = self.compute_metrics(
            metric_configurations=list(metrics.values()),
            runtime_configuration=None,
            min_graph_edges_pbar_enable=0,
        )
        return {
            metric_configuration.metric_name: resolved_metrics[metric_configuration.id]
            for metric_configuration in metrics.values()
        }

    def compute_metrics(
        self,
        metric_configurations: List[MetricConfiguration],
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,
        # Set to low number (e.g., 3) to suppress progress bar for small graphs.
    ) -> tuple[_MetricsDict, _AbortedMetricsInfoDict]:
        """
        Args:
            metric_configurations: List of desired MetricConfiguration objects to be resolved.
            runtime_configuration: Additional run-time settings (see "Validator.DEFAULT_RUNTIME_CONFIGURATION").
            min_graph_edges_pbar_enable: Minimum number of graph edges to warrant showing progress bars.

        Returns:
            Tuple of two elements, the first is a dictionary with requested metrics resolved,
            with unique metric ID as key and computed metric as value. The second is a dictionary of the
            aborted metrics information, with metric ID as key if any metrics were aborted.
        """  # noqa: E501 # FIXME CoP
        graph: ValidationGraph = self.build_metric_dependency_graph(
            metric_configurations=metric_configurations,
            runtime_configuration=runtime_configuration,
        )
        resolved_metrics: _MetricsDict
        aborted_metrics: _AbortedMetricsInfoDict
        (
            resolved_metrics,
            aborted_metrics,
        ) = self.resolve_validation_graph_and_handle_aborted_metrics_info(
            graph=graph,
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
        )
        return resolved_metrics, aborted_metrics

    def build_metric_dependency_graph(
        self,
        metric_configurations: List[MetricConfiguration],
        runtime_configuration: Optional[dict] = None,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
    ) -> ValidationGraph:
        """
        Obtain domain and value keys for metrics and proceeds to add these metrics to the validation graph
        until all metrics have been added.

        Args:
            metric_configurations: List of "MetricConfiguration" objects, for which to build combined "ValidationGraph".
            runtime_configuration: Additional run-time settings (see "Validator.DEFAULT_RUNTIME_CONFIGURATION").
            diagnostics_config: Optional expansion policy; when omitted it is derived from the new
                keys in ``runtime_configuration`` (if any).

        Returns:
            Resulting "ValidationGraph" object.
        """  # noqa: E501 # FIXME CoP
        if diagnostics_config is None:
            diagnostics_config = MetricDependencyDiagnosticsConfig.if_runtime_keys_present(
                runtime_configuration=runtime_configuration
            )

        graph: ValidationGraph = ValidationGraph(execution_engine=self._execution_engine)

        metric_configuration: MetricConfiguration
        for metric_configuration in metric_configurations:
            graph.build_metric_dependency_graph(
                metric_configuration=metric_configuration,
                runtime_configuration=runtime_configuration,
                diagnostics_config=diagnostics_config,
            )

        return graph

    def resolve_validation_graph_and_handle_aborted_metrics_info(
        self,
        graph: ValidationGraph,
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,
        # Set to low number (e.g., 3) to suppress progress bar for small graphs.
    ) -> Tuple[_MetricsDict, _AbortedMetricsInfoDict]:
        """
        Args:
            graph: "ValidationGraph" object, containing "metric_edge" structures with "MetricConfiguration" objects.
            runtime_configuration: Additional run-time settings (see "Validator.DEFAULT_RUNTIME_CONFIGURATION").
            min_graph_edges_pbar_enable: Minimum number of graph edges to warrant showing progress bars.

        Returns:
            Dictionary with requested metrics resolved, with unique metric ID as key and computed metric as value.
            Dictionary with aborted metrics information, with metric ID as key.
        """  # noqa: E501 # FIXME CoP
        resolved_metrics: _MetricsDict
        aborted_metrics_info: _AbortedMetricsInfoDict
        (
            resolved_metrics,
            aborted_metrics_info,
        ) = self.resolve_validation_graph(
            graph=graph,
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
        )

        if aborted_metrics_info:
            logger.warning(
                f"Exceptions\n{aborted_metrics_info!s}\noccurred while resolving metrics."
            )

        return (
            resolved_metrics,
            aborted_metrics_info,
        )

    def resolve_validation_graph(
        self,
        graph: ValidationGraph,
        runtime_configuration: Optional[dict] = None,
        min_graph_edges_pbar_enable: int = 0,
        # Set to low number (e.g., 3) to suppress progress bar for small graphs.
    ) -> Tuple[_MetricsDict, _AbortedMetricsInfoDict]:
        """
        Calls "ValidationGraph.resolve()" method with supplied arguments.

        Args:
            graph: "ValidationGraph" object, containing "metric_edge" structures with "MetricConfiguration" objects.
            runtime_configuration: Additional run-time settings (see "Validator.DEFAULT_RUNTIME_CONFIGURATION").
            min_graph_edges_pbar_enable: Minimum number of graph edges to warrant showing progress bars.

        Returns:
            Dictionary with requested metrics resolved, with unique metric ID as key and computed metric as value.
            Dictionary with aborted metrics information, with metric ID as key.
        """  # noqa: E501 # FIXME CoP
        resolved_metrics: _MetricsDict
        aborted_metrics_info: _AbortedMetricsInfoDict
        diagnostics_config = MetricDependencyDiagnosticsConfig.if_runtime_keys_present(
            runtime_configuration=runtime_configuration
        )
        resolved_metrics, aborted_metrics_info = graph.resolve(
            runtime_configuration=runtime_configuration,
            min_graph_edges_pbar_enable=min_graph_edges_pbar_enable,
            show_progress_bars=self._show_progress_bars,
            diagnostics_config=diagnostics_config,
        )
        return resolved_metrics, aborted_metrics_info

    # ------------------------------------------------------------------
    # Read-only metric dependency diagnostics
    # ------------------------------------------------------------------

    def diagnose_metric_dependencies(
        self,
        metric_configurations: List[MetricConfiguration],
        runtime_configuration: Optional[dict] = None,
        diagnostics_config: Optional[MetricDependencyDiagnosticsConfig] = None,
        resolve_metrics: Optional[bool] = None,
    ) -> MetricDependencyReport:
        """Builds a read-only report of the metric dependency graph.

        Args:
            metric_configurations: Desired metrics (or the validation dependencies
                derived from expectations).
            runtime_configuration: Additional run-time settings; diagnostics
                options may also be supplied via its keys.
            diagnostics_config: Explicit diagnostics options. Takes precedence
                over ``runtime_configuration`` when supplied.
            resolve_metrics: When ``True``, also solve the graph (requires data);
                when ``False`` only the structure and structural satisfiability
                are reported. Defaults to the config's value (structure-only).

        Returns:
            A ``MetricDependencyReport`` describing nodes, edges, cycles,
            unsatisfiable dependencies, provider errors, truncated branches and
            (optionally) resolution outcome. Never mutates caller configurations
            or the execution engine's metric cache.
        """  # FIXME CoP
        if diagnostics_config is None:
            config = MetricDependencyDiagnosticsConfig.from_runtime_configuration(
                runtime_configuration=runtime_configuration,
            )
        else:
            config = replace(diagnostics_config)
        if resolve_metrics is not None:
            config = replace(config, resolve_metrics=resolve_metrics)
        config.validate()

        # Expansion mutates default kwargs and the dependency dict of a config;
        # operate on copies so the caller's objects stay untouched.
        root_configurations: List[MetricConfiguration] = copy.deepcopy(list(metric_configurations))
        root_ids: List[MetricConfigurationID] = [
            metric_configuration.id for metric_configuration in root_configurations
        ]

        # Structural collection always isolates failures and skips cycles so the
        # whole graph can be described; raise policy is applied afterwards.
        build_config = replace(config, catch_exceptions=True, on_cycle=ON_CYCLE_SKIP)

        graph: ValidationGraph = ValidationGraph(execution_engine=self._execution_engine)
        for root in root_configurations:
            graph.build_metric_dependency_graph(
                metric_configuration=root,
                runtime_configuration=runtime_configuration,
                diagnostics_config=build_config,
            )

        report: MetricDependencyReport = self._build_structural_report(
            graph=graph,
            root_ids=root_ids,
        )

        if not config.catch_exceptions and not config.consolidate_exceptions:
            self._raise_first_failure(
                report=report,
                aborted_metrics_info={},
            )

        aborted_metrics_info: _AbortedMetricsInfoDict = {}
        if config.resolve_metrics:
            aborted_metrics_info = self._resolve_for_diagnostics(
                graph=graph,
                report=report,
                runtime_configuration=runtime_configuration,
                config=config,
            )

        self._finalize_report(report=report)

        if not config.catch_exceptions:
            if config.consolidate_exceptions:
                self._raise_consolidated_failure(report=report)
            else:
                self._raise_first_failure(
                    report=report,
                    aborted_metrics_info=aborted_metrics_info,
                )

        return report

    def _build_structural_report(
        self,
        graph: ValidationGraph,
        root_ids: List[MetricConfigurationID],
    ) -> MetricDependencyReport:
        cycles, cycles_enumeration_truncated = graph.find_dependency_cycles()

        unsatisfiable_dependencies: List[UnsatisfiableDependencyInfo] = []
        expansion_provider_errors: List[ProviderErrorInfo] = []

        metric_id: MetricConfigurationID
        error_info: MetricExpansionErrorResultValue
        for metric_id, error_info in graph.expansion_errors.items():
            exception_message = error_info["exception_info"].exception_message
            if error_info["error_type"] == "MetricProviderError":
                referenced_by = {
                    edge.left.id
                    for edge in graph.edges
                    if edge.right is not None and edge.right.id == metric_id
                }
                unsatisfiable_dependencies.append(
                    UnsatisfiableDependencyInfo(
                        metric_id=metric_id,
                        metric_name=metric_id.metric_name,
                        reason=exception_message,
                        referenced_by=sorted(referenced_by),
                    )
                )
            else:
                expansion_provider_errors.append(
                    ProviderErrorInfo(
                        metric_id=metric_id,
                        error_type=error_info["error_type"],
                        error_message=exception_message,
                        stage=STAGE_EXPANSION,
                    )
                )

        node_ids: List[MetricConfigurationID] = graph.metric_node_ids()
        for root_id in root_ids:
            if root_id not in node_ids:
                node_ids.append(root_id)

        return MetricDependencyReport(
            roots=list(root_ids),
            nodes=node_ids,
            edges=[
                (
                    edge.left.id,
                    edge.right.id if edge.right is not None else None,
                )
                for edge in graph.edges
            ],
            cycles=cycles,
            unsatisfiable_dependencies=sorted(
                unsatisfiable_dependencies, key=lambda item: item.metric_id
            ),
            provider_errors=sorted(
                expansion_provider_errors,
                key=lambda item: str(item.metric_id),
            ),
            truncated_nodes=sorted(
                graph.truncated_metric_infos.values(),
                key=lambda item: item.metric_id,
            ),
            structure_complete=len(graph.truncated_metric_infos) == 0,
            cycles_enumeration_truncated=cycles_enumeration_truncated,
        )

    def _resolve_for_diagnostics(
        self,
        graph: ValidationGraph,
        report: MetricDependencyReport,
        runtime_configuration: Optional[dict],
        config: MetricDependencyDiagnosticsConfig,
    ) -> _AbortedMetricsInfoDict:
        # Isolate the execution engine metric cache: solve into a throwaway cache
        # and restore the original so diagnostics never affects later runs.
        execution_engine: Any = self._execution_engine
        original_cache: Any = getattr(execution_engine, "_metric_cache", None)
        cache_swapped = isinstance(original_cache, dict)
        if cache_swapped:
            execution_engine._metric_cache = {}

        effective_catch_exceptions = config.catch_exceptions or config.consolidate_exceptions
        effective_runtime_configuration = dict(runtime_configuration or {})
        effective_runtime_configuration["catch_exceptions"] = effective_catch_exceptions

        try:
            (
                resolved_metrics,
                aborted_metrics_info,
                resolution_diagnostics,
            ) = graph.resolve_with_diagnostics(
                runtime_configuration=effective_runtime_configuration,
                min_graph_edges_pbar_enable=0,
                show_progress_bars=False,
                diagnostics_config=replace(config, catch_exceptions=effective_catch_exceptions),
            )
        finally:
            if cache_swapped:
                execution_engine._metric_cache = original_cache

        report.metrics_resolved = True
        report.resolved_metric_ids = sorted(resolved_metrics.keys())
        report.converged = resolution_diagnostics.converged
        report.blocked_nodes = sorted(
            resolution_diagnostics.blocked_metrics.values(),
            key=lambda item: item.metric_id,
        )
        if resolution_diagnostics.resolution_aborted is not None:
            report.provider_errors.append(resolution_diagnostics.resolution_aborted)
        for aborted_id, aborted_info in aborted_metrics_info.items():
            report.provider_errors.append(
                ProviderErrorInfo(
                    metric_id=aborted_id,
                    error_type="MetricResolutionError",
                    error_message=aborted_info["exception_info"].exception_message,
                    stage=STAGE_RESOLUTION,
                    num_failures=aborted_info["num_failures"],
                )
            )
        report.non_convergence_reasons = list(resolution_diagnostics.non_convergence_reasons)
        return aborted_metrics_info

    def _finalize_report(self, report: MetricDependencyReport) -> None:
        report.cycles = sorted(
            report.cycles,
            key=lambda cycle: (cycle.is_self_loop, tuple(cycle.path)),
        )
        report.provider_errors = sorted(
            report.provider_errors,
            key=lambda item: (
                item.metric_id is None,
                str(item.metric_id) if item.metric_id is not None else "",
            ),
        )

        reasons = set(report.non_convergence_reasons)
        if any(not cycle.is_self_loop for cycle in report.cycles):
            reasons.add(REASON_CYCLE)
        if report.unsatisfiable_dependencies:
            reasons.add(REASON_UNSATISFIABLE_DEPENDENCY)
        if report.truncated_nodes:
            reasons.add(REASON_DEPTH_TRUNCATION)
        if any(
            provider_error.stage == STAGE_EXPANSION for provider_error in report.provider_errors
        ):
            reasons.add(REASON_PROVIDER_ERROR)
        if report.metrics_resolved:
            if any(
                provider_error.stage == STAGE_RESOLUTION
                for provider_error in report.provider_errors
            ):
                reasons.add(REASON_PROVIDER_ERROR)
            if report.converged:
                # Structural blockers cannot coexist with a real fixpoint; keep
                # only reasons backed by actual evidence.
                reasons.discard(REASON_CYCLE)
                reasons.discard(REASON_UNSATISFIABLE_DEPENDENCY)
                reasons.discard(REASON_DEPTH_TRUNCATION)
        report.non_convergence_reasons = sorted(
            reason for reason in reasons if reason != REASON_RESOLUTION_ABORTED
        ) + ([REASON_RESOLUTION_ABORTED] if REASON_RESOLUTION_ABORTED in reasons else [])

    def _raise_first_failure(
        self,
        report: MetricDependencyReport,
        aborted_metrics_info: _AbortedMetricsInfoDict,
    ) -> None:
        # Multi-node cycles first.
        multi_node_cycles = [cycle for cycle in report.cycles if not cycle.is_self_loop]
        if multi_node_cycles:
            raise gx_exceptions.MetricDependencyCycleError(
                message=(
                    "Metric dependencies form one or more cycles: "
                    + "; ".join(
                        " -> ".join(str(node) for node in cycle.path) for cycle in multi_node_cycles
                    )
                ),
                cycle_paths=[cycle.path for cycle in multi_node_cycles],
            )

        # Unsatisfiable dependency (no provider registered).
        if report.unsatisfiable_dependencies:
            unsatisfiable = report.unsatisfiable_dependencies[0]
            message = unsatisfiable.reason
            prefix = "MetricProviderError: "
            if message.startswith(prefix):
                message = message[len(prefix) :]
            raise gx_exceptions.MetricProviderError(message)

        # Provider errors observed while resolving.
        if aborted_metrics_info:
            first_aborted_id = sorted(aborted_metrics_info.keys())[0]
            first_aborted_info = aborted_metrics_info[first_aborted_id]
            raise gx_exceptions.MetricResolutionError(
                message=first_aborted_info["exception_info"].exception_message,
                failed_metrics=[first_aborted_info["metric_configuration"]],
            )

        # Unattributable resolution abort.
        resolution_abort = next(
            (
                provider_error
                for provider_error in report.provider_errors
                if provider_error.metric_id is None and provider_error.stage == STAGE_RESOLUTION
            ),
            None,
        )
        if resolution_abort is not None:
            raise gx_exceptions.MetricError(resolution_abort.error_message)

        # Expansion-stage provider error.
        expansion_error = next(
            (
                provider_error
                for provider_error in report.provider_errors
                if provider_error.stage == STAGE_EXPANSION
            ),
            None,
        )
        if expansion_error is not None:
            raise gx_exceptions.MetricError(expansion_error.error_message)

    def _raise_consolidated_failure(self, report: MetricDependencyReport) -> None:
        failures: List[DiagnosticFailureEntry] = []

        for cycle in report.cycles:
            failures.append(
                DiagnosticFailureEntry(
                    category=CATEGORY_CYCLE,
                    metric_id=cycle.path[0],
                    cycle_path=list(cycle.path),
                    message=("Self-dependency on " if cycle.is_self_loop else "Dependency cycle ")
                    + " -> ".join(str(node) for node in cycle.path),
                )
            )

        for unsatisfiable_item in report.unsatisfiable_dependencies:
            failures.append(
                DiagnosticFailureEntry(
                    category=CATEGORY_UNSATISFIABLE_DEPENDENCY,
                    metric_id=unsatisfiable_item.metric_id,
                    message=(
                        f"No provider for metric {unsatisfiable_item.metric_name}: "
                        f"{unsatisfiable_item.reason}"
                    ),
                )
            )

        for provider_error_item in report.provider_errors:
            failures.append(
                DiagnosticFailureEntry(
                    category=CATEGORY_PROVIDER_ERROR,
                    metric_id=provider_error_item.metric_id,
                    message=(
                        f"{provider_error_item.error_type} during metric "
                        f"{provider_error_item.stage}: "
                        f"{provider_error_item.error_message}"
                    ),
                )
            )

        for blocked_item in report.blocked_nodes:
            failures.append(
                DiagnosticFailureEntry(
                    category=CATEGORY_BLOCKED,
                    metric_id=blocked_item.metric_id,
                    message=(
                        f"Metric blocked ({blocked_item.reason}) waiting on "
                        + ", ".join(str(node) for node in blocked_item.blocking_metric_ids)
                    ),
                )
            )

        if not failures:
            return

        message = f"Metric dependency diagnostics found {len(failures)} failure(s):\n" + "\n".join(
            f"- [{failure.category}] {failure.message}" for failure in failures
        )
        raise MetricDependencyDiagnosticsError(message=message, failures=failures)

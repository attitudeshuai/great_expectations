from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, cast

import pytest

import great_expectations.exceptions as gx_exceptions
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations.expectation_configuration import (
    ExpectationConfiguration,
)
from great_expectations.validator.metric_configuration import (
    MetricConfiguration,
    MetricConfigurationID,
)
from great_expectations.validator.metric_dependency_diagnostics import (
    ON_CYCLE_RAISE,
    MetricDependencyDiagnosticsConfig,
    MetricDependencyDiagnosticsError,
    find_cycles,
)
from great_expectations.validator.metrics_calculator import MetricsCalculator
from great_expectations.validator.validation_graph import ValidationGraph

if TYPE_CHECKING:
    from great_expectations.execution_engine import ExecutionEngine

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMetricProvider:
    dependency_map: Dict[str, List[str]] = {}

    default_kwarg_values: dict = {}
    domain_keys: tuple = ()
    value_keys: tuple = ()

    @classmethod
    def get_evaluation_dependencies(
        cls,
        metric: MetricConfiguration,
        configuration: Any = None,
        execution_engine: Any = None,
        runtime_configuration: Optional[dict] = None,
    ) -> Dict[str, MetricConfiguration]:
        return {
            f"dep_{dep_name}": MetricConfiguration(
                metric_name=dep_name,
                metric_domain_kwargs=dict(metric.metric_domain_kwargs),
            )
            for dep_name in cls.dependency_map.get(metric.metric_name, [])
        }


class FakeExecutionEngine:
    """Stand-in execution engine with a controllable resolve_metrics + cache."""

    def __init__(self, failing_metric_names: Optional[Iterable[str]] = None) -> None:
        self._failing_metric_names = set(failing_metric_names or [])
        self._caching = True
        self._metric_cache: dict = {}
        self.resolve_call_count = 0

    def resolve_metrics(
        self,
        metrics_to_resolve: Iterable[MetricConfiguration],
        metrics: Optional[dict] = None,
        runtime_configuration: Optional[dict] = None,
    ) -> Dict[MetricConfigurationID, Any]:
        self.resolve_call_count += 1
        resolved: Dict[MetricConfigurationID, Any] = {}
        failed_metrics: List[MetricConfiguration] = []
        for metric_configuration in metrics_to_resolve:
            if metric_configuration.metric_name in self._failing_metric_names:
                failed_metrics.append(metric_configuration)
            else:
                resolved[metric_configuration.id] = f"value_{metric_configuration.metric_name}"
        if failed_metrics:
            raise gx_exceptions.MetricResolutionError(
                message=(
                    "boom: provider failed for "
                    + ", ".join(metric.metric_name for metric in failed_metrics)
                ),
                failed_metrics=failed_metrics,
            )
        return resolved


@pytest.fixture
def metrics_calculator_factory(monkeypatch):
    def _factory(
        dependency_map: Dict[str, List[str]],
        missing_provider_names: Optional[Iterable[str]] = None,
        failing_metric_names: Optional[Iterable[str]] = None,
    ) -> MetricsCalculator:
        FakeMetricProvider.dependency_map = dependency_map
        missing = set(missing_provider_names or [])

        def _fake_lookup(graph_self, metric_configuration: MetricConfiguration):
            metric_name = metric_configuration.metric_name
            if metric_name in missing:
                raise gx_exceptions.MetricProviderError(f"No metric named {metric_name} found.")
            return FakeMetricProvider, None

        monkeypatch.setattr(
            ValidationGraph,
            "set_metric_configuration_default_kwargs_if_absent",
            _fake_lookup,
        )

        engine = FakeExecutionEngine(failing_metric_names=failing_metric_names)
        return MetricsCalculator(
            execution_engine=cast("ExecutionEngine", engine),
            show_progress_bars=False,
        )

    return _factory


def metric_id(name: str) -> MetricConfigurationID:
    return MetricConfiguration(metric_name=name, metric_domain_kwargs={}).id


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diagnostics_config_validates_values():
    MetricDependencyDiagnosticsConfig()
    MetricDependencyDiagnosticsConfig(max_dependency_depth=0)
    MetricDependencyDiagnosticsConfig(max_dependency_depth=5, on_cycle=ON_CYCLE_RAISE)

    with pytest.raises(ValueError, match="max_metric_dependency_depth"):
        MetricDependencyDiagnosticsConfig(max_dependency_depth=-1)

    with pytest.raises(ValueError, match="max_metric_dependency_depth"):
        MetricDependencyDiagnosticsConfig(max_dependency_depth=1.5)  # type: ignore[arg-type]

    # bool is a subclass of int but is not a meaningful depth value
    with pytest.raises(ValueError, match="max_metric_dependency_depth"):
        MetricDependencyDiagnosticsConfig(max_dependency_depth=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="on_cycle"):
        MetricDependencyDiagnosticsConfig(on_cycle="explode")


@pytest.mark.unit
def test_diagnostics_config_from_runtime_configuration():
    config = MetricDependencyDiagnosticsConfig.from_runtime_configuration(
        {
            "max_metric_dependency_depth": 3,
            "on_cycle": "raise",
            "consolidate_exceptions": True,
            "catch_exceptions": False,
            "result_format": "BASIC",  # unrelated key is ignored
        }
    )
    assert config.max_dependency_depth == 3
    assert config.on_cycle == ON_CYCLE_RAISE
    assert config.consolidate_exceptions is True
    assert config.catch_exceptions is False

    assert (
        MetricDependencyDiagnosticsConfig.if_runtime_keys_present({"catch_exceptions": False})
        is None
    )
    assert (
        MetricDependencyDiagnosticsConfig.if_runtime_keys_present({"on_cycle": "raise"}) is not None
    )


# ---------------------------------------------------------------------------
# Pure graph algorithms
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_cycles_two_ring_and_three_ring():
    adjacency = {
        metric_id("a"): {metric_id("b")},
        metric_id("b"): {metric_id("a")},
    }
    cycles, truncated = find_cycles(adjacency)
    assert truncated is False
    assert len(cycles) == 1
    assert cycles[0].path == [metric_id("a"), metric_id("b"), metric_id("a")]
    assert cycles[0].is_self_loop is False

    adjacency = {
        metric_id("a"): {metric_id("b")},
        metric_id("b"): {metric_id("c")},
        metric_id("c"): {metric_id("a")},
    }
    cycles, _ = find_cycles(adjacency)
    assert len(cycles) == 1
    assert len(cycles[0].path) == 4
    assert cycles[0].path[0] == cycles[0].path[-1]


@pytest.mark.unit
def test_find_cycles_diamond_is_acyclic_and_self_loops_listed():
    adjacency = {
        metric_id("a"): {metric_id("b"), metric_id("c")},
        metric_id("b"): {metric_id("d")},
        metric_id("c"): {metric_id("d")},
        metric_id("d"): set(),
    }
    cycles, _ = find_cycles(adjacency, self_dependency_ids={metric_id("d")})
    assert len(cycles) == 1
    assert cycles[0].is_self_loop is True
    assert cycles[0].path == [metric_id("d"), metric_id("d")]


@pytest.mark.unit
def test_find_cycles_enumerates_distinct_circuits_with_chords():
    adjacency = {
        metric_id("a"): {metric_id("b"), metric_id("c")},
        metric_id("b"): {metric_id("a")},
        metric_id("c"): {metric_id("a")},
    }
    cycles, _ = find_cycles(adjacency)
    paths = {tuple(cycle.path) for cycle in cycles}
    assert paths == {
        (metric_id("a"), metric_id("b"), metric_id("a")),
        (metric_id("a"), metric_id("c"), metric_id("a")),
    }


# ---------------------------------------------------------------------------
# Structural diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_structure_only_report_does_not_require_data(metrics_calculator_factory):
    calculator = metrics_calculator_factory(dependency_map={"root": []})
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="root", metric_domain_kwargs={})]
    )
    assert report.metrics_resolved is False
    assert report.converged is None
    assert report.structure_complete is True
    assert report.is_satisfiable() is True
    assert report.nodes == [metric_id("root")]
    assert report.edges == [(metric_id("root"), None)]
    assert report.cycles == []
    assert report.non_convergence_reasons == []
    assert report.to_json_dict()["satisfiable"] is True
    json.dumps(report.to_json_dict())
    assert "root" in str(report)


@pytest.mark.unit
def test_self_loop_is_skipped_and_marked(metrics_calculator_factory):
    calculator = metrics_calculator_factory(dependency_map={"selfish": ["selfish"]})
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="selfish", metric_domain_kwargs={})]
    )
    assert report.self_dependencies == [metric_id("selfish")]
    assert len(report.cycles) == 1
    assert report.cycles[0].is_self_loop is True
    # Legacy behavior: the self edge is not added to the graph.
    assert report.edges == []
    assert report.is_satisfiable() is False


@pytest.mark.unit
def test_two_node_cycle_reports_full_path(metrics_calculator_factory):
    calculator = metrics_calculator_factory(
        dependency_map={
            "root": ["a"],
            "a": ["b"],
            "b": ["a"],
            "ok": [],
        }
    )
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[
            MetricConfiguration(metric_name="root", metric_domain_kwargs={}),
            MetricConfiguration(metric_name="ok", metric_domain_kwargs={}),
        ],
        resolve_metrics=True,
    )

    multi_node_cycles = [cycle for cycle in report.cycles if not cycle.is_self_loop]
    assert len(multi_node_cycles) == 1
    assert multi_node_cycles[0].path == [
        metric_id("a"),
        metric_id("b"),
        metric_id("a"),
    ]

    # Independent nodes still solve.
    assert report.resolved_metric_ids == [metric_id("ok")]
    assert report.converged is False
    assert "cycle" in report.non_convergence_reasons

    blocked = {item.metric_id: item for item in report.blocked_nodes}
    assert blocked[metric_id("a")].reason == "cycle"
    assert blocked[metric_id("b")].reason == "cycle"
    assert blocked[metric_id("root")].reason == "cycle"
    assert blocked[metric_id("root")].blocking_metric_ids == [metric_id("a")]

    # No empty "success" result.
    assert report.provider_errors == []
    assert report.resolved_metric_ids != report.nodes


@pytest.mark.unit
def test_long_cycle_reports_complete_path_without_recursion_error(
    metrics_calculator_factory,
):
    ring_size = 50
    ring_names = [f"n{index}" for index in range(ring_size)]
    dependency_map = {
        name: [ring_names[(index + 1) % ring_size]] for index, name in enumerate(ring_names)
    }
    dependency_map["ok"] = []

    calculator = metrics_calculator_factory(dependency_map=dependency_map)
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[
            MetricConfiguration(metric_name=ring_names[0], metric_domain_kwargs={}),
            MetricConfiguration(metric_name="ok", metric_domain_kwargs={}),
        ],
        resolve_metrics=True,
    )

    multi_node_cycles = [cycle for cycle in report.cycles if not cycle.is_self_loop]
    assert len(multi_node_cycles) == 1
    # Full path: every ring node once, closing node repeated at the end.
    cycle_path = multi_node_cycles[0].path
    assert len(cycle_path) == ring_size + 1
    assert set(cycle_path[:-1]) == {metric_id(name) for name in ring_names}
    assert cycle_path[0] == cycle_path[-1]

    assert report.resolved_metric_ids == [metric_id("ok")]
    assert report.converged is False
    assert report.non_convergence_reasons == ["cycle"]


@pytest.mark.unit
def test_unsatisfiable_dependency_is_distinct_from_provider_error(
    metrics_calculator_factory,
):
    calculator = metrics_calculator_factory(
        dependency_map={"parent": ["missing"]},
        missing_provider_names={"missing"},
    )
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="parent", metric_domain_kwargs={})],
        resolve_metrics=True,
    )

    assert len(report.unsatisfiable_dependencies) == 1
    unsatisfiable = report.unsatisfiable_dependencies[0]
    assert unsatisfiable.metric_id == metric_id("missing")
    assert "No metric named missing found." in unsatisfiable.reason
    assert unsatisfiable.referenced_by == [metric_id("parent")]

    assert report.provider_errors == []
    assert report.converged is False
    assert "unsatisfiable_dependency" in report.non_convergence_reasons
    blocked = {item.metric_id: item for item in report.blocked_nodes}
    assert blocked[metric_id("parent")].reason == "unsatisfiable_dependency"
    assert blocked[metric_id("parent")].blocking_metric_ids == [metric_id("missing")]


@pytest.mark.unit
def test_provider_error_carries_original_message(metrics_calculator_factory):
    calculator = metrics_calculator_factory(
        dependency_map={"root": ["bad"], "bad": [], "ok": []},
        failing_metric_names={"bad"},
    )
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[
            MetricConfiguration(metric_name="root", metric_domain_kwargs={}),
            MetricConfiguration(metric_name="ok", metric_domain_kwargs={}),
        ],
        resolve_metrics=True,
    )

    assert report.unsatisfiable_dependencies == []
    assert len(report.provider_errors) == 1
    provider_error = report.provider_errors[0]
    assert provider_error.metric_id == metric_id("bad")
    assert provider_error.stage == "resolution"
    assert "boom" in provider_error.error_message
    assert provider_error.num_failures == 3

    assert report.resolved_metric_ids == [metric_id("ok")]
    assert report.converged is False
    assert "provider_error" in report.non_convergence_reasons
    blocked = {item.metric_id: item for item in report.blocked_nodes}
    assert blocked[metric_id("root")].reason == "provider_error"
    assert blocked[metric_id("root")].blocking_metric_ids == [metric_id("bad")]


# ---------------------------------------------------------------------------
# Depth limit
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normal_compute_path_skips_cycle_and_keeps_other_results(
    metrics_calculator_factory,
):
    # New options flow through the ordinary compute_metrics entry point as well.
    calculator = metrics_calculator_factory(
        dependency_map={"root": ["a"], "a": ["b"], "b": ["a"], "ok": []}
    )
    resolved, _aborted = calculator.compute_metrics(
        metric_configurations=[
            MetricConfiguration(metric_name="root", metric_domain_kwargs={}),
            MetricConfiguration(metric_name="ok", metric_domain_kwargs={}),
        ],
        runtime_configuration={
            "catch_exceptions": True,
            "on_cycle": "skip",
        },
    )
    assert metric_id("ok") in resolved
    assert metric_id("a") not in resolved
    assert metric_id("b") not in resolved
    assert metric_id("root") not in resolved


@pytest.mark.unit
def test_normal_compute_path_raises_cycle_per_runtime_configuration(
    metrics_calculator_factory,
):
    calculator = metrics_calculator_factory(dependency_map={"root": ["a"], "a": ["b"], "b": ["a"]})
    with pytest.raises(gx_exceptions.MetricDependencyCycleError):
        calculator.compute_metrics(
            metric_configurations=[
                MetricConfiguration(metric_name="root", metric_domain_kwargs={})
            ],
            runtime_configuration={
                "catch_exceptions": False,
                "on_cycle": "raise",
            },
        )


@pytest.mark.unit
def test_depth_limit_records_truncated_nodes(metrics_calculator_factory):
    calculator = metrics_calculator_factory(
        dependency_map={
            "root": ["a"],
            "a": ["b"],
            "b": ["leaf"],
            "leaf": [],
        }
    )
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="root", metric_domain_kwargs={})],
        diagnostics_config=MetricDependencyDiagnosticsConfig(max_dependency_depth=1),
    )

    assert [item.metric_id for item in report.truncated_nodes] == [metric_id("a")]
    assert report.truncated_nodes[0].depth == 1
    assert report.truncated_nodes[0].max_dependency_depth == 1
    assert report.structure_complete is False
    assert report.is_satisfiable() is None
    assert report.edges == [(metric_id("root"), metric_id("a"))]


@pytest.mark.unit
def test_depth_limit_does_not_raise_recursion_error(metrics_calculator_factory):
    # A chain far shallower than the interpreter limit still proves the cap path;
    # the important contract is a structured truncation entry, not an exception.
    calculator = metrics_calculator_factory(
        dependency_map={"root": ["a"], "a": ["b"], "b": ["leaf"], "leaf": []}
    )
    report = calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="root", metric_domain_kwargs={})],
        diagnostics_config=MetricDependencyDiagnosticsConfig(
            max_dependency_depth=2,
            resolve_metrics=True,
        ),
    )
    assert sorted(item.metric_id for item in report.truncated_nodes) == [metric_id("b")]
    assert report.converged is False
    assert "depth_truncation" in report.non_convergence_reasons


# ---------------------------------------------------------------------------
# Raise / consolidation policies
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_cycle_raise_policy_at_build(metrics_calculator_factory):
    calculator = metrics_calculator_factory(dependency_map={"root": ["a"], "a": ["b"], "b": ["a"]})
    graph = ValidationGraph(execution_engine=calculator._execution_engine)
    with pytest.raises(gx_exceptions.MetricDependencyCycleError) as exc_info:
        graph.build_metric_dependency_graph(
            metric_configuration=MetricConfiguration(metric_name="root", metric_domain_kwargs={}),
            diagnostics_config=MetricDependencyDiagnosticsConfig(
                catch_exceptions=False,
                on_cycle=ON_CYCLE_RAISE,
            ),
        )
    assert exc_info.value.cycle_paths[0][0] == metric_id("a")


@pytest.mark.unit
def test_diagnostics_native_raise_matches_report_entry(metrics_calculator_factory):
    calculator = metrics_calculator_factory(
        dependency_map={"root": ["a"], "a": ["b"], "b": ["a"], "ok": []}
    )
    with pytest.raises(gx_exceptions.MetricDependencyCycleError):
        calculator.diagnose_metric_dependencies(
            metric_configurations=[
                MetricConfiguration(metric_name="root", metric_domain_kwargs={})
            ],
            diagnostics_config=MetricDependencyDiagnosticsConfig(catch_exceptions=False),
        )


@pytest.mark.unit
def test_diagnostics_native_raise_unsatisfiable(metrics_calculator_factory):
    calculator = metrics_calculator_factory(
        dependency_map={"parent": ["missing"]},
        missing_provider_names={"missing"},
    )
    with pytest.raises(gx_exceptions.MetricProviderError, match="No metric named"):
        calculator.diagnose_metric_dependencies(
            metric_configurations=[
                MetricConfiguration(metric_name="parent", metric_domain_kwargs={})
            ],
            diagnostics_config=MetricDependencyDiagnosticsConfig(catch_exceptions=False),
        )


@pytest.mark.unit
def test_diagnostics_consolidated_exception_lists_every_failure(
    metrics_calculator_factory,
):
    calculator = metrics_calculator_factory(
        dependency_map={"root": ["bad"], "bad": []},
        failing_metric_names={"bad"},
    )
    with pytest.raises(MetricDependencyDiagnosticsError) as exc_info:
        calculator.diagnose_metric_dependencies(
            metric_configurations=[
                MetricConfiguration(metric_name="root", metric_domain_kwargs={})
            ],
            diagnostics_config=MetricDependencyDiagnosticsConfig(
                catch_exceptions=False,
                consolidate_exceptions=True,
                resolve_metrics=True,
            ),
        )

    categories = {failure.category for failure in exc_info.value.failures}
    assert "provider_error" in categories
    assert "blocked" in categories
    # Every failure maps to a metric id present in the would-be report.
    metric_categories = {failure.metric_id for failure in exc_info.value.failures}
    assert metric_categories == {metric_id("bad"), metric_id("root")}


# ---------------------------------------------------------------------------
# Side-effect guarantees
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_diagnostics_does_not_mutate_caller_config(metrics_calculator_factory):
    calculator = metrics_calculator_factory(dependency_map={"root": ["a"], "a": []})
    root = MetricConfiguration(metric_name="root", metric_domain_kwargs={})
    calculator.diagnose_metric_dependencies(metric_configurations=[root])
    assert dict(root.metric_dependencies) == {}


@pytest.mark.unit
def test_diagnostics_resolve_does_not_pollute_engine_cache(
    metrics_calculator_factory,
):
    calculator = metrics_calculator_factory(dependency_map={"ok": []})
    engine = cast("FakeExecutionEngine", calculator._execution_engine)
    sentinel_id = MetricConfigurationID("sentinel", "", "")
    engine._metric_cache = {sentinel_id: "preexisting"}

    calculator.diagnose_metric_dependencies(
        metric_configurations=[MetricConfiguration(metric_name="ok", metric_domain_kwargs={})],
        resolve_metrics=True,
    )

    assert engine._metric_cache == {sentinel_id: "preexisting"}
    assert metric_id("ok") not in engine._metric_cache


# ---------------------------------------------------------------------------
# End-to-end against the real pandas engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validator_diagnostics_real_pandas_engine(in_memory_runtime_context, monkeypatch):
    from great_expectations.self_check.util import get_test_validator_with_data

    validator = get_test_validator_with_data(
        execution_engine="pandas",
        data={"a": [1, 2, 3]},
        table_name="metric_dependency_diagnostics_test",
        context=in_memory_runtime_context,
    )
    configuration = ExpectationConfiguration(
        type="expect_table_row_count_to_equal",
        kwargs={"value": 3},
    )

    # Structure-only: nothing is computed.
    engine = validator.execution_engine

    def _forbidden_resolve(*args, **kwargs):
        raise AssertionError("structure-only diagnostics must not compute")

    monkeypatch.setattr(engine, "resolve_metrics", _forbidden_resolve)
    report = validator.diagnose_metric_dependencies(expectation_configurations=[configuration])
    assert report.metrics_resolved is False
    assert report.is_satisfiable() is True
    assert report.nodes
    assert any(node.metric_name == "table.row_count" for node in report.nodes)

    # An ExpectationSuite ("校验定义") is accepted too.
    suite_report = validator.diagnose_metric_dependencies(
        expectation_suite=ExpectationSuite(name="diagnostics_suite", expectations=[configuration])
    )
    assert suite_report.is_satisfiable() is True

    monkeypatch.undo()

    # Solve for real: converged, no failures, and the diagnostic solve never
    # writes into the execution engine's metric cache.
    import copy

    resolved_report = validator.diagnose_metric_dependencies(
        expectation_configurations=[copy.deepcopy(configuration)],
        resolve_metrics=True,
    )
    assert resolved_report.converged is True
    assert resolved_report.provider_errors == []
    assert resolved_report.blocked_nodes == []
    assert resolved_report.non_convergence_reasons == []
    row_count_id = next(
        node_id
        for node_id in resolved_report.resolved_metric_ids
        if node_id.metric_name == "table.row_count"
    )
    assert row_count_id not in validator.execution_engine._metric_cache

    # Parity: the normal validation path still succeeds and populates the cache.
    results = validator.graph_validate([copy.deepcopy(configuration)])
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].result["observed_value"] == 3
    assert validator.execution_engine._metric_cache[row_count_id] == 3

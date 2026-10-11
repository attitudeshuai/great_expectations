import pandas as pd

import great_expectations.expectations as gxe
from tests.integration.conftest import parameterize_batch_for_data_sources
from tests.integration.test_utils.data_source_config import (
    ALL_DATA_SOURCES,
    DataSourceTestConfig,
)

PANDAS_DATA_SOURCES: list[DataSourceTestConfig] = [
    ds for ds in ALL_DATA_SOURCES if ds.label.startswith("pandas")
]

_DATA = pd.DataFrame({"a": [1, 2, 3]})


class TestMetricDependencyDiagnostics:
    @parameterize_batch_for_data_sources(
        data_source_configs=PANDAS_DATA_SOURCES,
        data=_DATA,
    )
    def test_structure_report_needs_no_data(self, batch_for_datasource) -> None:
        expectation = gxe.ExpectTableRowCountToEqual(value=3)

        validator = batch_for_datasource._create_validator(result_format="BASIC")._wrapped_validator

        report = validator.diagnose_metric_dependencies(
            expectation_configurations=[expectation.configuration]
        )

        assert report.metrics_resolved is False
        assert report.converged is None
        assert report.is_satisfiable() is True
        assert report.cycles == []
        assert report.unsatisfiable_dependencies == []
        assert any(node.metric_name == "table.row_count" for node in report.nodes)

    @parameterize_batch_for_data_sources(
        data_source_configs=PANDAS_DATA_SOURCES,
        data=_DATA,
    )
    def test_resolve_report_converges_and_matches_validation(self, batch_for_datasource):
        expectation = gxe.ExpectTableRowCountToEqual(value=3)

        validator = batch_for_datasource._create_validator(result_format="BASIC")._wrapped_validator
        engine = validator.execution_engine

        cache_before = dict(engine._metric_cache)

        report = validator.diagnose_metric_dependencies(
            expectation_configurations=[expectation.configuration],
            resolve_metrics=True,
        )

        assert report.converged is True
        assert report.provider_errors == []
        assert report.blocked_nodes == []
        assert report.non_convergence_reasons == []

        row_count_id = next(
            node_id
            for node_id in report.resolved_metric_ids
            if node_id.metric_name == "table.row_count"
        )
        # The diagnostic solve must not change the engine's metric cache.
        assert engine._metric_cache == cache_before
        assert row_count_id not in engine._metric_cache

        # Parity with the normal validation path.
        result = batch_for_datasource.validate(expectation)
        assert result.success is True
        assert result.result["observed_value"] == 3

    @parameterize_batch_for_data_sources(
        data_source_configs=PANDAS_DATA_SOURCES,
        data=_DATA,
    )
    def test_provider_error_is_distinguished_from_unsatisfiable_dependency(
        self, batch_for_datasource
    ):
        # A structurally valid graph (all providers exist) whose computation fails
        # because the configured column is absent from the data.
        expectation = gxe.ExpectColumnMaxToBeBetween(
            column="not_in_table",
            min_value=0,
            max_value=10,
        )

        validator = batch_for_datasource._create_validator(result_format="BASIC")._wrapped_validator

        structural_report = validator.diagnose_metric_dependencies(
            expectation_configurations=[expectation.configuration]
        )
        assert structural_report.is_satisfiable() is True
        assert structural_report.unsatisfiable_dependencies == []

        resolved_report = validator.diagnose_metric_dependencies(
            expectation_configurations=[expectation.configuration],
            resolve_metrics=True,
        )

        assert resolved_report.unsatisfiable_dependencies == []
        assert len(resolved_report.provider_errors) >= 1
        assert all(
            provider_error.stage == "resolution"
            for provider_error in resolved_report.provider_errors
        )
        assert resolved_report.converged is False
        assert "provider_error" in resolved_report.non_convergence_reasons
        # No false empty success: something failed and it is explained.
        assert resolved_report.blocked_nodes or resolved_report.provider_errors

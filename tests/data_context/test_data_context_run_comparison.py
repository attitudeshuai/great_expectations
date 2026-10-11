from __future__ import annotations

import datetime as dt

import pytest

from great_expectations.core import (
    ComparisonThresholds,
    ExpectationSuiteValidationResult,
    ExpectationValidationResult,
    RunIdentifier,
)
from great_expectations.data_context.types.resource_identifiers import (
    ExpectationSuiteIdentifier,
    ValidationResultIdentifier,
)
from great_expectations.expectations.expectation_configuration import (
    ExpectationConfiguration,
)

pytestmark = pytest.mark.unit

SUITE = "my_suite"
OTHER_SUITE = "other_suite"
T0 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
DAY = dt.timedelta(days=1)


def _evr(success, observed, unexpected_count, unexpected_percent):
    return ExpectationValidationResult(
        success=success,
        expectation_config=ExpectationConfiguration(
            type="expect_table_row_count_to_equal", kwargs={"value": 100}
        ),
        result={
            "observed_value": observed,
            "unexpected_count": unexpected_count,
            "unexpected_percent": unexpected_percent,
        },
    )


def _suite_result(run_id, evrs, suite_name=SUITE, batch_parameters=None):
    total = len(evrs)
    successful = sum(1 for evr in evrs if evr.success)
    meta: dict = {"run_id": run_id, "expectation_suite_name": suite_name}
    if batch_parameters is not None:
        meta["batch_parameters"] = batch_parameters
    return ExpectationSuiteValidationResult(
        success=all(evr.success for evr in evrs),
        results=evrs,
        suite_name=suite_name,
        statistics={
            "evaluated_expectations": total,
            "successful_expectations": successful,
            "unsuccessful_expectations": total - successful,
            "success_percent": (successful / total * 100 if total else None),
        },
        meta=meta,
    )


def _save(context, run_id, batch_id, suite_result, suite_name=SUITE):
    store = context.stores[context.validation_results_store_name]
    store.set(
        ValidationResultIdentifier(ExpectationSuiteIdentifier(suite_name), run_id, batch_id),
        suite_result,
    )


@pytest.fixture
def two_runs(in_memory_runtime_context):
    context = in_memory_runtime_context
    r1 = RunIdentifier(run_name="run1", run_time=T0)
    r2 = RunIdentifier(run_name="run2", run_time=T0 + DAY)
    _save(context, r1, "batch_a", _suite_result(r1, [_evr(True, 100, 0, 0.0)]))
    _save(context, r2, "batch_a", _suite_result(r2, [_evr(False, 90, 10, 10.0)]))
    return context, r1, r2


def test_compare_validation_runs_end_to_end(two_runs):
    context, r1, r2 = two_runs

    report = context.compare_validation_runs(SUITE, r2, other_run_id=r1)

    assert report.old_run_id == r1
    assert report.new_run_id == r2
    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == "passed_to_failed"
    assert row.unexpected_count.delta == 10.0
    assert report.summary["flip_count"] == 1
    assert report.summary["failed_count"] == 1


def test_compare_validation_runs_orders_by_run_time(two_runs):
    # Args given newest-first; the report still labels earlier run as old.
    context, r1, r2 = two_runs

    report = context.compare_validation_runs(SUITE, r2, other_run_id=r1)
    assert report.old_run_id == r1
    assert report.new_run_id == r2


def test_compare_validation_runs_is_read_only(two_runs):
    context, r1, r2 = two_runs
    store = context.stores[context.validation_results_store_name]
    keys_before = sorted(key.to_fixed_length_tuple() for key in store.list_keys())
    result_before = store.get(
        ValidationResultIdentifier(ExpectationSuiteIdentifier(SUITE), r2, "batch_a")
    )

    context.compare_validation_runs(SUITE, r1, other_run_id=r2)

    keys_after = sorted(key.to_fixed_length_tuple() for key in store.list_keys())
    assert keys_before == keys_after
    result_after = store.get(
        ValidationResultIdentifier(ExpectationSuiteIdentifier(SUITE), r2, "batch_a")
    )
    assert result_after == result_before


def test_compare_validation_runs_missing_run_raises(two_runs):
    context, _r1, _r2 = two_runs
    missing = RunIdentifier(run_name="nope", run_time=T0 + 2 * DAY)

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(SUITE, missing, other_run_id=_r1)
    assert exc_info.type.__name__ == "ValidationRunNotFoundError"


def test_compare_validation_runs_cleaned_results_raises(in_memory_runtime_context):
    context = in_memory_runtime_context
    run_id = RunIdentifier(run_name="shared", run_time=T0)
    # The run exists, but only a result for a different suite was retained.
    _save(
        context,
        run_id,
        "batch_a",
        _suite_result(run_id, [_evr(True, 1, 0, 0.0)], suite_name=OTHER_SUITE),
        suite_name=OTHER_SUITE,
    )
    other = RunIdentifier(run_name="other", run_time=T0 + DAY)
    _save(context, other, "batch_a", _suite_result(other, [_evr(True, 1, 0, 0.0)]))

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(SUITE, other, other_run_id=run_id)
    assert exc_info.type.__name__ == "ValidationRunResultsCleanedError"


def test_compare_validation_runs_window_mode(two_runs):
    context, r1, r2 = two_runs

    report = context.compare_validation_runs(
        SUITE,
        r2,
        window_start=T0 - DAY,
        window_end=r2.run_time,
    )
    assert report.old_run_id == r1
    assert report.new_run_id == r2
    assert report.window == {
        "start": (T0 - DAY).isoformat(),
        "end": r2.run_time.isoformat(),
    }


def test_compare_validation_runs_empty_window(two_runs):
    context, _r1, r2 = two_runs

    report = context.compare_validation_runs(
        SUITE,
        r2,
        window_start=T0 + 10 * DAY,
        window_end=T0 + 11 * DAY,
    )
    assert report.batch_comparisons == []
    assert report.old_run_id is None


def test_compare_validation_runs_reversed_window_raises(two_runs):
    context, _r1, r2 = two_runs

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(
            SUITE,
            r2,
            window_start=T0 + 2 * DAY,
            window_end=T0,
        )
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_compare_validation_runs_rejects_exceeding_max_results(in_memory_runtime_context):
    context = in_memory_runtime_context
    r1 = RunIdentifier(run_name="run1", run_time=T0)
    r2 = RunIdentifier(run_name="run2", run_time=T0 + DAY)
    for batch_id in ("batch_a", "batch_b"):
        _save(context, r1, batch_id, _suite_result(r1, [_evr(True, 1, 0, 0.0)]))
        _save(context, r2, batch_id, _suite_result(r2, [_evr(True, 1, 0, 0.0)]))

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(SUITE, r1, other_run_id=r2, max_results=3)
    assert exc_info.type.__name__ == "InvalidConfigError"
    assert "max_results" in str(exc_info.value)


def test_compare_validation_runs_rejects_invalid_thresholds(two_runs):
    context, r1, r2 = two_runs

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(
            SUITE, r1, other_run_id=r2, thresholds={"unexpected_percent": -1}
        )
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_compare_validation_runs_same_run_rejected(two_runs):
    context, r1, _r2 = two_runs

    with pytest.raises(Exception) as exc_info:
        context.compare_validation_runs(SUITE, r1, other_run_id=r1)
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_compare_validation_runs_batch_parameters_mismatch(in_memory_runtime_context):
    context = in_memory_runtime_context
    r1 = RunIdentifier(run_name="run1", run_time=T0)
    r2 = RunIdentifier(run_name="run2", run_time=T0 + DAY)
    _save(
        context,
        r1,
        "batch_a",
        _suite_result(r1, [_evr(True, 1, 0, 0.0)], batch_parameters={"date": "2024-01-01"}),
    )
    _save(
        context,
        r2,
        "batch_a",
        _suite_result(r2, [_evr(False, 2, 1, 5.0)], batch_parameters={"date": "2024-02-01"}),
    )

    report = context.compare_validation_runs(SUITE, r1, other_run_id=r2)
    batch = report.batch_comparisons[0]
    assert batch.status == "not_comparable"
    assert batch.comparability_reason == "batch_parameters_mismatch"


def test_compare_validation_runs_accepts_thresholds_object(two_runs):
    context, r1, r2 = two_runs

    report = context.compare_validation_runs(
        SUITE,
        r1,
        other_run_id=r2,
        thresholds=ComparisonThresholds(
            observed_value=0, unexpected_count=20, unexpected_percent=20
        ),
    )
    # A pass -> fail flip is always surfaced regardless of numeric thresholds.
    assert report.batch_comparisons[0].expectation_comparisons[0].category == ("passed_to_failed")

from __future__ import annotations

import datetime as dt

import pytest

import great_expectations.exceptions as gx_exceptions
from great_expectations.core import (
    ComparisonThresholds,
    ExpectationComparison,
    ExpectationSuiteValidationResult,
    ExpectationValidationResult,
    FlipCategory,
    RunIdentifier,
    build_run_comparison,
)
from great_expectations.data_context.store import ValidationResultsStore
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


# -------------------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------------------
@pytest.fixture
def store() -> ValidationResultsStore:
    return ValidationResultsStore(
        store_backend={
            "module_name": "great_expectations.data_context.store",
            "class_name": "InMemoryStoreBackend",
        }
    )


def make_evr(
    expectation_type: str = "expect_column_to_exist",
    kwargs: dict | None = None,
    success: bool = True,
    result: dict | None = None,
) -> ExpectationValidationResult:
    return ExpectationValidationResult(
        success=success,
        expectation_config=ExpectationConfiguration(type=expectation_type, kwargs=kwargs or {}),
        result=result if result is not None else {},
    )


def numeric_evr(
    success: bool,
    observed: float,
    unexpected_count: int,
    unexpected_percent: float,
    expectation_type: str = "expect_table_row_count_to_equal",
    kwargs: dict | None = None,
) -> ExpectationValidationResult:
    return make_evr(
        expectation_type=expectation_type,
        kwargs=kwargs if kwargs is not None else {"value": 100},
        success=success,
        result={
            "observed_value": observed,
            "unexpected_count": unexpected_count,
            "unexpected_percent": unexpected_percent,
        },
    )


def stats_for(evrs: list[ExpectationValidationResult]) -> dict:
    total = len(evrs)
    successful = sum(1 for evr in evrs if evr.success)
    return {
        "evaluated_expectations": total,
        "successful_expectations": successful,
        "unsuccessful_expectations": total - successful,
        "success_percent": (successful / total * 100 if total else None),
    }


def save_run(
    store: ValidationResultsStore,
    run_id: RunIdentifier,
    batch_results: dict[str, ExpectationSuiteValidationResult],
) -> None:
    for batch_id, suite_result in batch_results.items():
        key = ValidationResultIdentifier(
            ExpectationSuiteIdentifier(suite_result.suite_name), run_id, batch_id
        )
        store.set(key, suite_result)


def suite_result(
    evrs: list[ExpectationValidationResult],
    run_id: RunIdentifier,
    suite_name: str = SUITE,
    statistics: dict | None = None,
    batch_parameters: dict | None = None,
) -> ExpectationSuiteValidationResult:
    meta: dict = {"run_id": run_id, "expectation_suite_name": suite_name}
    if batch_parameters is not None:
        meta["batch_parameters"] = batch_parameters
    return ExpectationSuiteValidationResult(
        success=all(evr.success for evr in evrs),
        results=evrs,
        suite_name=suite_name,
        statistics=stats_for(evrs) if statistics is None else statistics,
        meta=meta,
    )


def compare_pairwise(
    store: ValidationResultsStore,
    run_a: RunIdentifier,
    run_b: RunIdentifier,
    thresholds: ComparisonThresholds | None = None,
    max_results: int = 100,
):
    return build_run_comparison(
        store=store,
        expectation_suite_name=SUITE,
        anchor_run_id=run_a,
        other_run_id=run_b,
        window_start=None,
        window_end=None,
        time_window=DAY,
        max_results=max_results,
        thresholds=thresholds or ComparisonThresholds(),
    )


def make_run(name: str, time: dt.datetime) -> RunIdentifier:
    return RunIdentifier(run_name=name, run_time=time)


def rows_by_category(report) -> dict[str, ExpectationComparison]:
    rows = {}
    for batch in report.batch_comparisons:
        for row in batch.expectation_comparisons:
            rows.setdefault(row.category, row)
    return rows


# -------------------------------------------------------------------------------------------------
# Status flips
# -------------------------------------------------------------------------------------------------
def test_passed_to_failed_is_highest_priority(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 100, 0, 0.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 90, 10, 10.0)], r2)})

    report = compare_pairwise(store, r2, r1)

    assert report.old_run_id == r1
    assert report.new_run_id == r2
    batch = report.batch_comparisons[0]
    row = batch.expectation_comparisons[0]
    assert row.category == FlipCategory.PASSED_TO_FAILED
    assert row.old_success is True
    assert row.new_success is False
    assert row.success_changed is True
    assert row.flipped is True
    assert row.priority == 1
    assert row.comparable is True
    assert row.observed_value.delta == -10.0
    assert row.observed_value.changed is True
    assert row.unexpected_count.delta == 10.0
    assert row.unexpected_percent.delta == 10.0
    assert batch.summary["flip_count"] == 1
    assert batch.summary["failed_count"] == 1
    assert report.summary["passed_to_failed"] == 1
    assert report.summary["flip_count"] == 1


def test_failed_to_passed_is_lowest_priority_flip(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(False, 90, 10, 10.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 100, 0, 0.0)], r2)})

    report = compare_pairwise(store, r1, r2)

    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == FlipCategory.FAILED_TO_PASSED
    assert row.old_success is False
    assert row.new_success is True
    assert row.flipped is True
    assert row.priority == 5
    assert report.summary["failed_to_passed"] == 1


def test_pairwise_ordering_resolved_by_run_time(store):
    # Pass the older run as the anchor; the report must still label it as the old run.
    r_old = make_run("run1", T0)
    r_new = make_run("run2", T0 + DAY)
    save_run(store, r_old, {"batch_a": suite_result([numeric_evr(True, 100, 0, 0.0)], r_old)})
    save_run(store, r_new, {"batch_a": suite_result([numeric_evr(False, 90, 10, 10.0)], r_new)})

    report = compare_pairwise(store, r_old, r_new)

    assert report.old_run_id == r_old
    assert report.new_run_id == r_new


# -------------------------------------------------------------------------------------------------
# Still failing, numeric thresholds
# -------------------------------------------------------------------------------------------------
def test_still_failing_change_flagged_when_above_threshold(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(False, 100, 10, 10.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 100, 15, 15.0)], r2)})

    thresholds = ComparisonThresholds(unexpected_count=4, unexpected_percent=4)
    report = compare_pairwise(store, r1, r2, thresholds)

    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == FlipCategory.STILL_FAILING_CHANGED
    assert row.flipped is True
    assert row.priority == 2
    assert row.unexpected_count.delta == 5.0
    assert row.unexpected_percent.delta == 5.0


def test_still_failing_unchanged_when_within_threshold(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(False, 100, 10, 10.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 100, 15, 15.0)], r2)})

    # Strictly greater than: delta exactly equal to the threshold is not a flip.
    thresholds = ComparisonThresholds(unexpected_count=5, unexpected_percent=5)
    report = compare_pairwise(store, r1, r2, thresholds)

    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == FlipCategory.UNCHANGED
    assert row.flipped is False
    assert row.priority is None
    assert report.summary["flip_count"] == 0


def test_any_metric_beyond_threshold_triggers_change(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(False, 100, 10, 10.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 100, 15, 10.0)], r2)})

    thresholds = ComparisonThresholds(unexpected_count=4, unexpected_percent=5)
    report = compare_pairwise(store, r1, r2, thresholds)

    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == FlipCategory.STILL_FAILING_CHANGED


def test_unexpected_percent_compared_in_percentage_points(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(False, 100, 1, 1.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 100, 1, 2.0)], r2)})

    within = compare_pairwise(store, r1, r2, ComparisonThresholds(unexpected_percent=1))
    assert within.batch_comparisons[0].expectation_comparisons[0].category == FlipCategory.UNCHANGED

    beyond = compare_pairwise(store, r1, r2, ComparisonThresholds(unexpected_percent=0.5))
    assert (
        beyond.batch_comparisons[0].expectation_comparisons[0].category
        == FlipCategory.STILL_FAILING_CHANGED
    )


def test_non_numeric_observed_value_inequality_is_a_change(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    evr_old = make_evr(
        "expect_column_values_to_be_in_set",
        {"column": "c"},
        False,
        {"observed_value": "abc"},
    )
    evr_new = make_evr(
        "expect_column_values_to_be_in_set",
        {"column": "c"},
        False,
        {"observed_value": "xyz"},
    )
    save_run(store, r1, {"batch_a": suite_result([evr_old], r1)})
    save_run(store, r2, {"batch_a": suite_result([evr_new], r2)})

    report = compare_pairwise(store, r1, r2)

    metric = report.batch_comparisons[0].expectation_comparisons[0].observed_value
    assert metric.comparable is True
    assert metric.changed is True
    assert metric.delta is None
    assert (
        report.batch_comparisons[0].expectation_comparisons[0].category
        == FlipCategory.STILL_FAILING_CHANGED
    )


# -------------------------------------------------------------------------------------------------
# New / missing expectations and batches
# -------------------------------------------------------------------------------------------------
def test_new_and_missing_expectations_are_classified(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    exp_a_old = numeric_evr(True, 1, 0, 0.0, "expect_a", kwargs={})
    exp_b_old = numeric_evr(True, 2, 0, 0.0, "expect_b", kwargs={})
    exp_a_new = numeric_evr(True, 1, 0, 0.0, "expect_a", kwargs={})
    exp_c_new = numeric_evr(True, 3, 0, 0.0, "expect_c", kwargs={})
    save_run(store, r1, {"batch_a": suite_result([exp_a_old, exp_b_old], r1)})
    save_run(store, r2, {"batch_a": suite_result([exp_a_new, exp_c_new], r2)})

    report = compare_pairwise(store, r1, r2)
    categories = rows_by_category(report)
    assert FlipCategory.NEW_EXPECTATION in categories
    assert categories[FlipCategory.NEW_EXPECTATION].expectation_type == "expect_c"
    assert categories[FlipCategory.NEW_EXPECTATION].priority == 4
    assert FlipCategory.MISSING_EXPECTATION in categories
    assert categories[FlipCategory.MISSING_EXPECTATION].expectation_type == "expect_b"
    assert categories[FlipCategory.MISSING_EXPECTATION].priority == 3
    assert categories[FlipCategory.UNCHANGED].expectation_type == "expect_a"
    summary = report.summary
    assert summary["new_expectation"] == 1
    assert summary["missing_expectation"] == 1
    assert summary["flip_count"] == 2


def test_same_expectation_type_with_different_kwargs_are_distinct(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    exp_age = numeric_evr(
        True, 1, 0, 0.0, "expect_column_min_to_be_between", kwargs={"column": "age"}
    )
    exp_fare = numeric_evr(
        True, 1, 0, 0.0, "expect_column_min_to_be_between", kwargs={"column": "fare"}
    )
    save_run(store, r1, {"batch_a": suite_result([exp_age], r1)})
    save_run(store, r2, {"batch_a": suite_result([exp_age, exp_fare], r2)})

    report = compare_pairwise(store, r1, r2)
    row = rows_by_category(report)[FlipCategory.NEW_EXPECTATION]
    assert row.expectation_type == "expect_column_min_to_be_between"
    assert '"column"' in row.identity
    assert report.summary["new_expectation"] == 1


def test_new_and_missing_batches_do_not_compare_values(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1),
            "batch_c": suite_result([numeric_evr(True, 3, 0, 0.0)], r1),
        },
    )
    save_run(
        store,
        r2,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2),
            "batch_b": suite_result([numeric_evr(True, 2, 0, 0.0)], r2),
        },
    )

    report = compare_pairwise(store, r1, r2)
    batches = {batch.batch_identifier: batch for batch in report.batch_comparisons}
    assert list(batches.keys()) == ["batch_a", "batch_b", "batch_c"]
    assert batches["batch_a"].status == "aligned"
    assert batches["batch_b"].status == "new_batch"
    assert batches["batch_c"].status == "missing_batch"
    new_row = batches["batch_b"].expectation_comparisons[0]
    assert new_row.category == FlipCategory.NEW_EXPECTATION
    assert new_row.observed_value.new_value == 2.0
    assert new_row.observed_value.old_value is None
    missing_row = batches["batch_c"].expectation_comparisons[0]
    assert missing_row.category == FlipCategory.MISSING_EXPECTATION
    assert report.summary["new_batch_count"] == 1
    assert report.summary["missing_batch_count"] == 1


# -------------------------------------------------------------------------------------------------
# Non-comparable conditions
# -------------------------------------------------------------------------------------------------
def test_batch_parameters_mismatch_marks_entries_not_comparable(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_a": suite_result(
                [numeric_evr(True, 1, 0, 0.0)], r1, batch_parameters={"date": "2024-01-01"}
            )
        },
    )
    save_run(
        store,
        r2,
        {
            "batch_a": suite_result(
                [numeric_evr(False, 2, 1, 5.0)], r2, batch_parameters={"date": "2024-02-01"}
            )
        },
    )

    report = compare_pairwise(store, r1, r2)
    batch = report.batch_comparisons[0]
    assert batch.status == "not_comparable"
    assert batch.comparability_reason == "batch_parameters_mismatch"
    row = batch.expectation_comparisons[0]
    assert row.category == FlipCategory.NOT_COMPARABLE
    assert row.comparability_reason == "batch_parameters_mismatch"
    assert row.flipped is False
    # Raw outcomes remain visible, but no verdict is issued.
    assert row.old_success is True
    assert row.new_success is False


def test_equal_batch_parameters_align_normally(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    params = {"date": "2024-01-01"}
    save_run(
        store,
        r1,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1, batch_parameters=params)},
    )
    save_run(
        store,
        r2,
        {
            "batch_a": suite_result(
                [numeric_evr(True, 1, 0, 0.0)], r2, batch_parameters=dict(params)
            )
        },
    )

    report = compare_pairwise(store, r1, r2)
    assert report.batch_comparisons[0].status == "aligned"


def test_missing_suite_statistics_marks_batch_not_comparable(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1)})
    save_run(
        store,
        r2,
        {"batch_a": suite_result([numeric_evr(False, 2, 1, 5.0)], r2, statistics={})},
    )

    report = compare_pairwise(store, r1, r2)
    batch = report.batch_comparisons[0]
    assert batch.status == "not_comparable"
    assert batch.comparability_reason == "statistics_missing"
    assert batch.expectation_comparisons[0].category == FlipCategory.NOT_COMPARABLE


def test_missing_expectation_statistics_marks_row_not_comparable(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1)})
    empty = make_evr("expect_table_row_count_to_equal", {"value": 100}, False, result={})
    save_run(store, r2, {"batch_a": suite_result([empty], r2)})

    report = compare_pairwise(store, r1, r2)
    row = report.batch_comparisons[0].expectation_comparisons[0]
    assert row.category == FlipCategory.NOT_COMPARABLE
    assert row.comparability_reason == "missing_statistics"
    assert row.old_success is True
    assert row.new_success is False


def test_unidentifiable_expectation_is_not_silently_dropped(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    unnamed = ExpectationValidationResult(
        success=False, expectation_config=None, result={"observed_value": 1}
    )
    save_run(store, r1, {"batch_a": suite_result([unnamed], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2)})

    report = compare_pairwise(store, r1, r2)
    rows = report.batch_comparisons[0].expectation_comparisons
    reasons = [row.comparability_reason for row in rows]
    assert "missing_expectation_configuration" in reasons


# -------------------------------------------------------------------------------------------------
# Record errors
# -------------------------------------------------------------------------------------------------
def test_deserialization_failure_is_listed_with_reason(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1),
            "batch_bad": suite_result([numeric_evr(True, 2, 0, 0.0)], r1),
        },
    )
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2)})

    bad_key = ValidationResultIdentifier(ExpectationSuiteIdentifier(SUITE), r1, "batch_bad")
    store.store_backend.set(store.key_to_tuple(bad_key), b"not valid json")

    report = compare_pairwise(store, r1, r2)

    assert [batch.batch_identifier for batch in report.batch_comparisons] == ["batch_a"]
    assert len(report.record_errors) == 1
    error = report.record_errors[0]
    assert error.error_type == "deserialization_error"
    assert error.batch_identifier == "batch_bad"
    assert error.reason
    flat = report.to_flattened_dict()
    assert flat["error_rows"][0]["error_type"] == "deserialization_error"


def test_result_not_found_is_listed(store, monkeypatch):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1),
            "batch_gone": suite_result([numeric_evr(True, 2, 0, 0.0)], r1),
        },
    )
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2)})

    gone_key = ValidationResultIdentifier(ExpectationSuiteIdentifier(SUITE), r1, "batch_gone")
    original_get = store.get

    def patched_get(key):
        if key == gone_key:
            return None
        return original_get(key)

    monkeypatch.setattr(store, "get", patched_get)

    report = compare_pairwise(store, r1, r2)
    error = report.record_errors[0]
    assert error.error_type == "result_not_found"
    assert "cleaned" in error.reason


# -------------------------------------------------------------------------------------------------
# Missing run vs cleaned results
# -------------------------------------------------------------------------------------------------
def test_unknown_run_raises_run_not_found(store):
    r1 = make_run("run1", T0)
    r_missing = make_run("missing", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2)})

    with pytest.raises(Exception) as exc_info:
        compare_pairwise(store, r_missing, r1)
    assert exc_info.type.__name__ == "ValidationRunNotFoundError"
    assert "not found" in str(exc_info.value)


def test_run_exists_but_suite_results_cleaned_raises_cleaned_error(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    # The run exists, but only for a different suite.
    other = suite_result([numeric_evr(True, 1, 0, 0.0)], r1, suite_name=OTHER_SUITE)
    save_run(store, r1, {"batch_a": other})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2)})

    with pytest.raises(Exception) as exc_info:
        compare_pairwise(store, r1, r2)
    assert exc_info.type.__name__ == "ValidationRunResultsCleanedError"
    assert "cleaned" in str(exc_info.value)


def test_pairwise_other_run_missing_raises_not_found(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1)})

    with pytest.raises(Exception) as exc_info:
        compare_pairwise(store, r1, r2)
    assert exc_info.type.__name__ == "ValidationRunNotFoundError"


# -------------------------------------------------------------------------------------------------
# Threshold validation
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"observed_value": -1},
        {"observed_value": float("nan")},
        {"observed_value": float("inf")},
        {"observed_value": "x"},
        {"observed_value": True},
        {"unexpected_count": -0.5},
        {"unexpected_percent": -1},
    ],
)
def test_invalid_thresholds_raise(kwargs):
    with pytest.raises(Exception) as exc_info:
        ComparisonThresholds(**kwargs)
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_thresholds_from_dict_rejects_unknown_keys():
    with pytest.raises(Exception) as exc_info:
        ComparisonThresholds.from_dict({"bogus": 1})
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_thresholds_from_dict_rejects_non_mapping():
    with pytest.raises(Exception) as exc_info:
        ComparisonThresholds.from_dict("nope")
    assert exc_info.type.__name__ == "InvalidConfigError"


# -------------------------------------------------------------------------------------------------
# Time window
# -------------------------------------------------------------------------------------------------
def compare_window(store, anchor, start, end, **kwargs):
    return build_run_comparison(
        store=store,
        expectation_suite_name=SUITE,
        anchor_run_id=anchor,
        other_run_id=None,
        window_start=start,
        window_end=end,
        time_window=kwargs.pop("time_window", DAY),
        max_results=kwargs.pop("max_results", 100),
        thresholds=ComparisonThresholds(),
    )


def test_reversed_window_raises(store):
    r_anchor = make_run("anchor", T0 + 5 * DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_anchor)},
    )

    with pytest.raises(Exception) as exc_info:
        compare_window(store, r_anchor, T0 + 2 * DAY, T0 + DAY)
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_empty_window_returns_empty_report(store):
    r_anchor = make_run("anchor", T0 + 5 * DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_anchor)},
    )

    report = compare_window(store, r_anchor, T0 + 6 * DAY, T0 + 7 * DAY)

    assert report.batch_comparisons == []
    assert report.record_errors == []
    assert report.old_run_id is None
    assert report.new_run_id == r_anchor
    assert report.window == {
        "start": (T0 + 6 * DAY).isoformat(),
        "end": (T0 + 7 * DAY).isoformat(),
    }
    assert report.summary["flip_count"] == 0


def test_zero_length_window_is_empty_report(store):
    r_anchor = make_run("anchor", T0 + DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_anchor)},
    )

    report = compare_window(store, r_anchor, T0, T0)
    assert report.batch_comparisons == []


def test_window_is_half_open(store):
    r_at_start = make_run("at_start", T0)
    r_at_end = make_run("at_end", T0 + 2 * DAY)
    r_anchor = make_run("anchor", T0 + 3 * DAY)
    save_run(
        store,
        r_at_start,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_at_start)},
    )
    save_run(
        store,
        r_at_end,
        {"batch_a": suite_result([numeric_evr(True, 2, 0, 0.0)], r_at_end)},
    )
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 3, 0, 0.0)], r_anchor)},
    )

    report = compare_window(store, r_anchor, T0, T0 + 2 * DAY)

    # Only the run at the inclusive start qualifies; the run at the exclusive end does not.
    assert report.old_run_id == r_at_start


def test_window_picks_latest_candidate(store):
    runs = []
    for day in range(1, 4):
        run_id = make_run(f"run{day}", T0 + day * DAY)
        save_run(
            store,
            run_id,
            {"batch_a": suite_result([numeric_evr(True, day, 0, 0.0)], run_id)},
        )
        runs.append(run_id)
    r_anchor = make_run("anchor", T0 + 4 * DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 4, 0, 0.0)], r_anchor)},
    )

    report = compare_window(store, r_anchor, T0, T0 + 4 * DAY)

    assert report.old_run_id == runs[-1]


def test_window_bounds_accept_iso_strings(store):
    r_old = make_run("old", T0 + DAY)
    r_anchor = make_run("anchor", T0 + 3 * DAY)
    save_run(
        store,
        r_old,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_old)},
    )
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 3, 0, 0.0)], r_anchor)},
    )

    report = compare_window(
        store,
        r_anchor,
        (T0).isoformat(),
        (T0 + 3 * DAY).isoformat(),
    )
    assert report.old_run_id == r_old


def test_window_invalid_datetime_string_raises(store):
    r_anchor = make_run("anchor", T0 + DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_anchor)},
    )

    with pytest.raises(Exception) as exc_info:
        compare_window(store, r_anchor, "not-a-date", None)
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_default_window_used_when_bounds_omitted(store):
    r_in_window = make_run("in_window", T0 + DAY)
    r_outside = make_run("outside", T0)
    r_anchor = make_run("anchor", T0 + 8 * DAY)
    save_run(
        store,
        r_in_window,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_in_window)},
    )
    save_run(
        store,
        r_outside,
        {"batch_a": suite_result([numeric_evr(True, 0, 0, 0.0)], r_outside)},
    )
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 8, 0, 0.0)], r_anchor)},
    )

    # Default 7-day look-back window is [anchor - 7d, anchor).
    report = build_run_comparison(
        store=store,
        expectation_suite_name=SUITE,
        anchor_run_id=r_anchor,
        other_run_id=None,
        window_start=None,
        window_end=None,
        time_window=7 * DAY,
        max_results=100,
        thresholds=ComparisonThresholds(),
    )
    assert report.old_run_id == r_in_window


def test_negative_time_window_rejected(store):
    r_anchor = make_run("anchor", T0 + DAY)
    save_run(
        store,
        r_anchor,
        {"batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r_anchor)},
    )

    with pytest.raises(Exception) as exc_info:
        build_run_comparison(
            store=store,
            expectation_suite_name=SUITE,
            anchor_run_id=r_anchor,
            other_run_id=None,
            window_start=None,
            window_end=None,
            time_window=dt.timedelta(days=-1),
            max_results=100,
            thresholds=ComparisonThresholds(),
        )
    assert exc_info.type.__name__ == "InvalidConfigError"


# -------------------------------------------------------------------------------------------------
# Result count limit
# -------------------------------------------------------------------------------------------------
def test_exceeding_max_results_is_rejected(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1),
            "batch_b": suite_result([numeric_evr(True, 2, 0, 0.0)], r1),
            "batch_c": suite_result([numeric_evr(True, 3, 0, 0.0)], r1),
        },
    )
    save_run(
        store,
        r2,
        {
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r2),
            "batch_b": suite_result([numeric_evr(True, 2, 0, 0.0)], r2),
            "batch_c": suite_result([numeric_evr(True, 3, 0, 0.0)], r2),
        },
    )

    with pytest.raises(Exception) as exc_info:
        compare_pairwise(store, r1, r2, max_results=5)
    assert exc_info.type.__name__ == "InvalidConfigError"
    assert "max_results" in str(exc_info.value)

    # At the exact limit the comparison succeeds (no silent truncation either way).
    report = compare_pairwise(store, r1, r2, max_results=6)
    assert len(report.batch_comparisons) == 3


def test_invalid_max_results_rejected():
    from great_expectations.core.run_comparison import coerce_max_results

    with pytest.raises(gx_exceptions.InvalidConfigError):
        coerce_max_results(0)
    with pytest.raises(gx_exceptions.InvalidConfigError):
        coerce_max_results(True)
    with pytest.raises(gx_exceptions.InvalidConfigError):
        coerce_max_results("10")


# -------------------------------------------------------------------------------------------------
# Same-run, determinism, sorting, flattened export
# -------------------------------------------------------------------------------------------------
def test_comparing_run_against_itself_rejected():
    r1 = make_run("run1", T0)
    from great_expectations.core.run_comparison import order_runs

    with pytest.raises(Exception) as exc_info:
        order_runs(r1, r1)
    assert exc_info.type.__name__ == "InvalidConfigError"


def test_report_is_deterministic(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(
        store,
        r1,
        {
            "batch_b": suite_result([numeric_evr(True, 2, 0, 0.0)], r1),
            "batch_a": suite_result([numeric_evr(True, 1, 0, 0.0)], r1),
        },
    )
    save_run(
        store,
        r2,
        {
            "batch_b": suite_result([numeric_evr(False, 5, 3, 30.0)], r2),
            "batch_a": suite_result([numeric_evr(False, 0, 1, 10.0)], r2),
        },
    )

    first = compare_pairwise(store, r1, r2).to_json_dict()
    second = compare_pairwise(store, r1, r2).to_json_dict()
    assert first == second

    batch_ids = [batch["batch_identifier"] for batch in first["batch_comparisons"]]
    assert batch_ids == sorted(batch_ids)


def test_expectations_sorted_by_type_then_identity(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    evrs_old = [
        numeric_evr(True, 1, 0, 0.0, "expect_table_row_count_to_equal", kwargs={}),
        numeric_evr(True, 1, 0, 0.0, "expect_column_min_to_be_between", kwargs={"column": "b"}),
        numeric_evr(True, 1, 0, 0.0, "expect_column_min_to_be_between", kwargs={"column": "a"}),
    ]
    save_run(store, r1, {"batch_a": suite_result(evrs_old, r1)})
    save_run(store, r2, {"batch_a": suite_result(list(evrs_old), r2)})

    report = compare_pairwise(store, r1, r2)
    keys = [
        (row.expectation_type, row.identity)
        for row in report.batch_comparisons[0].expectation_comparisons
    ]
    assert keys == sorted(keys)
    assert keys[0][0] == "expect_column_min_to_be_between"
    assert '"a"' in keys[0][1]


def test_flattened_export_shape(store):
    r1 = make_run("run1", T0)
    r2 = make_run("run2", T0 + DAY)
    save_run(store, r1, {"batch_a": suite_result([numeric_evr(True, 100, 0, 0.0)], r1)})
    save_run(store, r2, {"batch_a": suite_result([numeric_evr(False, 90, 10, 10.0)], r2)})

    report = compare_pairwise(store, r1, r2)
    flat = report.to_flattened_dict()

    assert set(flat.keys()) == {
        "expectation_suite_name",
        "old_run_id",
        "new_run_id",
        "window",
        "summary",
        "expectation_rows",
        "error_rows",
    }
    row = flat["expectation_rows"][0]
    expected_columns = {
        "expectation_suite_name",
        "batch_identifier",
        "batch_status",
        "expectation_type",
        "expectation_identity",
        "old_success",
        "new_success",
        "success_changed",
        "category",
        "flipped",
        "priority",
        "observed_value_old",
        "observed_value_new",
        "observed_value_delta",
        "observed_value_changed",
        "observed_value_comparable",
        "unexpected_count_old",
        "unexpected_count_new",
        "unexpected_count_delta",
        "unexpected_count_changed",
        "unexpected_count_comparable",
        "unexpected_percent_old",
        "unexpected_percent_new",
        "unexpected_percent_delta",
        "unexpected_percent_changed",
        "unexpected_percent_comparable",
        "comparable",
        "comparability_reason",
    }
    assert set(row.keys()) == expected_columns
    assert row["category"] == "passed_to_failed"
    assert row["old_success"] is True
    assert row["new_success"] is False
    assert row["unexpected_count_new"] == 10.0
    assert flat["error_rows"] == []

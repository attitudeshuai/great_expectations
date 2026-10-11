from __future__ import annotations

import datetime
import json
import logging
import math
import numbers
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Optional, Union

from dateutil.parser import parse as dateutil_parse
from marshmallow import ValidationError as MarshmallowValidationError

import great_expectations.exceptions as gx_exceptions
from great_expectations._docs_decorators import public_api
from great_expectations.compatibility.pydantic import ValidationError as PydanticValidationError
from great_expectations.compatibility.typing_extensions import override
from great_expectations.core.run_identifier import RunIdentifier
from great_expectations.data_context.types.resource_identifiers import (
    ValidationResultIdentifier,
)
from great_expectations.types import SerializableDictDot
from great_expectations.util import (
    convert_to_json_serializable,  # noqa: TID251 # FIXME CoP
)

if TYPE_CHECKING:
    from great_expectations.core.expectation_validation_result import (
        ExpectationSuiteValidationResult,
        ExpectationValidationResult,
    )
    from great_expectations.data_context.store.validation_results_store import (
        ValidationResultsStore,
    )

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------
DEFAULT_TIME_WINDOW = datetime.timedelta(days=7)
DEFAULT_MAX_RESULTS = 100
DEFAULT_OBSERVED_VALUE_THRESHOLD = 0.0
DEFAULT_UNEXPECTED_COUNT_THRESHOLD = 0.0
DEFAULT_UNEXPECTED_PERCENT_THRESHOLD = 0.0  # percentage points, on the 0-100 scale


class FlipCategory:
    """Categories describing how a single expectation's outcome changed between two runs."""

    PASSED_TO_FAILED = "passed_to_failed"
    FAILED_TO_PASSED = "failed_to_passed"
    STILL_FAILING_CHANGED = "still_failing_changed"
    NEW_EXPECTATION = "new_expectation"
    MISSING_EXPECTATION = "missing_expectation"
    UNCHANGED = "unchanged"
    NOT_COMPARABLE = "not_comparable"


FLIP_CATEGORIES = frozenset(
    {
        FlipCategory.PASSED_TO_FAILED,
        FlipCategory.FAILED_TO_PASSED,
        FlipCategory.STILL_FAILING_CHANGED,
        FlipCategory.NEW_EXPECTATION,
        FlipCategory.MISSING_EXPECTATION,
    }
)

# Lower numbers indicate flips that deserve attention first.
FLIP_PRIORITY: dict[str, int] = {
    FlipCategory.PASSED_TO_FAILED: 1,
    FlipCategory.STILL_FAILING_CHANGED: 2,
    FlipCategory.MISSING_EXPECTATION: 3,
    FlipCategory.NEW_EXPECTATION: 4,
    FlipCategory.FAILED_TO_PASSED: 5,
}


class BatchStatus:
    """Whether a batch was present on both sides or only one side."""

    ALIGNED = "aligned"
    NEW_BATCH = "new_batch"
    MISSING_BATCH = "missing_batch"
    NOT_COMPARABLE = "not_comparable"


class RecordErrorType:
    RESULT_NOT_FOUND = "result_not_found"
    DESERIALIZATION_ERROR = "deserialization_error"
    STORE_BACKEND_ERROR = "store_backend_error"
    UNEXPECTED_ERROR = "unexpected_error"


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------
def _validate_threshold(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"Threshold '{field_name}' must be a non-negative number, not {value!r}."
        )
    if not math.isfinite(float(value)) or value < 0:
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"Threshold '{field_name}' must be a finite, non-negative number, not {value!r}."
        )
    return float(value)


@public_api
class ComparisonThresholds(SerializableDictDot):
    """Thresholds governing when a numeric change between two runs counts as a flip.

    A comparable item is flagged as changed only when the absolute difference between the
    new and old value strictly exceeds the corresponding threshold.

    Args:
        observed_value: Maximum absolute difference in an expectation's observed_value.
        unexpected_count: Maximum absolute difference in the number of unexpected rows.
        unexpected_percent: Maximum absolute difference in unexpected_percent, expressed in
            percentage points on the 0-100 scale.
    """

    def __init__(
        self,
        observed_value: float = DEFAULT_OBSERVED_VALUE_THRESHOLD,
        unexpected_count: float = DEFAULT_UNEXPECTED_COUNT_THRESHOLD,
        unexpected_percent: float = DEFAULT_UNEXPECTED_PERCENT_THRESHOLD,
    ) -> None:
        self.observed_value = _validate_threshold(observed_value, "observed_value")
        self.unexpected_count = _validate_threshold(unexpected_count, "unexpected_count")
        self.unexpected_percent = _validate_threshold(unexpected_percent, "unexpected_percent")

    @classmethod
    def from_dict(cls, thresholds: Optional[ComparisonThresholds | dict]) -> ComparisonThresholds:
        if thresholds is None:
            return cls()
        if isinstance(thresholds, ComparisonThresholds):
            return thresholds
        if not isinstance(thresholds, dict):
            raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
                "thresholds must be a dict or ComparisonThresholds, not "
                f"{type(thresholds).__name__}."
            )
        allowed_keys = {"observed_value", "unexpected_count", "unexpected_percent"}
        unknown_keys = set(thresholds.keys()) - allowed_keys
        if unknown_keys:
            raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
                f"Unknown threshold keys: {sorted(unknown_keys)}. "
                f"Allowed keys are: {sorted(allowed_keys)}."
            )
        return cls(**thresholds)

    @public_api
    @override
    def to_json_dict(self) -> dict:
        """Returns a JSON-serializable dict of the configured comparison thresholds."""
        return {
            "observed_value": self.observed_value,
            "unexpected_count": self.unexpected_count,
            "unexpected_percent": self.unexpected_percent,
        }


# -------------------------------------------------------------------------------------------------
# Comparison result types
# -------------------------------------------------------------------------------------------------
class MetricComparison:
    """The old/new values for a single comparable metric."""

    def __init__(  # noqa: PLR0913 # FIXME CoP
        self,
        name: str,
        old_value: Any,
        new_value: Any,
        delta: Optional[float],
        changed: Optional[bool],
        comparable: bool,
        reason: Optional[str] = None,
    ) -> None:
        self.name = name
        self.old_value = old_value
        self.new_value = new_value
        self.delta = delta
        self.changed = changed
        self.comparable = comparable
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "changed": self.changed,
            "comparable": self.comparable,
            "reason": self.reason,
        }


@public_api
class ExpectationComparison(SerializableDictDot):
    """The comparison of a single expectation within a batch between two runs."""

    def __init__(  # noqa: PLR0913 # FIXME CoP
        self,
        expectation_type: Optional[str],
        expectation_kwargs: dict,
        identity: str,
        old_success: Optional[bool],
        new_success: Optional[bool],
        success_changed: Optional[bool],
        category: str,
        flipped: bool,
        priority: Optional[int],
        observed_value: MetricComparison,
        unexpected_count: MetricComparison,
        unexpected_percent: MetricComparison,
        comparable: bool,
        comparability_reason: Optional[str] = None,
    ) -> None:
        self.expectation_type = expectation_type
        self.expectation_kwargs = expectation_kwargs
        self.identity = identity
        self.old_success = old_success
        self.new_success = new_success
        self.success_changed = success_changed
        self.category = category
        self.flipped = flipped
        self.priority = priority
        self.observed_value = observed_value
        self.unexpected_count = unexpected_count
        self.unexpected_percent = unexpected_percent
        self.comparable = comparable
        self.comparability_reason = comparability_reason

    @public_api
    @override
    def to_json_dict(self) -> dict:
        """Returns a JSON-serializable dict of this expectation-level comparison."""
        return {
            "expectation_type": self.expectation_type,
            "expectation_kwargs": self.expectation_kwargs,
            "identity": self.identity,
            "old_success": self.old_success,
            "new_success": self.new_success,
            "success_changed": self.success_changed,
            "category": self.category,
            "flipped": self.flipped,
            "priority": self.priority,
            "observed_value": self.observed_value.to_dict(),
            "unexpected_count": self.unexpected_count.to_dict(),
            "unexpected_percent": self.unexpected_percent.to_dict(),
            "comparable": self.comparable,
            "comparability_reason": self.comparability_reason,
        }


@public_api
class BatchComparison(SerializableDictDot):
    """The comparison of every expectation within a single batch."""

    def __init__(
        self,
        batch_identifier: str,
        status: str,
        expectation_comparisons: list[ExpectationComparison],
        comparability_reason: Optional[str] = None,
    ) -> None:
        self.batch_identifier = batch_identifier
        self.status = status
        self.comparability_reason = comparability_reason
        self.expectation_comparisons = expectation_comparisons
        self.summary: dict[str, int] = _summarize_expectations(expectation_comparisons)

    @public_api
    @override
    def to_json_dict(self) -> dict:
        """Returns a JSON-serializable dict of this batch-level comparison."""
        return {
            "batch_identifier": self.batch_identifier,
            "status": self.status,
            "comparability_reason": self.comparability_reason,
            "summary": self.summary,
            "expectation_comparisons": [r.to_json_dict() for r in self.expectation_comparisons],
        }


class RecordError:
    """A result record that could not be loaded for the comparison."""

    def __init__(
        self,
        key: ValidationResultIdentifier,
        error_type: str,
        reason: str,
    ) -> None:
        self.run_id: RunIdentifier = key.run_id
        self.batch_identifier = key.batch_identifier
        self.error_type = error_type
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id.to_json_dict(),
            "batch_identifier": self.batch_identifier,
            "error_type": self.error_type,
            "reason": self.reason,
        }


_EMPTY_COUNTS = {
    FlipCategory.PASSED_TO_FAILED: 0,
    FlipCategory.FAILED_TO_PASSED: 0,
    FlipCategory.STILL_FAILING_CHANGED: 0,
    FlipCategory.NEW_EXPECTATION: 0,
    FlipCategory.MISSING_EXPECTATION: 0,
    FlipCategory.UNCHANGED: 0,
    FlipCategory.NOT_COMPARABLE: 0,
}


def _summarize_expectations(rows: list[ExpectationComparison]) -> dict[str, int]:
    counts = dict(_EMPTY_COUNTS)
    for row in rows:
        counts[row.category] += 1
    return {
        "flip_count": sum(1 for row in rows if row.flipped),
        "failed_count": sum(1 for row in rows if row.new_success is False),
        **counts,
    }


@public_api
class RunComparisonReport(SerializableDictDot):
    """A read-only, per-batch/per-expectation comparison of validation results from two runs."""

    def __init__(  # noqa: PLR0913 # FIXME CoP
        self,
        expectation_suite_name: str,
        old_run_id: Optional[RunIdentifier],
        new_run_id: RunIdentifier,
        batch_comparisons: list[BatchComparison],
        record_errors: list[RecordError],
        window: Optional[dict[str, str]] = None,
    ) -> None:
        self.expectation_suite_name = expectation_suite_name
        self.old_run_id = old_run_id
        self.new_run_id = new_run_id
        self.window = window
        self.batch_comparisons = batch_comparisons
        self.record_errors = record_errors

    @property
    def summary(self) -> dict[str, int]:
        counts = dict(_EMPTY_COUNTS)
        for batch in self.batch_comparisons:
            for category, value in batch.summary.items():
                if category in counts:
                    counts[category] += value
        return {
            "batch_count": len(self.batch_comparisons),
            "new_batch_count": sum(
                1 for b in self.batch_comparisons if b.status == BatchStatus.NEW_BATCH
            ),
            "missing_batch_count": sum(
                1 for b in self.batch_comparisons if b.status == BatchStatus.MISSING_BATCH
            ),
            "flip_count": sum(batch.summary["flip_count"] for batch in self.batch_comparisons),
            "failed_count": sum(batch.summary["failed_count"] for batch in self.batch_comparisons),
            "record_error_count": len(self.record_errors),
            **counts,
        }

    @public_api
    @override
    def to_json_dict(self) -> dict:
        """Returns a JSON-serializable dict representation of the comparison report."""
        return {
            "expectation_suite_name": self.expectation_suite_name,
            "old_run_id": self.old_run_id.to_json_dict() if self.old_run_id else None,
            "new_run_id": self.new_run_id.to_json_dict(),
            "window": self.window,
            "summary": self.summary,
            "batch_comparisons": [b.to_json_dict() for b in self.batch_comparisons],
            "record_errors": [e.to_dict() for e in self.record_errors],
        }

    @public_api
    def to_flattened_dict(self) -> dict:
        """Export the report as flat, machine-readable rows.

        Returns:
            A dict with one flat row per expectation comparison (``expectation_rows``) and one
            flat row per unloadable result record (``error_rows``), along with the report-level
            metadata and summary.
        """
        expectation_rows: list[dict] = []
        for batch in self.batch_comparisons:
            for row in batch.expectation_comparisons:
                expectation_rows.append(
                    {
                        "expectation_suite_name": self.expectation_suite_name,
                        "batch_identifier": batch.batch_identifier,
                        "batch_status": batch.status,
                        "expectation_type": row.expectation_type,
                        "expectation_identity": row.identity,
                        "old_success": row.old_success,
                        "new_success": row.new_success,
                        "success_changed": row.success_changed,
                        "category": row.category,
                        "flipped": row.flipped,
                        "priority": row.priority,
                        "observed_value_old": row.observed_value.old_value,
                        "observed_value_new": row.observed_value.new_value,
                        "observed_value_delta": row.observed_value.delta,
                        "observed_value_changed": row.observed_value.changed,
                        "observed_value_comparable": row.observed_value.comparable,
                        "unexpected_count_old": row.unexpected_count.old_value,
                        "unexpected_count_new": row.unexpected_count.new_value,
                        "unexpected_count_delta": row.unexpected_count.delta,
                        "unexpected_count_changed": row.unexpected_count.changed,
                        "unexpected_count_comparable": row.unexpected_count.comparable,
                        "unexpected_percent_old": row.unexpected_percent.old_value,
                        "unexpected_percent_new": row.unexpected_percent.new_value,
                        "unexpected_percent_delta": row.unexpected_percent.delta,
                        "unexpected_percent_changed": row.unexpected_percent.changed,
                        "unexpected_percent_comparable": row.unexpected_percent.comparable,
                        "comparable": row.comparable,
                        "comparability_reason": row.comparability_reason,
                    }
                )

        error_rows: list[dict] = []
        for error in self.record_errors:
            error_dict = error.to_dict()
            run_id_dict = error_dict["run_id"]
            assert isinstance(run_id_dict, dict)
            error_rows.append(
                {
                    "expectation_suite_name": self.expectation_suite_name,
                    "run_name": run_id_dict["run_name"],
                    "run_time": run_id_dict["run_time"],
                    "batch_identifier": error_dict["batch_identifier"],
                    "error_type": error_dict["error_type"],
                    "reason": error_dict["reason"],
                }
            )

        return {
            "expectation_suite_name": self.expectation_suite_name,
            "old_run_id": self.old_run_id.to_json_dict() if self.old_run_id else None,
            "new_run_id": self.new_run_id.to_json_dict(),
            "window": self.window,
            "summary": self.summary,
            "expectation_rows": expectation_rows,
            "error_rows": error_rows,
        }


# -------------------------------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------------------------------
def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        return convert_to_json_serializable(value)
    except TypeError:
        return str(value)


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        return float(value)
    return None


def _canonical(value: Any) -> Any:
    """Return a deterministic, hashable, content-based representation of expectation kwargs."""
    if isinstance(value, dict):
        return (
            "d",
            tuple(sorted((str(k), _canonical(v)) for k, v in value.items())),
        )
    if isinstance(value, set):
        try:
            items = sorted(value)
        except TypeError:
            items = sorted(
                value,
                key=lambda x: json.dumps(
                    _json_value(x), default=str, sort_keys=True, ensure_ascii=False
                ),
            )
        return ("set", tuple(_canonical(item) for item in items))
    if isinstance(value, (list, tuple)):
        return ("l", tuple(_canonical(item) for item in value))
    converted = _json_value(value)
    if converted is not value:
        return _canonical(converted)
    if converted is None or isinstance(converted, (str, bool, int, float)):
        return ("s", converted)
    return ("s", json.dumps(converted, default=str, sort_keys=True, ensure_ascii=False))


def _identity_string(identity: Any) -> str:
    return json.dumps(identity, default=str, sort_keys=True, ensure_ascii=False)


def _config_parts(
    config: Any,
) -> tuple[Optional[str], Optional[dict]]:
    """Return (expectation_type, kwargs) from an expectation_config object or dict."""
    if config is None:
        return None, None
    if isinstance(config, dict):
        kwargs = config.get("kwargs")
        return config.get("type"), kwargs if isinstance(kwargs, dict) else {}
    return config.type, config.kwargs


def _expectation_identity(evr: ExpectationValidationResult) -> Optional[tuple]:
    expectation_type, kwargs = _config_parts(evr.expectation_config)
    if expectation_type is None:
        return None
    return (expectation_type, _canonical(kwargs))


def _coerce_datetime(value: Any, field_name: str) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = dateutil_parse(value)
        except (ValueError, TypeError) as parse_error:
            raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
                f"{field_name} is not a valid datetime: {value!r} ({parse_error})."
            )
    else:
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"{field_name} must be a datetime.datetime or ISO string, not {type(value).__name__}."
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _isoformat(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat()


def coerce_run_id(value: Any, field_name: str) -> RunIdentifier:
    if isinstance(value, RunIdentifier):
        return value
    if isinstance(value, dict):
        try:
            return RunIdentifier(**value)
        except TypeError as type_error:
            raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
                f"{field_name} is not a valid run identifier: {type_error}."
            )
    if value is None:
        raise gx_exceptions.InvalidConfigError(f"{field_name} is required.")  # noqa: TRY003 # FIXME CoP
    return RunIdentifier(run_name=str(value))


def coerce_time_window(value: Any) -> datetime.timedelta:
    if value is None:
        return DEFAULT_TIME_WINDOW
    if not isinstance(value, datetime.timedelta):
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"time_window must be a datetime.timedelta, not {type(value).__name__}."
        )
    if value < datetime.timedelta(0):
        raise gx_exceptions.InvalidConfigError("time_window must not be negative.")  # noqa: TRY003 # FIXME CoP
    return value


def coerce_max_results(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_RESULTS
    if isinstance(value, bool) or not isinstance(value, int):
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"max_results must be a positive integer, not {value!r}."
        )
    if value < 1:
        raise gx_exceptions.InvalidConfigError("max_results must be at least 1.")  # noqa: TRY003 # FIXME CoP
    return value


def order_runs(
    run_a: RunIdentifier,
    run_b: RunIdentifier,
) -> tuple[RunIdentifier, RunIdentifier]:
    """Return the two runs ordered as (old_run, new_run), earlier run_time first."""
    if run_a == run_b:
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            "Cannot compare a validation run against itself; supply two distinct run identifiers."
        )
    key_a = (run_a.run_time, run_a.run_name or "")
    key_b = (run_b.run_time, run_b.run_name or "")
    if key_a <= key_b:
        return run_a, run_b
    return run_b, run_a


# -------------------------------------------------------------------------------------------------
# Record loading
# -------------------------------------------------------------------------------------------------
def _load_record(
    store: ValidationResultsStore,
    key: ValidationResultIdentifier,
) -> tuple[Optional[ExpectationSuiteValidationResult], Optional[RecordError]]:
    try:
        record = store.get(key)
    except (MarshmallowValidationError, JSONDecodeError, PydanticValidationError) as e:
        return None, RecordError(key, RecordErrorType.DESERIALIZATION_ERROR, str(e))
    except gx_exceptions.StoreBackendError as e:
        return None, RecordError(key, RecordErrorType.STORE_BACKEND_ERROR, str(e))
    except Exception as e:  # defensive: one bad record must not break comparison
        return None, RecordError(key, RecordErrorType.UNEXPECTED_ERROR, f"{type(e).__name__}: {e}")

    if record is None:
        return None, RecordError(
            key,
            RecordErrorType.RESULT_NOT_FOUND,
            "The result key exists but the stored validation result could not be found; "
            "it may have been cleaned.",
        )
    return record, None


def _load_run_records(
    store: ValidationResultsStore,
    keys: list[ValidationResultIdentifier],
) -> tuple[dict[str, ExpectationSuiteValidationResult], list[RecordError]]:
    records: dict[str, ExpectationSuiteValidationResult] = {}
    errors: list[RecordError] = []
    for key in keys:
        record, error = _load_record(store, key)
        if error is not None:
            errors.append(error)
        if record is not None:
            records[str(key.batch_identifier)] = record
    return records, errors


def _raise_if_no_usable_records(
    run_id: RunIdentifier,
    errors: list[RecordError],
) -> None:
    reasons = "\n".join(
        f"  - batch {error.batch_identifier!s}: {error.error_type}: {error.reason}"
        for error in errors
    )
    if all(error.error_type == RecordErrorType.RESULT_NOT_FOUND for error in errors):
        raise gx_exceptions.ValidationRunResultsCleanedError(  # noqa: TRY003 # FIXME CoP
            f"No validation results could be loaded for run {run_id!r}; the results appear "
            f"to have been cleaned.\n{reasons}"
        )
    raise gx_exceptions.GreatExpectationsError(  # noqa: TRY003 # FIXME CoP
        f"No usable validation results could be loaded for run {run_id!r}.\n{reasons}"
    )


# -------------------------------------------------------------------------------------------------
# Row construction
# -------------------------------------------------------------------------------------------------
def _compare_numeric_metric(
    name: str,
    old_value: Any,
    new_value: Any,
    threshold: float,
) -> MetricComparison:
    old_number = _as_number(old_value)
    new_number = _as_number(new_value)
    if old_number is not None and new_number is not None:
        delta = new_number - old_number
        return MetricComparison(name, old_number, new_number, delta, abs(delta) > threshold, True)
    return MetricComparison(
        name,
        _json_value(old_value),
        _json_value(new_value),
        None,
        None,
        False,
        "missing_or_non_numeric",
    )


def _compare_observed_value(
    old_value: Any,
    new_value: Any,
    threshold: float,
) -> MetricComparison:
    old_number = _as_number(old_value)
    new_number = _as_number(new_value)
    if old_number is not None and new_number is not None:
        delta = new_number - old_number
        return MetricComparison(
            "observed_value",
            old_number,
            new_number,
            delta,
            abs(delta) > threshold,
            True,
        )
    old_json = _json_value(old_value)
    new_json = _json_value(new_value)
    if old_json is None or new_json is None:
        return MetricComparison(
            "observed_value", old_json, new_json, None, None, False, "missing_value"
        )
    if old_json == new_json:
        return MetricComparison("observed_value", old_json, new_json, None, False, True)
    # Non-numeric values cannot be quantified as a delta; any inequality exceeds the threshold.
    return MetricComparison("observed_value", old_json, new_json, None, True, True)


def _build_metrics(
    old_result: Optional[dict],
    new_result: Optional[dict],
    thresholds: ComparisonThresholds,
) -> tuple[MetricComparison, MetricComparison, MetricComparison]:
    old_result = old_result or {}
    new_result = new_result or {}
    observed = _compare_observed_value(
        old_result.get("observed_value"),
        new_result.get("observed_value"),
        thresholds.observed_value,
    )
    count = _compare_numeric_metric(
        "unexpected_count",
        old_result.get("unexpected_count"),
        new_result.get("unexpected_count"),
        thresholds.unexpected_count,
    )
    percent = _compare_numeric_metric(
        "unexpected_percent",
        old_result.get("unexpected_percent"),
        new_result.get("unexpected_percent"),
        thresholds.unexpected_percent,
    )
    return observed, count, percent


def _one_sided_metrics(
    result: Optional[dict],
    side: str,
) -> tuple[MetricComparison, MetricComparison, MetricComparison]:
    result = result or {}

    def _wrap(name: str, value: Any) -> MetricComparison:
        json_value = _json_value(value)
        return MetricComparison(
            name,
            json_value if side == "old" else None,
            json_value if side == "new" else None,
            None,
            None,
            False,
            "one_sided",
        )

    return (
        _wrap("observed_value", result.get("observed_value")),
        _wrap("unexpected_count", result.get("unexpected_count")),
        _wrap("unexpected_percent", result.get("unexpected_percent")),
    )


def _row_from_evr(
    evr: ExpectationValidationResult,
    category: str,
    side: str,
) -> ExpectationComparison:
    _, kwargs = _config_parts(evr.expectation_config)
    identity = _expectation_identity(evr)
    identity_str = _identity_string(identity) if identity is not None else ""
    observed, count, percent = _one_sided_metrics(evr.result, side)
    success = evr.success
    return ExpectationComparison(
        expectation_type=_config_parts(evr.expectation_config)[0],
        expectation_kwargs=_json_value(kwargs) or {},
        identity=identity_str,
        old_success=success if side == "old" else None,
        new_success=success if side == "new" else None,
        success_changed=None,
        category=category,
        flipped=True,
        priority=FLIP_PRIORITY[category],
        observed_value=observed,
        unexpected_count=count,
        unexpected_percent=percent,
        comparable=False,
        comparability_reason="one_sided",
    )


def _not_comparable_row(  # noqa: PLR0913 # FIXME CoP
    expectation_type: Optional[str],
    kwargs: Optional[dict],
    identity: Any,
    old_evr: Optional[ExpectationValidationResult],
    new_evr: Optional[ExpectationValidationResult],
    reason: str,
) -> ExpectationComparison:
    old_result = old_evr.result if old_evr is not None else None
    new_result = new_evr.result if new_evr is not None else None
    if old_evr is not None and new_evr is not None:
        observed, count, percent = _build_metrics(old_result, new_result, ComparisonThresholds())
    elif old_evr is not None:
        observed, count, percent = _one_sided_metrics(old_result, "old")
    else:
        assert new_evr is not None
        observed, count, percent = _one_sided_metrics(new_result, "new")

    old_success = old_evr.success if old_evr is not None else None
    new_success = new_evr.success if new_evr is not None else None
    success_changed: Optional[bool]
    if old_success is None or new_success is None:
        success_changed = None
    else:
        success_changed = old_success != new_success

    return ExpectationComparison(
        expectation_type=expectation_type,
        expectation_kwargs=_json_value(kwargs) or {},
        identity=_identity_string(identity) if identity is not None else "",
        old_success=old_success,
        new_success=new_success,
        success_changed=success_changed,
        category=FlipCategory.NOT_COMPARABLE,
        flipped=False,
        priority=None,
        observed_value=observed,
        unexpected_count=count,
        unexpected_percent=percent,
        comparable=False,
        comparability_reason=reason,
    )


def _aligned_row(
    old_evr: ExpectationValidationResult,
    new_evr: ExpectationValidationResult,
    thresholds: ComparisonThresholds,
) -> ExpectationComparison:
    expectation_type, kwargs = _config_parts(new_evr.expectation_config)
    identity = (expectation_type, _canonical(kwargs)) if expectation_type is not None else None

    old_result = old_evr.result or {}
    new_result = new_evr.result or {}
    if not old_result or not new_result:
        return _not_comparable_row(
            expectation_type, kwargs, identity, old_evr, new_evr, "missing_statistics"
        )

    observed, count, percent = _build_metrics(old_result, new_result, thresholds)
    old_success = old_evr.success
    new_success = new_evr.success

    category: str
    if old_success is True and new_success is False:
        category = FlipCategory.PASSED_TO_FAILED
    elif old_success is False and new_success is True:
        category = FlipCategory.FAILED_TO_PASSED
    elif old_success is False and new_success is False:
        metrics = (observed, count, percent)
        if not any(metric.comparable for metric in metrics):
            return _not_comparable_row(
                expectation_type, kwargs, identity, old_evr, new_evr, "no_comparable_metrics"
            )
        if any(metric.comparable and metric.changed for metric in metrics):
            category = FlipCategory.STILL_FAILING_CHANGED
        else:
            category = FlipCategory.UNCHANGED
    elif old_success is True and new_success is True:
        category = FlipCategory.UNCHANGED
    else:
        return _not_comparable_row(
            expectation_type, kwargs, identity, old_evr, new_evr, "success_unknown"
        )

    flipped = category in FLIP_CATEGORIES
    return ExpectationComparison(
        expectation_type=expectation_type,
        expectation_kwargs=_json_value(kwargs) or {},
        identity=_identity_string(identity) if identity is not None else "",
        old_success=old_success,
        new_success=new_success,
        success_changed=bool(old_success != new_success),
        category=category,
        flipped=flipped,
        priority=FLIP_PRIORITY.get(category),
        observed_value=observed,
        unexpected_count=count,
        unexpected_percent=percent,
        comparable=True,
    )


# -------------------------------------------------------------------------------------------------
# Batch construction
# -------------------------------------------------------------------------------------------------
def _batch_comparability_reason(
    old_record: ExpectationSuiteValidationResult,
    new_record: ExpectationSuiteValidationResult,
) -> Optional[str]:
    if not old_record.statistics or not new_record.statistics:
        return "statistics_missing"
    old_batch_parameters = old_record.meta.get("batch_parameters")
    new_batch_parameters = new_record.meta.get("batch_parameters")
    if (
        old_batch_parameters is not None
        and new_batch_parameters is not None
        and _json_value(old_batch_parameters) != _json_value(new_batch_parameters)
    ):
        return "batch_parameters_mismatch"
    return None


def _index_expectations(
    record: ExpectationSuiteValidationResult,
) -> tuple[dict[Any, ExpectationValidationResult], list[tuple[int, ExpectationValidationResult]]]:
    indexed: dict[Any, ExpectationValidationResult] = {}
    unidentifiable: list[tuple[int, ExpectationValidationResult]] = []
    for position, evr in enumerate(record.results):
        identity = _expectation_identity(evr)
        if identity is None:
            unidentifiable.append((position, evr))
        else:
            indexed[identity] = evr
    return indexed, unidentifiable


def _build_aligned_batch(
    batch_identifier: str,
    old_record: ExpectationSuiteValidationResult,
    new_record: ExpectationSuiteValidationResult,
    thresholds: ComparisonThresholds,
) -> BatchComparison:
    reason = _batch_comparability_reason(old_record, new_record)
    old_indexed, old_unidentified = _index_expectations(old_record)
    new_indexed, new_unidentified = _index_expectations(new_record)

    rows: list[ExpectationComparison] = []
    identities = set(old_indexed.keys()) | set(new_indexed.keys())
    for identity in identities:
        old_evr = old_indexed.get(identity)
        new_evr = new_indexed.get(identity)
        if old_evr is None:
            rows.append(_row_from_evr(new_evr, FlipCategory.NEW_EXPECTATION, "new"))  # type: ignore[arg-type] # FIXME CoP
        elif new_evr is None:
            rows.append(_row_from_evr(old_evr, FlipCategory.MISSING_EXPECTATION, "old"))
        elif reason is not None:
            expectation_type, kwargs = _config_parts(new_evr.expectation_config)
            rows.append(
                _not_comparable_row(expectation_type, kwargs, identity, old_evr, new_evr, reason)
            )
        else:
            rows.append(_aligned_row(old_evr, new_evr, thresholds))

    # Expectations without an identifiable configuration are never silently dropped.
    for _position, evr in old_unidentified:
        rows.append(
            _not_comparable_row(None, None, None, evr, None, "missing_expectation_configuration")
        )
    for _position, evr in new_unidentified:
        rows.append(
            _not_comparable_row(None, None, None, None, evr, "missing_expectation_configuration")
        )

    rows.sort(key=lambda row: (row.expectation_type or "", row.identity))
    status = BatchStatus.NOT_COMPARABLE if reason is not None else BatchStatus.ALIGNED
    return BatchComparison(str(batch_identifier), status, rows, reason)


def _build_one_sided_batch(
    batch_identifier: str,
    record: ExpectationSuiteValidationResult,
    side: str,
) -> BatchComparison:
    category = FlipCategory.NEW_EXPECTATION if side == "new" else FlipCategory.MISSING_EXPECTATION
    rows: list[ExpectationComparison] = []
    for evr in record.results:
        identity = _expectation_identity(evr)
        if identity is None:
            rows.append(
                _not_comparable_row(
                    None,
                    None,
                    None,
                    evr if side == "old" else None,
                    evr if side == "new" else None,
                    "missing_expectation_configuration",
                )
            )
        else:
            rows.append(_row_from_evr(evr, category, side))
    rows.sort(key=lambda row: (row.expectation_type or "", row.identity))
    status = BatchStatus.NEW_BATCH if side == "new" else BatchStatus.MISSING_BATCH
    return BatchComparison(str(batch_identifier), status, rows)


# -------------------------------------------------------------------------------------------------
# Public builder
# -------------------------------------------------------------------------------------------------
def build_run_comparison(  # noqa: C901, PLR0912, PLR0913, PLR0915 # FIXME CoP
    store: ValidationResultsStore,
    expectation_suite_name: str,
    anchor_run_id: RunIdentifier,
    other_run_id: Optional[RunIdentifier],
    window_start: Optional[Union[datetime.datetime, str]],
    window_end: Optional[Union[datetime.datetime, str]],
    time_window: datetime.timedelta,
    max_results: int,
    thresholds: ComparisonThresholds,
) -> RunComparisonReport:
    """Build a read-only comparison report from validation results in ``store``.

    This function performs only read operations (``list_keys`` / ``get``) against the store; it
    never writes results or modifies run statistics.
    """
    if other_run_id is not None:
        old_run_id, new_run_id = order_runs(anchor_run_id, other_run_id)
    else:
        old_run_id, new_run_id = None, anchor_run_id

    all_keys = store.list_keys()
    suite_keys = [
        key
        for key in all_keys
        if isinstance(key, ValidationResultIdentifier)
        and key.expectation_suite_identifier.name == expectation_suite_name
    ]

    new_keys = [key for key in suite_keys if key.run_id == new_run_id]
    if not new_keys:
        run_recorded_elsewhere = any(
            isinstance(key, ValidationResultIdentifier) and key.run_id == new_run_id
            for key in all_keys
        )
        if run_recorded_elsewhere:
            raise gx_exceptions.ValidationRunResultsCleanedError(  # noqa: TRY003 # FIXME CoP
                f"Run {new_run_id!r} exists, but no validation results were found for "
                f"expectation suite '{expectation_suite_name}'. The results may have been cleaned."
            )
        raise gx_exceptions.ValidationRunNotFoundError(  # noqa: TRY003 # FIXME CoP
            f"Run {new_run_id!r} was not found in the validation results store. "
            "Check the run_name and run_time and try again."
        )

    window_meta: Optional[dict[str, str]] = None
    if old_run_id is None:
        start_dt = (
            _coerce_datetime(window_start, "window_start") if window_start is not None else None
        )
        end_dt = _coerce_datetime(window_end, "window_end") if window_end is not None else None
        if start_dt is None and end_dt is None:
            start_dt = new_run_id.run_time - time_window
            end_dt = new_run_id.run_time
        elif start_dt is None:
            assert end_dt is not None
            start_dt = end_dt - time_window
        elif end_dt is None:
            end_dt = start_dt + time_window

        # Half-open window [start, end); start == end is a valid empty window.
        if start_dt > end_dt:
            raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
                "window_start must be earlier than or equal to window_end; "
                f"got start={_isoformat(start_dt)}, end={_isoformat(end_dt)}."
            )

        window_meta = {"start": _isoformat(start_dt), "end": _isoformat(end_dt)}
        candidate_run_ids = {
            key.run_id
            for key in suite_keys
            if key.run_id != new_run_id and start_dt <= key.run_id.run_time < end_dt
        }
        if not candidate_run_ids:
            # An empty window yields an empty report, not an error.
            return RunComparisonReport(
                expectation_suite_name=expectation_suite_name,
                old_run_id=None,
                new_run_id=new_run_id,
                batch_comparisons=[],
                record_errors=[],
                window=window_meta,
            )
        old_run_id = max(
            candidate_run_ids,
            key=lambda run: (run.run_time, run.run_name or ""),
        )

    assert old_run_id is not None
    old_keys = [key for key in suite_keys if key.run_id == old_run_id]
    if not old_keys:
        run_recorded_elsewhere = any(
            isinstance(key, ValidationResultIdentifier) and key.run_id == old_run_id
            for key in all_keys
        )
        if run_recorded_elsewhere:
            raise gx_exceptions.ValidationRunResultsCleanedError(  # noqa: TRY003 # FIXME CoP
                f"Run {old_run_id!r} exists, but no validation results were found for "
                f"expectation suite '{expectation_suite_name}'. The results may have been cleaned."
            )
        raise gx_exceptions.ValidationRunNotFoundError(  # noqa: TRY003 # FIXME CoP
            f"Run {old_run_id!r} was not found in the validation results store. "
            "Check the run_name and run_time and try again."
        )

    result_record_count = len(new_keys) + len(old_keys)
    if result_record_count > max_results:
        raise gx_exceptions.InvalidConfigError(  # noqa: TRY003 # FIXME CoP
            f"Comparison requires {result_record_count} validation result records "
            f"({len(new_keys)} new + {len(old_keys)} old), exceeding the configured "
            f"max_results={max_results}. Increase max_results or narrow the comparison scope; "
            "results are not truncated."
        )

    new_records, new_errors = _load_run_records(store, new_keys)
    if not new_records:
        _raise_if_no_usable_records(new_run_id, new_errors)
    old_records, old_errors = _load_run_records(store, old_keys)
    if not old_records:
        _raise_if_no_usable_records(old_run_id, old_errors)

    batch_identifiers = set(new_records.keys()) | set(old_records.keys())
    batch_comparisons: list[BatchComparison] = []
    for batch_identifier in sorted(batch_identifiers):
        old_record = old_records.get(batch_identifier)
        new_record = new_records.get(batch_identifier)
        if old_record is None:
            assert new_record is not None
            batch_comparisons.append(_build_one_sided_batch(batch_identifier, new_record, "new"))
        elif new_record is None:
            batch_comparisons.append(_build_one_sided_batch(batch_identifier, old_record, "old"))
        else:
            batch_comparisons.append(
                _build_aligned_batch(batch_identifier, old_record, new_record, thresholds)
            )

    record_errors = new_errors + old_errors
    record_errors.sort(
        key=lambda error: (
            error.run_id.run_time,
            error.run_id.run_name or "",
            str(error.batch_identifier),
        )
    )

    return RunComparisonReport(
        expectation_suite_name=expectation_suite_name,
        old_run_id=old_run_id,
        new_run_id=new_run_id,
        batch_comparisons=batch_comparisons,
        record_errors=record_errors,
        window=window_meta,
    )

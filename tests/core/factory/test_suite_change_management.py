import datetime
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import great_expectations as gx
import great_expectations.expectations as gxe
from great_expectations.core import ExpectationSuite
from great_expectations.core.factory import (
    ChangeSource,
    ChangeType,
    ConflictPolicy,
    DiffDetail,
    ReferenceRef,
    SuiteChangeManagementConfig,
)
from great_expectations.core.suite_change_management import (
    SuiteHistoryStore,
    _diff_suites,
)
from great_expectations.data_context import AbstractDataContext
from great_expectations.exceptions import (
    SuiteChangeManagementError,
    SuiteReferenceConflictError,
    SuiteVersionNotFoundError,
)

ENABLED_CONFIG = SuiteChangeManagementConfig(enabled=True, retention_limit=10)


def _enable(context: AbstractDataContext, **kwargs) -> None:
    config_kwargs = {"enabled": True}
    config_kwargs.update(kwargs)
    context.suites.configure_change_management(SuiteChangeManagementConfig(**config_kwargs))


def _identity_dict(expectation) -> dict:
    return expectation.dict(exclude={"id", "rendered_content", "notes", "meta"})


def _same_entry_content(left: ExpectationSuite, right: ExpectationSuite) -> bool:
    """Content equality independent of store-assigned entry ids."""
    left_entries = sorted(_canonical_entry(exp) for exp in left.expectations)
    right_entries = sorted(_canonical_entry(exp) for exp in right.expectations)
    return left_entries == right_entries


def _canonical_entry(expectation) -> str:
    return json.dumps(
        expectation.dict(exclude={"id", "rendered_content", "notes", "meta"}),
        sort_keys=True,
        default=str,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestSuiteChangeManagementConfig:
    @pytest.mark.unit
    def test_retention_limit_zero_is_rejected_when_building_factory(self):
        store = gx.get_context(mode="ephemeral").suites._store
        with pytest.raises(ValueError, match="retention_limit"):
            gx.core.factory.SuiteFactory(
                store=store,
                change_management=SuiteChangeManagementConfig(enabled=True, retention_limit=0),
            )

    @pytest.mark.unit
    def test_retention_limit_zero_rejected_even_when_disabled(self):
        # An explicitly malformed config is rejected regardless of enabled.
        with pytest.raises(ValueError, match="retention_limit"):
            SuiteChangeManagementConfig(enabled=False, retention_limit=0)

    @pytest.mark.unit
    @pytest.mark.parametrize("bad_retention", [-1, -10])
    def test_negative_retention_rejected(self, bad_retention):
        with pytest.raises(ValueError, match="retention_limit"):
            SuiteChangeManagementConfig(retention_limit=bad_retention)

    @pytest.mark.unit
    def test_bad_diff_detail_rejected(self):
        with pytest.raises(ValueError, match="diff_detail"):
            SuiteChangeManagementConfig(diff_detail="everything")

    @pytest.mark.unit
    def test_bad_conflict_policy_rejected(self):
        with pytest.raises(ValueError, match="conflict_policy"):
            SuiteChangeManagementConfig(conflict_policy="explode")

    @pytest.mark.unit
    def test_dict_config_accepted(self):
        store = gx.get_context(mode="ephemeral").suites._store
        factory = gx.core.factory.SuiteFactory(
            store=store,
            change_management={"enabled": True, "retention_limit": 3},
        )
        assert factory.change_management.enabled is True
        assert factory.change_management.retention_limit == 3


# ---------------------------------------------------------------------------
# Default-off behavior
# ---------------------------------------------------------------------------


class TestDefaultOff:
    @pytest.mark.unit
    def test_no_versions_written(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        suite = context.suites.add(ExpectationSuite(name="s"))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="a"))
        context.suites.add_or_update(suite)
        context.suites.add_or_update(context.suites.get("s"))
        assert context.suites.get_versions("s") == []

    @pytest.mark.unit
    def test_preview_still_available_when_disabled(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                ],
            )
        )
        preview = context.suites.preview_changes(ExpectationSuite(name="s", expectations=[]))
        assert preview.counts[ChangeType.REMOVED] == 1

    @pytest.mark.unit
    def test_rollback_disabled_raises(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        context.suites.add(ExpectationSuite(name="s"))
        with pytest.raises(SuiteChangeManagementError):
            context.suites.rollback("s", version=1)


# ---------------------------------------------------------------------------
# Pure diff engine
# ---------------------------------------------------------------------------


class TestDiffEngine:
    @pytest.mark.unit
    def test_four_categories(self):
        current = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=1),
                gxe.ExpectColumnValuesToBeInSet(column="c", value_set=["x"]),
            ],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[
                # unchanged
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                # modified: max_value changes
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=9),
                # removed: c
                # added: d
                gxe.ExpectColumnValuesToBeInSet(column="d", value_set=["y"]),
            ],
        )
        preview = _diff_suites(suite_name="s", current=current, pending=pending)
        counts = preview.counts
        assert counts == {"added": 1, "removed": 1, "modified": 1, "unchanged": 1}

        modified = preview.modified[0]
        assert modified.expectation_type == "expect_column_values_to_be_between"
        field_names = {change.field for change in modified.changed_fields}
        assert "kwargs.max_value" in field_names
        max_change = next(
            change for change in modified.changed_fields if change.field == "kwargs.max_value"
        )
        assert max_change.old == 1
        assert max_change.new == 9

        assert preview.removed[0].expectation_type == "expect_column_values_to_be_in_set"
        assert preview.added[0].expectation_type == "expect_column_values_to_be_in_set"
        assert preview.unchanged[0].expectation_type == "expect_column_values_to_not_be_null"

    @pytest.mark.unit
    def test_excluded_fields_surfaced_separately(self):
        # The only difference is meta, which equalish excludes from identity.
        current = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a", meta={"k": "old"}),
            ],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a", meta={"k": "new"}),
            ],
        )
        preview = _diff_suites(suite_name="s", current=current, pending=pending)
        assert preview.counts[ChangeType.UNCHANGED] == 1
        assert preview.counts[ChangeType.MODIFIED] == 0
        assert preview.has_changes is True
        excluded = preview.unchanged[0].excluded_field_changes
        assert [change.field for change in excluded] == ["meta"]
        assert excluded[0].old == {"k": "old"}
        assert excluded[0].new == {"k": "new"}

    @pytest.mark.unit
    def test_notes_change_excluded_field(self):
        current = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a", notes="pay attention")],
        )
        preview = _diff_suites(suite_name="s", current=current, pending=pending)
        fields = {change.field for change in preview.unchanged[0].excluded_field_changes}
        assert fields == {"notes"}

    @pytest.mark.unit
    def test_identical_suites(self):
        suite = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
        )
        preview = _diff_suites(suite_name="s", current=suite, pending=suite)
        assert preview.counts[ChangeType.UNCHANGED] == 1
        assert preview.is_identical() is True
        assert preview.has_changes is False

    @pytest.mark.unit
    def test_new_suite_all_added(self):
        pending = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["z"]),
            ],
        )
        preview = _diff_suites(suite_name="s", current=None, pending=pending)
        assert preview.counts[ChangeType.ADDED] == 2
        assert preview.counts[ChangeType.UNCHANGED] == 0

    @pytest.mark.unit
    def test_deterministic_order_and_output(self):
        current = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToBeInSet(column="c", value_set=["x"]),
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=1),
            ],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToBeInSet(column="z", value_set=["q"]),
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=2),
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
            ],
        )
        first = _diff_suites(suite_name="s", current=current, pending=pending).to_dict()
        second = _diff_suites(suite_name="s", current=current, pending=pending).to_dict()
        assert first == second
        # entries within each category are stably sorted by key
        for category in (
            ChangeType.ADDED,
            ChangeType.REMOVED,
            ChangeType.MODIFIED,
            ChangeType.UNCHANGED,
        ):
            keys = [entry["key"] for entry in first[category]]
            assert keys == sorted(keys)
        # machine-readable payload is JSON serializable and sortable
        serialized = json.dumps(first, sort_keys=True)
        assert json.loads(serialized) == first

    @pytest.mark.unit
    def test_summary_detail_hides_values_and_configs(self):
        current = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=1)],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=9)],
        )
        preview = _diff_suites(
            suite_name="s",
            current=current,
            pending=pending,
            diff_detail=DiffDetail.SUMMARY,
        )
        change = preview.modified[0]
        assert "old_config" not in change.to_dict()
        assert "new_config" not in change.to_dict()
        assert change.changed_fields[0].old is None
        assert change.changed_fields[0].new is None
        assert change.changed_fields[0].field == "kwargs.max_value"

    @pytest.mark.unit
    def test_references_mark_removed_and_modified_as_conflicts(self):
        current = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=1),
                gxe.ExpectColumnValuesToBeInSet(column="c", value_set=["x"]),
            ],
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=9),
            ],
        )
        refs = [
            ReferenceRef(kind="validation_definition", name="vd"),
            ReferenceRef(kind="checkpoint", name="cp"),
        ]
        preview = _diff_suites(suite_name="s", current=current, pending=pending, references=refs)
        assert preview.has_conflicts is True
        assert {(r.kind, r.name) for r in preview.affected_references} == {
            ("validation_definition", "vd"),
            ("checkpoint", "cp"),
        }
        assert len(preview.modified[0].conflicts) == 2
        assert len(preview.removed[0].conflicts) == 2
        assert preview.added == [] and preview.unchanged == []

    @pytest.mark.unit
    def test_added_entries_are_not_conflicts(self):
        current = ExpectationSuite(
            name="s", expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")]
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["z"]),
            ],
        )
        preview = _diff_suites(
            suite_name="s",
            current=current,
            pending=pending,
            references=[ReferenceRef(kind="validation_definition", name="vd")],
        )
        assert preview.has_conflicts is False
        assert preview.affected_references  # still reported
        assert preview.added[0].conflicts == []

    @pytest.mark.unit
    def test_suite_parameter_change_reported(self):
        current = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            suite_parameters={"min": 1},
        )
        pending = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            suite_parameters={"min": 2},
        )
        preview = _diff_suites(suite_name="s", current=current, pending=pending)
        assert preview.is_identical() is False
        assert [change.field for change in preview.suite_attribute_changes] == ["suite_parameters"]


# ---------------------------------------------------------------------------
# Preview through the factory (read-only)
# ---------------------------------------------------------------------------


class TestFactoryPreview:
    @pytest.mark.unit
    def test_preview_is_read_only(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        before = context.suites.get("s")
        pending = ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToBeBetween(column="b", min_value=0, max_value=1)],
        )
        preview = context.suites.preview_changes(pending)
        after = context.suites.get("s")
        assert before == after
        assert preview.counts[ChangeType.REMOVED] == 1
        assert preview.counts[ChangeType.ADDED] == 1
        # the in-memory pending suite was not mutated by preview
        assert len(pending.expectations) == 1
        assert context.suites.get_versions("s") == []


# ---------------------------------------------------------------------------
# Version retention
# ---------------------------------------------------------------------------


class TestVersionRetention:
    @pytest.mark.unit
    def test_adding_new_suite_creates_no_version(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(ExpectationSuite(name="s"))
        assert context.suites.get_versions("s") == []

    @pytest.mark.unit
    def test_consecutive_writes_create_consecutive_versions(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                ],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                    gxe.ExpectColumnMaxToBeBetween(column="c", max_value=5),
                ],
            )
        )
        versions = context.suites.get_versions("s")
        assert [v.version for v in versions] == [1, 2]
        for summary in versions:
            assert summary.created_at
            assert summary.source == ChangeSource.ADD_OR_UPDATE
            assert summary.operation == "write"

    @pytest.mark.unit
    def test_snapshots_contain_full_entries(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                ],
            )
        )
        v1 = context.suites.get_version("s", version=1)
        assert len(v1.suite.expectations) == 1
        assert v1.suite.expectations[0].expectation_type == "expect_column_values_to_not_be_null"

    @pytest.mark.unit
    def test_identical_overwrite_creates_no_history(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        stored = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        # semantically identical re-write
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        # still truthfully reported via preview
        preview = context.suites.preview_changes(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        assert preview.is_identical() is True
        assert context.suites.get_versions("s") == []
        # entry content untouched (ids are re-assigned by whole-suite overwrite)
        assert _same_entry_content(context.suites.get("s"), stored)

    @pytest.mark.unit
    def test_retention_limit_prunes_old_versions(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context, retention_limit=2)
        context.suites.add(ExpectationSuite(name="s"))
        for column in ("a", "b", "c", "d"):
            context.suites.add_or_update(
                ExpectationSuite(
                    name="s",
                    expectations=[
                        gxe.ExpectColumnValuesToNotBeNull(column=column),
                    ],
                )
            )
        versions = context.suites.get_versions("s")
        assert [v.version for v in versions] == [3, 4]

    @pytest.mark.unit
    def test_missing_version_error_states_retained_range(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context, retention_limit=2)
        context.suites.add(ExpectationSuite(name="s"))
        for column in ("a", "b", "c"):
            context.suites.add_or_update(
                ExpectationSuite(
                    name="s",
                    expectations=[gxe.ExpectColumnValuesToNotBeNull(column=column)],
                )
            )
        with pytest.raises(SuiteVersionNotFoundError, match="versions: 2 through 3"):
            context.suites.get_version("s", version=1)

    @pytest.mark.unit
    def test_get_version_by_time(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                ],
            )
        )
        versions = context.suites.get_versions("s")
        at = datetime.datetime.fromisoformat(versions[0].created_at)
        result = context.suites.get_version("s", at=at)
        assert result.version == 1

    @pytest.mark.unit
    def test_version_selector_validation(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(ExpectationSuite(name="s"))
        with pytest.raises(ValueError, match="Exactly one"):
            context.suites.get_version("s")
        with pytest.raises(ValueError, match="Only one"):
            context.suites.get_version("s", version=1, at="2020-01-01")

    @pytest.mark.unit
    def test_delete_suite_retains_version_and_rollback_recreates(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        context.suites.delete("s")
        versions = context.suites.get_versions("s")
        assert len(versions) == 1
        assert versions[0].operation == "delete"
        with pytest.raises(Exception, match="was not found"):
            context.suites.get("s")
        recreated = context.suites.rollback("s", version=1)
        assert len(recreated.expectations) == 1
        assert context.suites.get("s").name == "s"

    @pytest.mark.unit
    def test_failed_write_leaves_no_version(self, in_memory_runtime_context, mocker):
        context = in_memory_runtime_context
        _enable(context)
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        mocker.patch.object(context.suites._store, "update", side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            context.suites.add_or_update(
                ExpectationSuite(
                    name="s",
                    expectations=[gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"])],
                )
            )
        assert context.suites.get_versions("s") == []


# ---------------------------------------------------------------------------
# Expectation-level write paths
# ---------------------------------------------------------------------------


class TestExpectationLevelRetention:
    @pytest.mark.unit
    def test_add_expectation_retains_previous_version(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        suite = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]))
        versions = context.suites.get_versions("s")
        assert [v.version for v in versions] == [1]
        assert versions[0].source == ChangeSource.ADD_EXPECTATION
        record = context.suites.get_version("s", version=1)
        assert len(record.suite.expectations) == 1

    @pytest.mark.unit
    def test_delete_expectation_retains_previous_version(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        suite = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                ],
            )
        )
        suite.delete_expectation(suite.expectations[0])
        versions = context.suites.get_versions("s")
        assert [v.source for v in versions] == [ChangeSource.DELETE_EXPECTATION]
        record = context.suites.get_version("s", version=1)
        assert len(record.suite.expectations) == 2
        assert len(context.suites.get("s").expectations) == 1

    @pytest.mark.unit
    def test_failed_expectation_delete_leaves_no_version_and_restores_memory(
        self, in_memory_runtime_context, mocker
    ):
        context = in_memory_runtime_context
        _enable(context)
        suite = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        mocker.patch.object(
            context.suites._store,
            "delete_expectation",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            suite.delete_expectation(suite.expectations[0])
        assert len(suite.expectations) == 1  # in-memory rollback
        assert context.suites.get_versions("s") == []

    @pytest.mark.unit
    def test_suite_save_retains_previous_version(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        suite = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        suite.expectations.append(gxe.ExpectColumnMaxToBeBetween(column="c", max_value=3))
        suite.save()
        versions = context.suites.get_versions("s")
        assert [v.source for v in versions] == [ChangeSource.SAVE]
        record = context.suites.get_version("s", version=1)
        assert len(record.suite.expectations) == 1
        assert len(context.suites.get("s").expectations) == 2

    @pytest.mark.unit
    def test_expectation_save_retains_previous_version(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        suite = context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToBeBetween(column="a", min_value=0, max_value=1)
                ],
            )
        )
        expectation = suite.expectations[0]
        expectation.max_value = 9
        expectation.save()
        versions = context.suites.get_versions("s")
        assert [v.source for v in versions] == [ChangeSource.UPDATE_EXPECTATION]
        record = context.suites.get_version("s", version=1)
        stored_old = record.suite.expectations[0]
        assert stored_old.configuration.kwargs["max_value"] == 1
        assert context.suites.get("s").expectations[0].configuration.kwargs["max_value"] == 9


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    @staticmethod
    def _build_history(context):
        context.suites.add(
            ExpectationSuite(
                name="s",
                expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                ],
            )
        )
        context.suites.add_or_update(
            ExpectationSuite(
                name="s",
                expectations=[
                    gxe.ExpectColumnValuesToNotBeNull(column="a"),
                    gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
                    gxe.ExpectColumnMaxToBeBetween(column="c", max_value=5),
                ],
            )
        )

    @pytest.mark.unit
    def test_rollback_restores_entries_and_preserves_ids(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        self._build_history(context)

        target = context.suites.get_version("s", version=1)
        current_before = context.suites.get("s")
        # unchanged expectation keeps its identifier across the rollback
        unchanged_id_before = next(
            exp.id
            for exp in current_before.expectations
            if exp.expectation_type == "expect_column_values_to_not_be_null"
        )

        rolled_back = context.suites.rollback("s", version=1)

        assert len(rolled_back.expectations) == 1
        # entry-by-entry equality with the chosen historical version
        for restored, historical in zip(
            sorted(rolled_back.expectations),
            sorted(target.suite.expectations),
            strict=True,
        ):
            assert _identity_dict(restored) == _identity_dict(historical)
        # unchanged entry keeps its live (pre-rollback) identifier
        assert rolled_back.expectations[0].id == unchanged_id_before

    @pytest.mark.unit
    def test_rollback_records_itself_and_can_be_rolled_back(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        self._build_history(context)

        context.suites.rollback("s", version=1)
        versions = context.suites.get_versions("s")
        assert [v.version for v in versions] == [1, 2, 3]
        rollback_entry = versions[-1]
        assert rollback_entry.operation == "rollback"
        assert rollback_entry.restored_from_version == 1

        # the replaced (3-expectation) version was preserved by the rollback
        replaced = context.suites.get_version("s", version=3)
        assert replaced.operation == "rollback"
        assert len(replaced.suite.expectations) == 3

        # rolling forward again restores it
        restored_again = context.suites.rollback("s", version=3)
        assert len(restored_again.expectations) == 3
        versions = context.suites.get_versions("s")
        assert [v.operation for v in versions] == [
            "write",
            "write",
            "rollback",
            "rollback",
        ]
        assert versions[-1].restored_from_version == 3

    @pytest.mark.unit
    def test_rollback_unknown_version_mentions_range(self, in_memory_runtime_context):
        context = in_memory_runtime_context
        _enable(context)
        self._build_history(context)
        with pytest.raises(SuiteVersionNotFoundError, match="99"):
            context.suites.rollback("s", version=99)


# ---------------------------------------------------------------------------
# Reference conflicts end to end
# ---------------------------------------------------------------------------


@pytest.fixture
def referenced_suite_context(in_memory_runtime_context):
    context = in_memory_runtime_context
    data_source = context.data_sources.add_pandas(name="pd")
    asset = data_source.add_dataframe_asset(name="asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("bd")

    # default off while building references
    suite = context.suites.add(
        ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
        )
    )
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(name="vd", data=batch_definition, suite=suite)
    )
    context.checkpoints.add(
        gx.Checkpoint(name="cp", validation_definitions=[validation_definition])
    )
    return context


class TestReferenceConflicts:
    @pytest.mark.unit
    def test_preview_flags_validation_definition_and_checkpoint(self, referenced_suite_context):
        context = referenced_suite_context
        preview = context.suites.preview_changes(ExpectationSuite(name="s"))
        assert preview.has_conflicts
        kinds = {ref.kind for ref in preview.affected_references}
        assert kinds == {"validation_definition", "checkpoint"}

    @pytest.mark.unit
    def test_warn_policy_records_affected_parties(self, referenced_suite_context):
        context = referenced_suite_context
        _enable(context, conflict_policy=ConflictPolicy.WARN)
        context.suites.add_or_update(ExpectationSuite(name="s"))
        [summary] = context.suites.get_versions("s")
        record = context.suites.get_version("s", version=summary.version)
        assert {ref["kind"] for ref in record.affected_references} == {
            "validation_definition",
            "checkpoint",
        }
        assert len(context.suites.get("s").expectations) == 0

    @pytest.mark.unit
    def test_raise_policy_blocks_write(self, referenced_suite_context):
        context = referenced_suite_context
        _enable(context, conflict_policy=ConflictPolicy.RAISE)
        with pytest.raises(SuiteReferenceConflictError):
            context.suites.add_or_update(ExpectationSuite(name="s"))
        # nothing changed, no history written
        assert len(context.suites.get("s").expectations) == 1
        assert context.suites.get_versions("s") == []

    @pytest.mark.unit
    def test_raise_policy_blocks_delete(self, referenced_suite_context):
        context = referenced_suite_context
        _enable(context, conflict_policy=ConflictPolicy.RAISE)
        with pytest.raises(SuiteReferenceConflictError):
            context.suites.delete("s")
        assert context.suites.get("s") is not None

    @pytest.mark.unit
    def test_ignore_policy_does_not_flag_but_records(self, referenced_suite_context):
        context = referenced_suite_context
        _enable(context, conflict_policy=ConflictPolicy.IGNORE)
        preview = context.suites.preview_changes(ExpectationSuite(name="s"))
        assert preview.has_conflicts is False
        assert len(preview.affected_references) == 2
        context.suites.add_or_update(ExpectationSuite(name="s"))
        [summary] = context.suites.get_versions("s")
        record = context.suites.get_version("s", version=summary.version)
        assert len(record.affected_references) == 2


# ---------------------------------------------------------------------------
# Filesystem persistence / concurrency
# ---------------------------------------------------------------------------


@pytest.mark.filesystem
def test_disabled_context_writes_no_history_files(tmp_path):
    # Default-off: saves/overwrites/deletes must not create any extra storage.
    context = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    suite = context.suites.add(
        ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
        )
    )
    suite.add_expectation(gxe.ExpectColumnMaxToBeBetween(column="c", max_value=1))
    context.suites.add_or_update(
        ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="b")],
        )
    )
    context.suites.delete("s")

    expectations_base = Path(context.expectations_store.store_backend.full_base_directory)
    history_directory = expectations_base.parent / "suite_versions"
    assert not history_directory.exists()


@pytest.mark.filesystem
def test_history_shared_across_contexts(tmp_path):
    context_one = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    _enable(context_one)
    context_one.suites.add(
        ExpectationSuite(
            name="s",
            expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")],
        )
    )
    context_one.suites.add_or_update(
        ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeInSet(column="b", value_set=["x"]),
            ],
        )
    )

    context_two = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    _enable(context_two)
    assert [v.version for v in context_two.suites.get_versions("s")] == [1]

    context_two.suites.add_or_update(
        ExpectationSuite(
            name="s",
            expectations=[
                gxe.ExpectColumnValuesToNotBeNull(column="a"),
                gxe.ExpectColumnValuesToBeInSet(column="c", value_set=["y"]),
            ],
        )
    )
    assert [v.version for v in context_one.suites.get_versions("s")] == [1, 2]
    # history never appears in suite listing
    assert {suite.name for suite in context_one.suites.all()} == {"s"}


@pytest.mark.filesystem
def test_concurrent_version_claims_do_not_overwrite(tmp_path):
    context = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    _enable(context)
    other = gx.get_context(mode="file", project_root_dir=str(tmp_path))
    _enable(other)
    history_one = SuiteHistoryStore(context.suites._store)
    history_two = SuiteHistoryStore(other.suites._store)

    def write(store, index):
        store.write_version(
            "s",
            {"suite_name": "s", "index": index, "source": "test"},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(write, history_one, i) for i in range(10)]
        futures += [pool.submit(write, history_two, i) for i in range(10, 20)]
        for future in futures:
            future.result()

    numbers = history_one.list_version_numbers("s")
    assert numbers == list(range(1, 21))
    bodies = {json.dumps(history_one.read_version("s", n), sort_keys=True) for n in numbers}
    assert len(bodies) == 20

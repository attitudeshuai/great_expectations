"""Change preview, historical version retention and rollback for ExpectationSuites.

This capability is **off by default**. When enabled on a ``SuiteFactory`` it:

* previews the per-expectation diff between a pending suite and the stored one
  (added / removed / modified / unchanged, with field-level before/after values
  and a separate view of the fields excluded from the suite's equality check);
* records a fully readable snapshot of the old version immediately before every
  successful write or delete, timestamped and attributed to a source;
* restores any retained version (by version number or point in time) while
  preserving the version being replaced, so rollbacks can themselves be rolled
  back.

The module is deliberately storage-agnostic: version blobs live next to (not
inside) the configured ExpectationsStore, in immutable per-version keys, which
is what keeps concurrent writers from overwriting each other's history.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import posixpath
import re
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Union

from great_expectations.compatibility.typing_extensions import override
from great_expectations.data_context.data_context.context_factory import project_manager
from great_expectations.exceptions import (
    StoreBackendError,
    SuiteChangeManagementError,
    SuiteReferenceConflictError,
    SuiteVersionNotFoundError,
)
from great_expectations.util import convert_to_json_serializable  # noqa: TID251

if TYPE_CHECKING:
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.data_context.store.expectations_store import (
        ExpectationsStore,
    )
    from great_expectations.data_context.store.gx_cloud_store_backend import (
        GXCloudStoreBackend,
    )
    from great_expectations.data_context.store.in_memory_store_backend import (
        InMemoryStoreBackend,
    )
    from great_expectations.data_context.store.tuple_store_backend import (
        TupleFilesystemStoreBackend,
        TupleStoreBackend,
    )
    from great_expectations.expectations.expectation import Expectation

logger = logging.getLogger(__name__)


# Store backend classes are imported lazily (inside _store_backend_classes())
# because this module is imported while great_expectations.core is still being
# initialized; importing the data_context.store package there is circular.


def _store_backend_classes() -> tuple[
    type[GXCloudStoreBackend],
    type[InMemoryStoreBackend],
    type[TupleFilesystemStoreBackend],
    type[TupleStoreBackend],
]:
    from great_expectations.data_context.store.gx_cloud_store_backend import (
        GXCloudStoreBackend,
    )
    from great_expectations.data_context.store.in_memory_store_backend import (
        InMemoryStoreBackend,
    )
    from great_expectations.data_context.store.tuple_store_backend import (
        TupleFilesystemStoreBackend,
        TupleStoreBackend,
    )

    return (
        GXCloudStoreBackend,
        InMemoryStoreBackend,
        TupleFilesystemStoreBackend,
        TupleStoreBackend,
    )


def _is_cloud_backend(backend: Any) -> bool:
    GXCloudStoreBackend, _, _, _ = _store_backend_classes()
    return isinstance(backend, GXCloudStoreBackend)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mirrors ExpectationSuite._expectations_are_equalish: these fields do not
# participate in the "same expectation" determination. The field selection is
# intentionally not changed here.
EQUALITY_EXCLUDED_FIELDS: frozenset[str] = frozenset({"id", "rendered_content", "notes", "meta"})
# Of the excluded fields, these are human-meaningful and surfaced separately.
# ``id`` is store-assigned and ``rendered_content`` is derived, so they are
# only surfaced when both sides carry a concrete value.
REPORTED_EXCLUDED_FIELDS: tuple[str, ...] = ("meta", "notes")

HISTORY_DIR_NAME = "suite_versions"
_HISTORY_KEY_PREFIX = ("__suite_versions__",)
_VERSION_FILE_RE = re.compile(r"v(\d{12})\.json$")
_VERSION_FILE_TEMPLATE = "v{number:012d}.json"
_MAX_VERSION_CLAIM_ATTEMPTS = 25


class ChangeType:
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class DiffDetail:
    SUMMARY = "summary"
    FULL = "full"
    _ALL = frozenset({SUMMARY, FULL})


class ConflictPolicy:
    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"
    _ALL = frozenset({RAISE, WARN, IGNORE})


class ChangeOperation:
    WRITE = "write"
    DELETE = "delete"
    ROLLBACK = "rollback"


class ChangeSource:
    SAVE = "save"
    ADD_OR_UPDATE = "add_or_update"
    DELETE_SUITE = "delete_suite"
    ADD_EXPECTATION = "add_expectation"
    UPDATE_EXPECTATION = "update_expectation"
    DELETE_EXPECTATION = "delete_expectation"
    ROLLBACK = "rollback"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteChangeManagementConfig:
    """Configuration for suite change preview and historical version retention.

    Args:
        enabled: Master switch. When False (default), suite save/add/update/delete
            behave exactly as before and produce no extra reads or writes.
        retention_limit: Number of historical versions to keep per suite. Must be
            a positive integer; ``0`` is rejected when the SuiteFactory is built.
        diff_detail: ``"full"`` includes field-level before/after values and full
            expectation configs; ``"summary"`` includes only change categories and
            changed field names.
        conflict_policy: How writes handle expectations that would be modified or
            removed while still held by a validation definition or checkpoint:
            ``"raise"`` blocks the write, ``"warn"`` (default) proceeds and records
            the affected parties, ``"ignore"`` proceeds without flagging a
            conflict (affected parties are still recorded for audit).
    """

    enabled: bool = False
    retention_limit: int = 10
    diff_detail: str = DiffDetail.FULL
    conflict_policy: str = ConflictPolicy.WARN

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Suite change management 'enabled' must be a boolean.")  # noqa: TRY003
        if isinstance(self.retention_limit, bool) or not isinstance(self.retention_limit, int):
            raise TypeError(  # noqa: TRY003
                "Suite change management 'retention_limit' must be a positive integer "
                f"(got {self.retention_limit!r})."
            )
        if self.retention_limit <= 0:
            raise ValueError(  # noqa: TRY003
                "Suite change management 'retention_limit' must be greater than 0; "
                f"got {self.retention_limit}. A retention limit of 0 would silently "
                "discard every historical version."
            )
        if self.diff_detail not in DiffDetail._ALL:
            raise ValueError(  # noqa: TRY003
                f"Suite change management 'diff_detail' must be one of "
                f"{sorted(DiffDetail._ALL)}, got {self.diff_detail!r}."
            )
        if self.conflict_policy not in ConflictPolicy._ALL:
            raise ValueError(  # noqa: TRY003
                f"Suite change management 'conflict_policy' must be one of "
                f"{sorted(ConflictPolicy._ALL)}, got {self.conflict_policy!r}."
            )


# ---------------------------------------------------------------------------
# Machine-readable diff models
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    return convert_to_json_serializable(value)


def _canonical(value: Any) -> str:
    """Stable canonical JSON string used for equality and ordering."""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass(frozen=True)
class FieldChange:
    """A single field that differs between the old and the new expectation."""

    field: str
    old: Any = None
    new: Any = None
    include_values: bool = True

    def to_dict(self) -> dict:
        result: dict[str, Any] = {"field": self.field}
        if self.include_values:
            result["old"] = _jsonable(self.old)
            result["new"] = _jsonable(self.new)
        return result


@dataclass(frozen=True)
class ReferenceRef:
    """Something (a validation definition or checkpoint) that holds the suite."""

    kind: str
    name: str

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name}


@dataclass
class ExpectationChange:
    """The change for one expectation entry."""

    change_type: str
    expectation_type: str
    key: str
    expectation_id: Optional[str] = None
    changed_fields: list[FieldChange] = field(default_factory=list)
    excluded_field_changes: list[FieldChange] = field(default_factory=list)
    conflicts: list[ReferenceRef] = field(default_factory=list)
    old_config: Optional[dict] = None
    new_config: Optional[dict] = None
    include_configs: bool = True

    def to_dict(self) -> dict:
        result: dict[str, Any] = {
            "change_type": self.change_type,
            "expectation_type": self.expectation_type,
            "expectation_id": self.expectation_id,
            "key": self.key,
            "changed_fields": [change.to_dict() for change in self.changed_fields],
            "excluded_field_changes": [change.to_dict() for change in self.excluded_field_changes],
            "conflicts": [ref.to_dict() for ref in self.conflicts],
        }
        if self.include_configs:
            result["old_config"] = (
                _jsonable(self.old_config) if self.old_config is not None else None
            )
            result["new_config"] = (
                _jsonable(self.new_config) if self.new_config is not None else None
            )
        return result


@dataclass
class SuiteChangePreview:
    """The per-entry diff of a pending suite against the currently stored one."""

    suite_name: str
    added: list[ExpectationChange] = field(default_factory=list)
    removed: list[ExpectationChange] = field(default_factory=list)
    modified: list[ExpectationChange] = field(default_factory=list)
    unchanged: list[ExpectationChange] = field(default_factory=list)
    suite_attribute_changes: list[FieldChange] = field(default_factory=list)
    affected_references: list[ReferenceRef] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            ChangeType.ADDED: len(self.added),
            ChangeType.REMOVED: len(self.removed),
            ChangeType.MODIFIED: len(self.modified),
            ChangeType.UNCHANGED: len(self.unchanged),
        }

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added
            or self.removed
            or self.modified
            or self.excluded_field_changes_present
            or self.suite_attribute_changes
        )

    @property
    def excluded_field_changes_present(self) -> bool:
        return any(
            change.excluded_field_changes
            for change in self.added + self.removed + self.modified + self.unchanged
        )

    @property
    def has_conflicts(self) -> bool:
        return any(change.conflicts for change in self.removed + self.modified)

    def conflicting_changes(self) -> list[ExpectationChange]:
        """Changes that remove or rewrite entries (the conflict candidates)."""
        return self.removed + self.modified

    def is_identical(self) -> bool:
        """True when the pending content is semantically identical to stored.

        Store-assigned ids and derived rendered content are ignored; everything
        else (including excluded-but-represented meta/notes changes) must match.
        """
        return not self.has_changes

    def to_dict(self) -> dict:
        return {
            "suite_name": self.suite_name,
            "counts": self.counts,
            "has_changes": self.has_changes,
            "has_conflicts": self.has_conflicts,
            ChangeType.ADDED: [change.to_dict() for change in self.added],
            ChangeType.REMOVED: [change.to_dict() for change in self.removed],
            ChangeType.MODIFIED: [change.to_dict() for change in self.modified],
            ChangeType.UNCHANGED: [change.to_dict() for change in self.unchanged],
            "suite_attribute_changes": [
                change.to_dict() for change in self.suite_attribute_changes
            ],
            "affected_references": [ref.to_dict() for ref in self.affected_references],
        }


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------


def _equalish_canonical(expectation: Expectation) -> str:
    """Canonical form of the fields participating in the suite's equality check.

    Mirrors ExpectationSuite._expectations_are_equalish exactly; the field
    selection is intentionally not changed here.
    """
    identity = expectation.dict(exclude=set(EQUALITY_EXCLUDED_FIELDS))
    return _canonical(identity)


def _entry_key(expectation: Expectation) -> str:
    """Stable identity of an entry.

    Two expectations are the same entry when they share expectation type and
    domain kwargs (the data the expectation applies to). Success/runtime kwargs
    are the entry's *content*, so changing them is a modification rather than a
    remove + add.
    """
    domain_kwargs = {
        key: expectation.configuration.kwargs.get(key) for key in expectation.domain_keys
    }
    return f"{expectation.expectation_type}|{_canonical(domain_kwargs)}"


def _is_equalish(left: Expectation, right: Expectation) -> bool:
    return left.expectation_type == right.expectation_type and _equalish_canonical(
        left
    ) == _equalish_canonical(right)


def _config_json(expectation: Expectation) -> dict:
    return expectation.configuration.to_json_dict()


def _normalize_excluded_value(value: Any) -> Any:
    """Falsy meta/notes are treated as absent so comparison is stable."""
    if value is None:
        return None
    if isinstance(value, (dict, list, str)) and len(value) == 0:
        return None
    return _jsonable(value)


def _field_changes_for_pair(old: Expectation, new: Expectation) -> list[FieldChange]:
    """Field-level changes in the fields that participate in equality."""
    old_identity = old.dict(exclude=set(EQUALITY_EXCLUDED_FIELDS))
    new_identity = new.dict(exclude=set(EQUALITY_EXCLUDED_FIELDS))
    old_kwargs_keys = set(_config_json(old).get("kwargs", {}).keys())
    new_kwargs_keys = set(_config_json(new).get("kwargs", {}).keys())
    kwargs_keys = old_kwargs_keys | new_kwargs_keys

    changes: list[FieldChange] = []
    for field_name in sorted(set(old_identity) | set(new_identity)):
        old_value = old_identity.get(field_name)
        new_value = new_identity.get(field_name)
        if _canonical(old_value) == _canonical(new_value):
            continue
        label = f"kwargs.{field_name}" if field_name in kwargs_keys else field_name
        changes.append(FieldChange(field=label, old=old_value, new=new_value))
    return changes


def _excluded_field_changes_for_pair(
    old: Expectation, new: Expectation, *, include_values: bool
) -> list[FieldChange]:
    """Changes in fields excluded from equality, surfaced separately."""
    old_config = _config_json(old)
    new_config = _config_json(new)
    changes: list[FieldChange] = []
    for field_name in REPORTED_EXCLUDED_FIELDS:
        old_value = _normalize_excluded_value(old_config.get(field_name))
        new_value = _normalize_excluded_value(new_config.get(field_name))
        if _canonical(old_value) != _canonical(new_value):
            changes.append(
                FieldChange(
                    field=field_name,
                    old=old_value,
                    new=new_value,
                    include_values=include_values,
                )
            )
    # id only carries meaning when both sides already have one.
    old_id, new_id = old_config.get("id"), new_config.get("id")
    if old_id and new_id and old_id != new_id:
        changes.append(
            FieldChange(field="id", old=old_id, new=new_id, include_values=include_values)
        )
    return changes


def _sort_changes(changes: list[ExpectationChange]) -> list[ExpectationChange]:
    changes.sort(key=lambda change: (change.key, change.expectation_id or ""))
    return changes


def _pair_old_expectation(
    candidates: list[Expectation], new_expectation: Expectation
) -> Optional[Expectation]:
    """Pick the stored entry that best matches a pending entry of the same domain."""
    # Prefer an exact equalish match (unchanged entry), then an id match, then
    # the first remaining entry of the same domain.
    candidates.sort(
        key=lambda exp: (
            0 if _is_equalish(exp, new_expectation) else 1,
            0 if exp.id and new_expectation.id and exp.id == new_expectation.id else 1,
            exp.id or "",
            id(exp),
        )
    )
    return candidates.pop(0)


def _classify_paired_change(
    *,
    key: str,
    old: Expectation,
    new: Expectation,
    include_values: bool,
) -> ExpectationChange:
    old_config = _config_json(old)
    new_config = _config_json(new)
    excluded_changes = _excluded_field_changes_for_pair(old, new, include_values=include_values)
    if not _is_equalish(old, new):
        field_changes = _field_changes_for_pair(old, new)
        if not include_values:
            field_changes = [
                FieldChange(field=change.field, include_values=False) for change in field_changes
            ]
        return ExpectationChange(
            change_type=ChangeType.MODIFIED,
            expectation_type=new.expectation_type,
            key=key,
            expectation_id=old.id,
            changed_fields=field_changes,
            excluded_field_changes=excluded_changes,
            old_config=old_config if include_values else None,
            new_config=new_config if include_values else None,
            include_configs=include_values,
        )
    return ExpectationChange(
        change_type=ChangeType.UNCHANGED,
        expectation_type=new.expectation_type,
        key=key,
        expectation_id=old.id,
        excluded_field_changes=excluded_changes,
        old_config=old_config if include_values else None,
        new_config=new_config if include_values else None,
        include_configs=include_values,
    )


def _suite_attribute_changes(
    current: ExpectationSuite, pending: ExpectationSuite, include_values: bool
) -> list[FieldChange]:
    changes: list[FieldChange] = []
    for attribute in ("suite_parameters", "notes", "meta"):
        old_value = _normalize_excluded_value(getattr(current, attribute, None))
        new_value = _normalize_excluded_value(getattr(pending, attribute, None))
        if _canonical(old_value) != _canonical(new_value):
            changes.append(
                FieldChange(field=attribute, old=old_value, new=new_value)
                if include_values
                else FieldChange(field=attribute, include_values=False)
            )
    changes.sort(key=lambda change: change.field)
    return changes


def _diff_suites(  # noqa: C901, PLR0913
    *,
    suite_name: str,
    current: Optional[ExpectationSuite],
    pending: Optional[ExpectationSuite],
    diff_detail: str = DiffDetail.FULL,
    references: Optional[list[ReferenceRef]] = None,
    mark_conflicts: bool = True,
) -> SuiteChangePreview:
    """Pure diff computation. Read-only; does not touch storage or memory."""
    include_values = diff_detail == DiffDetail.FULL
    references = sorted(references or [], key=lambda ref: (ref.kind, ref.name))

    old_expectations = list(current.expectations) if current else []
    new_expectations = list(pending.expectations) if pending else []

    old_pool: dict[str, list[Expectation]] = {}
    for expectation in old_expectations:
        old_pool.setdefault(_entry_key(expectation), []).append(expectation)

    added: list[ExpectationChange] = []
    removed: list[ExpectationChange] = []
    modified: list[ExpectationChange] = []
    unchanged: list[ExpectationChange] = []
    paired_old: set[int] = set()

    for new_expectation in new_expectations:
        key = _entry_key(new_expectation)
        candidates = old_pool.get(key)
        old_expectation = _pair_old_expectation(candidates, new_expectation) if candidates else None
        if old_expectation is None:
            added.append(
                ExpectationChange(
                    change_type=ChangeType.ADDED,
                    expectation_type=new_expectation.expectation_type,
                    key=key,
                    expectation_id=new_expectation.id,
                    new_config=_config_json(new_expectation) if include_values else None,
                    include_configs=include_values,
                )
            )
            continue

        paired_old.add(id(old_expectation))
        change = _classify_paired_change(
            key=key,
            old=old_expectation,
            new=new_expectation,
            include_values=include_values,
        )
        if change.change_type == ChangeType.MODIFIED:
            modified.append(change)
        else:
            unchanged.append(change)

    for old_expectation in old_expectations:
        if id(old_expectation) in paired_old:
            continue
        removed.append(
            ExpectationChange(
                change_type=ChangeType.REMOVED,
                expectation_type=old_expectation.expectation_type,
                key=_entry_key(old_expectation),
                expectation_id=old_expectation.id,
                old_config=_config_json(old_expectation) if include_values else None,
                include_configs=include_values,
            )
        )

    preview = SuiteChangePreview(
        suite_name=suite_name,
        added=_sort_changes(added),
        removed=_sort_changes(removed),
        modified=_sort_changes(modified),
        unchanged=_sort_changes(unchanged),
        affected_references=references,
    )

    # Anything modifying or removing entries is a conflict when references hold
    # the suite. Under the "ignore" policy affected_references are still
    # reported but individual changes are not marked as conflicts.
    if mark_conflicts and references and (preview.removed or preview.modified):
        for change in preview.conflicting_changes():
            change.conflicts = list(references)

    if current is not None and pending is not None:
        preview.suite_attribute_changes = _suite_attribute_changes(current, pending, include_values)

    return preview


# ---------------------------------------------------------------------------
# Reference discovery
# ---------------------------------------------------------------------------


def _safe_iter(factory: Any) -> list:
    try:
        return list(factory.all())
    except Exception:
        logger.warning(
            "Could not enumerate objects while looking up ExpectationSuite references; "
            "conflict detection may be incomplete.",
            exc_info=True,
        )
        return []


def _modern_checkpoint_references_suite(checkpoint: Any, suite_name: str) -> bool:
    # Modern checkpoints hold validation definitions, which in turn hold suites.
    try:
        for validation_definition in getattr(checkpoint, "validation_definitions", []) or []:
            suite = getattr(validation_definition, "suite", None)
            if suite is not None and getattr(suite, "name", None) == suite_name:
                return True
    except Exception:
        logger.debug("Unable to inspect checkpoint validation definitions.", exc_info=True)
    return False


def _legacy_validation_references_suite(validation: Any, suite_name: str) -> bool:
    if isinstance(validation, dict):
        if validation.get("expectation_suite_name") == suite_name:
            return True
        identifier = validation.get("expectation_suite_identifier")
        if isinstance(identifier, (tuple, list)) and suite_name in identifier:
            return True
        return bool(
            isinstance(identifier, dict) and identifier.get("expectation_suite_name") == suite_name
        )
    return getattr(validation, "expectation_suite_name", None) == suite_name


def _checkpoint_references_suite(checkpoint: Any, suite_name: str) -> bool:
    if _modern_checkpoint_references_suite(checkpoint, suite_name):
        return True
    # Legacy checkpoints may reference suites by name / identifier directly.
    try:
        validations = getattr(checkpoint, "validations", []) or []
        return any(
            _legacy_validation_references_suite(validation, suite_name)
            for validation in validations
        )
    except Exception:
        logger.debug("Unable to inspect legacy checkpoint validations.", exc_info=True)
        return False


def _discover_validation_definition_refs(project: Any, suite_name: str) -> list[ReferenceRef]:
    try:
        validation_factory = project.validation_definitions
    except Exception:
        return []
    refs: list[ReferenceRef] = []
    for validation_definition in _safe_iter(validation_factory):
        try:
            suite = getattr(validation_definition, "suite", None)
            if suite is not None and getattr(suite, "name", None) == suite_name:
                refs.append(
                    ReferenceRef(
                        kind="validation_definition",
                        name=getattr(validation_definition, "name", str(validation_definition)),
                    )
                )
        except Exception:
            continue
    return refs


def _discover_checkpoint_refs(project: Any, suite_name: str) -> list[ReferenceRef]:
    try:
        checkpoint_factory = project._checkpoints
    except Exception:
        return []
    refs: list[ReferenceRef] = []
    for checkpoint in _safe_iter(checkpoint_factory):
        try:
            if _checkpoint_references_suite(checkpoint, suite_name):
                refs.append(
                    ReferenceRef(
                        kind="checkpoint",
                        name=getattr(checkpoint, "name", str(checkpoint)),
                    )
                )
        except Exception:
            continue
    return refs


def discover_references(suite_name: str) -> list[ReferenceRef]:
    """Find validation definitions and checkpoints holding the named suite.

    Best effort: holders are read through the current project's stores; any
    failure is logged and yields no references rather than breaking a write.
    """
    try:
        project = project_manager._project
    except Exception:
        return []

    refs = _discover_validation_definition_refs(project, suite_name)
    refs.extend(_discover_checkpoint_refs(project, suite_name))

    # Deduplicate while keeping output stable.
    unique = {(ref.kind, ref.name): ref for ref in refs}
    return [unique[key] for key in sorted(unique)]


# ---------------------------------------------------------------------------
# Historical version storage
# ---------------------------------------------------------------------------


@dataclass
class SuiteVersionSummary:
    version: int
    suite_name: str
    created_at: str
    source: str
    operation: str
    restored_from_version: Optional[int]
    change_counts: dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class SuiteVersionRecord(SuiteVersionSummary):
    affected_references: list[dict] = field(default_factory=list)
    restored_from_at: Optional[str] = None
    suite: Optional[ExpectationSuite] = None

    @override
    def to_dict(self) -> dict:
        result = super().to_dict()
        result["affected_references"] = self.affected_references
        result["restored_from_at"] = self.restored_from_at
        result["suite"] = self.suite.to_json_dict() if self.suite is not None else None
        return result


class _PrefixedStoreBackend:
    """Proxy that namespaces tuple keys, used for custom/unknown backends."""

    def __init__(self, backend: Any, prefix: tuple[str, ...]):
        self._backend = backend
        self._prefix = prefix

    def _qualify(self, key: tuple[str, ...]) -> tuple[str, ...]:
        return self._prefix + tuple(key)

    def has_key(self, key: tuple[str, ...]) -> bool:
        return self._backend.has_key(self._qualify(key))

    def add(self, key: tuple[str, ...], value: str) -> Any:
        return self._backend.add(self._qualify(key), value)

    def get(self, key: tuple[str, ...]) -> str:
        return self._backend.get(self._qualify(key))

    def remove_key(self, key: tuple[str, ...]) -> None:
        self._backend.remove_key(self._qualify(key))

    def list_keys(self, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        qualified_prefix = self._qualify(prefix)
        results: list[tuple[str, ...]] = []
        for key in self._backend.list_keys(prefix=qualified_prefix):
            if key[: len(qualified_prefix)] == qualified_prefix:
                results.append(tuple(key[len(self._prefix) :]))
        return results


class SuiteHistoryStore:
    """Stores immutable per-suite version blobs alongside the ExpectationsStore.

    Each version is its own key (never overwritten), so concurrent writers
    cannot clobber each other's history: version numbers are claimed with the
    backend's create-only ``add`` and collisions are retried.
    """

    def __init__(self, expectations_store: ExpectationsStore) -> None:
        self._backend: Any = self._build_backend(expectations_store.store_backend)
        # In-process mutual exclusion. Cross-process safety comes from the
        # atomic O_CREAT|O_EXCL file claim on filesystem backends.
        self._claim_lock = threading.Lock()

    @staticmethod
    def _build_backend(backend: Any) -> Any:
        (
            GXCloudStoreBackend,
            InMemoryStoreBackend,
            TupleFilesystemStoreBackend,
            TupleStoreBackend,
        ) = _store_backend_classes()

        if isinstance(backend, GXCloudStoreBackend):
            raise SuiteChangeManagementError(  # noqa: TRY003
                "ExpectationSuite version retention is not supported against a GX Cloud backend."
            )
        if isinstance(backend, TupleFilesystemStoreBackend):
            # Sibling directory so the expectations store's recursive listing
            # never picks up version blobs as suites. normpath first: the
            # configured base directory may carry a trailing separator.
            expectations_base = Path(backend.full_base_directory)
            sibling_directory = str(expectations_base.parent / HISTORY_DIR_NAME)
            return TupleFilesystemStoreBackend(
                base_directory=sibling_directory,
                suppress_store_backend_id=True,
                store_name=HISTORY_DIR_NAME,
            )
        if isinstance(backend, InMemoryStoreBackend):
            # A separate in-memory backend per process (ephemeral contexts are
            # themselves per-process); sharing the expectations backend's dict
            # would pollute suite listing.
            return InMemoryStoreBackend(
                suppress_store_backend_id=True,
                store_name=HISTORY_DIR_NAME,
            )
        if isinstance(backend, TupleStoreBackend):
            config = {k: v for k, v in dict(backend.config).items() if v is not None}
            base = str(config.get("base_directory", ""))
            parent = posixpath.dirname(base)
            config["base_directory"] = (
                posixpath.join(parent, HISTORY_DIR_NAME) if parent else HISTORY_DIR_NAME
            )
            config.pop("store_name", None)
            config.setdefault("suppress_store_backend_id", True)
            try:
                return type(backend)(**config)
            except Exception as exc:
                raise SuiteChangeManagementError(  # noqa: TRY003
                    f"Cannot provision version storage for backend {type(backend).__name__}: {exc}"
                ) from exc
        return _PrefixedStoreBackend(backend, _HISTORY_KEY_PREFIX)

    @staticmethod
    def _version_key(suite_name: str, version: int) -> tuple[str, str]:
        return (suite_name, _VERSION_FILE_TEMPLATE.format(number=version))

    def list_version_numbers(self, suite_name: str) -> list[int]:
        numbers: set[int] = set()
        for key in self._backend.list_keys(prefix=(suite_name,)):
            match = _VERSION_FILE_RE.fullmatch(key[-1])
            if match:
                numbers.add(int(match.group(1)))
        return sorted(numbers)

    def read_version(self, suite_name: str, version: int) -> dict:
        raw = self._backend.get(self._version_key(suite_name, version))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _claim_version(self, suite_name: str, version: int, body: str) -> bool:
        """Atomically create the version blob.

        Returns False when another writer already claimed this version number.
        Filesystem backends use O_CREAT|O_EXCL (atomic across processes);
        other backends fall back to their create-only ``add``.
        """
        key = self._version_key(suite_name, version)
        _, _, TupleFilesystemStoreBackend, _ = _store_backend_classes()
        if isinstance(self._backend, TupleFilesystemStoreBackend):
            filepath = Path(
                self._backend.full_base_directory
            ) / self._backend._convert_key_to_filepath(key)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(filepath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body.encode("utf-8"))
            return True

        try:
            self._backend.add(key, body)
        except StoreBackendError:
            return False
        return True

    def write_version(self, suite_name: str, record: dict) -> int:
        body = json.dumps(_jsonable(record), indent=2, sort_keys=True, default=str)
        with self._claim_lock:
            for _ in range(_MAX_VERSION_CLAIM_ATTEMPTS):
                version_numbers = self.list_version_numbers(suite_name)
                next_version = (max(version_numbers) if version_numbers else 0) + 1
                if not self._claim_version(suite_name, next_version, body):
                    # Another process/thread claimed this number; re-read and retry.
                    continue
                self._prune(suite_name, record.get("retention_limit"), keep=next_version)
                return next_version
        raise SuiteChangeManagementError(  # noqa: TRY003
            f"Could not claim a new version number for ExpectationSuite '{suite_name}' "
            "after repeated concurrent-write collisions."
        )

    def _prune(
        self,
        suite_name: str,
        retention_limit: Optional[int],
        *,
        keep: Optional[int] = None,
    ) -> None:
        if not retention_limit or retention_limit < 1:
            return
        version_numbers = self.list_version_numbers(suite_name)
        stale = version_numbers[:-retention_limit]
        for version in stale:
            if version == keep:
                continue
            try:
                self._backend.remove_key(self._version_key(suite_name, version))
            except (StoreBackendError, KeyError):
                logger.debug("Version %s of suite '%s' was already pruned.", version, suite_name)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


@dataclass
class _PendingWrite:
    suite_name: str
    operation: str
    source: str
    record: dict
    affected_references: list[ReferenceRef]


def _parse_point_in_time(value: Union[dt.datetime, str]) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        parsed = dt.datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


class SuiteChangeManager:
    """Owns preview, retention and rollback behavior for one ExpectationsStore."""

    def __init__(
        self,
        store: ExpectationsStore,
        config: Optional[SuiteChangeManagementConfig] = None,
    ) -> None:
        self._store = store
        self._config = config or SuiteChangeManagementConfig()
        self._history: Optional[SuiteHistoryStore] = None
        self._suppression = threading.local()
        if self._config.enabled and _is_cloud_backend(store.store_backend):
            # Fail fast at factory construction time rather than mid-write; do
            # not create local history storage until the first retained write.
            raise SuiteChangeManagementError(  # noqa: TRY003
                "ExpectationSuite change management cannot be enabled against a GX Cloud backend."
            )

    @property
    def config(self) -> SuiteChangeManagementConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def configure(self, config: SuiteChangeManagementConfig) -> None:
        if config.enabled and _is_cloud_backend(self._store.store_backend):
            raise SuiteChangeManagementError(  # noqa: TRY003
                "ExpectationSuite change management cannot be enabled against a GX Cloud backend."
            )
        self._config = config
        self._history = None

    @property
    def _history_store(self) -> SuiteHistoryStore:
        # Lazily constructed so the default-off path creates no files/dirs.
        if self._history is None:
            self._history = SuiteHistoryStore(self._store)
        return self._history

    # -- suppression of nested write hooks (factory wrapping suite.save()) --

    @contextmanager
    def suppressed(self) -> Iterator[None]:
        count = getattr(self._suppression, "count", 0)
        self._suppression.count = count + 1
        try:
            yield
        finally:
            self._suppression.count = count

    def is_suppressed(self) -> bool:
        return getattr(self._suppression, "count", 0) > 0

    # -- current state helpers ---------------------------------------------

    def get_current(self, suite_name: str) -> Optional[ExpectationSuite]:
        key = self._store.get_key(name=suite_name, id=None)
        if not self._store.has_key(key=key):
            return None
        suite_dict = self._store.get(key=key)
        return self._store.deserialize_suite_dict(suite_dict=suite_dict)

    # -- preview ------------------------------------------------------------

    def preview(
        self,
        pending: ExpectationSuite,
        *,
        current: Optional[ExpectationSuite] = None,
    ) -> SuiteChangePreview:
        if current is None:
            current = self.get_current(pending.name)
        references = discover_references(pending.name)
        return _diff_suites(
            suite_name=pending.name,
            current=current,
            pending=pending,
            diff_detail=self._config.diff_detail,
            references=references,
            mark_conflicts=self._config.conflict_policy != ConflictPolicy.IGNORE,
        )

    # -- write/delete retention ---------------------------------------------

    def begin_write(
        self,
        *,
        pending: Optional[ExpectationSuite],
        current: Optional[ExpectationSuite],
        source: str,
    ) -> Optional[_PendingWrite]:
        """Capture the pre-write snapshot. Returns None when nothing is retained.

        Raises before any mutation when conflict_policy is ``"raise"``.
        """
        if not self.enabled or self.is_suppressed() or current is None or pending is None:
            return None
        preview = self.preview(pending, current=current)
        if not preview.has_changes:
            # Identical content is still reported via preview(); writes create
            # no new history entry.
            return None
        if self._config.conflict_policy == ConflictPolicy.RAISE and preview.has_conflicts:
            raise SuiteReferenceConflictError(
                name=pending.name,
                affected_references=[ref.to_dict() for ref in preview.affected_references],
            )
        return self._build_pending_write(
            current=current,
            preview=preview,
            operation=ChangeOperation.WRITE,
            source=source,
        )

    def begin_delete(self, *, current: ExpectationSuite, source: str) -> Optional[_PendingWrite]:
        if not self.enabled or self.is_suppressed():
            return None
        references = discover_references(current.name)
        preview = _diff_suites(
            suite_name=current.name,
            current=current,
            pending=None,
            diff_detail=self._config.diff_detail,
            references=references,
            mark_conflicts=self._config.conflict_policy != ConflictPolicy.IGNORE,
        )
        if self._config.conflict_policy == ConflictPolicy.RAISE and preview.has_conflicts:
            raise SuiteReferenceConflictError(
                name=current.name,
                affected_references=[ref.to_dict() for ref in preview.affected_references],
            )
        return self._build_pending_write(
            current=current,
            preview=preview,
            operation=ChangeOperation.DELETE,
            source=source,
        )

    def _build_pending_write(
        self,
        *,
        current: ExpectationSuite,
        preview: SuiteChangePreview,
        operation: str,
        source: str,
    ) -> _PendingWrite:
        record = {
            "suite_name": current.name,
            "source": source,
            "operation": operation,
            "change_counts": preview.counts,
            "affected_references": [ref.to_dict() for ref in preview.affected_references],
            "retention_limit": self._config.retention_limit,
            "restored_from_version": None,
            "restored_from_at": None,
            "suite": current.to_json_dict(),
        }
        return _PendingWrite(
            suite_name=current.name,
            operation=operation,
            source=source,
            record=record,
            affected_references=list(preview.affected_references),
        )

    def commit(self, token: Optional[_PendingWrite]) -> Optional[int]:
        """Persist the captured old version after the main write succeeded."""
        if token is None:
            return None
        token.record["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        return self._history_store.write_version(token.suite_name, token.record)

    def discard(self, token: Optional[_PendingWrite]) -> None:
        # Snapshot is discarded simply by not persisting it.
        return None

    # -- read history -------------------------------------------------------

    def list_versions(self, suite_name: str) -> list[SuiteVersionSummary]:
        if not self.enabled:
            return []
        summaries: list[SuiteVersionSummary] = []
        for version in self._history_store.list_version_numbers(suite_name):
            record = self._history_store.read_version(suite_name, version)
            summaries.append(
                SuiteVersionSummary(
                    version=version,
                    suite_name=suite_name,
                    created_at=record.get("created_at", ""),
                    source=record.get("source", ""),
                    operation=record.get("operation", ""),
                    restored_from_version=record.get("restored_from_version"),
                    change_counts=record.get("change_counts", {}),
                )
            )
        summaries.sort(key=lambda summary: summary.version)
        return summaries

    def get_version(
        self,
        suite_name: str,
        *,
        version: Optional[int] = None,
        at: Union[dt.datetime, str, None] = None,
    ) -> SuiteVersionRecord:
        self._validate_version_selector(version=version, at=at)
        numbers = self._history_store.list_version_numbers(suite_name)
        if not numbers:
            raise SuiteVersionNotFoundError(
                name=suite_name, version=version, at=str(at) if at else None
            )
        if version is not None:
            if version not in numbers:
                raise SuiteVersionNotFoundError(
                    name=suite_name, version=version, retained_versions=numbers
                )
            target_number: int = version
        else:
            # Selector validation guarantees exactly one of version/at is set.
            assert at is not None
            target_time = _parse_point_in_time(at)
            selected: Optional[int] = None
            for number in numbers:
                record = self._history_store.read_version(suite_name, number)
                created_at_raw = record.get("created_at")
                if not created_at_raw:
                    continue
                created_at = _parse_point_in_time(created_at_raw)
                if created_at <= target_time:
                    selected = number
            if selected is None:
                raise SuiteVersionNotFoundError(
                    name=suite_name,
                    at=str(at),
                    retained_versions=numbers,
                )
            target_number = selected

        record = self._history_store.read_version(suite_name, target_number)
        suite = self._store.deserialize_suite_dict(suite_dict=deepcopy(record["suite"]))
        return SuiteVersionRecord(
            version=target_number,
            suite_name=suite_name,
            created_at=record.get("created_at", ""),
            source=record.get("source", ""),
            operation=record.get("operation", ""),
            restored_from_version=record.get("restored_from_version"),
            change_counts=record.get("change_counts", {}),
            affected_references=record.get("affected_references", []),
            restored_from_at=record.get("restored_from_at"),
            suite=suite,
        )

    @staticmethod
    def _validate_version_selector(
        *, version: Optional[int], at: Union[dt.datetime, str, None]
    ) -> None:
        if version is None and at is None:
            raise ValueError(  # noqa: TRY003
                "Exactly one of 'version' or 'at' must be provided."
            )
        if version is not None and at is not None:
            raise ValueError(  # noqa: TRY003
                "Only one of 'version' or 'at' may be provided, not both."
            )
        if version is not None and (isinstance(version, bool) or not isinstance(version, int)):
            raise ValueError(f"'version' must be an integer, got {version!r}.")  # noqa: TRY003

    # -- rollback -----------------------------------------------------------

    @staticmethod
    def _stabilize_restored_ids(*, current: ExpectationSuite, restored: ExpectationSuite) -> None:
        """Reuse live identifiers for entries unchanged after rollback."""
        current_pool: dict[str, list[Expectation]] = {}
        for expectation in current.expectations:
            current_pool.setdefault(_entry_key(expectation), []).append(expectation)
        for expectation in restored.expectations:
            candidates = current_pool.get(_entry_key(expectation))
            if not candidates:
                continue
            candidates.sort(
                key=lambda exp: (
                    0 if _is_equalish(exp, expectation) else 1,
                    exp.id or "",
                    id(exp),
                )
            )
            match = candidates.pop(0)
            if match.id:
                expectation.id = match.id

    def rollback(
        self,
        suite_name: str,
        *,
        version: Optional[int] = None,
        at: Union[dt.datetime, str, None] = None,
        source: str = ChangeSource.ROLLBACK,
    ) -> ExpectationSuite:
        if not self.enabled:
            raise SuiteChangeManagementError(  # noqa: TRY003
                "Cannot roll back ExpectationSuite versions because change management is "
                "disabled. Enable it on the SuiteFactory via configure_change_management()."
            )
        target = self.get_version(suite_name, version=version, at=at)
        restored = self._store.deserialize_suite_dict(
            suite_dict=deepcopy(target.to_dict()["suite"])
        )

        current = self.get_current(suite_name)

        # Conflict handling between what is live and what rollback restores.
        references = discover_references(suite_name)
        preview = _diff_suites(
            suite_name=suite_name,
            current=current,
            pending=restored,
            diff_detail=self._config.diff_detail,
            references=references,
            mark_conflicts=self._config.conflict_policy != ConflictPolicy.IGNORE,
        )
        if self._config.conflict_policy == ConflictPolicy.RAISE and preview.has_conflicts:
            raise SuiteReferenceConflictError(
                name=suite_name,
                affected_references=[ref.to_dict() for ref in preview.affected_references],
            )

        rollback_record: dict
        if current is not None:
            # Entries unchanged relative to the live suite keep their live
            # identifiers; entries the live suite no longer has carry the
            # identifiers from the historical snapshot.
            self._stabilize_restored_ids(current=current, restored=restored)
            # Preserve the version being replaced; the rollback is itself a
            # version entry, which is what makes rollbacks re-rollback-able.
            rollback_record = {
                "suite_name": suite_name,
                "source": source,
                "operation": ChangeOperation.ROLLBACK,
                "change_counts": preview.counts,
                "affected_references": [ref.to_dict() for ref in preview.affected_references],
                "retention_limit": self._config.retention_limit,
                "restored_from_version": target.version,
                "restored_from_at": target.created_at,
                "suite": current.to_json_dict(),
            }
            restored.id = current.id
            key = self._store.get_key(name=suite_name, id=current.id)
            self._store.update(key=key, value=restored)
        else:
            # Suite had been deleted; re-create it from the historical version.
            rollback_record = {
                "suite_name": suite_name,
                "source": source,
                "operation": ChangeOperation.ROLLBACK,
                "change_counts": preview.counts,
                "affected_references": [ref.to_dict() for ref in preview.affected_references],
                "retention_limit": self._config.retention_limit,
                "restored_from_version": target.version,
                "restored_from_at": target.created_at,
                "suite": {"name": suite_name, "expectations": []},
            }
            key = self._store.get_key(name=suite_name, id=None)
            self._store.add(key=key, value=restored)

        rollback_record["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self._history_store.write_version(suite_name, rollback_record)

        return self.get_current(suite_name)  # type: ignore[return-value]

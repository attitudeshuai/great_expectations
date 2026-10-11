from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

_PRIMITIVES = (str, int, float, bool, type(None))


class ConfigCheckupError(Exception):
    """Raised when a config checkup cannot even be started.

    Used for bad checkup inputs (unknown strictness, a config file that cannot be
    read, content that is not valid YAML, or a YAML document that is not a mapping).
    It is never raised for problems found *inside* an otherwise readable config;
    those are collected in the returned ``ConfigCheckupReport``.
    """


class CheckupCategory(str, enum.Enum):
    """The four independently-collected groups of checkup findings."""

    VERSION = "version"
    STRUCTURE = "structure"
    REFERENCE = "reference"
    LOADING = "loading"


# Fixed display/iteration order; independent of dict insertion order.
CATEGORY_ORDER: Mapping[CheckupCategory, int] = {
    CheckupCategory.VERSION: 0,
    CheckupCategory.STRUCTURE: 1,
    CheckupCategory.REFERENCE: 2,
    CheckupCategory.LOADING: 3,
}


class Severity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Strictness(str, enum.Enum):
    """How advisory (warning-level) findings affect the checkup verdict.

    LENIENT: warnings are reported but do not fail the checkup.
    STRICT:  every warning is escalated to an error and fails the checkup.
    """

    LENIENT = "lenient"
    STRICT = "strict"


class IssueCode:
    """Stable, machine-readable identifiers for every checkup finding."""

    # version
    VERSION_MISSING = "version.missing"
    VERSION_INVALID_TYPE = "version.invalid_type"
    VERSION_TOO_LOW = "version.too_low"
    VERSION_LEGACY_SERIES = "version.legacy_series"
    VERSION_TOO_HIGH = "version.too_high"
    VERSION_OUTDATED_SUPPORTED = "version.outdated_supported"

    # structure
    STRUCTURE_UNKNOWN_KEY = "structure.unknown_key"
    STRUCTURE_MISSING_KEY = "structure.missing_key"
    STRUCTURE_WRONG_TYPE = "structure.wrong_type"

    # reference
    REFERENCE_UNDEFINED_STORE = "reference.undefined_store"
    REFERENCE_UNUSED_STORE = "reference.unused_store"

    # loading
    LOADING_MODULE_NOT_FOUND = "loading.module_not_found"
    LOADING_CLASS_NOT_FOUND = "loading.class_not_found"
    LOADING_ABSTRACT_CLASS = "loading.abstract_class"
    LOADING_INSTANTIATION_FAILED = "loading.instantiation_failed"
    LOADING_DATASOURCE_TYPE_UNKNOWN = "loading.datasource_type_unknown"
    LOADING_DATASOURCE_INSTANTIATION_FAILED = "loading.datasource_instantiation_failed"


@dataclass(frozen=True)
class ConfigIssue:
    """A single checkup finding.

    Args:
        category: Which of the four check groups produced the finding.
        severity: error, warning (advisory), or info.
        code: Stable machine-readable identifier (see ``IssueCode``).
        location: Structured path into the config (e.g.
            ``("stores", "my_store", "class_name")``).
        message: Human-readable explanation.
        details: Additional structured, JSON-serializable context.
    """

    category: CheckupCategory
    severity: Severity
    code: str
    location: tuple[str, ...]
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "code": self.code,
            "location": list(self.location),
            "message": self.message,
            "details": _to_jsonable(self.details),
        }


@dataclass(frozen=True)
class VersionVerdict:
    """The explicit conclusion about the config's ``config_version`` value."""

    status: str
    observed: Any
    minimum_supported: int
    current: int
    message: str
    migration_guide_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observed": _to_jsonable(self.observed),
            "minimum_supported": self.minimum_supported,
            "current": self.current,
            "message": self.message,
            "migration_guide_url": self.migration_guide_url,
        }


def _to_jsonable(value: Any) -> Any:
    """Coerce arbitrary values into JSON-native types for machine output."""

    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items(), key=_sort_key)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _sort_key(item: Any) -> str:
    key = item[0]
    return key if isinstance(key, str) else str(key)


@dataclass(frozen=True)
class ConfigCheckupReport:
    """The complete, read-only result of a config checkup run.

    Issues are stored in a fixed deterministic order so repeated runs over the
    same config produce byte-identical machine output.
    """

    strictness: Strictness
    config_path: str | None
    version_verdict: VersionVerdict
    issues: tuple[ConfigIssue, ...]

    @property
    def errors(self) -> tuple[ConfigIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ConfigIssue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def success(self) -> bool:
        """True when no error-severity findings remain (warnings do not count)."""

        return not self.errors

    def issues_by_category(self) -> dict[CheckupCategory, tuple[ConfigIssue, ...]]:
        grouped: dict[CheckupCategory, list[ConfigIssue]] = {
            category: [] for category in sorted(CATEGORY_ORDER, key=CATEGORY_ORDER.get)
        }
        for issue in self.issues:
            grouped[issue.category].append(issue)
        return {category: tuple(items) for category, items in grouped.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "strictness": self.strictness.value,
            "config_path": self.config_path,
            "version_verdict": self.version_verdict.to_dict(),
            "issue_counts": {
                Severity.ERROR.value: len(self.errors),
                Severity.WARNING.value: len(self.warnings),
                Severity.INFO.value: sum(1 for i in self.issues if i.severity is Severity.INFO),
                "total": len(self.issues),
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render_text(self) -> str:
        """Render a human-readable summary grouped by category."""

        target = self.config_path or "in-memory config"
        verdict = "PASSED" if self.success else "FAILED"
        lines = [
            f"GX Config Checkup: {target}",
            f"Result: {verdict} ({len(self.errors)} error(s), {len(self.warnings)} warning(s))",
            f"Strictness: {self.strictness.value}",
            f"Config version: {self.version_verdict.message}",
        ]

        for category, category_issues in self.issues_by_category().items():
            lines.append("")
            lines.append(f"[{category.value}]")
            if not category_issues:
                lines.append("  (no issues)")
                continue
            for issue in category_issues:
                location = ".".join(issue.location) if issue.location else "<root>"
                lines.append(f"  - [{issue.severity.value.upper()}] {location} ({issue.code})")
                lines.append(f"      {issue.message}")
                for key in sorted(issue.details):
                    lines.append(f"      {key}: {issue.details[key]}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render_text()

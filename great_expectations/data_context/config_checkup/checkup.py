from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import os
import pathlib
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.constructor import DuplicateKeyError

# Registering the fluent package populates ``DataSourceManager.type_lookup``.
import great_expectations.datasource.fluent  # noqa: F401
from great_expectations.data_context.config_checkup.models import (
    CATEGORY_ORDER,
    CheckupCategory,
    ConfigCheckupError,
    ConfigCheckupReport,
    ConfigIssue,
    IssueCode,
    Severity,
    Strictness,
    VersionVerdict,
)
from great_expectations.data_context.constants import (
    CURRENT_GX_CONFIG_VERSION,
    MINIMUM_SUPPORTED_CONFIG_VERSION,
)
from great_expectations.data_context.types.base import (
    DataContextConfig,
    DataContextConfigSchema,
)
from great_expectations.data_context.util import instantiate_class_from_config
from great_expectations.datasource.fluent.sources import DataSourceManager

yaml = YAML()

DEFAULT_STORE_MODULE = "great_expectations.data_context.store"
DEFAULT_SITE_MODULE = "great_expectations.render.renderer.site_builder"
DEFAULT_RENDERER_MODULE = "great_expectations.render.renderer"

# Mirrors the guidance linked by the existing load-path version errors.
MIGRATION_GUIDE_URL = (
    "https://docs.greatexpectations.io/docs/guides/miscellaneous/"
    "migration_guide#migrating-to-the-batch-request-v3-api"
)

# Top-level keys that are genuinely required at context-construction time.
# ``config_version`` is intentionally omitted: the version check group owns it.
REQUIRED_TOP_LEVEL_KEYS = (
    "expectations_store_name",
    "validation_results_store_name",
    "stores",
)

STORE_NAME_KEYS = (
    "expectations_store_name",
    "validation_results_store_name",
    "checkpoint_store_name",
)

# Store names the context always consumes by convention even though no top-level
# ``*_store_name`` key points at them; they must not be reported as unused.
IMPLICITLY_REFERENCED_STORES = ("validation_definition_store",)

# Expected value type and nullability for each top-level key (excluding
# ``config_version``, which is owned by the version check).
_TOP_LEVEL_TYPE_SPEC: Mapping[str, tuple[Any, bool]] = {
    "fluent_datasources": (Mapping, True),
    "expectations_store_name": (str, False),
    "validation_results_store_name": (str, False),
    "checkpoint_store_name": (str, True),
    "plugins_directory": (str, True),
    "stores": (Mapping, False),
    "data_docs_sites": (Mapping, True),
    "config_variables_file_path": (str, True),
    "analytics_enabled": (bool, True),
    "data_context_id": (str, True),
    "progress_bars": (Mapping, True),
}

_TYPE_LABELS: Mapping[Any, str] = {
    str: "string",
    bool: "boolean",
    Mapping: "mapping",
    (int, float): "number",
}


def run_config_checkup(
    config: str | os.PathLike | Mapping[str, Any] | DataContextConfig,
    *,
    strictness: str | Strictness = Strictness.LENIENT,
) -> ConfigCheckupReport:
    """Run a read-only, all-problems-at-once checkup of a project config.

    Accepts either a path to a ``great_expectations.yml`` file, an already-parsed
    config mapping, or a ``DataContextConfig``. No files are written, no
    long-lived stores or datasources are constructed, and the on-disk config is
    never modified.

    Args:
        config: Path to the YAML config, a parsed mapping, or a DataContextConfig.
        strictness: ``"lenient"`` (default) reports warnings without failing;
            ``"strict"`` escalates every warning to an error.

    Returns:
        A ``ConfigCheckupReport`` with findings grouped by category and a
        machine-readable / human-readable rendering.

    Raises:
        ConfigCheckupError: if the strictness value is invalid, the config file
            cannot be read, the content is not valid YAML, or the YAML document
            does not contain a top-level mapping.
    """
    strict = _coerce_strictness(strictness)
    raw, config_path = _load_config_input(config)

    # The temp root absorbs the (rare) constructor-time directory creation done
    # by some filesystem backends so the user's project directory is never touched.
    with tempfile.TemporaryDirectory(prefix="gx_config_checkup_") as probe_root:
        checkup = _ConfigCheckup(
            raw=raw,
            config_path=config_path,
            strictness=strict,
            probe_root=probe_root,
        )
        issues = checkup.run()

    issues = _finalize_issues(issues, strictness=strict)
    return ConfigCheckupReport(
        strictness=strict,
        config_path=config_path,
        version_verdict=checkup.version_verdict,
        issues=tuple(issues),
    )


def _coerce_strictness(strictness: str | Strictness) -> Strictness:
    if isinstance(strictness, Strictness):
        return strictness
    if isinstance(strictness, str):
        try:
            return Strictness(strictness)
        except ValueError:
            pass
    raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
        f"Invalid strictness {strictness!r}: expected one of {[s.value for s in Strictness]}."
    )


def _load_config_input(  # noqa: C901 # FIXME CoP
    config: str | os.PathLike | Mapping[str, Any] | DataContextConfig,
) -> tuple[dict[str, Any], str | None]:
    if isinstance(config, DataContextConfig):
        return dict(config.to_json_dict()), None

    if isinstance(config, Mapping):
        return dict(config), None

    if isinstance(config, (str, os.PathLike)):
        path = pathlib.Path(os.fspath(config))
        if not path.exists():
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file does not exist: {path}. "
                "Please provide the path to an existing great_expectations.yml file."
            )
        if not path.is_file():
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration path is not a file: {path}."
            )

        try:
            with path.open(encoding="utf-8") as f:
                loaded = yaml.load(f)
        except DuplicateKeyError:
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file {path} contains a duplicate key; "
                "each YAML key may only appear once per mapping."
            ) from None
        except YAMLError as err:
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file {path} is not a valid YAML file:\n{err}"
            ) from None
        except UnicodeDecodeError as err:
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file {path} is not valid UTF-8 and could not be read: {err}"
            ) from None
        except OSError as err:
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file {path} could not be read: {err}"
            ) from None

        if not isinstance(loaded, Mapping):
            raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
                f"Configuration file {path} must contain a YAML mapping at the top "
                f"level, but got a {type(loaded).__name__}."
            )
        return dict(loaded), str(path)

    raise ConfigCheckupError(  # noqa: TRY003 # FIXME CoP
        "Unsupported config input: expected a file path (str or pathlib.Path), "
        "a parsed config mapping, or a DataContextConfig instance, but got "
        f"{type(config).__name__}."
    )


@dataclass(frozen=True)
class _ResolvedClass:
    cls: type | None
    failure_code: str | None = None
    failure_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.cls is not None


class _ConfigCheckup:
    """Collects findings across all four independent check groups."""

    def __init__(
        self,
        raw: Mapping[str, Any],
        config_path: str | None,
        strictness: Strictness,
        probe_root: str,
    ) -> None:
        self._raw = raw
        self._config_path = config_path
        self._strictness = strictness
        self._probe_root = probe_root
        self._issues: list[ConfigIssue] = []
        self.version_verdict: VersionVerdict = VersionVerdict(
            status="missing",
            observed=None,
            minimum_supported=MINIMUM_SUPPORTED_CONFIG_VERSION,
            current=CURRENT_GX_CONFIG_VERSION,
            message=(
                "config_version is missing; supported versions are "
                f"{MINIMUM_SUPPORTED_CONFIG_VERSION}-{CURRENT_GX_CONFIG_VERSION}"
            ),
        )

    def run(self) -> list[ConfigIssue]:
        # Every group runs unconditionally against the raw mapping; a failure in
        # one group never short-circuits the others.
        self._check_version()
        self._check_structure()
        self._check_references()
        self._check_loading()
        return list(self._issues)

    # ------------------------------------------------------------------
    # Issue collection helper
    # ------------------------------------------------------------------

    def _issue(  # noqa: PLR0913 # FIXME CoP
        self,
        category: CheckupCategory,
        code: str,
        location: tuple[str, ...],
        message: str,
        *,
        severity: Severity = Severity.ERROR,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._issues.append(
            ConfigIssue(
                category=category,
                severity=severity,
                code=code,
                location=tuple(location),
                message=message,
                details=dict(sorted((details or {}).items())),
            )
        )

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------

    def _check_version(self) -> None:
        raw = self._raw
        minimum = MINIMUM_SUPPORTED_CONFIG_VERSION
        current = CURRENT_GX_CONFIG_VERSION

        if "config_version" not in raw:
            message = (
                "The key `config_version` is missing; supported config versions are "
                f"{minimum}-{current} (current: {current})."
            )
            self.version_verdict = VersionVerdict(
                status="missing",
                observed=None,
                minimum_supported=minimum,
                current=current,
                message=message,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_MISSING,
                ("config_version",),
                message,
            )
            return

        observed = raw["config_version"]
        number_like = isinstance(observed, (int, float)) and not isinstance(observed, bool)
        if not number_like:
            message = (
                "The key `config_version` must be a number, but got "
                f"{type(observed).__name__} ({observed!r}); supported config versions "
                f"are {minimum}-{current} (current: {current})."
            )
            self.version_verdict = VersionVerdict(
                status="invalid_type",
                observed=observed,
                minimum_supported=minimum,
                current=current,
                message=message,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_INVALID_TYPE,
                ("config_version",),
                message,
                details={"actual_type": type(observed).__name__},
            )
            return

        if observed == 0 and self._has_legacy_validation_results_store():
            message = (
                "You appear to be using a config version from the 0.7.x series. "
                "This version is no longer supported. Please see the migration guide: "
                f"{MIGRATION_GUIDE_URL}"
            )
            self.version_verdict = VersionVerdict(
                status="legacy_0_7_x",
                observed=observed,
                minimum_supported=minimum,
                current=current,
                message=message,
                migration_guide_url=MIGRATION_GUIDE_URL,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_LEGACY_SERIES,
                ("config_version",),
                message,
                details={"series": "0.7.x", "migration_guide_url": MIGRATION_GUIDE_URL},
            )
            return

        if observed < minimum:
            message = (
                f"You appear to have an invalid config version ({observed}). "
                f"The version number must be at least {minimum} and the current "
                f"version is {current}. Please see the migration guide: "
                f"{MIGRATION_GUIDE_URL}"
            )
            self.version_verdict = VersionVerdict(
                status="too_low",
                observed=observed,
                minimum_supported=minimum,
                current=current,
                message=message,
                migration_guide_url=MIGRATION_GUIDE_URL,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_TOO_LOW,
                ("config_version",),
                message,
                details={
                    "minimum_supported": minimum,
                    "current": current,
                    "migration_guide_url": MIGRATION_GUIDE_URL,
                },
            )
            return

        if observed > current:
            message = (
                f"You appear to have an invalid config version ({observed}). "
                f"Supported config versions are {minimum}-{current}; the maximum "
                f"valid version is {current}."
            )
            self.version_verdict = VersionVerdict(
                status="too_high",
                observed=observed,
                minimum_supported=minimum,
                current=current,
                message=message,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_TOO_HIGH,
                ("config_version",),
                message,
                details={"maximum_supported": current, "minimum_supported": minimum},
            )
            return

        if observed < current:
            message = (
                f"Config version {observed} is supported (supported range: "
                f"{minimum}-{current}), but newer config version {current} is available."
            )
            self.version_verdict = VersionVerdict(
                status="supported_outdated",
                observed=observed,
                minimum_supported=minimum,
                current=current,
                message=message,
            )
            self._issue(
                CheckupCategory.VERSION,
                IssueCode.VERSION_OUTDATED_SUPPORTED,
                ("config_version",),
                message,
                severity=Severity.WARNING,
                details={"minimum_supported": minimum, "current": current},
            )
            return

        message = (
            f"Config version {observed} is supported (supported range: "
            f"{minimum}-{current}; current: {current})."
        )
        self.version_verdict = VersionVerdict(
            status="supported",
            observed=observed,
            minimum_supported=minimum,
            current=current,
            message=message,
        )

    def _has_legacy_validation_results_store(self) -> bool:
        stores = self._raw.get("stores")
        if not isinstance(stores, Mapping):
            return False
        return any(
            isinstance(store_cfg, Mapping)
            and store_cfg.get("class_name") == "ValidationResultsStore"
            for store_cfg in stores.values()
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def _check_structure(self) -> None:
        raw = self._raw
        schema_fields = set(DataContextConfigSchema().fields)

        for key in sorted(raw):
            if key not in schema_fields:
                self._issue(
                    CheckupCategory.STRUCTURE,
                    IssueCode.STRUCTURE_UNKNOWN_KEY,
                    (str(key),),
                    f"Unknown top-level config key `{key}`; it is not a recognized "
                    "DataContext config key.",
                    details={"key": str(key)},
                )

        for key in REQUIRED_TOP_LEVEL_KEYS:
            if key not in raw:
                self._issue(
                    CheckupCategory.STRUCTURE,
                    IssueCode.STRUCTURE_MISSING_KEY,
                    (key,),
                    f"Required top-level config key `{key}` is missing.",
                    details={"key": key},
                )

        for key in sorted(_TOP_LEVEL_TYPE_SPEC):
            if key not in raw:
                continue
            expected, nullable = _TOP_LEVEL_TYPE_SPEC[key]
            self._check_value_type(
                value=raw[key],
                expected=expected,
                nullable=nullable,
                location=(key,),
            )

        self._check_stores_structure()
        self._check_sites_structure()
        self._check_fluent_datasources_structure()
        self._check_progress_bars_structure()

    def _check_value_type(
        self,
        value: Any,
        expected: Any,
        nullable: bool,
        location: tuple[str, ...],
    ) -> None:
        if value is None:
            if nullable:
                return
            self._wrong_type(location, expected, value)
            return
        if not _matches_type(value, expected):
            self._wrong_type(location, expected, value)

    def _wrong_type(self, location: tuple[str, ...], expected: Any, value: Any) -> None:
        expected_label = _TYPE_LABELS.get(expected, getattr(expected, "__name__", "value"))
        key_name = ".".join(location)
        self._issue(
            CheckupCategory.STRUCTURE,
            IssueCode.STRUCTURE_WRONG_TYPE,
            location,
            f"Config key `{key_name}` has an invalid value type: expected "
            f"{expected_label}, got {type(value).__name__}.",
            details={
                "expected": expected_label,
                "actual": type(value).__name__,
            },
        )

    def _check_stores_structure(self) -> None:
        stores = self._raw.get("stores")
        if not isinstance(stores, Mapping):
            return
        for name in sorted(stores):
            location = ("stores", name)
            store_cfg = stores[name]
            if not isinstance(store_cfg, Mapping):
                self._wrong_type(location, Mapping, store_cfg)
                continue

            self._require_str(store_cfg, "class_name", location, required=True)
            self._require_str(store_cfg, "module_name", location, required=False)

            if "store_backend" in store_cfg:
                backend_cfg = store_cfg["store_backend"]
                backend_location = location + ("store_backend",)
                if not isinstance(backend_cfg, Mapping):
                    self._wrong_type(backend_location, Mapping, backend_cfg)
                else:
                    self._require_str(backend_cfg, "class_name", backend_location, required=True)
                    self._require_str(backend_cfg, "module_name", backend_location, required=False)

    def _check_sites_structure(self) -> None:  # noqa: C901, PLR0912 # FIXME CoP
        sites = self._raw.get("data_docs_sites")
        if not isinstance(sites, Mapping):
            return
        for site_name in sorted(sites):
            location = ("data_docs_sites", site_name)
            site_cfg = sites[site_name]
            if not isinstance(site_cfg, Mapping):
                self._wrong_type(location, Mapping, site_cfg)
                continue

            self._require_str(site_cfg, "class_name", location, required=False)
            self._require_str(site_cfg, "module_name", location, required=False)

            if "store_backend" in site_cfg:
                backend_location = location + ("store_backend",)
                backend_cfg = site_cfg["store_backend"]
                if not isinstance(backend_cfg, Mapping):
                    self._wrong_type(backend_location, Mapping, backend_cfg)
                else:
                    self._require_str(backend_cfg, "class_name", backend_location, required=False)
                    self._require_str(backend_cfg, "module_name", backend_location, required=False)

            if "site_index_builder" in site_cfg:
                index_location = location + ("site_index_builder",)
                index_cfg = site_cfg["site_index_builder"]
                if not isinstance(index_cfg, Mapping):
                    self._wrong_type(index_location, Mapping, index_cfg)
                else:
                    self._require_str(index_cfg, "class_name", index_location, required=False)
                    self._require_str(index_cfg, "module_name", index_location, required=False)

            sections = site_cfg.get("site_section_builders")
            if isinstance(sections, Mapping):
                for section_name in sorted(sections):
                    section_location = location + (
                        "site_section_builders",
                        section_name,
                    )
                    section_cfg = sections[section_name]
                    if not isinstance(section_cfg, Mapping):
                        # Falsy scalars (e.g. False) are used to disable a section.
                        if section_cfg:
                            self._wrong_type(section_location, Mapping, section_cfg)
                        continue
                    self._require_str(section_cfg, "class_name", section_location, required=False)
                    self._require_str(section_cfg, "module_name", section_location, required=False)
                    self._require_str(
                        section_cfg,
                        "source_store_name",
                        section_location,
                        required=False,
                    )

    def _check_fluent_datasources_structure(self) -> None:
        datasources = self._raw.get("fluent_datasources")
        if not isinstance(datasources, Mapping):
            return
        for name in sorted(datasources):
            location = ("fluent_datasources", name)
            ds_cfg = datasources[name]
            if not isinstance(ds_cfg, Mapping):
                self._wrong_type(location, Mapping, ds_cfg)
                continue
            self._require_str(ds_cfg, "type", location, required=True)

    def _check_progress_bars_structure(self) -> None:
        progress_bars = self._raw.get("progress_bars")
        if not isinstance(progress_bars, Mapping):
            return
        for key in sorted(progress_bars):
            if key not in ("globally", "metric_calculations"):
                self._issue(
                    CheckupCategory.STRUCTURE,
                    IssueCode.STRUCTURE_UNKNOWN_KEY,
                    ("progress_bars", str(key)),
                    f"Unknown `progress_bars` key `{key}`; supported keys are "
                    "`globally` and `metric_calculations`.",
                    details={"key": str(key)},
                )
                continue
            value = progress_bars[key]
            if value is not None and not isinstance(value, bool):
                self._wrong_type(("progress_bars", str(key)), bool, value)

    def _require_str(
        self,
        cfg: Mapping[str, Any],
        key: str,
        parent_location: tuple[str, ...],
        required: bool,
    ) -> None:
        location = parent_location + (key,)
        if key not in cfg:
            if required:
                self._issue(
                    CheckupCategory.STRUCTURE,
                    IssueCode.STRUCTURE_MISSING_KEY,
                    location,
                    f"Required config key `{'.'.join(location)}` is missing.",
                    details={"key": key},
                )
            return
        value = cfg[key]
        if not isinstance(value, str):
            self._wrong_type(location, str, value)

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def _check_references(self) -> None:
        stores = self._raw.get("stores")
        defined = set(stores) if isinstance(stores, Mapping) else set()
        referenced: set[str] = set()

        for name_key in STORE_NAME_KEYS:
            value = self._raw.get(name_key)
            if isinstance(value, str):
                referenced.add(value)
                if value not in defined:
                    self._issue(
                        CheckupCategory.REFERENCE,
                        IssueCode.REFERENCE_UNDEFINED_STORE,
                        (name_key,),
                        f"`{name_key}` references store `{value}`, but no store with "
                        "that name is defined under `stores`.",
                        details={
                            "referenced": value,
                            "reference_kind": name_key,
                            "defined_stores": sorted(defined),
                        },
                    )

        referenced |= self._check_site_store_references(defined)
        referenced |= set(IMPLICITLY_REFERENCED_STORES) & defined

        for name in sorted(defined - referenced):
            self._issue(
                CheckupCategory.REFERENCE,
                IssueCode.REFERENCE_UNUSED_STORE,
                ("stores", name),
                f"Store `{name}` is defined but is not used by "
                "`expectations_store_name`, `validation_results_store_name`, "
                "`checkpoint_store_name`, or any data docs site section.",
                severity=Severity.WARNING,
                details={"store": name},
            )

    def _check_site_store_references(  # noqa: C901 # FIXME CoP
        self, defined_stores: set[str]
    ) -> set[str]:
        referenced: set[str] = set()
        sites = self._raw.get("data_docs_sites")
        if not isinstance(sites, Mapping):
            return referenced

        for site_name in sorted(sites):
            site_cfg = sites[site_name]
            if not isinstance(site_cfg, Mapping):
                continue
            sections = site_cfg.get("site_section_builders")
            if not isinstance(sections, Mapping):
                continue
            for section_name in sorted(sections):
                section_cfg = sections[section_name]
                if not isinstance(section_cfg, Mapping):
                    continue
                source_store_name = section_cfg.get("source_store_name")
                if not isinstance(source_store_name, str):
                    continue
                referenced.add(source_store_name)
                if source_store_name not in defined_stores:
                    self._issue(
                        CheckupCategory.REFERENCE,
                        IssueCode.REFERENCE_UNDEFINED_STORE,
                        (
                            "data_docs_sites",
                            site_name,
                            "site_section_builders",
                            section_name,
                            "source_store_name",
                        ),
                        f"Data docs site `{site_name}` section `{section_name}` "
                        f"references store `{source_store_name}`, but no store with "
                        "that name is defined under `stores`.",
                        details={
                            "referenced": source_store_name,
                            "reference_kind": "data_docs_site_section",
                            "site": site_name,
                            "section": section_name,
                            "defined_stores": sorted(defined_stores),
                        },
                    )
        return referenced

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _check_loading(self) -> None:
        self._check_stores_loading()
        self._check_sites_loading()
        self._check_fluent_datasources_loading()

    def _check_stores_loading(self) -> None:  # noqa: C901 # FIXME CoP
        stores = self._raw.get("stores")
        if not isinstance(stores, Mapping):
            return

        for name in sorted(stores):
            store_cfg = stores[name]
            if not isinstance(store_cfg, Mapping):
                continue
            class_name = store_cfg.get("class_name")
            if not isinstance(class_name, str):
                # Reported as a structural issue; nothing loadable to probe.
                continue
            module_name = _module_name(store_cfg, DEFAULT_STORE_MODULE)

            store_resolved = self._resolve_and_check_class(
                ("stores", name, "class_name"),
                module_name=module_name,
                class_name=class_name,
            )

            backend_failed = False
            backend_cfg = store_cfg.get("store_backend")
            if "store_backend" in store_cfg:
                if not isinstance(backend_cfg, Mapping):
                    backend_failed = True
                else:
                    backend_class_name = backend_cfg.get("class_name")
                    if not isinstance(backend_class_name, str):
                        backend_failed = True
                    else:
                        backend_module = _module_name(backend_cfg, DEFAULT_STORE_MODULE)
                        backend_resolved = self._resolve_and_check_class(
                            ("stores", name, "store_backend", "class_name"),
                            module_name=backend_module,
                            class_name=backend_class_name,
                        )
                        if backend_resolved is None or not self._probe_instantiation(
                            location=("stores", name, "store_backend"),
                            config=backend_cfg,
                            runtime_environment={"root_directory": self._probe_root},
                            config_defaults={
                                "module_name": backend_module,
                                "store_name": name,
                                "suppress_store_backend_id": True,
                            },
                        ):
                            backend_failed = True

            if store_resolved is not None and not backend_failed:
                full_config = copy.deepcopy(store_cfg)
                nested_backend = full_config.get("store_backend")
                if isinstance(nested_backend, Mapping):
                    nested_backend["suppress_store_backend_id"] = True
                self._probe_instantiation(
                    location=("stores", name),
                    config=full_config,
                    runtime_environment={"root_directory": self._probe_root},
                    config_defaults={
                        "store_name": name,
                        "module_name": module_name,
                    },
                )

    def _check_sites_loading(self) -> None:  # noqa: C901, PLR0912 # FIXME CoP
        sites = self._raw.get("data_docs_sites")
        if not isinstance(sites, Mapping):
            return

        for site_name in sorted(sites):
            site_cfg = sites[site_name]
            if not isinstance(site_cfg, Mapping):
                continue

            site_class_name = site_cfg.get("class_name", "SiteBuilder")
            if isinstance(site_class_name, str):
                site_module = _module_name(site_cfg, DEFAULT_SITE_MODULE)
                self._resolve_and_check_class(
                    ("data_docs_sites", site_name, "class_name"),
                    module_name=site_module,
                    class_name=site_class_name,
                )

            backend_cfg = site_cfg.get("store_backend")
            if isinstance(backend_cfg, Mapping):
                backend_class_name = backend_cfg.get("class_name", "TupleFilesystemStoreBackend")
                backend_location = (
                    "data_docs_sites",
                    site_name,
                    "store_backend",
                )
                if isinstance(backend_class_name, str):
                    backend_module = _module_name(backend_cfg, DEFAULT_STORE_MODULE)
                    backend_resolved = self._resolve_and_check_class(
                        backend_location + ("class_name",),
                        module_name=backend_module,
                        class_name=backend_class_name,
                    )
                    if backend_resolved is not None:
                        self._probe_instantiation(
                            location=backend_location,
                            config=backend_cfg,
                            runtime_environment={"root_directory": self._probe_root},
                            config_defaults={
                                "module_name": backend_module,
                                "store_name": f"data_docs_{site_name}",
                                "suppress_store_backend_id": True,
                            },
                        )

            index_cfg = site_cfg.get("site_index_builder")
            if isinstance(index_cfg, Mapping):
                index_class_name = index_cfg.get("class_name", "DefaultSiteIndexBuilder")
                if isinstance(index_class_name, str):
                    self._resolve_and_check_class(
                        (
                            "data_docs_sites",
                            site_name,
                            "site_index_builder",
                            "class_name",
                        ),
                        module_name=_module_name(index_cfg, DEFAULT_SITE_MODULE),
                        class_name=index_class_name,
                    )

            sections = site_cfg.get("site_section_builders")
            if isinstance(sections, Mapping):
                for section_name in sorted(sections):
                    section_cfg = sections[section_name]
                    if not isinstance(section_cfg, Mapping):
                        continue
                    section_class_name = section_cfg.get("class_name")
                    if isinstance(section_class_name, str):
                        self._resolve_and_check_class(
                            (
                                "data_docs_sites",
                                site_name,
                                "site_section_builders",
                                section_name,
                                "class_name",
                            ),
                            module_name=_module_name(section_cfg, DEFAULT_SITE_MODULE),
                            class_name=section_class_name,
                        )
                    renderer_cfg = section_cfg.get("renderer")
                    if isinstance(renderer_cfg, Mapping):
                        renderer_class_name = renderer_cfg.get("class_name")
                        if isinstance(renderer_class_name, str):
                            self._resolve_and_check_class(
                                (
                                    "data_docs_sites",
                                    site_name,
                                    "site_section_builders",
                                    section_name,
                                    "renderer",
                                    "class_name",
                                ),
                                module_name=_module_name(renderer_cfg, DEFAULT_RENDERER_MODULE),
                                class_name=renderer_class_name,
                            )

    def _check_fluent_datasources_loading(self) -> None:
        datasources = self._raw.get("fluent_datasources")
        if not isinstance(datasources, Mapping):
            return

        for name in sorted(datasources):
            ds_cfg = datasources[name]
            if not isinstance(ds_cfg, Mapping):
                continue
            ds_type_name = ds_cfg.get("type")
            if not isinstance(ds_type_name, str):
                # Structural check already reported the missing/invalid type.
                continue

            location = ("fluent_datasources", name)
            try:
                ds_type = DataSourceManager.type_lookup[ds_type_name]
            except LookupError:
                available = sorted(str(t) for t in DataSourceManager.type_lookup.type_names())
                self._issue(
                    CheckupCategory.LOADING,
                    IssueCode.LOADING_DATASOURCE_TYPE_UNKNOWN,
                    location + ("type",),
                    f"Datasource `{name}` references unknown datasource type `{ds_type_name}`.",
                    details={
                        "datasource": name,
                        "type": ds_type_name,
                        "available_types": available,
                    },
                )
                continue

            kwargs = _prepare_datasource_kwargs(ds_cfg, name)
            try:
                ds_type(**kwargs)
            except Exception as exc:
                reason = _summarize_exception(exc)
                self._issue(
                    CheckupCategory.LOADING,
                    IssueCode.LOADING_DATASOURCE_INSTANTIATION_FAILED,
                    location,
                    f"Datasource `{name}` (type `{ds_type_name}`) could not be "
                    f"instantiated: {reason}",
                    details={
                        "datasource": name,
                        "type": ds_type_name,
                        "error_type": type(exc).__name__,
                        "reason": reason,
                    },
                )

    def _resolve_and_check_class(
        self, location: tuple[str, ...], *, module_name: str, class_name: str
    ) -> type | None:
        """Resolve a configured class; report resolution/abstractness findings.

        Returns the resolved concrete class, or None when a finding was reported.
        """
        resolved = _resolve_class(module_name=module_name, class_name=class_name)
        if resolved.cls is None:
            code = resolved.failure_code or IssueCode.LOADING_CLASS_NOT_FOUND
            self._issue(
                CheckupCategory.LOADING,
                code,
                location,
                resolved.failure_reason or f"Could not resolve `{class_name}`.",
                details={
                    "module_name": module_name,
                    "class_name": class_name,
                },
            )
            return None

        cls = resolved.cls
        if inspect.isabstract(cls):
            self._issue(
                CheckupCategory.LOADING,
                IssueCode.LOADING_ABSTRACT_CLASS,
                location,
                f"`{module_name}.{class_name}` is an abstract class and cannot be "
                "instantiated; configure a concrete subclass instead.",
                details={
                    "module_name": module_name,
                    "class_name": class_name,
                    "abstract_methods": sorted(cls.__abstractmethods__),
                },
            )
            return None
        return cls

    def _probe_instantiation(
        self,
        *,
        location: tuple[str, ...],
        config: Mapping[str, Any],
        runtime_environment: Mapping[str, Any],
        config_defaults: Mapping[str, Any],
    ) -> bool:
        """Attempt a throwaway instantiation; report failures.

        Returns True on success, False when an instantiation finding was reported.
        The instance is only held in a local variable and never registered, so no
        long-lived store survives the checkup.
        """
        try:
            # The returned instance is deliberately discarded: the checkup only
            # needs to know whether construction succeeds and must not retain a
            # long-lived store.
            instantiate_class_from_config(
                config=copy.deepcopy(dict(config)),
                runtime_environment=dict(runtime_environment),
                config_defaults=dict(config_defaults),
            )
        except Exception as exc:
            reason = _summarize_exception(exc)
            class_name = config.get("class_name", "<unknown>")
            self._issue(
                CheckupCategory.LOADING,
                IssueCode.LOADING_INSTANTIATION_FAILED,
                location,
                f"`{class_name}` resolved but could not be instantiated: {reason}",
                details={
                    "class_name": class_name,
                    "error_type": type(exc).__name__,
                    "reason": reason,
                },
            )
            return False
        return True


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _module_name(cfg: Mapping[str, Any], default: str) -> str:
    module_name = cfg.get("module_name", default)
    return module_name if isinstance(module_name, str) else default


def _matches_type(value: Any, expected: Any) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if isinstance(expected, tuple):
        # bool is a subclass of int; never accept it where a number is expected.
        if isinstance(value, bool):
            return False
        return isinstance(value, expected)
    return isinstance(value, expected)


def _resolve_class(*, module_name: str, class_name: str) -> _ResolvedClass:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ModuleNotFoundError, ValueError):
        spec = None

    if spec is None:
        return _ResolvedClass(
            cls=None,
            failure_code=IssueCode.LOADING_MODULE_NOT_FOUND,
            failure_reason=(
                f"No module named `{module_name}` could be found; please verify the "
                "`module_name` in the config and that the module is importable."
            ),
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return _ResolvedClass(
            cls=None,
            failure_code=IssueCode.LOADING_MODULE_NOT_FOUND,
            failure_reason=(
                f"Module `{module_name}` could not be imported: {_summarize_exception(exc)}"
            ),
        )

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        return _ResolvedClass(
            cls=None,
            failure_code=IssueCode.LOADING_CLASS_NOT_FOUND,
            failure_reason=(
                f"Module `{module_name}` does not contain a class named "
                f"`{class_name}`; please verify the `class_name` in the config."
            ),
        )

    if not isinstance(cls, type):
        return _ResolvedClass(
            cls=None,
            failure_code=IssueCode.LOADING_CLASS_NOT_FOUND,
            failure_reason=(
                f"`{module_name}.{class_name}` exists but is not a class "
                f"(got {type(cls).__name__})."
            ),
        )

    return _ResolvedClass(cls=cls)


def _summarize_exception(exc: BaseException, limit: int = 300) -> str:
    text = str(exc).strip()
    if not text:
        return f"{type(exc).__name__}: <no error message>"
    # GX's instantiate_class_from_config wraps TypeErrors as a multi-line message
    # whose final line is the underlying constructor error; prefer that line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = lines[-1] if lines else text
    if len(summary) > limit:
        return summary[: limit - 3] + "..."
    return summary


def _prepare_datasource_kwargs(ds_cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Mirror the YAML->internal conversion done by ``GxConfig.parse_yaml``."""

    kwargs: dict[str, Any] = dict(ds_cfg)
    kwargs["name"] = name

    assets = kwargs.get("assets")
    if isinstance(assets, Mapping):
        converted_assets: list[dict[str, Any]] = []
        for asset_name in sorted(assets):
            asset_cfg = assets[asset_name]
            if not isinstance(asset_cfg, Mapping):
                converted_assets.append(asset_cfg)
                continue
            asset_kwargs = dict(asset_cfg)
            asset_kwargs["name"] = asset_name
            batch_definitions = asset_kwargs.get("batch_definitions")
            if isinstance(batch_definitions, Mapping):
                asset_kwargs["batch_definitions"] = [
                    {**batch_definitions[bd_name], "name": bd_name}
                    for bd_name in sorted(batch_definitions)
                    if isinstance(batch_definitions[bd_name], Mapping)
                ]
            converted_assets.append(asset_kwargs)
        kwargs["assets"] = converted_assets

    return kwargs


def _finalize_issues(issues: list[ConfigIssue], *, strictness: Strictness) -> list[ConfigIssue]:
    finalized: list[ConfigIssue] = []
    for issue in issues:
        if strictness is Strictness.STRICT and issue.severity is Severity.WARNING:
            finalized.append(replace(issue, severity=Severity.ERROR))
        else:
            finalized.append(issue)
    finalized.sort(
        key=lambda i: (
            CATEGORY_ORDER[i.category],
            tuple(part for part in i.location),
            i.code,
        )
    )
    return finalized

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest
from ruamel.yaml import YAML

import great_expectations.exceptions as gx_exceptions
from great_expectations.data_context import FileDataContext, run_config_checkup
from great_expectations.data_context.config_checkup import (
    CheckupCategory,
    ConfigCheckupError,
    Severity,
    Strictness,
)
from great_expectations.data_context.constants import (
    CURRENT_GX_CONFIG_VERSION,
    MINIMUM_SUPPORTED_CONFIG_VERSION,
)
from great_expectations.data_context.types.base import DataContextConfig

pytestmark = pytest.mark.unit

yaml = YAML()
FIXTURES_DIR = pathlib.Path(__file__).parents[1] / "fixtures"


def _valid_config() -> dict:
    return {
        "config_version": 4,
        "expectations_store_name": "expectations_store",
        "validation_results_store_name": "validation_results_store",
        "checkpoint_store_name": "checkpoint_store",
        "stores": {
            "expectations_store": {
                "class_name": "ExpectationsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "expectations/",
                },
            },
            "validation_results_store": {
                "class_name": "ValidationResultsStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "uncommitted/validations/",
                },
            },
            "checkpoint_store": {
                "class_name": "CheckpointStore",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "checkpoints/",
                },
            },
        },
        "data_docs_sites": {
            "local_site": {
                "class_name": "SiteBuilder",
                "store_backend": {
                    "class_name": "TupleFilesystemStoreBackend",
                    "base_directory": "uncommitted/data_docs/local_site/",
                },
                "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            }
        },
    }


def _write_yml(tmp_path: pathlib.Path, config: dict | str) -> pathlib.Path:
    path = tmp_path / "great_expectations.yml"
    if isinstance(config, str):
        path.write_text(config, encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(config, f)
    return path


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def _issues_at(report, location: tuple[str, ...]):
    return [i for i in report.issues if tuple(i.location) == location]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_missing_config_file_raises_explicit_error(tmp_path):
    missing = tmp_path / "nope.yml"
    with pytest.raises(ConfigCheckupError, match="does not exist"):
        run_config_checkup(missing)
    with pytest.raises(ConfigCheckupError, match="does not exist"):
        run_config_checkup(str(missing))


def test_directory_path_raises_explicit_error(tmp_path):
    with pytest.raises(ConfigCheckupError, match="not a file"):
        run_config_checkup(tmp_path)


def test_invalid_yaml_raises_explicit_error_not_parser_exception(tmp_path):
    source = FIXTURES_DIR / "bad_yml" / "great_expectations.yml"
    target = _write_yml(tmp_path, source.read_text(encoding="utf-8"))
    with pytest.raises(ConfigCheckupError, match="not a valid YAML"):
        run_config_checkup(target)


@pytest.mark.parametrize("content", ["just a string\n", "- 1\n- 2\n"])
def test_non_mapping_yaml_raises_explicit_error(tmp_path, content):
    path = _write_yml(tmp_path, content)
    with pytest.raises(ConfigCheckupError, match="top level"):
        run_config_checkup(path)


def test_invalid_strictness_is_rejected_before_reading_config(tmp_path):
    with pytest.raises(ConfigCheckupError, match="Invalid strictness"):
        run_config_checkup(tmp_path / "missing.yml", strictness="banana")
    with pytest.raises(ConfigCheckupError, match="Invalid strictness"):
        run_config_checkup(_valid_config(), strictness=42)


def test_accepts_mapping_data_context_config_and_paths(tmp_path):
    path = _write_yml(tmp_path, _valid_config())

    from_mapping = run_config_checkup(_valid_config())
    from_path = run_config_checkup(path)
    assert from_mapping.success
    assert from_path.success

    dcc = DataContextConfig(
        config_version=4,
        expectations_store_name="expectations_store",
        validation_results_store_name="validation_results_store",
        stores={
            "expectations_store": {"class_name": "ExpectationsStore"},
            "validation_results_store": {"class_name": "ValidationResultsStore"},
        },
    )
    from_dcc = run_config_checkup(dcc)
    assert from_dcc.success
    assert from_dcc.config_path is None


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_current_version_is_supported_with_no_version_issue():
    report = run_config_checkup(_valid_config())
    assert report.version_verdict.status == "supported"
    assert report.version_verdict.minimum_supported == MINIMUM_SUPPORTED_CONFIG_VERSION
    assert report.version_verdict.current == CURRENT_GX_CONFIG_VERSION
    assert not [i for i in report.issues if i.category is CheckupCategory.VERSION]


def test_minimum_supported_version_is_an_advisory_in_lenient_mode():
    config = _valid_config()
    config["config_version"] = 3
    report = run_config_checkup(config)
    assert report.version_verdict.status == "supported_outdated"
    issue = [i for i in report.issues if i.code == "version.outdated_supported"]
    assert len(issue) == 1
    assert issue[0].severity is Severity.WARNING
    assert report.success


def test_old_version_reports_too_low_with_migration_guidance():
    config = _valid_config()
    config["config_version"] = 2
    report = run_config_checkup(config)
    assert report.version_verdict.status == "too_low"
    issue = [i for i in report.issues if i.code == "version.too_low"][0]
    assert issue.severity is Severity.ERROR
    assert issue.details["migration_guide_url"].startswith("https://")
    assert not report.success


def test_version_zero_legacy_0_7_x_series_is_distinct_from_generic_too_low():
    # The 0.7.x series is identified exactly like the load path does:
    # version 0 plus at least one ValidationResultsStore class.
    legacy = {
        "config_version": 0,
        "expectations_store_name": "expectations_store",
        "validation_results_store_name": "expectations_store",
        "stores": {
            "expectations_store": {"class_name": "ValidationResultsStore"},
        },
    }
    report = run_config_checkup(legacy)
    assert report.version_verdict.status == "legacy_0_7_x"
    issue = [i for i in report.issues if i.code == "version.legacy_series"][0]
    assert issue.details["series"] == "0.7.x"
    assert issue.details["migration_guide_url"]

    # Version 0 without that signature class falls through to generic too_low.
    non_legacy = _valid_config()
    non_legacy["config_version"] = 0
    non_legacy["stores"] = {
        "expectations_store": {"class_name": "ExpectationsStore"},
        "validation_results_store": {"class_name": "ExpectationsStore"},
        "checkpoint_store": {"class_name": "CheckpointStore"},
    }
    report2 = run_config_checkup(non_legacy)
    assert report2.version_verdict.status == "too_low"


def test_too_high_version_is_reported():
    config = _valid_config()
    config["config_version"] = 9999
    report = run_config_checkup(config)
    assert report.version_verdict.status == "too_high"
    assert "version.too_high" in _codes(report)


def test_missing_and_wrong_type_version_are_reported():
    config = _valid_config()
    config.pop("config_version")
    report = run_config_checkup(config)
    assert report.version_verdict.status == "missing"
    assert "version.missing" in _codes(report)

    config["config_version"] = "foo"
    report = run_config_checkup(config)
    assert report.version_verdict.status == "invalid_type"
    issue = [i for i in report.issues if i.code == "version.invalid_type"][0]
    assert issue.details["actual_type"] == "str"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_is_reported():
    config = _valid_config()
    config["this_is_a_wildly_invalid_top_level_key"] = "stuff"
    report = run_config_checkup(config)
    issue = _issues_at(report, ("this_is_a_wildly_invalid_top_level_key",))
    assert issue and issue[0].code == "structure.unknown_key"


def test_missing_required_top_level_keys_are_reported():
    config = _valid_config()
    config.pop("expectations_store_name")
    config.pop("stores")
    report = run_config_checkup(config)
    missing = {i.location[0] for i in report.issues if i.code == "structure.missing_key"}
    assert "expectations_store_name" in missing
    assert "stores" in missing


def test_wrong_top_level_value_types_are_reported():
    config = _valid_config()
    config["stores"] = "foo"
    config["data_docs_sites"] = "foo"
    config["analytics_enabled"] = "yes"
    report = run_config_checkup(config)
    wrong = {tuple(i.location) for i in report.issues if i.code == "structure.wrong_type"}
    assert ("stores",) in wrong
    assert ("data_docs_sites",) in wrong
    assert ("analytics_enabled",) in wrong


def test_store_definition_structure_is_checked():
    config = _valid_config()
    config["stores"]["bad_store"] = "not_a_mapping"
    config["stores"]["no_class"] = {"store_backend": {"class_name": "InMemoryStoreBackend"}}
    config["stores"]["int_class"] = {"class_name": 5}
    config["stores"]["bad_backend"] = {
        "class_name": "ExpectationsStore",
        "store_backend": "not_a_mapping",
    }
    config["stores"]["backend_no_class"] = {
        "class_name": "ExpectationsStore",
        "store_backend": {"base_directory": "x/"},
    }
    report = run_config_checkup(config)
    assert _issues_at(report, ("stores", "bad_store"))[0].code == "structure.wrong_type"
    assert (
        _issues_at(report, ("stores", "no_class", "class_name"))[0].code == "structure.missing_key"
    )
    assert (
        _issues_at(report, ("stores", "int_class", "class_name"))[0].code == "structure.wrong_type"
    )
    assert (
        _issues_at(report, ("stores", "bad_backend", "store_backend"))[0].code
        == "structure.wrong_type"
    )
    assert (
        _issues_at(report, ("stores", "backend_no_class", "store_backend", "class_name"))[0].code
        == "structure.missing_key"
    )


def test_data_docs_site_structure_is_checked():
    config = _valid_config()
    config["data_docs_sites"]["not_a_mapping"] = 5
    config["data_docs_sites"]["bad_builder"] = {
        "class_name": "SiteBuilder",
        "store_backend": {"class_name": "TupleFilesystemStoreBackend"},
        "site_index_builder": "nope",
    }
    report = run_config_checkup(config)
    assert (
        _issues_at(report, ("data_docs_sites", "not_a_mapping"))[0].code == "structure.wrong_type"
    )
    assert (
        _issues_at(report, ("data_docs_sites", "bad_builder", "site_index_builder"))[0].code
        == "structure.wrong_type"
    )


def test_fluent_datasource_structure_is_checked():
    config = _valid_config()
    config["fluent_datasources"] = {
        "good": {"type": "pandas_filesystem", "base_directory": "/tmp/data"},
        "no_type": {"base_directory": "/tmp/data"},
        "scalar": 5,
    }
    report = run_config_checkup(config)
    assert (
        _issues_at(report, ("fluent_datasources", "no_type", "type"))[0].code
        == "structure.missing_key"
    )
    assert _issues_at(report, ("fluent_datasources", "scalar"))[0].code == "structure.wrong_type"
    assert not _issues_at(report, ("fluent_datasources", "good"))


def test_progress_bars_structure_is_checked():
    config = _valid_config()
    config["progress_bars"] = {"globally": "yes", "unknown_toggle": False}
    report = run_config_checkup(config)
    assert _issues_at(report, ("progress_bars", "globally"))[0].code == "structure.wrong_type"
    assert (
        _issues_at(report, ("progress_bars", "unknown_toggle"))[0].code == "structure.unknown_key"
    )


def test_staged_fixture_unknown_key_and_wrong_type_coexist():
    report = run_config_checkup(FIXTURES_DIR / "invalid_top_level_key" / "great_expectations.yml")
    assert "structure.unknown_key" in _codes(report)
    assert "structure.wrong_type" in _codes(report)
    # referenced store "foo" is not defined; reference checks still run
    assert "reference.undefined_store" in _codes(report)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_all_three_primary_store_name_references_are_checked():
    config = _valid_config()
    config["expectations_store_name"] = "nope_exp"
    config["validation_results_store_name"] = "nope_val"
    config["checkpoint_store_name"] = "nope_cp"
    report = run_config_checkup(config)
    locations = {tuple(i.location) for i in report.issues if i.code == "reference.undefined_store"}
    assert ("expectations_store_name",) in locations
    assert ("validation_results_store_name",) in locations
    assert ("checkpoint_store_name",) in locations
    for issue in report.issues:
        if issue.code == "reference.undefined_store":
            assert issue.details["referenced"].startswith("nope_")


def test_reference_checks_run_even_when_stores_is_not_a_mapping():
    config = _valid_config()
    config["stores"] = "foo"
    config["expectations_store_name"] = "foo"
    report = run_config_checkup(config)
    assert "structure.wrong_type" in _codes(report)
    assert "reference.undefined_store" in _codes(report)


def test_data_docs_site_section_store_reference_is_checked_independently():
    config = _valid_config()
    # An unrelated loading problem must still be reported alongside the site
    # reference problem.
    config["stores"]["broken_store"] = {"class_name": "NoSuchStoreClass"}
    config["data_docs_sites"]["local_site"]["site_section_builders"] = {
        "expectations": {
            "class_name": "DefaultSiteSectionBuilder",
            "source_store_name": "ghost_store",
        }
    }
    report = run_config_checkup(config)
    site_issues = [
        i
        for i in report.issues
        if i.code == "reference.undefined_store" and "ghost_store" in i.message
    ]
    assert len(site_issues) == 1
    assert site_issues[0].location == (
        "data_docs_sites",
        "local_site",
        "site_section_builders",
        "expectations",
        "source_store_name",
    )
    assert "loading.class_not_found" in _codes(report)


def test_unused_store_is_an_advisory():
    config = _valid_config()
    config["stores"]["lonely_store"] = {"class_name": "ExpectationsStore"}
    report = run_config_checkup(config)
    issue = _issues_at(report, ("stores", "lonely_store"))[0]
    assert issue.code == "reference.unused_store"
    assert issue.severity is Severity.WARNING
    assert report.success


def test_validation_definition_store_is_implicitly_referenced():
    config = _valid_config()
    config["stores"]["validation_definition_store"] = {"class_name": "ValidationDefinitionStore"}
    report = run_config_checkup(config)
    assert not _issues_at(report, ("stores", "validation_definition_store"))
    assert report.success


def test_scaffolded_project_passes_checkup(tmp_path):
    context = FileDataContext._create(project_root_dir=str(tmp_path))
    yml_path = pathlib.Path(context.root_directory) / FileDataContext.GX_YML

    report = run_config_checkup(yml_path)
    assert report.success
    assert report.issues == ()
    assert report.version_verdict.status == "supported"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_unresolvable_and_uninstantiable_store_classes_are_distinct_findings():
    config = _valid_config()
    config["stores"] = {
        "expectations_store": {"class_name": "ExpectationsStore"},
        "validation_results_store": {"class_name": "ValidationResultsStore"},
        "missing_module": {
            "class_name": "ExpectationsStore",
            "module_name": "no.such.module",
        },
        "missing_class": {"class_name": "NoSuchStore"},
        "abstract_store": {"class_name": "TupleStoreBackend"},
        "broken_backend": {
            "class_name": "ExpectationsStore",
            "store_backend": {"class_name": "TupleFilesystemStoreBackend"},
        },
        "abstract_backend": {
            "class_name": "ExpectationsStore",
            "store_backend": {"class_name": "StoreBackend"},
        },
    }
    report = run_config_checkup(config)

    module_issue = _issues_at(report, ("stores", "missing_module", "class_name"))
    assert module_issue[0].code == "loading.module_not_found"

    class_issue = _issues_at(report, ("stores", "missing_class", "class_name"))
    assert class_issue[0].code == "loading.class_not_found"

    abstract_issue = _issues_at(report, ("stores", "abstract_store", "class_name"))
    assert abstract_issue[0].code == "loading.abstract_class"

    abstract_backend_issue = _issues_at(
        report, ("stores", "abstract_backend", "store_backend", "class_name")
    )
    assert abstract_backend_issue[0].code == "loading.abstract_class"

    inst_issue = _issues_at(report, ("stores", "broken_backend", "store_backend"))
    assert inst_issue[0].code == "loading.instantiation_failed"
    assert "base_directory" in inst_issue[0].details["reason"]
    assert inst_issue[0].details["error_type"] == "TypeError"
    # the concrete-but-failing backend does not get a duplicate store-level failure
    store_level = [
        i
        for i in _issues_at(report, ("stores", "broken_backend"))
        if i.code == "loading.instantiation_failed"
    ]
    assert not store_level


def test_data_docs_site_classes_are_resolved_but_context_is_not_built():
    config = _valid_config()
    config["data_docs_sites"]["local_site"]["class_name"] = "NoSuchSiteBuilder"
    config["data_docs_sites"]["local_site"]["store_backend"] = {"class_name": "NoSuchBackend"}
    report = run_config_checkup(config)
    assert (
        _issues_at(report, ("data_docs_sites", "local_site", "class_name"))[0].code
        == "loading.class_not_found"
    )
    assert (
        _issues_at(
            report,
            ("data_docs_sites", "local_site", "store_backend", "class_name"),
        )[0].code
        == "loading.class_not_found"
    )


def test_fluent_datasource_type_and_instantiation_are_checked():
    config = _valid_config()
    config["fluent_datasources"] = {
        "good": {"type": "pandas_filesystem", "base_directory": "/tmp/data"},
        "unknown_type": {"type": "not_a_real_type"},
        "broken": {
            "type": "pandas_filesystem",
            "base_directory": "/tmp/data",
            "assets": {"a": {"type": "not_an_asset_type"}},
        },
    }
    report = run_config_checkup(config)

    assert (
        _issues_at(report, ("fluent_datasources", "unknown_type", "type"))[0].code
        == "loading.datasource_type_unknown"
    )
    inst_issue = _issues_at(report, ("fluent_datasources", "broken"))
    assert inst_issue[0].code == "loading.datasource_instantiation_failed"
    assert inst_issue[0].details["reason"]
    assert not _issues_at(report, ("fluent_datasources", "good"))


# ---------------------------------------------------------------------------
# Report shape, determinism and strictness
# ---------------------------------------------------------------------------


def test_report_machine_output_is_json_serializable_and_complete():
    config = _valid_config()
    config["stores"]["lonely_store"] = {"class_name": "ExpectationsStore"}
    config["unknown"] = True
    report = run_config_checkup(config)
    payload = report.to_dict()

    assert set(payload) == {
        "success",
        "strictness",
        "config_path",
        "version_verdict",
        "issue_counts",
        "issues",
    }
    json.dumps(payload)  # all values are JSON-native
    assert payload["issue_counts"]["total"] == len(payload["issues"])
    for issue in payload["issues"]:
        assert set(issue) == {
            "category",
            "severity",
            "code",
            "location",
            "message",
            "details",
        }
        assert isinstance(issue["location"], list)

    categories = list(report.issues_by_category())
    assert categories == list(CheckupCategory)


def test_human_readable_report_renders_groups():
    config = _valid_config()
    config["config_version"] = 2
    text = run_config_checkup(config).render_text()
    assert "FAILED" in text
    assert "[version]" in text
    assert "[structure]" in text
    assert "[reference]" in text
    assert "[loading]" in text
    assert "version.too_low" in text
    # str() and a passing report do not raise
    assert "PASSED" in str(run_config_checkup(_valid_config()))


def test_repeated_runs_are_deterministic():
    config = _valid_config()
    config["stores"]["lonely_store"] = {"class_name": "ExpectationsStore"}
    config["config_version"] = 3
    first = run_config_checkup(copy.deepcopy(config)).to_dict()
    second = run_config_checkup(copy.deepcopy(config)).to_dict()
    assert first == second
    # insertion order within the issues list is the stable sorted order
    order = [category.value for category in CheckupCategory]
    categories = [i["category"] for i in first["issues"]]
    assert categories == sorted(categories, key=order.index)


def test_strict_mode_escalates_warnings_to_errors():
    config = _valid_config()
    config["config_version"] = 3
    config["stores"]["lonely_store"] = {"class_name": "ExpectationsStore"}

    lenient = run_config_checkup(config, strictness=Strictness.LENIENT)
    assert lenient.success
    assert lenient.warnings and not lenient.errors

    strict = run_config_checkup(config, strictness="strict")
    assert not strict.success
    assert strict.errors
    assert not strict.warnings
    escalated_codes = {i.code for i in strict.errors}
    assert "version.outdated_supported" in escalated_codes
    assert "reference.unused_store" in escalated_codes


# ---------------------------------------------------------------------------
# Read-only behavior
# ---------------------------------------------------------------------------


def test_checkup_is_read_only(tmp_path):
    config = _valid_config()
    # use a relative plugins directory too, to prove nothing is scaffolded
    config["plugins_directory"] = "plugins/"
    path = _write_yml(tmp_path, config)

    def snapshot():
        return {
            str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tmp_path.rglob("*")
            if p.is_file()
        }

    before = snapshot()
    run_config_checkup(path)
    run_config_checkup(config)
    after = snapshot()
    assert before == after
    assert not list(tmp_path.rglob(".ge_store_backend_id"))
    assert not (tmp_path / "expectations").exists()
    assert not (tmp_path / "uncommitted").exists()


# ---------------------------------------------------------------------------
# Default load path is untouched
# ---------------------------------------------------------------------------


def test_default_load_path_error_types_are_unchanged(tmp_path):
    gx_dir = tmp_path / "gx"
    gx_dir.mkdir()
    shutil_copy_fixture("invalid_top_level_key", gx_dir)
    with pytest.raises(gx_exceptions.InvalidDataContextConfigError):
        FileDataContext(context_root_dir=str(gx_dir))

    gx_dir2 = tmp_path / "gx2"
    gx_dir2.mkdir()
    shutil_copy_fixture("bad_yml", gx_dir2)
    with pytest.raises(gx_exceptions.InvalidConfigurationYamlError):
        FileDataContext(context_root_dir=str(gx_dir2))


def shutil_copy_fixture(name: str, gx_dir: pathlib.Path) -> None:
    import shutil

    shutil.copy(
        FIXTURES_DIR / name / "great_expectations.yml",
        gx_dir / "great_expectations.yml",
    )

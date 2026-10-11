import pytest

import great_expectations.exceptions as gx_exceptions
from great_expectations.datasource.fluent import BatchRequest
from great_expectations.datasource.fluent.batch_pagination import (
    InvalidPageCursorError,
    InvalidPageSizeError,
)
from great_expectations.datasource.fluent.data_connector import FilesystemDataConnector
from great_expectations.datasource.fluent.pandas_filesystem_datasource import (
    PandasFilesystemDatasource,
)
from tests.test_utils import create_files_in_directory

FILE_NAMES = [
    "yellow_tripdata_sample_2019-01.csv",
    "yellow_tripdata_sample_2019-02.csv",
    "yellow_tripdata_sample_2019-03.csv",
    "yellow_tripdata_sample_2020-01.csv",
    "yellow_tripdata_sample_2020-02.csv",
    "yellow_tripdata_sample_2020-03.csv",
]
MONTHLY_REGEX = r"yellow_tripdata_sample_(?P<year>\d{4})-(?P<month>\d{2})\.csv"


@pytest.fixture
def datasource_with_asset(tmp_path) -> PandasFilesystemDatasource:
    create_files_in_directory(directory=str(tmp_path), file_name_list=FILE_NAMES)
    datasource = PandasFilesystemDatasource(name="pandas_fs_datasource", base_directory=tmp_path)
    asset = datasource.add_csv_asset(name="csv_asset")
    asset.add_batch_definition_monthly(name="monthly", regex=MONTHLY_REGEX)
    return datasource


@pytest.fixture
def monthly_batch_definition(datasource_with_asset: PandasFilesystemDatasource):
    asset = datasource_with_asset.get_asset("csv_asset")
    return asset.get_batch_definition("monthly")


def _walk_all_pages(asset, batch_request, page_size, **kwargs):
    pages = []
    cursor = None
    while True:
        page = asset.get_batch_identifiers_page(
            batch_request, page_size=page_size, cursor=cursor, **kwargs
        )
        pages.append(page)
        if not page.has_more:
            return pages
        cursor = page.next_cursor


def _build_request(batch_definition):
    return batch_definition.build_batch_request()


@pytest.mark.filesystem
def test_default_page_is_identical_to_full_enumeration(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()

    expected = asset.get_batch_identifiers_list(batch_request)
    page = asset.get_batch_identifiers_page(batch_request)

    assert page.batch_identifiers == expected
    assert page.page_count == len(expected) == 6
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.page_size is None
    assert page.total_count is None


@pytest.mark.filesystem
@pytest.mark.parametrize("sort_ascending", [True, False])
def test_paged_walk_covers_all_matches_in_enumeration_order(datasource_with_asset, sort_ascending):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_definition = asset.add_batch_definition_monthly(
        name=f"monthly_{sort_ascending}", regex=MONTHLY_REGEX, sort_ascending=sort_ascending
    )
    batch_request = batch_definition.build_batch_request()

    expected = asset.get_batch_identifiers_list(batch_request)

    pages = _walk_all_pages(asset, batch_request, page_size=2, include_total_count=True)
    seen = [identifier for page in pages for identifier in page.batch_identifiers]

    assert seen == expected
    assert len(seen) == len({identifier["path"] for identifier in seen})
    assert [page.page_count for page in pages] == [2, 2, 2]
    assert all(page.total_count == 6 for page in pages)


@pytest.mark.filesystem
def test_page_metadata_reports_has_more_and_cursor(datasource_with_asset, monthly_batch_definition):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()

    page_1 = asset.get_batch_identifiers_page(batch_request, page_size=4, include_total_count=True)
    assert page_1.page_count == 4
    assert page_1.page_size == 4
    assert page_1.has_more is True
    assert page_1.next_cursor is not None
    assert page_1.total_count == 6

    page_2 = asset.get_batch_identifiers_page(
        batch_request,
        page_size=4,
        cursor=page_1.next_cursor,
        include_total_count=True,
    )
    assert page_2.page_count == 2
    assert page_2.has_more is False
    assert page_2.next_cursor is None
    assert page_2.total_count == 6


@pytest.mark.filesystem
def test_pages_do_not_overlap_across_options_filter(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request(batch_parameters={"year": 2020})

    expected = asset.get_batch_identifiers_list(batch_request)
    pages = _walk_all_pages(asset, batch_request, page_size=2)
    seen = [identifier for page in pages for identifier in page.batch_identifiers]

    assert seen == expected
    assert len(seen) == 3
    assert all(identifier["year"] == "2020" for identifier in seen)


@pytest.mark.filesystem
def test_invalid_page_sizes_raise(datasource_with_asset, monthly_batch_definition):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()

    for bad_size in (0, -1):
        with pytest.raises(InvalidPageSizeError):
            asset.get_batch_identifiers_page(batch_request, page_size=bad_size)


@pytest.mark.filesystem
def test_invalid_cursor_raises_instead_of_restarting(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()

    with pytest.raises(InvalidPageCursorError):
        asset.get_batch_identifiers_page(batch_request, page_size=2, cursor="garbage")


@pytest.mark.filesystem
def test_cursor_one_past_end_returns_empty_page_with_total(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()
    full_list = asset.get_batch_identifiers_list(batch_request)
    # a legal cursor exactly one position past the end
    from great_expectations.datasource.fluent.batch_pagination import encode_page_cursor

    end_cursor = encode_page_cursor(offset=len(full_list), anchor=full_list[-1])

    trailing_page = asset.get_batch_identifiers_page(
        batch_request, page_size=3, cursor=end_cursor, include_total_count=True
    )
    assert trailing_page.batch_identifiers == []
    assert trailing_page.page_count == 0
    assert trailing_page.has_more is False
    assert trailing_page.total_count == 6


@pytest.mark.filesystem
def test_no_matches_is_empty_page_with_zero_total(datasource_with_asset, monthly_batch_definition):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request(batch_parameters={"year": 1999})

    page = asset.get_batch_identifiers_page(batch_request, page_size=10, include_total_count=True)
    assert page.batch_identifiers == []
    assert page.page_count == 0
    assert page.has_more is False
    assert page.total_count == 0


@pytest.mark.filesystem
def test_invalid_batch_request_is_distinguished_from_empty_results(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    bad_request = BatchRequest(
        datasource_name="some_other_datasource",
        data_asset_name=asset.name,
        options={},
        partitioner=monthly_batch_definition.partitioner,
    )

    with pytest.raises(gx_exceptions.InvalidBatchRequestError):
        asset.get_batch_identifiers_page(bad_request, page_size=2)
    with pytest.raises(gx_exceptions.InvalidBatchRequestError):
        asset.get_batch_identifiers_count(bad_request)


@pytest.mark.filesystem
def test_count_matches_full_enumeration_and_does_not_read_data(
    datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request()

    result = asset.get_batch_identifiers_count(batch_request)
    expected = asset.get_batch_identifiers_list(batch_request)

    assert result.count == len(expected) == 6
    # enumeration never instantiates an execution engine or reads batch data
    assert datasource_with_asset._execution_engine is None


@pytest.mark.filesystem
def test_count_respects_options_filter(datasource_with_asset, monthly_batch_definition):
    asset = datasource_with_asset.get_asset("csv_asset")
    batch_request = monthly_batch_definition.build_batch_request(batch_parameters={"year": 2020})

    assert asset.get_batch_identifiers_count(batch_request).count == 3


@pytest.mark.filesystem
def test_cache_is_built_once_and_reuse_is_reported(
    datasource_with_asset, monthly_batch_definition, mocker
):
    asset = datasource_with_asset.get_asset("csv_asset")
    connector = asset._data_connector
    batch_request = monthly_batch_definition.build_batch_request()

    # cold cache: built once, in the existing manner, during this call
    first_page = asset.get_batch_identifiers_page(batch_request, page_size=2)
    assert first_page.data_reference_cache_hit is False

    second_page = asset.get_batch_identifiers_page(
        batch_request, page_size=2, cursor=first_page.next_cursor
    )
    assert second_page.data_reference_cache_hit is True

    # count reports the same cache state for the same (regex, options) match set
    warmed_count = asset.get_batch_identifiers_count(batch_request)
    assert warmed_count.data_reference_cache_hit is True
    assert warmed_count.count == 6

    # the match-all cache was already warmed by test_connection() when the asset was added
    match_all_count = asset.get_batch_identifiers_count(asset.build_batch_request())
    assert match_all_count.data_reference_cache_hit is True
    assert match_all_count.count == 6

    # on a warm cache, counting constructs no new batch definitions
    spy = mocker.spy(connector, "_build_batch_definition")
    result = asset.get_batch_identifiers_count(batch_request)
    assert result.count == 6
    spy.assert_not_called()


@pytest.mark.filesystem
def test_pages_resume_correctly_after_cache_refresh(
    tmp_path, datasource_with_asset, monthly_batch_definition
):
    asset = datasource_with_asset.get_asset("csv_asset")
    connector = asset._data_connector
    batch_request = monthly_batch_definition.build_batch_request()

    page_1 = asset.get_batch_identifiers_page(batch_request, page_size=3)
    first_page_identifiers = page_1.batch_identifiers
    original_full_list = asset.get_batch_identifiers_list(batch_request)

    # a new file is discovered and the data-reference cache is refreshed before page 2
    create_files_in_directory(
        directory=str(tmp_path),
        file_name_list=["yellow_tripdata_sample_2018-12.csv"],
    )
    connector._data_references_cache.clear()

    page_2 = asset.get_batch_identifiers_page(batch_request, page_size=3, cursor=page_1.next_cursor)

    # resumes immediately after page 1's anchor, no overlap or gap despite the shifted cache
    assert page_2.batch_identifiers[0] == original_full_list[3]
    assert not any(identifier in page_2.batch_identifiers for identifier in first_page_identifiers)


@pytest.mark.filesystem
def test_datasource_level_pagination_delegates_to_asset(
    datasource_with_asset, monthly_batch_definition
):
    batch_request = _build_request(monthly_batch_definition)

    page = datasource_with_asset.get_batch_identifiers_page(batch_request, page_size=2)
    assert page.page_count == 2
    assert page.has_more is True

    count = datasource_with_asset.get_batch_identifiers_count(batch_request)
    assert count.count == 6


@pytest.mark.filesystem
def test_batch_definition_level_pagination_and_count(monthly_batch_definition):
    page = monthly_batch_definition.get_batch_identifiers_page(page_size=4)
    assert page.page_count == 4
    assert page.has_more is True

    count = monthly_batch_definition.get_batch_identifiers_count()
    assert count.count == 6


@pytest.mark.unit
def test_dataframe_asset_pagination_uses_generic_enumeration_path():
    import pandas as pd

    from great_expectations.datasource.fluent import PandasDatasource

    datasource = PandasDatasource(name="pandas_datasource")
    asset = datasource.add_dataframe_asset(name="dataframe_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")
    batch_request = batch_definition.build_batch_request(
        batch_parameters={"dataframe": pd.DataFrame({"a": [1, 2]})}
    )

    page = asset.get_batch_identifiers_page(batch_request)
    assert page.page_count == 1
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.data_reference_cache_hit is None

    count = asset.get_batch_identifiers_count(batch_request)
    assert count.count == 1
    assert count.data_reference_cache_hit is None


@pytest.mark.filesystem
def test_whole_directory_connector_count_and_pagination(tmp_path):
    create_files_in_directory(directory=str(tmp_path), file_name_list=FILE_NAMES)
    connector = FilesystemDataConnector(
        datasource_name="ds",
        data_asset_name="asset",
        base_directory=tmp_path,
        whole_directory_path_override=tmp_path,
    )
    batch_request = BatchRequest(datasource_name="ds", data_asset_name="asset", options={})

    matched, cache_hit = connector.get_matched_batch_definitions(batch_request)
    count, count_cache_hit = connector.get_matched_batch_definition_count(batch_request)

    assert len(matched) == 1
    assert count == 1
    assert cache_hit is False
    assert count_cache_hit is False

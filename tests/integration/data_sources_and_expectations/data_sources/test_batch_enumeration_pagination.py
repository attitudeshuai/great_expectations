from datetime import datetime, timezone

import pandas as pd
import pytest

from great_expectations.datasource.fluent.batch_pagination import (
    InvalidPageCursorError,
    InvalidPageSizeError,
    encode_page_cursor,
)
from great_expectations.datasource.fluent.sql_datasource import TableAsset
from tests.integration.conftest import parameterize_batch_for_data_sources
from tests.integration.test_utils.data_source_config import SqliteDatasourceTestConfig

DATE_COL = "date"
VALUE_COL = "value"

DATA = pd.DataFrame(
    {
        DATE_COL: [
            datetime(year=2023, month=1, day=1, tzinfo=timezone.utc).date(),
            datetime(year=2024, month=1, day=1, tzinfo=timezone.utc).date(),
            datetime(year=2024, month=2, day=1, tzinfo=timezone.utc).date(),
            datetime(year=2024, month=2, day=2, tzinfo=timezone.utc).date(),
        ],
        VALUE_COL: ["a", "b", "c", "d"],
    }
)

JUST_SQLITE = [SqliteDatasourceTestConfig()]


class TestBatchEnumerationPagination:
    """End-to-end paged batch-identifier enumeration against a real backend."""

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_default_page_matches_full_enumeration(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)

        expected = batch_def.get_batch_identifiers_list()
        page = batch_def.get_batch_identifiers_page()

        assert page.batch_identifiers == expected
        assert page.page_count == 4
        assert page.has_more is False
        assert page.next_cursor is None

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_pages_walk_all_batches_in_order_without_gaps_or_duplicates(
        self, asset_for_datasource: TableAsset
    ) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)
        expected = batch_def.get_batch_identifiers_list()

        seen = []
        cursor = None
        page_sizes = []
        while True:
            page = batch_def.get_batch_identifiers_page(
                page_size=2, cursor=cursor, include_total_count=True
            )
            assert page.total_count == 4
            page_sizes.append(page.page_count)
            seen.extend(page.batch_identifiers)
            if not page.has_more:
                assert page.next_cursor is None
                break
            cursor = page.next_cursor

        assert seen == expected
        assert len(seen) == len({tuple(sorted(item.items())) for item in seen})
        assert page_sizes == [2, 2]

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_descending_order_is_preserved(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily(
            "daily_descending", column=DATE_COL, sort_ascending=False
        )
        expected = batch_def.get_batch_identifiers_list()

        page = batch_def.get_batch_identifiers_page(page_size=2)
        assert page.batch_identifiers == expected[:2]

        page_2 = batch_def.get_batch_identifiers_page(page_size=2, cursor=page.next_cursor)
        assert page_2.batch_identifiers == expected[2:]

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_count_only(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)

        count = batch_def.get_batch_identifiers_count()
        assert count.count == 4

        filtered_count = batch_def.get_batch_identifiers_count(batch_parameters={"year": 2024})
        assert filtered_count.count == 3

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_options_filter_applied_to_pages(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)
        batch_request = batch_def.build_batch_request(batch_parameters={"year": 2024})

        page = asset_for_datasource.get_batch_identifiers_page(
            batch_request, page_size=2, include_total_count=True
        )
        assert page.total_count == 3
        assert all(identifier["year"] == 2024 for identifier in page.batch_identifiers)

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_invalid_page_size_raises(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)
        with pytest.raises(InvalidPageSizeError):
            batch_def.get_batch_identifiers_page(page_size=0)

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_invalid_cursor_raises(self, asset_for_datasource: TableAsset) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)
        with pytest.raises(InvalidPageCursorError):
            batch_def.get_batch_identifiers_page(page_size=2, cursor="not-a-cursor")

    @parameterize_batch_for_data_sources(
        data_source_configs=JUST_SQLITE,
        data=pd.DataFrame(DATA),
    )
    def test_cursor_past_end_returns_empty_page_with_total(
        self, asset_for_datasource: TableAsset
    ) -> None:
        batch_def = asset_for_datasource.add_batch_definition_daily("daily", column=DATE_COL)
        full_list = batch_def.get_batch_identifiers_list()
        end_cursor = encode_page_cursor(offset=len(full_list), anchor=full_list[-1])

        page = batch_def.get_batch_identifiers_page(
            page_size=2, cursor=end_cursor, include_total_count=True
        )
        assert page.batch_identifiers == []
        assert page.page_count == 0
        assert page.has_more is False
        assert page.total_count == 4

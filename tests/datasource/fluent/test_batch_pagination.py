import pytest

from great_expectations.datasource.fluent.batch_pagination import (
    DEFAULT_BATCH_IDENTIFIERS_PAGE_SIZE,
    MAX_BATCH_IDENTIFIERS_PAGE_SIZE,
    BatchPaginationConfigError,
    InvalidPageCursorError,
    InvalidPageSizeError,
    decode_page_cursor,
    encode_page_cursor,
    get_pagination_settings,
    paginate_batch_identifiers,
)


def _identifiers(n: int) -> list[dict]:
    return [{"path": f"file_{i:03d}.csv", "n": i} for i in range(n)]


@pytest.mark.unit
def test_no_pagination_params_returns_every_match_in_one_page():
    identifiers = _identifiers(10)

    page = paginate_batch_identifiers(identifiers)

    assert page.batch_identifiers == identifiers
    assert page.page_count == 10
    assert page.page_size is None
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.total_count is None
    assert page.data_reference_cache_hit is None


@pytest.mark.unit
@pytest.mark.parametrize("page_size", [1, 2, 3, 5, 7, 10, 13])
def test_walking_pages_covers_all_matches_without_duplicates_or_gaps(page_size):
    identifiers = _identifiers(10)

    seen: list[dict] = []
    cursor = None
    page_count = 0
    while True:
        page = paginate_batch_identifiers(
            identifiers, page_size=page_size, cursor=cursor, include_total_count=True
        )
        assert page.total_count == 10
        assert page.page_count == len(page.batch_identifiers)
        assert page.page_count <= page_size
        seen.extend(page.batch_identifiers)
        page_count += 1
        if not page.has_more:
            assert page.next_cursor is None
            break
        assert page.next_cursor is not None
        cursor = page.next_cursor

    assert seen == identifiers
    assert len(seen) == len({id_["path"] for id_ in seen})
    assert page_count == -(-10 // page_size)  # ceil(10 / page_size)


@pytest.mark.unit
def test_total_count_only_populated_when_requested():
    identifiers = _identifiers(3)

    assert paginate_batch_identifiers(identifiers, page_size=1).total_count is None
    page_with_total = paginate_batch_identifiers(identifiers, page_size=1, include_total_count=True)
    assert page_with_total.total_count == 3


@pytest.mark.unit
def test_last_page_has_no_more_and_no_cursor_when_items_fill_page_exactly():
    identifiers = _identifiers(4)

    page_1 = paginate_batch_identifiers(identifiers, page_size=2, include_total_count=True)
    assert page_1.has_more is True
    page_2 = paginate_batch_identifiers(
        identifiers, page_size=2, cursor=page_1.next_cursor, include_total_count=True
    )
    assert page_2.batch_identifiers == identifiers[2:]
    assert page_2.has_more is False
    assert page_2.next_cursor is None
    assert page_2.total_count == 4


@pytest.mark.unit
def test_cursor_one_past_end_returns_empty_page_with_total():
    identifiers = _identifiers(5)
    end_cursor = encode_page_cursor(offset=5, anchor=identifiers[-1])

    page = paginate_batch_identifiers(
        identifiers, page_size=2, cursor=end_cursor, include_total_count=True
    )

    assert page.batch_identifiers == []
    assert page.page_count == 0
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.total_count == 5


@pytest.mark.unit
def test_empty_match_set_is_empty_page_with_zero_total():
    page = paginate_batch_identifiers([], page_size=10, include_total_count=True)

    assert page.batch_identifiers == []
    assert page.page_count == 0
    assert page.has_more is False
    assert page.next_cursor is None
    assert page.total_count == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_size",
    [
        0,
        -1,
        -100,
        True,
        False,
        "2",
        1.5,
    ],
)
def test_invalid_page_size_raises_clear_error(bad_size):
    with pytest.raises(InvalidPageSizeError, match="page_size must be a positive integer"):
        paginate_batch_identifiers(_identifiers(5), page_size=bad_size)


@pytest.mark.unit
def test_page_size_over_maximum_raises(monkeypatch):
    monkeypatch.setenv("GX_BATCH_ENUMERATION_MAX_PAGE_SIZE", "10")
    with pytest.raises(InvalidPageSizeError, match="no greater than 10"):
        paginate_batch_identifiers(_identifiers(50), page_size=11)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_cursor",
    [
        "",
        "not-a-cursor",
        "gxp1:!!!not-base64",
        "gxp2:AAAA",
        42,
        object(),
    ],
)
def test_malformed_cursor_raises(bad_cursor):
    with pytest.raises(InvalidPageCursorError):
        paginate_batch_identifiers(_identifiers(5), page_size=2, cursor=bad_cursor)


@pytest.mark.unit
def test_tampered_cursor_payload_raises():
    import base64

    not_json_token = "gxp1:" + base64.urlsafe_b64encode(b"not-json-payload").decode("ascii").rstrip(
        "="
    )
    for bad_token in ("gxp1:@@@@", not_json_token):
        with pytest.raises(InvalidPageCursorError):
            decode_page_cursor(bad_token)


@pytest.mark.unit
def test_cursor_with_unsupported_version_raises():
    import base64
    import json

    payload = json.dumps({"v": 99, "o": 2, "a": {"path": "a"}}).encode("utf-8")
    token = "gxp1:" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    with pytest.raises(InvalidPageCursorError, match="version"):
        decode_page_cursor(token)


@pytest.mark.unit
def test_cursor_with_missing_anchor_raises_instead_of_restarting():
    identifiers = _identifiers(10)
    # points at offset 4 but the anchor no longer exists anywhere in the match set
    stale_cursor = encode_page_cursor(offset=4, anchor={"path": "deleted.csv"})

    with pytest.raises(InvalidPageCursorError, match="no longer present"):
        paginate_batch_identifiers(identifiers, page_size=2, cursor=stale_cursor)


@pytest.mark.unit
def test_cursor_resumes_without_gap_when_new_items_are_added_before_anchor():
    identifiers = _identifiers(10)
    page_1 = paginate_batch_identifiers(identifiers, page_size=5)
    assert page_1.batch_identifiers == identifiers[:5]

    # cache refresh: two new files sort before the existing anchor
    refreshed = [
        {"path": "file_000a.csv", "n": -2},
        {"path": "file_000b.csv", "n": -1},
    ] + identifiers
    page_2 = paginate_batch_identifiers(refreshed, page_size=5, cursor=page_1.next_cursor)

    # resumes immediately after page 1's anchor: no overlap, no gap vs the original traversal
    assert page_2.batch_identifiers[0] == identifiers[5]
    assert page_1.batch_identifiers[-1] not in page_2.batch_identifiers


@pytest.mark.unit
def test_cursor_resumes_when_items_are_appended():
    identifiers = _identifiers(10)
    page_1 = paginate_batch_identifiers(identifiers, page_size=7)

    refreshed = identifiers + [{"path": "file_010.csv", "n": 10}]
    page_2 = paginate_batch_identifiers(refreshed, page_size=7, cursor=page_1.next_cursor)

    assert page_2.batch_identifiers == [
        identifiers[7],
        identifiers[8],
        identifiers[9],
        refreshed[10],
    ]
    assert page_2.has_more is False


@pytest.mark.unit
def test_default_configuration_matches_module_constants():
    settings = get_pagination_settings()
    assert settings.default_page_size == DEFAULT_BATCH_IDENTIFIERS_PAGE_SIZE
    assert settings.max_page_size == MAX_BATCH_IDENTIFIERS_PAGE_SIZE


@pytest.mark.unit
def test_invalid_max_page_size_config_raises_when_paginating(monkeypatch):
    monkeypatch.setenv("GX_BATCH_ENUMERATION_MAX_PAGE_SIZE", "not-an-int")
    with pytest.raises(BatchPaginationConfigError):
        paginate_batch_identifiers(_identifiers(3), page_size=1)


@pytest.mark.unit
def test_invalid_config_does_not_change_default_unpaginated_behavior(monkeypatch):
    monkeypatch.setenv("GX_BATCH_ENUMERATION_MAX_PAGE_SIZE", "not-an-int")

    page = paginate_batch_identifiers(_identifiers(3))

    assert page.batch_identifiers == _identifiers(3)
    assert page.has_more is False

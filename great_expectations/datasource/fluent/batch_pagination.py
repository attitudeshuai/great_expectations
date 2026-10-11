from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from great_expectations._docs_decorators import public_api

# Pagination configuration. The maximum page size can be overridden with an environment variable so
# that deployments can tune the limit without code changes. It is validated lazily (only when a
# paginated enumeration method is actually called) so that a bad configuration never changes the
# behavior of callers that do not paginate.
DEFAULT_BATCH_IDENTIFIERS_PAGE_SIZE: Optional[int] = None
"""Requests made without an explicit ``page_size`` return every match in a single page, which is
identical (in order, count and content) to the unpaginated enumeration."""

MAX_BATCH_IDENTIFIERS_PAGE_SIZE: int = 1000

_MAX_PAGE_SIZE_ENV_VAR = "GX_BATCH_ENUMERATION_MAX_PAGE_SIZE"

# An opaque, server-interpreted continuation token. The version prefix lets us evolve the
# encoding without silently interpreting old/new tokens as offsets into the wrong list.
_CURSOR_PREFIX = "gxp1:"
_CURSOR_VERSION = 1


class BatchPaginationError(ValueError):
    """Base class for errors raised while paging through batch identifiers."""


class InvalidPageSizeError(BatchPaginationError):
    """Raised when a requested page size is not a positive integer within the configured
    maximum."""

    def __init__(self, page_size: Any, max_page_size: int) -> None:
        self.page_size = page_size
        self.max_page_size = max_page_size
        super().__init__(
            "page_size must be a positive integer no greater than "
            f"{max_page_size}, but got {page_size!r}."
        )


class InvalidPageCursorError(BatchPaginationError):
    """Raised when a page cursor cannot be interpreted or no longer points into the set of
    matched batches.

    A malformed or stale cursor never causes enumeration to silently restart at the first
    page.
    """


class BatchPaginationConfigError(BatchPaginationError):
    """Raised when the configured default/maximum page sizes are not valid."""


@public_api
@dataclasses.dataclass(frozen=True)
class BatchIdentifiersPage:
    """A single page of batch identifiers returned by a paged batch enumeration.

    Args:
        batch_identifiers: The batch identifiers on this page, in the same order that an
            unpaginated enumeration would return them in.
        page_size: The effective page size used to produce this page. ``None`` means no
            limit was applied and every match is on this page.
        next_cursor: An opaque token to pass as ``cursor`` to fetch the next page. It is
            ``None`` when ``has_more`` is ``False``.
        has_more: Whether at least one more page of matched batches exists.
        total_count: The total number of matched batches across all pages, when requested
            with ``include_total_count=True``; otherwise ``None``.
        data_reference_cache_hit: Whether the data asset's data-reference cache was
            reused while producing this page. ``None`` for asset types without a
            data-reference cache; ``False`` means the cache was (re)built once, in the
            existing manner, during this call.
    """

    batch_identifiers: List[dict]
    page_size: Optional[int]
    next_cursor: Optional[str]
    has_more: bool
    total_count: Optional[int] = None
    data_reference_cache_hit: Optional[bool] = None

    @property
    def page_count(self) -> int:
        """The number of batch identifiers on this page."""
        return len(self.batch_identifiers)


@public_api
@dataclasses.dataclass(frozen=True)
class BatchIdentifiersCount:
    """The result of a count-only batch enumeration.

    Constructing a count never loads batch data and never builds additional batch
    definitions; if the data-reference cache is cold it is established once in the
    existing manner.

    Args:
        count: The number of batches matching the batch request.
        data_reference_cache_hit: Whether the data asset's data-reference cache was
            reused while computing the count. ``None`` for asset types without a
            data-reference cache.
    """

    count: int
    data_reference_cache_hit: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class _PaginationSettings:
    default_page_size: Optional[int]
    max_page_size: int


def _parse_positive_int(raw_value: str, env_var: str) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        raise BatchPaginationConfigError(  # noqa: TRY003 # FIXME CoP
            f"Environment variable {env_var} must be a positive integer, but got {raw_value!r}."
        ) from None
    if value <= 0:
        raise BatchPaginationConfigError(  # noqa: TRY003 # FIXME CoP
            f"Environment variable {env_var} must be a positive integer, but got {raw_value!r}."
        )
    return value


def get_pagination_settings() -> _PaginationSettings:
    """Resolve the effective default and maximum page sizes from configuration.

    The maximum is read (and validated) on every paginated call so configuration changes take
    effect without a restart, but callers that never paginate never pay this cost or risk a
    configuration failure.
    """
    max_raw = os.environ.get(  # noqa: TID251 # os.environ allowed in config files
        _MAX_PAGE_SIZE_ENV_VAR
    )
    max_page_size = (
        MAX_BATCH_IDENTIFIERS_PAGE_SIZE
        if max_raw is None or max_raw.strip() == ""
        else _parse_positive_int(max_raw, _MAX_PAGE_SIZE_ENV_VAR)
    )

    return _PaginationSettings(
        default_page_size=DEFAULT_BATCH_IDENTIFIERS_PAGE_SIZE,
        max_page_size=max_page_size,
    )


def validate_page_size(
    page_size: Optional[int],
    settings: Optional[_PaginationSettings] = None,
) -> Optional[int]:
    """Validate a caller-supplied page size, falling back to the configured default.

    Returns ``None`` when no limit should be applied (all matches in a single page).
    """
    if settings is None:
        settings = get_pagination_settings()

    if page_size is None:
        return settings.default_page_size

    # bool is a subclass of int but is never a meaningful page size.
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise InvalidPageSizeError(page_size, settings.max_page_size)

    if page_size <= 0:
        raise InvalidPageSizeError(page_size, settings.max_page_size)

    if page_size > settings.max_page_size:
        raise InvalidPageSizeError(page_size, settings.max_page_size)

    return page_size


def encode_page_cursor(offset: int, anchor: Optional[dict]) -> str:
    """Build an opaque continuation token for the item at ``offset - 1``.

    ``anchor`` is the batch identifier of the last item already returned. It lets the
    server resume after that exact item even if the cached match set was refreshed (and
    therefore shifted) between requests.
    """
    payload = {"v": _CURSOR_VERSION, "o": offset, "a": anchor}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{_CURSOR_PREFIX}{token}"


_INVALID_CURSOR_MESSAGE = (
    "cursor is not a valid batch-enumeration continuation token; "
    "use the next_cursor returned with a previous page."
)


def decode_page_cursor(cursor: str) -> Tuple[int, Optional[dict]]:
    """Decode and structurally validate a continuation token.

    Raises:
        InvalidPageCursorError: If the token is not a well-formed cursor produced by this
            module.
    """
    if not isinstance(cursor, str) or not cursor.startswith(_CURSOR_PREFIX):
        raise InvalidPageCursorError(_INVALID_CURSOR_MESSAGE)
    token = cursor[len(_CURSOR_PREFIX) :]
    # restore stripped padding
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload: Dict[str, Any] = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError) as err:
        raise InvalidPageCursorError(_INVALID_CURSOR_MESSAGE) from err

    offset, anchor = _validate_cursor_payload(payload)
    return offset, anchor


def _validate_cursor_payload(payload: Any) -> Tuple[int, Optional[dict]]:
    if not isinstance(payload, dict):
        raise InvalidPageCursorError("Malformed page cursor: payload must be an object.")  # noqa: TRY003 # FIXME CoP
    if payload.get("v") != _CURSOR_VERSION:
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            f"Unsupported page cursor version {payload.get('v')!r}; expected {_CURSOR_VERSION}."
        )

    offset = payload.get("o")
    anchor = payload.get("a")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            "Malformed page cursor: offset must be a non-negative integer."
        )
    if anchor is not None and not isinstance(anchor, dict):
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            "Malformed page cursor: anchor must be an object or null."
        )
    if offset == 0 and anchor is not None:
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            "Malformed page cursor: a cursor at offset 0 cannot carry an anchor."
        )
    if offset > 0 and anchor is None:
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            "Malformed page cursor: a cursor past offset 0 must carry an anchor."
        )

    return offset, anchor


def _resolve_page_start(
    batch_identifiers: List[dict],
    cursor: Optional[str],
) -> int:
    """Return the index at which the requested page begins.

    Anchoring on the last-returned identifier (rather than on a bare offset) means a
    refreshed cache cannot make adjacent pages overlap or skip items: when the match set
    has shifted, the anchor is relocated and enumeration resumes right after it.
    """
    if cursor is None:
        return 0

    offset, anchor = decode_page_cursor(cursor)
    total = len(batch_identifiers)

    if offset == 0:
        return 0
    # decode_page_cursor guarantees a non-null anchor whenever offset > 0
    assert anchor is not None

    boundary_index = offset - 1
    if boundary_index < total and batch_identifiers[boundary_index] == anchor:
        return offset

    # The boundary has shifted; relocate the anchor to preserve point-in-time continuity.
    try:
        anchor_index = batch_identifiers.index(anchor)
    except ValueError:
        # A well-formed cursor one position past the end yields an empty trailing page
        # rather than an "asset is empty" error. Anything else is a stale anchor.
        if offset >= total:
            return total
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            "The page cursor refers to a batch that is no longer present in the matched "
            "set; restart paging from the first page."
        ) from None

    return anchor_index + 1


def paginate_batch_identifiers(
    batch_identifiers: List[dict],
    page_size: Optional[int] = None,
    cursor: Optional[str] = None,
    include_total_count: bool = False,
    data_reference_cache_hit: Optional[bool] = None,
) -> BatchIdentifiersPage:
    """Slice a fully ordered list of batch identifiers into a validated page.

    The input list must already be in the same order that an unpaginated enumeration
    returns. This helper is deterministic: the same input and cursor always produce the
    same page.

    When neither ``page_size`` nor ``cursor`` is supplied the configuration is not even
    consulted, so a misconfigured page-size limit can never change the default,
    unpaginated behavior.
    """
    if cursor is not None and not isinstance(cursor, str):
        raise InvalidPageCursorError(  # noqa: TRY003 # FIXME CoP
            f"cursor must be a string returned as next_cursor on a previous page, got {cursor!r}."
        )

    if page_size is None and cursor is None:
        # No pagination requested: behave exactly like the unpaginated enumeration and never
        # touch the (potentially invalid) pagination configuration.
        effective_page_size = None
    else:
        effective_page_size = validate_page_size(page_size, get_pagination_settings())

    total = len(batch_identifiers)
    start = _resolve_page_start(batch_identifiers, cursor)

    if effective_page_size is None:
        end = total
    else:
        end = min(start + effective_page_size, total)

    page_items = batch_identifiers[start:end]
    has_more = end < total
    next_cursor = (
        encode_page_cursor(offset=end, anchor=batch_identifiers[end - 1]) if has_more else None
    )

    return BatchIdentifiersPage(
        batch_identifiers=page_items,
        page_size=effective_page_size,
        next_cursor=next_cursor,
        has_more=has_more,
        total_count=total if include_total_count else None,
        data_reference_cache_hit=data_reference_cache_hit,
    )

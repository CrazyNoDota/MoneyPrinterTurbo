"""Retry helpers with exponential backoff + jitter.

Used to wrap the project's external network calls (Pinterest scraping, stock
footage search/download, the LLM, the vision model, AI video generation) so a
single transient failure -- a timeout, a dropped connection, an HTTP 429/5xx --
does not abort a whole video generation. Permanent failures (e.g. an HTTP 4xx
other than 429) are not retried, so we fail fast on real misconfiguration.
"""

import random
import time
from functools import wraps
from typing import Callable, Tuple, Type

import requests
from loguru import logger

# Default set of exceptions worth retrying. ``is_retryable_http`` refines this
# for ``requests`` HTTP errors (only 429/5xx, never other 4xx).
_TRANSIENT_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def is_retryable_http(exc: BaseException) -> bool:
    """Whether an exception from a ``requests`` call is worth retrying.

    Connection/timeout errors are always transient. For ``HTTPError`` (raised by
    ``raise_for_status``) only 429 (rate limit) and 5xx (server) are retried; a
    400/401/403/404 indicates a real problem that a retry will not fix.
    """
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            return True  # no response attached -> treat as transient
        return status == 429 or 500 <= status < 600
    return False


def call_with_retry(
    fn: Callable,
    *args,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_retryable_http,
    description: str = "",
    **kwargs,
):
    """Call ``fn(*args, **kwargs)``, retrying on transient failures.

    Args:
        attempts: total number of tries (>= 1).
        base_delay: first backoff, doubled each retry, capped at ``max_delay``.
        jitter: add random 0..1x of the delay to avoid thundering herds.
        retry_on: predicate deciding whether a raised exception is retryable.
        description: label used in log messages.

    The last exception is re-raised once attempts are exhausted, so callers keep
    their existing try/except fallbacks.
    """
    attempts = max(int(attempts), 1)
    label = description or getattr(fn, "__name__", "call")
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last_exc = exc
            if attempt >= attempts or not retry_on(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay += random.random() * delay
            logger.warning(
                f"{label} failed (attempt {attempt}/{attempts}): {exc}; "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)

    # Unreachable (the loop either returns or raises), but keeps type-checkers happy.
    assert last_exc is not None
    raise last_exc


def with_retry(
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: Callable[[BaseException], bool] = is_retryable_http,
):
    """Decorator form of :func:`call_with_retry`."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_retry(
                fn,
                *args,
                attempts=attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter=jitter,
                retry_on=retry_on,
                description=fn.__name__,
                **kwargs,
            )

        return wrapper

    return decorator


__all__ = ["call_with_retry", "with_retry", "is_retryable_http"]

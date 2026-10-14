"""In-memory registry of user stop requests, keyed by thread id.

POST /threads/{id}/stop sets a flag here and the streaming loop polls it
between chunks. The flags are process-local, so this assumes a single API
worker.
"""

import threading
from typing import Callable

_flags: dict[str, bool] = {}
_lock = threading.Lock()


class UserStoppedException(Exception):
    """Raised inside a pipeline step when the user has requested a stop.

    Handled by ``agent_api.stream_generator`` like a stop seen between chunks.
    """


def make_stop_check(thread_id: str) -> Callable[[], None]:
    """Return a callable that raises UserStoppedException if a stop was requested for ``thread_id``.

    Long-running pipeline steps call it so a stop takes effect before the tool finishes.
    """
    def _check() -> None:
        if is_stop_requested(thread_id):
            raise UserStoppedException(thread_id)
    return _check


def request_stop(thread_id: str) -> None:
    """Flag ``thread_id`` for stopping (idempotent)."""
    with _lock:
        _flags[thread_id] = True


def is_stop_requested(thread_id: str) -> bool:
    """Return whether a stop is pending for ``thread_id``, without clearing it."""
    with _lock:
        return _flags.get(thread_id, False)


def consume_stop(thread_id: str) -> bool:
    """Atomically read and clear the stop flag; return True if it was set.

    Called when a stream ends so a stale flag cannot abort the next stream.
    """
    with _lock:
        was_set = _flags.pop(thread_id, False)
        return was_set


def clear_stop(thread_id: str) -> None:
    """Clear the stop flag for ``thread_id``, if any."""
    with _lock:
        _flags.pop(thread_id, None)

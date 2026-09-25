"""Make viewer failures visible: a full traceback on disk, a readable line in the GUI.

The GUI status bar is a viser markdown handle. Exception text carries brackets, backticks,
angle brackets and newlines, and interpolating it straight into markdown made the renderer
give up -- failures showed as "Markdown Failed to Render" with the real message lost, which
hid errors that already said exactly which extra to install. :func:`status_error` wraps the
message in a fenced block, which is opaque to the parser, and :func:`log_error` writes the
stack to :data:`~.config.LOG_PATH` and echoes it to stderr.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

from .config import LOG_PATH

__all__ = ["log_error", "status_error"]

_FENCE = "```"


def log_error(what: str, exc: BaseException, *, log_path: Path = LOG_PATH) -> str:
    """Append the full traceback to ``log_path``, echo it to stderr, return the summary line.

    ``what`` names the action that failed. The returned summary is what the GUI shows; the
    stack is read back from the log.
    """
    summary = f"{type(exc).__name__}: {exc}"
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    record = f"\n===== {datetime.now().isoformat(timespec='seconds')}  {what}\n{detail}"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(record)
    except OSError:  # a read-only home must never mask the original error
        pass
    print(record, file=sys.stderr, flush=True)
    return summary


def status_error(summary: str, *, log_path: Path = LOG_PATH) -> str:
    """Render ``summary`` as markdown that cannot break the renderer.

    Only a fence inside the message itself needs escaping; everything else survives
    verbatim, newlines included, so a multi-line repair hint arrives intact.
    """
    body = summary.replace(_FENCE, "'''")
    return f"**Error** (full traceback in `{log_path}`)\n\n{_FENCE}\n{body}\n{_FENCE}"

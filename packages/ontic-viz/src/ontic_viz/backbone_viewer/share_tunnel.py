"""Keep the viser share tunnel alive for a long-running viewer session.

viser's ``_simple_proxy`` only suppresses ``ConnectionError`` on teardown, so an
unclean TLS close (``ssl.SSLError``) kills the relay while the local server keeps
serving a dead public URL; the relay also expires URLs after 24 hours.
:func:`patch_viser_tunnel` retries the relay (same URL survives) and
:func:`start_share_watchdog` re-requests a fresh URL when the tunnel is gone.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

import viser

RETRY_DELAY_S = 0.5


def patch_viser_tunnel() -> bool:
    """Wrap viser's tunnel relay in a retry loop. True if this call patched it."""
    import viser._tunnel as viser_tunnel

    original = viser_tunnel._simple_proxy
    if getattr(original, "_ontic_resilient", False):
        return False

    async def resilient_simple_proxy(
        local_host: str,
        local_port: int,
        remote_host: str,
        remote_port: int,
        close_event: asyncio.Event,
    ) -> None:
        while True:
            try:
                await original(local_host, local_port, remote_host, remote_port, close_event)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                if close_event.is_set():
                    return
                await asyncio.sleep(RETRY_DELAY_S)

    resilient_simple_proxy._ontic_resilient = True  # type: ignore[attr-defined]
    viser_tunnel._simple_proxy = resilient_simple_proxy
    return True


def _tunnel_alive(server: viser.ViserServer) -> bool:
    tunnel = getattr(server, "_share_tunnel", None)
    if tunnel is None:
        return False
    if tunnel.get_status() not in ("ready", "connecting", "connected"):
        return False
    worker = getattr(tunnel, "_process", None) or getattr(tunnel, "_thread", None)
    return worker is None or worker.is_alive()


def start_share_watchdog(
    server: viser.ViserServer,
    on_url: Callable[[str | None], None],
    poll_seconds: float = 30.0,
) -> threading.Thread:
    """Re-request a share URL whenever the tunnel goes down, reporting it via ``on_url``."""

    def loop() -> None:
        while True:
            time.sleep(poll_seconds)
            try:
                if _tunnel_alive(server):
                    continue
                server._share_tunnel = None  # viser only builds a new tunnel when unset
                on_url(server.request_share_url(verbose=False))
            except Exception:
                pass

    thread = threading.Thread(target=loop, daemon=True, name="viser-share-watchdog")
    thread.start()
    return thread

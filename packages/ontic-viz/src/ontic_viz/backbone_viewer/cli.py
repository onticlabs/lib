"""Launch the backbone 3D viewer: ``ontic-backbone-viewer --port 8080``.

Port-forward (``ssh -L 8080:localhost:8080 <host>``) and open the printed URL, or
pass ``--share`` for a public share.viser.studio tunnel URL (kept alive and re-issued
by :mod:`share_tunnel`; watch stdout for ``SHARE URL:`` lines). Dataset roots default
to the dev-box paths in :data:`data_source.DEFAULT_ROOTS`; override per dataset with
``--<name>-root``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import viser

from .app import BackboneViewer
from .data_source import DEFAULT_ROOTS, dataset_names
from .share_tunnel import patch_viser_tunnel, start_share_watchdog


def build_parser(names: list[str]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu")
    p.add_argument("--stage", default="val", choices=["train", "val", "test"])
    p.add_argument(
        "--demo", action="store_true", help="open the analytic tracking demo (no weights required)"
    )
    p.add_argument(
        "--recording",
        action="append",
        default=[],
        help="saved viewer .npz run; repeat for comparisons",
    )
    p.add_argument(
        "--model-config", type=Path, help="JSON with backbone checkpoints and tracker defaults"
    )
    p.add_argument("--share", action="store_true", help="request a public viser share URL")
    p.add_argument("--presets", default=None, help="presets JSON store (default: user config)")
    for name in names:
        p.add_argument(
            f"--{name}-root", default=DEFAULT_ROOTS.get(name), help=f"{name} dataset root"
        )
    return p


def apply_model_config(viewer: BackboneViewer, config: dict) -> None:
    viewer._backbone_paths.update(config.get("backbone_checkpoints", {}))
    viewer._on_backbone_change()
    trackers = dict(config.get("trackers", {}))
    # Older configs describe MVTracker alone; never apply these paths to other trackers.
    legacy = config.get("tracker")
    if legacy:
        name = legacy.get("name", "mvtracker")
        trackers[name] = {**legacy, **trackers.get(name, {})}
    viewer.tracking.configure_model_files(trackers)


def main(argv: list[str] | None = None) -> None:
    names = dataset_names()
    args = build_parser(names).parse_args(argv)
    roots = {
        name: root
        for name in names
        if (root := getattr(args, f"{name.replace('-', '_')}_root")) is not None
    }

    server = viser.ViserServer(host=args.host, port=args.port)
    viewer = BackboneViewer(
        server, device=args.device, roots=roots, stage=args.stage, presets_path=args.presets
    )
    if args.model_config:
        config = json.loads(args.model_config.read_text())
        apply_model_config(viewer, config)
    if args.recording:
        viewer.register_recordings(args.recording)
    elif args.demo:
        viewer.load_demo()
    print(
        f"[backbone-viewer] serving on port {server.get_port()} (device={args.device}). "
        "Ctrl-C to quit."
    )

    def report_share_url(url: str | None) -> None:
        if url:
            print(f"[backbone-viewer] SHARE URL: {url}", flush=True)
        else:
            print(
                "[backbone-viewer] share URL request failed (no outbound access?); "
                "use SSH port-forward.",
                flush=True,
            )

    if args.share:
        patch_viser_tunnel()
        print("[backbone-viewer] requesting public share URL (share.viser.studio)...", flush=True)
        report_share_url(server.request_share_url())
        start_share_watchdog(server, report_share_url)
    try:
        server.sleep_forever()
    finally:
        viewer.close()
        server.stop()


if __name__ == "__main__":
    main()

"""CLI script to deterministically build SUMO city grid & OSM Palo Alto networks and scenario JSONs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm.config import load_scenario_config  # noqa: E402
from swarm.env.corridors import load_or_build_topology  # noqa: E402
from swarm.env.network_builder import build_all_networks_and_scenarios  # noqa: E402
from swarm.logging_utils import configure_logging, fail_loud_entrypoint, get_logger  # noqa: E402


@fail_loud_entrypoint(REPO_ROOT)
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build SUMO procedural grids, Downtown Palo Alto OSM network, and scenario JSONs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory (default: parent of scripts/).",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Require cached OSM XML without making external HTTP requests.",
    )
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="Build network XMLs only without overwriting configs/scenarios/*.json.",
    )
    args = parser.parse_args(argv)

    configure_logging("INFO")
    log = get_logger("scripts.build_networks")
    outputs = build_all_networks_and_scenarios(
        repo_root=args.repo_root,
        allow_osm_download=not args.offline_only,
        write_scenarios=not args.skip_scenarios,
    )

    for key, path in sorted(outputs.items()):
        if key.startswith("scenario:"):
            scen = load_scenario_config(path, must_exist_network_files=True, repo_root=args.repo_root)
            topo = load_or_build_topology(scen, repo_root=args.repo_root)
            tls_count = sum(1 for j in topo.junctions.values() if j.tls_id is not None)
            log.info(
                "Validated %s -> corridors=%d, junctions=%d, tls=%d",
                scen.scenario_id,
                len(topo.corridors),
                len(topo.junctions),
                tls_count,
            )
        else:
            log.info("Built asset %s -> %s (%.1f KB)", key, path, path.stat().st_size / 1024.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

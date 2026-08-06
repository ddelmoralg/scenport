"""Small command line helper. The real entry points are the scripts in scripts/."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scenport", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version")
    sub.add_parser("list", help="show the registered forecasters and allocators")

    check = sub.add_parser("check", help="load a config and report what it resolves to")
    check.add_argument("config")
    check.add_argument("--set", dest="overrides", action="append", default=[])

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "list":
        from . import alloc
        from .forecast import CPU_FORECASTERS, GPU_FORECASTERS

        print("forecasters (cpu):", ", ".join(CPU_FORECASTERS))
        print("forecasters (gpu):", ", ".join(GPU_FORECASTERS))
        print("allocators       :", ", ".join(alloc.available()))
        return 0

    from . import config as cfgmod

    cfg = cfgmod.load(args.config, args.overrides)
    root = cfgmod.repo_root()
    for key in ("paths.panel", "paths.runs", "paths.artifacts"):
        path = cfg.resolve(key, root)
        print(f"{key:20s} {path}  {'(exists)' if path.exists() else '(missing)'}")
    print(f"{'rebalance window':20s} {cfg['backtest.first_rebalance']} .. "
          f"{cfg['backtest.last_rebalance']} every {cfg['backtest.rebalance']}")
    print(f"{'horizon / train':20s} {cfg['backtest.horizon_days']}d / "
          f"{cfg['backtest.train_days']}d")
    print(f"{'scenarios':20s} {cfg['scenarios.n_paths']} paths, "
          f"{cfg['scenarios.coupling']} coupling")
    print(f"{'constraints':20s} w<={cfg['alloc.w_max']}, w>={cfg['alloc.w_min']}, "
          f"K={cfg['alloc.cardinality']}, alpha={cfg['alloc.cvar_alpha']}, "
          f"gamma={cfg['alloc.gamma']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI for inspecting a disk-backed `ReplayBuffer`.

Usage:
    python -m proteinfoundation.replay.buffer_cli stats <dir>
"""

from __future__ import annotations

import argparse
import json

from proteinfoundation.replay.buffer import ReplayBuffer


def _cmd_stats(args: argparse.Namespace) -> None:
    buffer = ReplayBuffer.load(args.dir)
    print(json.dumps(buffer.stats().as_dict(), indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a CPSea replay buffer.")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_parser = sub.add_parser("stats", help="Print buffer statistics as JSON.")
    stats_parser.add_argument("dir", help="Replay buffer directory (as written by ReplayBuffer.save).")
    stats_parser.set_defaults(func=_cmd_stats)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

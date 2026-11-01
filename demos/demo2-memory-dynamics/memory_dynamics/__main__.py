"""Command-line entry point for the memory dynamics experiment."""

import argparse
import json
from pathlib import Path
import sqlite3

from .model import DynamicsConfig
from .providers import APIProvider, OfflineProvider
from .runtime import Runtime


def _read_json(path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Memory weight/stability experiment (Python 3.10+)")
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--db", default="data/runtime.sqlite3")
    parser.add_argument("--dynamics", choices=("weight-only", "two-state"), default="two-state")
    parser.add_argument("--decay-rate", type=float, default=0.12)
    parser.add_argument("--stability-protection", type=float, default=0.85)
    parser.add_argument("--recovery-rate", type=float, default=0.65)
    parser.add_argument("--stability-gain", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=3)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Initialize memories from a scenario JSON file")
    init.add_argument("scenario")

    run = sub.add_parser("run", help="Run a complete scripted scenario")
    run.add_argument("scenario")
    run.add_argument("--output", help="Optional path for a JSON result file")

    step = sub.add_parser("step", help="Run one cognitive round; omit text for an internal tick")
    step.add_argument("text", nargs="?")
    retrieve = sub.add_parser("retrieve", help="Inspect retrieval without advancing the clock")
    retrieve.add_argument("text")
    sub.add_parser("tick", help="Run one internal round without external input")
    sub.add_parser("state", help="Show current state and recent traces")

    args = parser.parse_args()
    try:
        provider = APIProvider() if args.mode == "api" else OfflineProvider()
        config = DynamicsConfig(
            mode=args.dynamics,
            decay_rate=args.decay_rate,
            stability_protection=args.stability_protection,
            recovery_rate=args.recovery_rate,
            stability_gain=args.stability_gain,
            top_k=args.top_k,
        )
        runtime = Runtime(args.db, provider, config)

        if args.command == "init":
            scenario = _read_json(args.scenario)
            memories = scenario.get("memories") if isinstance(scenario, dict) else scenario
            result = runtime.initialize(memories)
        elif args.command == "run":
            scenario = _read_json(args.scenario)
            if not isinstance(scenario, dict):
                raise ValueError("Scenario must be a JSON object")
            runtime.initialize(scenario.get("memories"))
            turns = scenario.get("turns")
            if not isinstance(turns, list):
                raise ValueError("Scenario turns must be a list")
            traces = []
            for turn in turns:
                if isinstance(turn, str):
                    text = turn
                elif isinstance(turn, dict):
                    text = turn.get("input")
                elif turn is None:
                    text = None
                else:
                    raise ValueError("Each turn must be text, an object with input, or null")
                traces.append(runtime.step(text))
            result = {
                "scenario": scenario.get("name", Path(args.scenario).stem),
                "mode": provider.mode,
                "provider": {
                    "chat": getattr(provider, "chat_profile", None),
                    "embedding": provider.profile,
                },
                "config": config.to_dict(),
                "traces": traces,
                "final": runtime.inspect()["state"],
            }
            if args.output:
                _write_json(args.output, result)
        elif args.command == "step":
            result = runtime.step(args.text)
        elif args.command == "tick":
            result = runtime.step()
        elif args.command == "retrieve":
            result = runtime.retrieve(args.text)
        else:
            result = runtime.inspect()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

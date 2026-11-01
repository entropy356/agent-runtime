import argparse
import json
import sqlite3
from .providers import APIProvider, OfflineProvider
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description="Inspectable emotion/memory experiment (Python 3.10+)")
    parser.add_argument("--mode", choices=("offline", "api"), default="offline")
    parser.add_argument("--db", default="data/runtime.sqlite3")
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--recall-limit", type=int, default=2,
                        help="Model-initiated extra retrievals per round (0 disables recall)")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Open the local dashboard")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--interval", type=float, default=10)
    sub.add_parser("seed", help="Initialize hand-authored example; requires empty DB")
    sub.add_parser("state")
    sub.add_parser("replaced", help="Debug log of memories removed by replacement")
    step = sub.add_parser("step")
    step.add_argument("text", nargs="?", help="Omit for a tick without external input")
    step.add_argument("--no-emotion", action="store_true")
    for name in ("save", "restore"):
        sub.add_parser(name).add_argument("name")
    compare = sub.add_parser("compare")
    compare.add_argument("text", nargs="?", default="我们再一起试一次吧。")
    compare.add_argument("--baseline", default="baseline")
    compare.add_argument("--a", default="A")
    compare.add_argument("--b", default="B")
    args = parser.parse_args()
    try:
        runtime = Runtime(args.db, APIProvider() if args.mode == "api" else OfflineProvider(),
                          args.rho, args.recall_limit)
        if args.command == "serve":
            from .server import serve
            serve(runtime, args.port, args.interval)
            return
        if args.command == "seed":
            result = runtime.seed()
        elif args.command == "state":
            result = runtime.inspect()
        elif args.command == "replaced":
            result = runtime.replaced_log()
        elif args.command == "step":
            result = runtime.step(args.text, not args.no_emotion)
        elif args.command == "compare":
            result = runtime.compare(args.text, args.baseline, args.a, args.b)
        else:
            result = getattr(runtime, args.command)(args.name)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.IntegrityError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

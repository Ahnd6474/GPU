import argparse
import json

from jakal_runtime import Runtime, default_host_only_options


def main() -> int:
    parser = argparse.ArgumentParser(description="Jakal runtime app adapter")
    parser.add_argument("command", choices=["doctor", "paths"])
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()

    runtime = Runtime(default_host_only_options() if args.host_only else None)
    if args.command == "doctor":
        print(json.dumps({"paths": runtime.paths(), "backends": runtime.backend_statuses()}, indent=2))
        return 0
    if args.command == "paths":
        print(json.dumps(runtime.paths(), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

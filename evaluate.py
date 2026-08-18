import argparse
import json
from pathlib import Path

from kegcl.metrics import evaluate_complexes, read_complexes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output")
    args = parser.parse_args()
    metrics = evaluate_complexes(
        read_complexes(args.predictions),
        read_complexes(args.gold),
        args.threshold,
    )
    payload = {"threshold": args.threshold, **metrics}
    print(json.dumps(payload, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

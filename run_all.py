import argparse
import subprocess
import sys
from pathlib import Path

from kegcl.config import load_config


def run(script: Path, arguments) -> None:
    subprocess.run([sys.executable, str(script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=["bio", "col", "dip", "k14"])
    parser.add_argument("--gold")
    parser.add_argument("--device")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    project_root = Path(config["project_root"])
    if not args.skip_preprocess:
        run(project_root / "preprocess.py", ["--config", str(config_path), "--all"])
    for dataset in args.datasets:
        train_args = ["--config", str(config_path), "--dataset", dataset]
        if args.device:
            train_args.extend(["--device", args.device])
        run(project_root / "train.py", train_args)
        run(
            project_root / "cluster.py",
            ["--config", str(config_path), "--dataset", dataset],
        )
        if args.gold:
            prediction = Path(config["output_root"]) / "runs" / dataset / "complexes.txt"
            output = Path(config["output_root"]) / "runs" / dataset / "evaluation.json"
            gold = Path(args.gold)
            if not gold.is_absolute():
                gold = project_root / gold
            run(
                project_root / "evaluate.py",
                [
                    "--predictions",
                    str(prediction),
                    "--gold",
                    str(gold),
                    "--threshold",
                    str(config["evaluation"]["overlap_threshold"]),
                    "--output",
                    str(output),
                ]
            )


if __name__ == "__main__":
    main()

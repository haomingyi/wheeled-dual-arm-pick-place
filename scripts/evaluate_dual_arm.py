"""Run repeated headless evaluations for the wheeled dual-arm demo."""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dual_arm_pick_place import run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the wheeled dual-arm pick-and-place demo.")
    parser.add_argument("--runs", type=int, default=5, help="Number of headless runs to execute.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/eval_dual_arm"))
    parser.add_argument("--summary-file", type=Path, default=Path("logs/eval_dual_arm_summary.csv"))
    parser.add_argument("--log-every", type=int, default=10, help="Record one row every N simulation steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for run_idx in range(args.runs):
        log_file = args.output_dir / f"run_{run_idx:03d}.csv"
        print(f"Run {run_idx + 1}/{args.runs}: {log_file}", flush=True)
        summary = run(headless=True, log_file=log_file, log_every=args.log_every)
        summary["run"] = run_idx
        summary["log_file"] = str(log_file)
        results.append(summary)
        print(
            "  success={success} final_placement_error={final_placement_error:.4f} "
            "max_workpiece_height={max_workpiece_height:.4f} final_base_error={final_base_error:.4f}".format(
                **summary
            ),
            flush=True,
        )

    with args.summary_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run",
                "success",
                "final_placement_error",
                "max_workpiece_height",
                "final_base_error",
                "final_stage",
                "log_file",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    successes = sum(1 for result in results if result["success"])
    print("\nEvaluation summary")
    print(f"Runs: {args.runs}")
    print(f"Successes: {successes}")
    print(f"Success rate: {successes / args.runs:.2%}")
    print(f"Summary CSV: {args.summary_file}")


if __name__ == "__main__":
    main()

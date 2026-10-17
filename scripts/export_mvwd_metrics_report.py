"""Add METRICS.md beside an existing metrics.json without recomputing metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvwd_metrics.report import write_metrics_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Experiment snapshot recorded by the runner")
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() != args.evaluation.resolve():
        parser.error("The report must be next to the existing evaluation metrics.json")
    print(write_metrics_markdown(args.evaluation))


if __name__ == "__main__":
    main()

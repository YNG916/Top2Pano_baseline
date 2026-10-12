"""Add GT/coarse/prediction reports to existing finished inference without a GPU."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvwd_runtime import load_config
from mvwd_prediction_comparison import export_comparisons, export_bevs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--bev-only", action="store_true", help="Add recorded BEV to existing GT comparisons")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.bev_only:
        export_bevs(args.predictions, args.output)
    else:
        export_comparisons(args.predictions, args.output, config["data"].get("cache_root"))


if __name__ == "__main__":
    main()

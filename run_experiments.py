"""
Convenience script: run all 4 experiments sequentially,
then generate learning curves and comparison charts.

Usage:
    python run_experiments.py --data_dir path/to/pokemon/dataset
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the Pokemon dataset folder")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training and only generate evaluation plots")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_train:
        print("Starting training for all experiments...")
        ret = subprocess.run(
            [python, "train.py", "--data_dir", args.data_dir, "--output_dir", args.output_dir],
        )
        if ret.returncode != 0:
            print("Training failed.")
            sys.exit(1)

    print("\nGenerating evaluation plots and classification reports...")
    subprocess.run(
        [python, "evaluate.py",
         "--data_dir", args.data_dir,
         "--results_dir", args.output_dir],
    )

    print("\nDone! Launch the demo with:")
    print("  streamlit run demo.py")


if __name__ == "__main__":
    main()

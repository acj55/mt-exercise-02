
""" The script takes one or more TSV log files as input and creates:
- tables for training perplexity by epoch
- tables for validation perplexity by epoch
- a table with final test perplexity
- a line plot for training perplexity
- a line plot for validation perplexity """

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def infer_model_label(path: Path) -> str:
    """
    infers model label from the filename.
    Examples:
        log_dp_0.0.tsv -> Dropout 0.0
        run_dropout_0.3.txt -> Dropout 0.3
        anything_else.tsv -> anything_else
    """
    stem = path.stem
    match = re.search(r'(?:dp|dropout)[-_]?([0-9]*\.?[0-9]+)', stem, flags=re.IGNORECASE)
    if match:
        return f"Dropout {match.group(1)}"
    return stem


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"split", "epoch", "batch", "loss", "ppl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["epoch"] = df["epoch"].astype(str).str.strip()
    df["batch"] = df["batch"].astype(str).str.strip()
    df["ppl"] = pd.to_numeric(df["ppl"], errors="raise")
    df["loss"] = pd.to_numeric(df["loss"], errors="raise")

    # Numeric epoch for train/valid rows; test rows like "final" become NaN.
    df["epoch_num"] = pd.to_numeric(df["epoch"], errors="coerce")

    # Numeric batch for sorting train rows; "end" becomes NaN.
    df["batch_num"] = pd.to_numeric(df["batch"], errors="coerce")
    return df


def epoch_level_series(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, float]:
    """
    Returns:
        train_epoch_ppl: pd.Series indexed by epoch number
        valid_epoch_ppl: pd.Series indexed by epoch number
        test_ppl: float
    """
    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()

    # For each epoch, keep the last train entry (highest batch number).
    train_last = (
        train.sort_values(["epoch_num", "batch_num"])
             .groupby("epoch_num", as_index=False)
             .tail(1)
             .sort_values("epoch_num")
    )
    train_epoch_ppl = train_last.set_index("epoch_num")["ppl"]
    train_epoch_ppl.index = train_epoch_ppl.index.astype(int)

    # Valid usually has one row per epoch; if not, keep the last one per epoch.
    valid_last = (
        valid.sort_values(["epoch_num"])
             .groupby("epoch_num", as_index=False)
             .tail(1)
             .sort_values("epoch_num")
    )
    valid_epoch_ppl = valid_last.set_index("epoch_num")["ppl"]
    valid_epoch_ppl.index = valid_epoch_ppl.index.astype(int)

    if test.empty:
        test_ppl = float("nan")
    else:
        test_ppl = float(test.iloc[-1]["ppl"])

    return train_epoch_ppl, valid_epoch_ppl, test_ppl


def build_tables(log_paths: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_table = pd.DataFrame()
    valid_table = pd.DataFrame()
    test_table = pd.DataFrame(index=["Final"])

    for path in log_paths:
        label = infer_model_label(path)
        df = load_log(path)
        train_series, valid_series, test_ppl = epoch_level_series(df)

        train_table[label] = train_series
        valid_table[label] = valid_series
        test_table[label] = [test_ppl]

    train_table.index.name = "Epoch"
    valid_table.index.name = "Epoch"
    test_table.index.name = "Split"

    return train_table.sort_index(), valid_table.sort_index(), test_table


def save_tables(train_table: pd.DataFrame, valid_table: pd.DataFrame, test_table: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    train_table.to_csv(outdir /  "train_ppl.tsv", sep="\t", float_format="%.6f")
    valid_table.to_csv(outdir / "valid_ppl.tsv", sep="\t", float_format="%.6f")
    test_table.to_csv(outdir / "test_ppl.tsv", sep="\t", float_format="%.6f")

    train_table.to_markdown(outdir / "train_ppl.md", floatfmt=".2f")
    valid_table.to_markdown(outdir / "valid_ppl.md", floatfmt=".2f")
    test_table.to_markdown(outdir /  "test_ppl.md", floatfmt=".2f")


def plot_table(df: pd.DataFrame, title: str, ylabel: str, outfile: Path) -> None:
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], marker="o", label=column)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(df.index)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create perplexity tables and training/validation line plots from dropout experiment log files."
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help="Paths to log TSV files, e.g. logs/log_dp_0.0.tsv logs/log_dp_0.2.tsv ..."
    )
    parser.add_argument(
        "--outdir",
        default="./results",
        help="Output directory (default: results)"
    )
    args = parser.parse_args()

    log_paths = [Path(p) for p in args.logs]
    outdir = Path(args.outdir)

    train_table, valid_table, test_table = build_tables(log_paths)
    save_tables(train_table, valid_table, test_table, outdir)

    plot_table(
        train_table,
        title="Training Perplexity by Epoch",
        ylabel="Training Perplexity",
        outfile=outdir / "training_ppl.png",
    )
    plot_table(
        valid_table,
        title="Validation Perplexity by Epoch",
        ylabel="Validation Perplexity",
        outfile=outdir / "validation_ppl.png",
    )

    print(f"Saved outputs to: {outdir}")
    print(f"- {outdir / 'train_ppl_by_epoch.tsv'}")
    print(f"- {outdir / 'valid_ppl_by_epoch.tsv'}")
    print(f"- {outdir / 'test_ppl.tsv'}")
    print(f"- {outdir / 'training_ppl.png'}")
    print(f"- {outdir / 'validation_ppl.png'}")


if __name__ == "__main__":
    main()
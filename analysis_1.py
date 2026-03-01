import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from utility import STOCK_LIST


def descriptive_analysis(data_root: Path) -> pd.DataFrame:
    input_factor_return = data_root / "factor_return_df.csv"
    input_stock_data_folder = data_root / "stock_data_folder"

    factor_return_df = pd.read_csv(input_factor_return, index_col=0)
    factor_return_df.index.name = "Date"
    base_index = factor_return_df.index

    stock_return_factor_dic = {}
    for stock in STOCK_LIST:
        stock_df = pd.read_csv(input_stock_data_folder / f"{stock}.csv", index_col=0)
        stock_df["return"] = (stock_df["Close"] - stock_df["Close"].shift(1)) / stock_df["Close"].shift(1)
        stock_df = stock_df.merge(pd.DataFrame(index=base_index), how="right", on="Date")
        stock_df.dropna(inplace=True)
        stock_return_factor_dic[stock] = stock_df["return"]

    stock_summary_df = pd.DataFrame(stock_return_factor_dic)
    descriptive_table = stock_summary_df.describe().T
    descriptive_table["skewness"] = stock_summary_df.skew().tolist()
    descriptive_table["kurtosis"] = stock_summary_df.kurtosis().tolist()
    return descriptive_table[["min", "max", "mean", "std", "skewness", "kurtosis"]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Descriptive statistics for stock returns")
    parser.add_argument("--data-root", type=Path, default=Path("data_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/descriptive_stats.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    table = descriptive_analysis(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(tabulate(table, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()

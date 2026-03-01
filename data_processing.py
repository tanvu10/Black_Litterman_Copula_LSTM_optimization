import argparse
from pathlib import Path

import pandas as pd

from utility import STOCK_LIST, get_technical_indicators


def process_stock_datasets(data_root: Path) -> None:
    input_sub_factor_folder = data_root / "sub_factor_folder"
    input_factor_return = data_root / "factor_return_df.csv"
    input_stock_data_folder = data_root / "stock_data_folder"
    output_stock_data_dl_folder = data_root / "stock_data_DL_folder"
    output_stock_return_df = data_root / "stock_return_df.csv"

    factor_return_df = pd.read_csv(input_factor_return, index_col=0)
    factor_return_df.index.name = "Date"
    base_index = factor_return_df.index
    factor_list = factor_return_df.columns

    sub_factor_dic = {}
    for factor in factor_list:
        factor_df = pd.read_csv(input_sub_factor_folder / f"{factor}.csv", index_col=0)
        sub_factor_dic[factor] = factor_df.merge(pd.DataFrame(index=base_index), how="right", on="Date")

    output_stock_data_dl_folder.mkdir(parents=True, exist_ok=True)

    stock_return_factor_dic = {}
    for stock in STOCK_LIST:
        stock_df = pd.read_csv(input_stock_data_folder / f"{stock}.csv", index_col=0)
        stock_df["return"] = (stock_df["Close"] - stock_df["Close"].shift(1)) / stock_df["Close"].shift(1)
        stock_df = get_technical_indicators(stock_df)

        for factor in factor_list:
            stock_df[factor] = sub_factor_dic[factor][stock] * factor_return_df[factor]

        stock_df = stock_df.merge(pd.DataFrame(index=base_index), how="right", on="Date")
        stock_df.dropna(inplace=True)
        stock_df.drop(columns=["Close", "High", "Low", "Open", "Volume"], inplace=True)
        stock_return_factor_dic[stock] = stock_df
        stock_df.to_csv(output_stock_data_dl_folder / f"{stock}.csv")

    stock_return_df = {}
    for stock in STOCK_LIST:
        stock_df = pd.read_csv(input_stock_data_folder / f"{stock}.csv", index_col=0)
        stock_df[stock] = (stock_df["Close"] - stock_df["Close"].shift(1)) / stock_df["Close"].shift(1)
        stock_df = stock_df.merge(pd.DataFrame(index=base_index), how="right", on="Date")
        stock_return_df[stock] = stock_df[stock]

    pd.DataFrame(stock_return_df).to_csv(output_stock_return_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DL and return datasets from raw factor and stock inputs")
    parser.add_argument("--data-root", type=Path, default=Path("data_v2"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_stock_datasets(args.data_root)


if __name__ == "__main__":
    main()

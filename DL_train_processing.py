import argparse
from pathlib import Path

import pandas as pd

from LTSM_train import DL_train
from utility import STOCK_LIST


def generate_predictions(
    data_root: Path,
    start_date: str,
    end_date: str,
    time_step: int,
    epochs: int,
    batch_size: int,
) -> pd.DataFrame:
    input_stock_data_dl_folder = data_root / "stock_data_DL_folder"

    combine_dict = {}

    for stock in STOCK_LIST:
        stock_df = pd.read_csv(input_stock_data_dl_folder / f"{stock}.csv")
        stock_df["Date"] = stock_df["Date"].apply(pd.Timestamp)
        stock_df.set_index("Date", inplace=True)

        combine_dict[stock] = {f"{stock}_pred": [], "Date": []}

        for dt in pd.bdate_range(start_date, end_date, freq="B"):
            train_df = stock_df[stock_df.index < dt]
            if len(train_df) <= time_step + 1:
                continue
            prediction = DL_train(
                train_df,
                time_step=time_step,
                epochs=epochs,
                batch_size=batch_size,
            )
            combine_dict[stock][f"{stock}_pred"].append(prediction)
            combine_dict[stock]["Date"].append(dt)

    current_df = pd.DataFrame(combine_dict[STOCK_LIST[0]]).set_index("Date")

    for stock in STOCK_LIST[1:]:
        prediction_df = pd.DataFrame(combine_dict[stock]).set_index("Date")
        current_df = current_df.merge(prediction_df, how="left", on="Date")

    return current_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate stock return forecasts with LSTM")
    parser.add_argument("--data-root", type=Path, default=Path("data_v2"))
    parser.add_argument("--start-date", type=str, default="2021-10-01")
    parser.add_argument("--end-date", type=str, default="2022-04-01")
    parser.add_argument("--time-step", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions = generate_predictions(
        data_root=args.data_root,
        start_date=args.start_date,
        end_date=args.end_date,
        time_step=args.time_step,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    predictions.to_csv(args.data_root / "stock_prediction.csv")


if __name__ == "__main__":
    main()

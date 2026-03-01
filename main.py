import argparse
from pathlib import Path

from BL_optimization import BacktestConfig, run_backtest
from DL_train_processing import generate_predictions
from analysis_1 import descriptive_analysis
from analysis_2 import sensitive_check
from data_processing import process_stock_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research pipeline for BL-Copula-LSTM optimization")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-data")
    prep.add_argument("--data-root", type=Path, default=Path("data_v2"))

    predict = subparsers.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("data_v2"))
    predict.add_argument("--start-date", type=str, default="2021-10-01")
    predict.add_argument("--end-date", type=str, default="2022-04-01")
    predict.add_argument("--time-step", type=int, default=30)
    predict.add_argument("--epochs", type=int, default=100)
    predict.add_argument("--batch-size", type=int, default=64)

    backtest = subparsers.add_parser("backtest")
    backtest.add_argument("--data-root", type=Path, default=Path("data_v2"))
    backtest.add_argument("--output-dir", type=Path, default=Path("outputs"))
    backtest.add_argument("--upper-bound", type=float, default=0.5)
    backtest.add_argument("--tau", type=float, default=0.01)
    backtest.add_argument("--delta", type=float, default=2.5)
    backtest.add_argument("--confidence-level", type=float, default=0.9)
    backtest.add_argument("--plot", action="store_true")

    desc = subparsers.add_parser("describe")
    desc.add_argument("--data-root", type=Path, default=Path("data_v2"))
    desc.add_argument("--output", type=Path, default=Path("outputs/descriptive_stats.csv"))

    sense = subparsers.add_parser("sensitivity")
    sense.add_argument("--data-root", type=Path, default=Path("data_v2"))
    sense.add_argument("--output-root", type=Path, default=Path("outputs/sensitive_check"))
    sense.add_argument("--upper-bound", type=float, default=0.5)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare-data":
        process_stock_datasets(data_root=args.data_root)
    elif args.command == "predict":
        predictions = generate_predictions(
            data_root=args.data_root,
            start_date=args.start_date,
            end_date=args.end_date,
            time_step=args.time_step,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        predictions.to_csv(args.data_root / "stock_prediction.csv")
    elif args.command == "backtest":
        config = BacktestConfig(
            data_root=args.data_root,
            upper_bound=args.upper_bound,
            tau=args.tau,
            delta=args.delta,
            confidence_level=args.confidence_level,
        )
        run_backtest(config=config, output_dir=args.output_dir, plot=args.plot)
    elif args.command == "describe":
        table = descriptive_analysis(data_root=args.data_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output)
    elif args.command == "sensitivity":
        sensitive_check(data_root=args.data_root, output_root=args.output_root, upper_bound=args.upper_bound)


if __name__ == "__main__":
    main()

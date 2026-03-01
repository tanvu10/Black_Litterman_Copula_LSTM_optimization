import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility import BL_max_sharpe, BL_processing, copula_max_sharpe, drawdown, normal_max_sharpe, sharpe


@dataclass
class BacktestConfig:
    data_root: Path
    upper_bound: float = 0.5
    tau: float = 0.01
    delta: float = 2.5
    confidence_level: float = 0.9

    @property
    def cov_dirs(self) -> Dict[str, Path]:
        return {
            "Clayton": self.data_root / "cov_matrix" / "Clayton",
            "Gauss": self.data_root / "cov_matrix" / "Gauss",
            "Frank": self.data_root / "cov_matrix" / "Frank",
            "Gumbel": self.data_root / "cov_matrix" / "Gumbel",
        }


def _load_series(csv_path: Path, index_col: int = 0) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=index_col)


def _date_list(gauss_cov_dir: Path) -> List[str]:
    files = sorted(gauss_cov_dir.glob("*.csv"))
    return [file.stem for file in files]


def run_backtest(config: BacktestConfig, output_dir: Path, plot: bool = False) -> None:
    mkt_cap_weight_df = _load_series(config.data_root / "mkt_cap_weight.csv")
    stock_prediction = _load_series(config.data_root / "stock_prediction.csv")
    stock_return_df = _load_series(config.data_root / "stock_return_df.csv").dropna()

    dates = _date_list(config.cov_dirs["Gauss"])
    if not dates:
        raise FileNotFoundError("No covariance files found in Gauss directory.")

    strategy_returns = {
        "normal": [0.0],
        "BL_clayton_copula": [0.0],
        "BL_gauss_copula": [0.0],
        "BL_frank_copula": [0.0],
        "BL_gumbel_copula": [0.0],
        "only_BL": [0.0],
    }

    for date in dates:
        current_return_df = stock_return_df.loc[:date].iloc[:-1, :]
        if current_return_df.empty:
            continue

        prior_cov = {
            copula: pd.read_csv(path / f"{date}.csv", index_col=0).to_numpy(dtype=np.float64)
            for copula, path in config.cov_dirs.items()
        }

        mkt_weight = np.array(mkt_cap_weight_df.loc[date]).reshape(-1, 1)
        q_view = np.array(stock_prediction.loc[date]).reshape(-1, 1)

        bl_results = {}
        for copula, cov in prior_cov.items():
            bl_results[copula] = BL_processing(
                prior_cov=cov,
                tau=config.tau,
                delta=config.delta,
                mkt_weight=mkt_weight,
                confidence_level=config.confidence_level,
                Q=q_view,
            )

        norm_mean_bl, norm_cov_bl = BL_processing(
            prior_cov=current_return_df.cov().to_numpy(dtype=np.float64),
            tau=config.tau,
            delta=config.delta,
            mkt_weight=mkt_weight,
            confidence_level=config.confidence_level,
            Q=q_view,
        )

        normal_weight = normal_max_sharpe(current_return_df, config.upper_bound)
        strategy_returns["normal"].append(float(stock_return_df.loc[date].dot(normal_weight)))

        copula_to_key = {
            "Clayton": "BL_clayton_copula",
            "Gauss": "BL_gauss_copula",
            "Frank": "BL_frank_copula",
            "Gumbel": "BL_gumbel_copula",
        }

        for copula, key in copula_to_key.items():
            mu_bl, cov_bl = bl_results[copula]
            bl_weight = BL_max_sharpe(upperbound=config.upper_bound, Q=cov_bl, mu=mu_bl)
            strategy_returns[key].append(float(stock_return_df.loc[date].dot(bl_weight)))

        only_bl_weight = BL_max_sharpe(upperbound=config.upper_bound, Q=norm_cov_bl, mu=norm_mean_bl)
        strategy_returns["only_BL"].append(float(stock_return_df.loc[date].dot(only_bl_weight)))

        # Also solve copula-only portfolios to keep methodology complete.
        _ = {
            copula: copula_max_sharpe(
                upperbound=config.upper_bound,
                Q=prior_cov[copula],
                dataframe=current_return_df,
            )
            for copula in prior_cov
        }

    indexed_dates = ["start"] + dates[: len(strategy_returns["normal"]) - 1]
    portfolio_df = pd.DataFrame({"Date": indexed_dates, **strategy_returns}).set_index("Date")

    stat_index = list(strategy_returns.keys())
    stat_df = pd.DataFrame(index=stat_index, columns=["Sharpe", "Average Drawdown", "Max Drawdown"])

    for strategy in stat_index:
        dd = drawdown(portfolio_df[strategy])
        stat_df.loc[strategy, "Sharpe"] = sharpe(portfolio_df[strategy])
        stat_df.loc[strategy, "Average Drawdown"] = float(np.mean(dd))
        stat_df.loc[strategy, "Max Drawdown"] = float(np.max(dd))

    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio_df.to_csv(output_dir / "portfolio_returns.csv")
    stat_df.to_csv(output_dir / "portfolio_stats.csv")

    if plot:
        ((portfolio_df + 1).cumprod() - 1).plot(figsize=(12, 6), title="Cumulative Strategy Returns")
        plt.tight_layout()
        plt.savefig(output_dir / "cumulative_returns.png", dpi=200)
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Black-Litterman + Copula portfolio backtest")
    parser.add_argument("--data-root", type=Path, default=Path("data_v2"), help="Folder containing prepared CSV inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Folder to save backtest outputs")
    parser.add_argument("--upper-bound", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=2.5)
    parser.add_argument("--confidence-level", type=float, default=0.9)
    parser.add_argument("--plot", action="store_true", help="Save cumulative return plot")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BacktestConfig(
        data_root=args.data_root,
        upper_bound=args.upper_bound,
        tau=args.tau,
        delta=args.delta,
        confidence_level=args.confidence_level,
    )
    run_backtest(config=config, output_dir=args.output_dir, plot=args.plot)


if __name__ == "__main__":
    main()

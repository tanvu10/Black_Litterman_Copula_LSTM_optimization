import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utility import BL_max_sharpe, BL_processing, sharpe


def _load_cov(cov_dir: Path, date: str) -> np.ndarray:
    return pd.read_csv(cov_dir / f"{date}.csv", index_col=0).to_numpy(dtype=np.float64)


def sharpe_check(
    data_root: Path,
    dates: List[str],
    tau: float,
    delta: float,
    confidence_level: float,
    upper_bound: float,
) -> Dict[str, float]:
    mkt_cap_weight_df = pd.read_csv(data_root / "mkt_cap_weight.csv", index_col=0)
    stock_prediction = pd.read_csv(data_root / "stock_prediction.csv", index_col=0)
    stock_return_df = pd.read_csv(data_root / "stock_return_df.csv", index_col=0).dropna()

    method_returns = {
        "BL_clayton_copula": [0.0],
        "BL_gauss_copula": [0.0],
        "BL_frank_copula": [0.0],
        "BL_gumbel_copula": [0.0],
    }

    cov_dirs = {
        "Clayton": data_root / "cov_matrix" / "Clayton",
        "Gauss": data_root / "cov_matrix" / "Gauss",
        "Frank": data_root / "cov_matrix" / "Frank",
        "Gumbel": data_root / "cov_matrix" / "Gumbel",
    }

    key_map = {
        "Clayton": "BL_clayton_copula",
        "Gauss": "BL_gauss_copula",
        "Frank": "BL_frank_copula",
        "Gumbel": "BL_gumbel_copula",
    }

    for date in dates:
        mkt_weight = np.array(mkt_cap_weight_df.loc[date]).reshape(-1, 1)
        q_view = np.array(stock_prediction.loc[date]).reshape(-1, 1)

        for copula, cov_dir in cov_dirs.items():
            prior_cov = _load_cov(cov_dir, date)
            mean_bl, cov_bl = BL_processing(
                prior_cov=prior_cov,
                tau=tau,
                delta=delta,
                mkt_weight=mkt_weight,
                confidence_level=confidence_level,
                Q=q_view,
            )
            weight = BL_max_sharpe(Q=cov_bl, mu=mean_bl, upperbound=upper_bound)
            method_returns[key_map[copula]].append(float(stock_return_df.loc[date].dot(weight)))

    return {method: sharpe(ret) for method, ret in method_returns.items()}


def sensitive_check(data_root: Path, output_root: Path, upper_bound: float) -> None:
    gauss_dir = data_root / "cov_matrix" / "Gauss"
    dates = [file.stem for file in sorted(gauss_dir.glob("*.csv"))]

    tau_range = [0.01, 0.05, 0.1, 0.3, 0.5]
    delta_range = [1, 3, 5, 7, 9]
    confidence_level_range = [0.90, 0.95, 0.99]
    methods = ["BL_gauss_copula", "BL_clayton_copula", "BL_frank_copula", "BL_gumbel_copula"]

    for confidence_level in confidence_level_range:
        data_dic = {method: pd.DataFrame(columns=delta_range, index=tau_range) for method in methods}

        for tau in tau_range:
            for delta in delta_range:
                sharpe_dict = sharpe_check(
                    data_root=data_root,
                    dates=dates,
                    tau=tau,
                    delta=delta,
                    confidence_level=confidence_level,
                    upper_bound=upper_bound,
                )
                for method in methods:
                    data_dic[method].loc[tau, delta] = sharpe_dict[method]

        confidence_pct = int(confidence_level * 100)
        for method in methods:
            out_dir = output_root / f"conf_{confidence_pct}"
            out_dir.mkdir(parents=True, exist_ok=True)
            data_dic[method].to_csv(out_dir / f"{method}.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sensitivity check for BL-Copula parameters")
    parser.add_argument("--data-root", type=Path, default=Path("data_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sensitive_check"))
    parser.add_argument("--upper-bound", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sensitive_check(data_root=args.data_root, output_root=args.output_root, upper_bound=args.upper_bound)


if __name__ == "__main__":
    main()

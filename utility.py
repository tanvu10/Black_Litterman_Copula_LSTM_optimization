import math
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# VN30-focused universe used across the project.
STOCK_LIST = [
    "HPG", "TCB", "VPB", "VNM", "VIC", "MBB", "FPT", "STB", "VHM", "NVL",
    "MSN", "MWG", "VCB", "CTG", "HDB", "VJC", "TPB", "PNJ", "SSI", "VRE",
    "PDR", "KDH", "PLX", "REE", "GAS", "BID", "POW", "CII", "SBT", "BVH",
]


def get_technical_indicators(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build technical indicators from OHLCV data."""
    data = dataset.copy()

    data["ma3"] = np.log(data["Close"].rolling(window=3).mean() / data["Close"].rolling(window=3).mean().shift(1))
    data["ma5"] = np.log(data["Close"].rolling(window=5).mean() / data["Close"].rolling(window=5).mean().shift(1))
    data["26ema"] = np.log(data["Close"].ewm(span=26).mean() / data["Close"].ewm(span=26).mean().shift(1))
    data["12ema"] = np.log(data["Close"].ewm(span=12).mean() / data["Close"].ewm(span=12).mean().shift(1))
    data["MACD"] = data["12ema"] - data["26ema"]
    data["5_day_momentum"] = np.log(data["Close"] / data["Close"].shift(5))
    data["1_day_volume"] = np.log(data["Volume"] / data["Volume"].shift(1))
    data["1_day_return"] = np.log(data["Close"] / data["Close"].shift(1))
    return data


def BL_processing(
    prior_cov: np.ndarray,
    tau: float,
    delta: float,
    mkt_weight: np.ndarray,
    confidence_level: float,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Black-Litterman posterior mean/covariance under identity pick matrix."""
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    n = len(mkt_weight)
    p_matrix = np.identity(n)
    sigma = ((1 / confidence_level) - 1) * p_matrix.dot(prior_cov).dot(p_matrix.T)

    capm_mean_ret = delta * prior_cov.dot(mkt_weight)
    left = np.linalg.inv(tau * prior_cov)
    right = p_matrix.T.dot(np.linalg.inv(sigma)).dot(p_matrix)

    mu_BL = np.linalg.inv(left + right).dot(
        left.dot(capm_mean_ret) + p_matrix.T.dot(np.linalg.inv(sigma)).dot(Q)
    )

    cov_BL = (1 + tau) * prior_cov - (tau ** 2) * prior_cov.dot(p_matrix.T).dot(
        np.linalg.inv(tau * p_matrix.dot(prior_cov).dot(p_matrix.T) + sigma)
    ).dot(p_matrix).dot(prior_cov)

    return mu_BL, cov_BL


def _optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, upperbound: float) -> np.ndarray:
    mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
    cov_mat = np.asarray(cov, dtype=np.float64)
    n = mu_vec.shape[0]

    if n == 0:
        raise ValueError("mu must be non-empty")
    if upperbound <= 0:
        raise ValueError("upperbound must be positive")

    def objective(weights: np.ndarray) -> float:
        numerator = float(mu_vec.T @ weights)
        denom = float(np.sqrt(max(weights.T @ cov_mat @ weights, 1e-12)))
        return -(numerator / denom)

    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
    bounds = tuple((0.0, upperbound) for _ in range(n))
    initial = np.array([1.0 / n] * n)

    solution = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not solution.success:
        raise RuntimeError(f"Optimization failed: {solution.message}")

    return solution.x


def normal_max_sharpe(dataframe: pd.DataFrame, upperbound: float) -> np.ndarray:
    cov = dataframe.cov().to_numpy(dtype=np.float64)
    mu = dataframe.mean().to_numpy(dtype=np.float64)
    return _optimize_max_sharpe(mu=mu, cov=cov, upperbound=upperbound)


def BL_max_sharpe(upperbound: float, Q: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return _optimize_max_sharpe(mu=mu, cov=Q, upperbound=upperbound)


def copula_max_sharpe(upperbound: float, Q: np.ndarray, dataframe: pd.DataFrame) -> np.ndarray:
    mu = dataframe.mean().to_numpy(dtype=np.float64)
    return _optimize_max_sharpe(mu=mu, cov=Q, upperbound=upperbound)


def sharpe(input_list: Iterable[float], annualization: int = 252) -> float:
    returns = np.asarray(list(input_list), dtype=np.float64)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(annualization))


def drawdown(input_list: Iterable[float]) -> pd.Series:
    returns = pd.Series(input_list, dtype=np.float64)
    cum_return = (returns + 1.0).cumprod() - 1.0
    peaks = cum_return.cummax()
    return peaks - cum_return

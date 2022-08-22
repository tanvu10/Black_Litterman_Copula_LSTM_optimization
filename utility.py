import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math

def get_technical_indicators(dataset):
    # Create 3, 7 and 21 days Moving Average
    dataset['ma3'] = np.log(
        dataset['Close'].rolling(window=3).mean() / dataset['Close'].rolling(window=3).mean().shift(1))
    dataset['ma5'] = np.log(
        dataset['Close'].rolling(window=5).mean() / dataset['Close'].rolling(window=5).mean().shift(1))
    # Create MACD
    dataset['26ema'] = np.log(dataset['Close'].ewm(span=26).mean() / dataset['Close'].ewm(span=26).mean().shift(1))
    dataset['12ema'] = np.log(dataset['Close'].ewm(span=12).mean() / dataset['Close'].ewm(span=12).mean().shift(1))
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    # Create Momentum
    dataset['5_day_momentum'] = np.log(dataset['Close'] / dataset['Close'].shift(5))
    # Create 1 day volume and 3 days Moving Average volumne
    dataset['1_day_volume'] = np.log(dataset['Volume'] / dataset['Volume'].shift(1))
    # Create 1 day return
    dataset['1_day_return'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
    return dataset

def BL_processing(prior_cov, tau, delta, mkt_weight, confidence_level, Q):
    # all vectors are col vector
    n = len(mkt_weight)
    P = np.identity(n)
    sigma = ((1/confidence_level) - 1) * P.dot(prior_cov).dot(P.T)
    CAPM_mean_ret = delta*prior_cov.dot(mkt_weight)
    mu_BL = np.linalg.inv(np.linalg.inv(tau*prior_cov) + P.T.dot(np.linalg.inv(sigma)).dot(P)).\
        dot(np.linalg.inv(tau*prior_cov).dot(CAPM_mean_ret) + P.T.dot(np.linalg.inv(sigma)).dot(Q))
    cov_BL = (1+tau)*prior_cov - (tau**2)*prior_cov.dot(P.T).dot(np.linalg.inv(tau*P.dot(prior_cov).dot(P.T) + sigma)).dot(P).dot(prior_cov)
    return mu_BL, cov_BL

def normal_max_sharpe(dataframe, upperbound):
    Q = (dataframe.cov()).to_numpy()
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(dataframe.columns)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1}]
    bnds = [(0, upperbound)] * n
    bnds = tuple(bnds)
    cons = tuple(cons)
    inital_point = np.array([1 / n] * n)
    sol = minimize(f, x0=inital_point, method='SLSQP', bounds=bnds, constraints= cons)
    if sol.success == False:
        print('NOT EXISTS OPTIMIZED WEIGHT')
    else:
        print('EXISTS OPTIMIZED WEIGHT')
    return sol.x

def BL_max_sharpe(upperbound, Q, mu):
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(mu)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1}]
    bnds = [(0, upperbound)] * n
    bnds = tuple(bnds)
    cons = tuple(cons)
    inital_point = np.array([1 / n] * n)
    sol = minimize(f, x0=inital_point, method='SLSQP', bounds=bnds, constraints= cons)
    # if sol.success == False:
    #     print('NOT EXISTS OPTIMIZED WEIGHT')
    # else:
    #     print('EXISTS OPTIMIZED WEIGHT')
    return sol.x

def copula_max_sharpe(upperbound, Q, dataframe):
    mu = (dataframe.mean()).to_numpy()
    f = lambda x: -(mu.T @ x) / np.sqrt(x.T @ Q @ x)
    n = len(mu)
    cons = [{'type': 'eq', 'fun': lambda x: sum(x[i] for i in range(n)) - 1}]
    bnds = [(0, upperbound)] * n
    bnds = tuple(bnds)
    cons = tuple(cons)
    inital_point = np.array([1 / n] * n)
    sol = minimize(f, x0=inital_point, method='SLSQP', bounds=bnds, constraints= cons)
    if sol.success == False:
        print('NOT EXISTS OPTIMIZED WEIGHT')
    else:
        print('EXISTS OPTIMIZED WEIGHT')
    return sol.x

def sharpe(input_list):
    return np.mean(input_list)/np.std(input_list) * math.sqrt(252)

def drawdown(input_list):
    cum_return = (input_list + 1).cumprod() -1
    peaks = cum_return.cummax()
    drawdown_vec = (peaks - cum_return)
    return drawdown_vec


# stock_list = ['CII', 'DIG', 'HPG', 'HT1', 'HSG', 'GAS', 'GVR', 'TPB', 'TCB', 'MSN']
stock_list = ['HPG', 'TCB', 'VPB', 'VNM', 'VIC', 'MBB', 'FPT', 'STB', 'VHM', 'NVL', 'MSN', 'MWG', 'VCB', 'CTG', 'HDB',
              'VJC', 'TPB', 'PNJ', 'SSI', 'VRE', 'PDR', 'KDH', 'PLX', 'REE', 'GAS', 'BID', 'POW', 'CII', 'SBT', 'BVH']

base_stock = pd.read_csv('C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v1/stock_data_DL_folder/CII.csv', index_col=0)
base_index = base_stock.index


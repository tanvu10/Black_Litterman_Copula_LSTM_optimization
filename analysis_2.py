import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import BL_processing, BL_max_sharpe, normal_max_sharpe, copula_max_sharpe, sharpe

INPUT_CLAYTON_COV_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Clayton/*.csv'
INPUT_GAUSSIAN_COV_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gauss/*.csv'
INPUT_FRANK_COV_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Frank/*.csv'
INPUT_GUMBEL_COV_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gumbel/*.csv'
INPUT_MKT_CAP_WEIGHT = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/mkt_cap_weight.csv'
INPUT_CONFIDENCE_DF = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/confidence_df.csv'
INPUT_STOCK_PREDICTION_DF = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_prediction.csv'
INPUT_STOCK_RETURN_DF = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_return_df.csv'
INPUT_VN30INDEX = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_folder/VN30.csv'
INPUT_VNINDEX = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_folder/VNINDEX.csv'
OUTPUT_SENSITIVE_CHECK = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/sensitive_check'

# INPUT_CLAYTON_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Clayton/*.csv'
# INPUT_GAUSSIAN_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gauss/*.csv'
# INPUT_FRANK_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Frank/*.csv'
# INPUT_GUMBEL_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gumbel/*.csv'
# INPUT_MKT_CAP_WEIGHT = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/mkt_cap_weight.csv'
# INPUT_CONFIDENCE_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/confidence_df.csv'
# INPUT_STOCK_PREDICTION_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_prediction.csv'
# INPUT_STOCK_RETURN_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_return_df.csv'


mkt_cap_weight_df = pd.read_csv(f'{INPUT_MKT_CAP_WEIGHT}', index_col=0)
stock_prediction = pd.read_csv(f'{INPUT_STOCK_PREDICTION_DF}', index_col=0)
stock_return_df = pd.read_csv(f'{INPUT_STOCK_RETURN_DF}', index_col=0)
stock_return_df.dropna(inplace=True)

# start = '2021-10-20'
VN30_index = pd.read_csv(INPUT_VN30INDEX, index_col=0)
VNINDEX = pd.read_csv(INPUT_VNINDEX, index_col=0)

VN30_index['VN30INDEX'] = (VN30_index['Close'] - VN30_index['Close'].shift(1))/VN30_index['Close'].shift(1)
VNINDEX['VNINDEX'] = (VNINDEX['Close'] - VNINDEX['Close'].shift(1))/VNINDEX['Close'].shift(1)


data_factor_folder = sorted(glob.glob(INPUT_GAUSSIAN_COV_FOLDER))  # sort datetime
print(len(data_factor_folder))
date_list = [os.path.splitext(os.path.basename(sub_path))[0] for sub_path in data_factor_folder]
print(date_list)
norm_return_list = [0]
gauss_copula_return_list = [0]
clayton_copula_return_list = [0]
frank_copula_return_list = [0]
gumbel_copula_return_list = [0]
gauss_BL_return_list = [0]
clayton_BL_return_list = [0]
frank_BL_return_list = [0]
gumbel_BL_return_list = [0]
norm_BL_return_list = [0]
upper_bound = 0.5 # best with 0.5
# start_index = date_list.index(start)

def sharpe_check(tau, delta, confidence_level):
    for date in date_list:
        # print(date)
        current_return_df = stock_return_df.loc[:date].iloc[:-1,:]  # drop current date

        clayton_prior_cov = pd.read_csv(f'{INPUT_CLAYTON_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
        clayton_prior_cov = np.array(clayton_prior_cov, dtype=np.float64)
        clayton_confidence_level = confidence_level

        gauss_prior_cov = pd.read_csv(f'{INPUT_GAUSSIAN_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
        gauss_prior_cov = np.array(gauss_prior_cov, dtype=np.float64)
        gauss_confidence_level = confidence_level

        frank_prior_cov = pd.read_csv(f'{INPUT_FRANK_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
        frank_prior_cov = np.array(frank_prior_cov, dtype=np.float64)
        frank_confidence_level = confidence_level

        gumbel_prior_cov = pd.read_csv(f'{INPUT_GUMBEL_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
        gumbel_prior_cov = np.array(gumbel_prior_cov, dtype=np.float64)
        gumbel_confidence_level = confidence_level

        mkt_weight = np.array(mkt_cap_weight_df.loc[date]).reshape(-1, 1)
        Q = np.array(stock_prediction.loc[date]).reshape(-1, 1)

        gauss_mean_BL, gauss_cov_BL = BL_processing(prior_cov=gauss_prior_cov, tau=tau, delta=delta, mkt_weight=mkt_weight,
                                                    confidence_level=gauss_confidence_level, Q=Q)

        clayton_mean_BL, clayton_cov_BL = BL_processing(prior_cov=clayton_prior_cov, tau=tau, delta=delta,
                                                        mkt_weight=mkt_weight, confidence_level=clayton_confidence_level,
                                                        Q=Q)

        frank_mean_BL, frank_cov_BL = BL_processing(prior_cov=frank_prior_cov, tau=tau, delta=delta,
                                                        mkt_weight=mkt_weight, confidence_level=frank_confidence_level,
                                                        Q=Q)

        gumbel_mean_BL, gumbel_cov_BL = BL_processing(prior_cov=gumbel_prior_cov, tau=tau, delta=delta,
                                                        mkt_weight=mkt_weight, confidence_level=gumbel_confidence_level,
                                                        Q=Q)

        BL_gauss_weight = BL_max_sharpe(Q=gauss_cov_BL, mu=gauss_mean_BL, upperbound=upper_bound)
        BL_clayton_weight = BL_max_sharpe(Q=clayton_cov_BL, mu=clayton_mean_BL, upperbound=upper_bound)
        BL_frank_weight = BL_max_sharpe(Q=frank_cov_BL, mu=frank_mean_BL, upperbound=upper_bound)
        BL_gumbel_weight = BL_max_sharpe(Q=gumbel_cov_BL, mu=gumbel_mean_BL, upperbound=upper_bound)

        clayton_BL_return_list.append(stock_return_df.loc[date].dot(BL_clayton_weight))
        gauss_BL_return_list.append(stock_return_df.loc[date].dot(BL_gauss_weight))
        frank_BL_return_list.append(stock_return_df.loc[date].dot(BL_frank_weight))
        gumbel_BL_return_list.append(stock_return_df.loc[date].dot(BL_gumbel_weight))

    # date_list.insert(0, 'start')
    portfolio_dic = {
        'BL_clayton_copula': sharpe(clayton_BL_return_list),
        'BL_gauss_copula': sharpe(gauss_BL_return_list),
        'BL_frank_copula': sharpe(frank_BL_return_list),
        'BL_gumbel_copula': sharpe(gumbel_BL_return_list),
    }
    return portfolio_dic

def sensitive_check():
    tau_range = [0.01, 0.05, 0.1, 0.3, 0.5]
    delta_range = [1, 3, 5, 7, 9]
    confidence_level_range = [90, 95, 99]
    method_list = ['BL_gauss_copula', 'BL_clayton_copula', 'BL_frank_copula', 'BL_gumbel_copula']
    for confidence_level in confidence_level_range:
        data_dic = {}
        for method in method_list:
            data_dic[method] = pd.DataFrame(columns=delta_range, index=tau_range)

        for tau in tau_range:
            for delta in delta_range:
                sharpe_dict = sharpe_check(tau=tau, delta=delta, confidence_level=confidence_level/100)
                for method in method_list:
                    data_dic[method].loc[tau, delta] = sharpe_dict[method]

        print(f'{confidence_level}% confidence')
        for method in method_list:
            data_dic[method].to_csv(f'{OUTPUT_SENSITIVE_CHECK}/conf_{confidence_level}/{method}.csv')
            print(method)
            print(data_dic[method])
if __name__ == '__main__':
    sensitive_check()
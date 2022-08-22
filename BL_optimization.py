import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import BL_processing, BL_max_sharpe, normal_max_sharpe, copula_max_sharpe, sharpe, drawdown

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

# INPUT_CLAYTON_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Clayton/*.csv'
# INPUT_GAUSSIAN_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gauss/*.csv'
# INPUT_FRANK_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Frank/*.csv'
# INPUT_GUMBEL_COV_FOLDER = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/cov_matrix/Gumbel/*.csv'
# INPUT_MKT_CAP_WEIGHT = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/mkt_cap_weight.csv'
# INPUT_CONFIDENCE_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/confidence_df.csv'
# INPUT_STOCK_PREDICTION_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_prediction.csv'
# INPUT_STOCK_RETURN_DF = '/mnt/c/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_return_df.csv'


mkt_cap_weight_df = pd.read_csv(f'{INPUT_MKT_CAP_WEIGHT}', index_col=0)
# confidence_df = pd.read_csv(f'{INPUT_CONFIDENCE_DF}')
# confidence_df = confidence_df.iloc[:, 1:]
# confidence_df.set_index('Date', inplace=True)
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
VN30 = list(VN30_index.loc[date_list, 'VN30INDEX'])
VN30.insert(0, 0)
VNINDEX = list(VNINDEX.loc[date_list, 'VNINDEX'])
VNINDEX.insert(0, 0)
upper_bound = 0.5 #best with 0.5
# start_index = date_list.index(start)


for date in date_list:
    print(date)
    current_return_df = stock_return_df.loc[:date].iloc[:-1,:]  # drop current date

    clayton_prior_cov = pd.read_csv(f'{INPUT_CLAYTON_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
    clayton_prior_cov = np.array(clayton_prior_cov, dtype=np.float64)
    # clayton_confidence_level = 1 - confidence_df['Clayton_conf'].loc[date]
    clayton_confidence_level = 0.9

    gauss_prior_cov = pd.read_csv(f'{INPUT_GAUSSIAN_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
    gauss_prior_cov = np.array(gauss_prior_cov, dtype=np.float64)
    # gauss_confidence_level = 1 - confidence_df['Gauss_conf'].loc[date]
    gauss_confidence_level = 0.9

    frank_prior_cov = pd.read_csv(f'{INPUT_FRANK_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
    frank_prior_cov = np.array(frank_prior_cov, dtype=np.float64)
    # frank_confidence_level = 1 - confidence_df['Gauss_conf'].loc[date]
    frank_confidence_level = 0.9

    gumbel_prior_cov = pd.read_csv(f'{INPUT_GUMBEL_COV_FOLDER[:-6]}/{date}.csv', index_col=0)
    gumbel_prior_cov = np.array(gumbel_prior_cov, dtype=np.float64)
    # gumbel_confidence_level = 1 - confidence_df['Gauss_conf'].loc[date]
    gumbel_confidence_level = 0.9

    mkt_weight = np.array(mkt_cap_weight_df.loc[date]).reshape(-1, 1)
    Q = np.array(stock_prediction.loc[date]).reshape(-1, 1)

    gauss_mean_BL, gauss_cov_BL = BL_processing(prior_cov=gauss_prior_cov, tau=0.01, delta=2.5, mkt_weight=mkt_weight,
                                                confidence_level=gauss_confidence_level, Q=Q)

    clayton_mean_BL, clayton_cov_BL = BL_processing(prior_cov=clayton_prior_cov, tau=0.01, delta=2.5,
                                                    mkt_weight=mkt_weight, confidence_level=clayton_confidence_level,
                                                    Q=Q)

    frank_mean_BL, frank_cov_BL = BL_processing(prior_cov=frank_prior_cov, tau=0.01, delta=2.5,
                                                    mkt_weight=mkt_weight, confidence_level=frank_confidence_level,
                                                    Q=Q)

    gumbel_mean_BL, gumbel_cov_BL = BL_processing(prior_cov=gumbel_prior_cov, tau=0.01, delta=2.5,
                                                    mkt_weight=mkt_weight, confidence_level=gumbel_confidence_level,
                                                    Q=Q)

    norm_mean_BL, norm_cov_BL = BL_processing(prior_cov=(current_return_df.cov()).to_numpy(), tau=0.01, delta=2.5,
                                                    mkt_weight=mkt_weight, confidence_level=gumbel_confidence_level,
                                                    Q=Q)

    normal_weight = normal_max_sharpe(current_return_df, upper_bound)

    clayton_copula_weight = copula_max_sharpe(upperbound=upper_bound, Q=clayton_prior_cov, dataframe=current_return_df)
    gauss_copula_weight = copula_max_sharpe(upperbound=upper_bound, Q=gauss_prior_cov, dataframe=current_return_df)
    frank_copula_weight = copula_max_sharpe(upperbound=upper_bound, Q=frank_prior_cov, dataframe=current_return_df)
    gumbel_copula_weight = copula_max_sharpe(upperbound=upper_bound, Q=gumbel_prior_cov, dataframe=current_return_df)

    BL_gauss_weight = BL_max_sharpe(Q=gauss_cov_BL, mu=gauss_mean_BL, upperbound=upper_bound)
    BL_clayton_weight = BL_max_sharpe(Q=clayton_cov_BL, mu=clayton_mean_BL, upperbound=upper_bound)
    BL_frank_weight = BL_max_sharpe(Q=frank_cov_BL, mu=frank_mean_BL, upperbound=upper_bound)
    BL_gumbel_weight = BL_max_sharpe(Q=gumbel_cov_BL, mu=gumbel_mean_BL, upperbound=upper_bound)
    BL_norm_weight = BL_max_sharpe(Q=norm_cov_BL, mu=norm_mean_BL, upperbound=upper_bound)
    print(normal_weight)
    print(clayton_copula_weight)
    print(gauss_copula_weight)
    print(BL_gauss_weight)
    print(BL_clayton_weight)

    norm_return_list.append(stock_return_df.loc[date].dot(normal_weight))

    clayton_copula_return_list.append(stock_return_df.loc[date].dot(clayton_copula_weight))
    gauss_copula_return_list.append(stock_return_df.loc[date].dot(gauss_copula_weight))
    frank_copula_return_list.append(stock_return_df.loc[date].dot(frank_copula_weight))
    gumbel_copula_return_list.append(stock_return_df.loc[date].dot(gumbel_copula_weight))

    clayton_BL_return_list.append(stock_return_df.loc[date].dot(BL_clayton_weight))
    gauss_BL_return_list.append(stock_return_df.loc[date].dot(BL_gauss_weight))
    frank_BL_return_list.append(stock_return_df.loc[date].dot(BL_frank_weight))
    gumbel_BL_return_list.append(stock_return_df.loc[date].dot(BL_gumbel_weight))
    norm_BL_return_list.append(stock_return_df.loc[date].dot(BL_norm_weight))

date_list.insert(0, 'start')
portfolio_dic = {
    'Date': date_list,
    'normal': norm_return_list,
    # 'clayton_copula': clayton_copula_return_list,
    # 'gauss_copula': gauss_copula_return_list,
    # 'frank_copula': frank_copula_return_list,
    # 'gumbel_copula': gumbel_copula_return_list,
    'BL_clayton_copula': clayton_BL_return_list,
    'BL_gauss_copula': gauss_BL_return_list,
    'BL_frank_copula': frank_BL_return_list,
    'BL_gumbel_copula': gumbel_BL_return_list,
    'only_BL': norm_BL_return_list,
    # 'VN30': VN30,
    # 'VNINDEX': VNINDEX
}

stat_df = pd.DataFrame(index=['normal', 'BL_clayton_copula', 'BL_gauss_copula', 'BL_frank_copula', 'BL_gumbel_copula',
                              'only_BL'],
                       columns=['Sharpe', 'Average Drawdown', 'Max Drawdown'])

portfolio_df = pd.DataFrame(portfolio_dic)
portfolio_df.set_index(['Date'], inplace=True)

for key in portfolio_dic.keys():
    if key != 'Date':
        drawdown_vec = drawdown(portfolio_df[key])
        stat_df.loc[key, 'Sharpe'] = sharpe(portfolio_dic[key])
        stat_df.loc[key, 'Average Drawdown'] = np.mean(drawdown_vec)
        stat_df.loc[key, 'Max Drawdown'] = np.max(drawdown_vec)


print(stat_df)

# portfolio_df.cumsum().plot()
((portfolio_df + 1).cumprod() -1).plot()
plt.plot()
plt.show()
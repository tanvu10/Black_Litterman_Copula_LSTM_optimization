import os
import pandas as pd
import numpy as np
import glob
from utility import get_technical_indicators, stock_list
from tabulate import tabulate
INPUT_SUB_FACTOR_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/sub_factor_folder'
INPUT_FACTOR_RETURN = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/factor_return_df.csv'
INPUT_STOCK_DATA_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_folder'
OUTPUT_STOCK_DATA_DL_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_DL_folder'
OUTPUT_STOCK_RETURN_DF = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_return_df.csv'

factor_return_df = pd.read_csv(INPUT_FACTOR_RETURN, index_col=0)
factor_return_df.index.name = 'Date'
base_index = factor_return_df.index
factor_list = factor_return_df.columns


stock_return_factor_dic = {}
for stock in stock_list:
    stock_df = pd.read_csv(f'{INPUT_STOCK_DATA_FOLDER}/{stock}.csv', index_col=0)
    stock_df['return'] = (stock_df['Close'] - stock_df['Close'].shift(1))/stock_df['Close'].shift(1)
    stock_df = stock_df.merge(pd.DataFrame(index=base_index), how='right', on='Date')
    stock_df.dropna(inplace=True)
    stock_return_factor_dic[stock] = stock_df['return']

stock_summary_df = pd.DataFrame(stock_return_factor_dic)
descriptive_table = stock_summary_df.describe().T
print(descriptive_table)
descriptive_table['skewness'] = stock_summary_df.skew().tolist()
descriptive_table['kurtosis'] = stock_summary_df.kurtosis().tolist()
descriptive_table = descriptive_table[['min', 'max', 'mean', 'std', 'skewness', 'kurtosis']]
print(tabulate(descriptive_table))



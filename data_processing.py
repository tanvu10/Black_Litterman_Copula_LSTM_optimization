import os
import pandas as pd
import numpy as np
import glob
from utility import get_technical_indicators, stock_list

INPUT_SUB_FACTOR_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/sub_factor_folder'
INPUT_FACTOR_RETURN = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/factor_return_df.csv'
INPUT_STOCK_DATA_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_folder'
OUTPUT_STOCK_DATA_DL_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_DL_folder'
OUTPUT_STOCK_RETURN_DF = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_return_df.csv'

factor_return_df = pd.read_csv(INPUT_FACTOR_RETURN, index_col=0)
factor_return_df.index.name = 'Date'
base_index = factor_return_df.index
factor_list = factor_return_df.columns

sub_factor_dic = {}
for factor in factor_list:
    sub_factor_dic[factor] = pd.read_csv(f"{INPUT_SUB_FACTOR_FOLDER}/{factor}.csv", index_col=0)
    sub_factor_dic[factor] = sub_factor_dic[factor].merge(pd.DataFrame(index=base_index), how='right', on='Date')
    # sub_factor_dic[factor].fillna(0, inplace=True)

stock_return_factor_dic = {}
for stock in stock_list:
    stock_df = pd.read_csv(f'{INPUT_STOCK_DATA_FOLDER}/{stock}.csv', index_col=0)
    stock_df['return'] = (stock_df['Close'] - stock_df['Close'].shift(1))/stock_df['Close'].shift(1)
    # get technical indicator
    stock_df = get_technical_indicators(stock_df)
    # add factor to df
    for factor in factor_list:
        stock_df[factor] = sub_factor_dic[factor][stock]*factor_return_df[factor]
    stock_df = stock_df.merge(pd.DataFrame(index=base_index), how='right', on='Date')
    stock_df.dropna(inplace=True)
    stock_df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
    stock_return_factor_dic[stock] = stock_df

for stock in stock_return_factor_dic.keys():
    stock_return_factor_dic[stock].to_csv(f'{OUTPUT_STOCK_DATA_DL_FOLDER}/{stock}.csv')

# calculate stock_return:

stock_return_df = {}
for stock in stock_list:
    stock_df = pd.read_csv(f'{INPUT_STOCK_DATA_FOLDER}/{stock}.csv', index_col=0)
    stock_df[stock] = (stock_df['Close'] - stock_df['Close'].shift(1))/stock_df['Close'].shift(1)
    stock_df = stock_df.merge(pd.DataFrame(index=base_index), how='right', on='Date')
    stock_return_df[stock] = stock_df[stock]

stock_return_df = pd.DataFrame(stock_return_df)
stock_return_df.to_csv(OUTPUT_STOCK_RETURN_DF)
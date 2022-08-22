import pandas as pd
from utility import stock_list, base_index
from LTSM_train import DL_train

INPUT_STOCK_DATA_DL_FOLDER = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_DL_folder'
OUTPUT_STOCK_PREDICTION_FILE = 'C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2'
start_date = '2021-10-01'
end_date = '2022-04-01'

combine_dict = {}

for stock in stock_list:
    stock_df = pd.read_csv(f'{INPUT_STOCK_DATA_DL_FOLDER}/{stock}.csv')
    stock_df.merge(pd.DataFrame(index=base_index), how='right', on='Date')
    stock_df['Date'] = stock_df['Date'].apply(pd.Timestamp)
    stock_df.set_index('Date', inplace=True)

    combine_dict[stock] = {
        f'{stock}_pred': [],
        'Date': []
    }

    for dt in pd.bdate_range(start_date, end_date, freq='B'):
        train_df = stock_df[stock_df.index < dt]
        print(train_df)
        prediction = DL_train(train_df)
        combine_dict[stock][f'{stock}_pred'].append(prediction)
        combine_dict[stock]['Date'].append(dt)

cur_df = pd.DataFrame(combine_dict[stock_list[0]])
cur_df.set_index('Date', inplace=True)
cur_df.index.astype(str)
for stock in stock_list[1:]:
    pre_df = pd.DataFrame(combine_dict[stock])
    pre_df.set_index('Date', inplace=True)
    pre_df.index.astype(str)
    cur_df = cur_df.merge(pre_df, how='left', on='Date')

cur_df.to_csv(f'{OUTPUT_STOCK_PREDICTION_FILE}/stock_prediction.csv')

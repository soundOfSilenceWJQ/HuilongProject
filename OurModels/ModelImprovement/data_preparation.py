# 整理出一些小的时间片段的数据
import pickle
import os
from datetime import date

import pandas as pd
import numpy as np
import torch
from util import get_data_Xy


def get_snippets(data: pd.DataFrame, TN=3, is_index_info_needed=False):
    symbols = data.index.get_level_values('symbol').unique()
    dates = data.index.get_level_values('date').unique()
    snip_tensor_list = []
    # index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    index_info = []

    for i in range(len(dates) - TN):
        start_date = dates[i]
        print("{} has loaded.".format(start_date))
        end_date = dates[i + TN - 1]
        data_snippet = data[(data.index.get_level_values('date') >= start_date) & (data.index.get_level_values('date') <= end_date)]
        for symbol in symbols:
            try:
                single_stock_snippet = data_snippet.xs(symbol, level='symbol')
            except KeyError:
                continue
            if len(single_stock_snippet) < TN:
                continue

            sss_array = np.array(single_stock_snippet.values).astype(np.float32)
            single_stock_snippet_tensor = torch.from_numpy(sss_array)
            snip_tensor_list.append(single_stock_snippet_tensor)
            if is_index_info_needed:
                index_info.append((end_date, symbol))

    # 将snip_tensor_list和ret_tensor_list转换为tensor
    snip_tensor = torch.stack(snip_tensor_list)
    if is_index_info_needed:
        return snip_tensor, index_info
    else:
        return snip_tensor


def get_snippets_for_a_year(df: pd.DataFrame, year, TN, base_path):
    '''生成开始月份在这一年的时间片段的个股因子数据以及收益率数据，还有index_info，存储到本地'''
    # 截取这一年的dataframe
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    sub_df = df[(df.index.get_level_values('date') >= start_date) & (df.index.get_level_values('date') <= end_date)]
    data_X, data_y, _ = get_data_Xy(sub_df, str(year))
    snip_tensor_x, index_info = get_snippets(data_X, TN, True)
    snip_tensor_y = get_snippets(data_y, TN, False)
    tensor_y = snip_tensor_y[:, -1]
    path = base_path + '\\' + str(year)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(snip_tensor_x, path + '\\snip_tensor_X.pt')
    torch.save(tensor_y, path + '\\tensor_y.pt')

    with open(path + '\\index_info.pkl', 'wb') as f:
        pickle.dump(index_info, f)
    return


if __name__ == '__main__':
    # snip_tensor, ret_tensor = get_snippets('C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf')
    # print
    data: pd.DataFrame = pd.read_hdf("C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf")
    get_snippets_for_a_year(data, 2009, 3, 'C:\\Users\\ipwx\\Desktop\\testing\\')

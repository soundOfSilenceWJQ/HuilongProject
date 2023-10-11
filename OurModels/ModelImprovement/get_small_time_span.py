# 整理出一些小的时间片段的数据

import pandas as pd
import numpy as np
import torch


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


if __name__ == '__main__':
    snip_tensor, ret_tensor = get_snippets('C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf')
    print(snip_tensor)
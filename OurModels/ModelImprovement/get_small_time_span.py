# 整理出一些小的时间片段的数据

import pandas as pd
import numpy as np
import torch


def get_snippets(data: pd.DataFrame, TN=3):
    symbols = data.index.get_level_values('symbol').unique()
    dates = data.index.get_level_values('date').unique()
    # 构造一个dataframe, 使用multiindex，index为symbol，date，TN
    # columns为所有的columns
    # 然后对于每一个symbol，每一个date，取出这个symbol在这个date之前的TN个数据
    # 构造一个三级multiindex，第一级为symbol，第二级为date，第三级为TN
    index = pd.MultiIndex.from_product([dates, symbols, range(TN)], names=['symbol', 'date', 'TN'])
    df = pd.DataFrame(index=index, columns=data.columns)

    for i in range(len(dates) - TN):
        start_date = dates[i]
        print(start_date)
        end_date = dates[i + TN]
        data_snippet = data[(data.index.get_level_values('date') >= start_date) & (data.index.get_level_values('date') < end_date)]
        for symbol in symbols:
            try:
                single_stock_snippet = data_snippet.xs(symbol, level='symbol')
            except:
                continue
            if(len(single_stock_snippet) < TN):
                continue

            sss_array = np.array(single_stock_snippet.values).astype(np.float32)
            df.loc[(start_date, symbol, slice(None)), :] = sss_array

    return df


if __name__ == '__main__':
    data: pd.DataFrame = pd.read_hdf("C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf")
    df = get_snippets(data, 3)

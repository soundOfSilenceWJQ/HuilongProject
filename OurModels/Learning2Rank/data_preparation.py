"""准备这样的数据：X:两个股票的因子序列，Y:收益率的关系，是否有必要限制在同一天？试试限制在同一天的办法"""
import pickle
import os
from datetime import date
from quant_stock.pipeline import extract_factor_columns
import pandas as pd
import numpy as np
import torch


def get_order(data: pd.DataFrame, sample_frac1=0.3, sample_frac2=0.9):
    data = data.filter(regex='Alpha.*|NEXT_RET')
    # 去掉data中含nan的行
    data = data.dropna(axis=0, how='any')
    # 做时间维度的截断，同时做截面维度的截断，各随机取一部分数据，生成两个X向量以及一个y向量，进行保存
    # 生成一个时间维度的索引
    dates = data.index.get_level_values('date').unique()
    # 生成一个截面维度的索引
    symbols = data.index.get_level_values('symbol').unique()

    X1_tensor = torch.empty(0, 158)
    X2_tensor = torch.empty(0, 158)
    y_tensor = torch.empty(0)

    # # 先处理时间维度
    # for symbol in symbols:
    #     print("begin to load:", symbol)
    #     symbol_data = data.xs(symbol, level='symbol')
    #     sample_df1 = symbol_data.sample(frac=sample_frac1)
    #     sample_df2 = symbol_data.sample(frac=sample_frac1)
    #     # 判断X1和X2中是否有nan值
    #     # if (X1.isnull().values.any() or X2.isnull().values.any() or next_ret1.isna().any()
    #     #         or next_ret2.isna().any()):
    #     #     continue
    #     X1_tensor = torch.cat((X1_tensor, torch.from_numpy(np.array(sample_df1.filter(regex='Alpha.*')).astype(np.float32))))
    #     X2_tensor = torch.cat((X2_tensor, torch.from_numpy(np.array(sample_df2.filter(regex='Alpha.*')).astype(np.float32))))
    #     # 将sample_df1和sample_df2转换成numpy数组
    #     array1 = torch.from_numpy(np.array(sample_df1['NEXT_RET']).astype(np.float32))
    #     array2 = torch.from_numpy(np.array(sample_df2['NEXT_RET']).astype(np.float32))
    #     for i in range(0, min(len(array1), len(array2))):
    #         if array1[i] > array2[i]:
    #             y_tensor = torch.cat((y_tensor, torch.tensor([1])))
    #         elif array1[i] < array2[i]:
    #             y_tensor = torch.cat((y_tensor, torch.tensor([-1])))
    #         else:
    #             y_tensor = torch.cat((y_tensor, torch.tensor([0])))


    # 再处理截面维度
    for date in dates:
        print("begin to load:", date)
        date_data = data.xs(date, level='date')     # 这一时间截面的所有股票的数据
        sample_df1 = date_data.sample(frac=sample_frac2)
        sample_df2 = date_data.sample(frac=sample_frac2)
        X1_tensor = torch.cat((X1_tensor, torch.from_numpy(np.array(sample_df1.filter(regex='Alpha.*')).astype(np.float32))))
        X2_tensor = torch.cat((X2_tensor, torch.from_numpy(np.array(sample_df2.filter(regex='Alpha.*')).astype(np.float32))))
        # 将sample_df1和sample_df2转换成numpy数组
        array1 = torch.from_numpy(np.array(sample_df1['NEXT_RET']).astype(np.float32))
        array2 = torch.from_numpy(np.array(sample_df2['NEXT_RET']).astype(np.float32))
        for i in range(0, min(len(array1), len(array2))):
            if array1[i] > array2[i]:
                y_tensor = torch.cat((y_tensor, torch.tensor([1])))
            elif array1[i] < array2[i]:
                y_tensor = torch.cat((y_tensor, torch.tensor([-1])))
            else:
                y_tensor = torch.cat((y_tensor, torch.tensor([0])))

    return X1_tensor, X2_tensor, y_tensor


if __name__ == '__main__':
    base_path = 'C:\\Users\\ipwx\\Desktop\\testing\\'
    df = pd.read_hdf(base_path + "_Alpha158_Financial01_Barra_HT_proceed.hdf")
    X1_tensor, X2_tensor, y_tensor = get_order(df)
    # 将X1_tensor, X2_tensor, y_tensor进行保存
    torch.save(X1_tensor, base_path + 'Ranking\\X1_tensor.pt')
    torch.save(X2_tensor, base_path + 'Ranking\\X2_tensor.pt')
    torch.save(y_tensor, base_path + 'Ranking\\y_tensor.pt')

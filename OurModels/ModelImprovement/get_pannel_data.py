from datetime import date,datetime
import pandas as pd
import numpy as np
from loguru import logger
from quant_stock.backtest import easy_factor_test
from quant_stock.core import StockDataLoaderV2
from quant_stock.pipeline import *

from quant_stock.pipeline.op_cross_section import cs_fill_nan


def get_data_Xy(df, tag, factor_cols=None):
    '''划分label和feature'''
    if factor_cols is None:
        # 在训练集中只采纳有效的列
        df = extract_factor_columns(df, extra=['INDUSTRY', 'NEXT_RET'])
        factor_cols = [
            col for col in df.columns
            if (col not in ('INDUSTRY', 'NEXT_RET')) and (df[col].isna().sum() == 0)
        ]
    else:
        # 在测试集中如果对应存在无效列，则表明 cs_fill_nan 失败，只能填充
        factor_cols = list(factor_cols)
        df = df[factor_cols + ['INDUSTRY', 'NEXT_RET']]
        for col in factor_cols:
            na_sum = df[col].isna().sum()
            if na_sum > 0:
                logger.warning(f'在 {tag!r} 中，所需因子列 {col!r} 存在空值，只能填充为中值或 0。')
                if na_sum < len(df[col]):
                    col_val = df[col].fillna(df[col].median())
                else:
                    col_val = 0
                df = df.assign(**{col: col_val})

    # 抽取非 NaN 的行（主要针对 NEXT_RET）
    df = df[factor_cols + ['INDUSTRY', 'NEXT_RET']]
    df = df.dropna(how='any')

    # 抽取 X 和 y
    X = extract_factor_columns(df)
    y = cs_clip_extreme(
        df[['NEXT_RET']],
        columns=['NEXT_RET'],
    )['NEXT_RET']

    # 返回数据
    return X, y, factor_cols


def get_factor_data(file_path):
    data: pd.DataFrame = pd.read_hdf(file_path)
    # 获取data第二级索引的所有可能的值
    symbols = data.index.get_level_values('symbol').unique()
    dates = data.index.get_level_values('date').unique()
    data_all = pd.DataFrame(index=data.index, columns=data.columns)

    for i in range(len(dates)):
        date_i_data = data.xs(dates[i], level='date')
        pass_cnt = 0
    # 将first_date_data中的数据添加到new_data中
        for symbol in symbols:
            try:
                data_all.loc[(dates[i], symbol)] = date_i_data.loc[symbol]
            except KeyError:
                pass_cnt += 1
        print("{}:缺失{}只股票数据".format(dates[i], pass_cnt))

    return data_all


if __name__ == '__main__':
    data = get_factor_data('C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf')
    cs_fill_nan(data=data, inplace=True)
    print(data)

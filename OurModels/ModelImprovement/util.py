import os
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
from quant_stock.pipeline import *

def expo(x):
    return np.exp(x) - 1

def log(x):
    return np.log(x + 1)

def get_data_Xy(df, tag, factor_cols=None):
    # 抽取有效的列
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
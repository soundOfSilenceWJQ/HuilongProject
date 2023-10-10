import os

import pandas as pd
from loguru import logger
from quant_stock.backtest import easy_factor_test
from quant_stock.core import StockDataLoaderV2
from quant_stock.pipeline import *


if __name__ == '__main__':
    if not os.path.isfile('./123123.hdf'):
        # 载入因子
        data: pd.DataFrame = pd.read_hdf('./_factor_origin_statues.hdf')
        dates = data.index.get_level_values('date')
        start_date, end_date = dates.min(), dates.max()
        logger.info('因子已载入。')

        # 载入行业
        loader = StockDataLoaderV2(start_date=start_date, end_date=end_date)
        indmap = loader.load_industry_mapping()

        # 因子预处理
        logger.info('因子预处理 ...')

        pipelines = [
            cs_fill_nan,
            cs_clip_extreme,
            cs_neutralize,
            cs_standardize,
        ]
        data = cs_pipeline(
            data=data,
            pipelines=pipelines,
            progress_desc='因子预处理',
        )
        dates = data.index.get_level_values('date')

        # 因子正交
        logger.info('因子正交化 ...')
        data = symmetric_orthogonalize(data)

        # 存储因子
        logger.info('存储因子 ...')
        data.to_hdf('./_factor_processed_statues.hdf', key='data')

import os
from datetime import date,datetime
import pandas as pd
import numpy as np
from loguru import logger
from quant_stock.backtest import easy_factor_test
from quant_stock.core import StockDataLoaderV2
from quant_stock.pipeline import *
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import time
import gc
from torch.optim.lr_scheduler import MultiStepLR
from get_small_time_span import get_snippets

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

# 模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 假设输出是一个数值

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 只取最后一个时间步的输出进行全连接
        return out.squeeze()

if __name__ == '__main__':

    # 数据准备
    # 载入因子
    data: pd.DataFrame = pd.read_hdf("C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf")

    # 对'NEXT_RET'列进行+1然后取对数操作
    data['NEXT_RET'] = np.log(data['NEXT_RET'] + 1)


    # 构造截面数据
    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()
    logger.info('因子已载入。')


    fac = pd.Series()
    for year in range(2009, 2010):  # 从2009开始滚动，到2022
        logger.info(f"Training model for {year}...")

        # 定义滚动窗口
        train_start = date(year, 1, 1)
        train_end = date(year + 9, 12, 31)
        valid_start = date(year + 10, 1, 1)
        valid_end = date(year + 10, 12, 31)
        test_start = date(year + 11, 1, 1)
        if year == 2012:
            test_end = date(2023, 6, 1)
        else:
            test_end = date(year + 11, 12, 31)

        # 根据滚动窗口划分数据集
        train_data_df = data[(dates >= train_start) & (dates < train_end)]
        valid_data_df = data[(dates >= valid_start) & (dates < valid_end)]
        test_data_df = data[(dates >= test_start) & (dates < test_end)]

        X_train, y_train, factor_cols = get_data_Xy(train_data_df, 'train')
        # X_valid, y_valid, _ = get_data_Xy(valid_data_df, 'valid', factor_cols)
        X_test, y_test, _ = get_data_Xy(test_data_df, 'test', factor_cols)
        # 得到snippets
        # 如果没有找到相应文件，就重新生成
        base_path = 'C:\\Users\ipwx\Desktop\\testing\\'
        if os.path.exists(base_path + '\X_train_tensor' + str(year) + '.pt') == False:
            X_train_tensor = get_snippets(X_train, 3)
            y_train_tensor = get_snippets(y_train, 3)
            y_train_tensor = y_train_tensor[:, -1]
            torch.save(X_train_tensor, base_path + '\X_train_tensor' + str(year) + '.pt')
            torch.save(y_train_tensor, base_path + '\y_train_tensor' + str(year) + '.pt')
        else:
            X_train_tensor = torch.load(base_path + '\X_train_tensor' + str(year) + '.pt')
            y_train_tensor = torch.load(base_path + '\y_train_tensor' + str(year) + '.pt')
        # if os.path.exists(base_path + '\X_valid_tensor.pt') == False:
        #     X_valid_tensor = get_snippets(X_valid, 3)
        #     y_valid_tensor = get_snippets(y_valid, 3)
        #     y_valid_tensor = y_valid_tensor[:, -1]
        #     torch.save(X_valid_tensor, base_path + '\X_valid_tensor.pt')
        #     torch.save(y_valid_tensor, base_path + '\y_valid_tensor.pt')
        # else:
        #     X_valid_tensor = torch.load(base_path + '\X_valid_tensor.pt')
        #     y_valid_tensor = torch.load(base_path + '\y_valid_tensor.pt')
        if os.path.exists(base_path + '\X_test_tensor' + str(year) + '.pt') == False:
            X_test_tensor = get_snippets(X_test, 3)
            y_test_tensor = get_snippets(y_test, 3)
            y_test_tensor = y_test_tensor[:, -1]
            torch.save(X_test_tensor, base_path + '\X_test_tensor' + str(year) + '.pt')
            torch.save(y_test_tensor, base_path + '\y_test_tensor' + str(year) + '.pt')
        else:
            X_test_tensor = torch.load(base_path + '\X_test_tensor' + str(year) + '.pt')
            y_test_tensor = torch.load(base_path + '\y_test_tensor' + str(year) + '.pt')


        # 定义GRU模型
        batch_size = 256
        input_size = len(factor_cols)  # 根据训练数据更新input_size
        hidden_size = 128
        num_layers = 2
        dropout = 0.2
        num_epochs = 5

        model = GRUModel(input_size, hidden_size, num_layers, dropout)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第10和20个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        # 开始统计模型运行时间
        start_time = time.time()

        for epoch in range(num_epochs):
            for i in range(0, len(X_train_tensor), batch_size):
                batch_inputs = X_train_tensor[i:i + batch_size]
                batch_targets = y_train_tensor[i:i + batch_size]

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

            # with torch.no_grad():
            #     valid_loss = 0
            #     for i in range(0, len(X_valid_tensor), batch_size):
            #         # batch_x要是序列数据（前几个月的因子值）
            #         batch_inputs = X_valid_tensor[i:i + batch_size]
            #         batch_targets = y_train_tensor[i:i + batch_size]
            #         outputs = model(batch_inputs)
            #         loss = criterion(outputs, batch_targets)
            #         valid_loss += loss.item()
            # print(f"Year: {year}, Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss / len(X_valid_tensor)}")

            # 测试模型
            with torch.no_grad():
                model.eval()
                y_test_pred = model(X_test_tensor)
                # 计算MSE Loss
                loss = criterion(y_test_pred, y_test_tensor)
                print(f"Year: {year}, Epoch: {epoch + 1}/{num_epochs}, Test Loss: {loss.item()}")
                # 计算MAE Loss
                mae_loss = nn.L1Loss()
                loss = mae_loss(y_test_pred, y_test_tensor)
                print(f"Year: {year}, Epoch: {epoch + 1}/{num_epochs}, Test Loss: {loss.item()}")

            scheduler.step()

        end_time = time.time()

        # 同样的处理，合并预测结果到 fac

        predictions = pd.Series(y_test_pred.numpy(), index=X_test.index, name=f'Alpha158_GRU_{year}')

        if not predictions.empty:
            if fac.empty:
                fac = predictions
            else:
                fac = pd.concat([fac, predictions])

    # 在循环结束后释放其它的大数据结构
    del data
    gc.collect()

    # 结束时间并计算运行时间
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60  # 这会给出时间差，以分钟为单位

    print(f"模型运行时间为: {elapsed_time_minutes:.2f} 分钟")

    fac[:] = np.exp(fac) - 1
    index_df = pd.DataFrame(fac.index.tolist(), columns=['date', 'stock'])
    start_date_filter = datetime(2020, 1, 1).date()
    end_date_filter = datetime(2023, 6, 1).date()
    filtered_indices = index_df[
        (index_df['date'] >= start_date_filter) &
        (index_df['date'] <= end_date_filter)
        ].index

    fac_filtered = fac.iloc[filtered_indices]
    # Reset index to MultiIndex with proper names
    fac_filtered.index = pd.MultiIndex.from_tuples(
        fac_filtered.index.tolist(),
        names=['date', 'symbol']
    )

    # 首先，将 'date' 索引级别的日期字符串转换为 datetime.date 对象
    new_levels = pd.to_datetime(data.index.levels[0], format='%Y-%m-%d').date
    data.index =data.index.set_levels(new_levels, level='date')

    # 定义开始日期和结束日期
    start_date = datetime.strptime('2020-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2023-06-01', '%Y-%m-%d').date()

    # 使用多级索引的日期进行筛选
    data_filtered =data[(data.index.get_level_values('date') >= start_date) & (data.index.get_level_values('date') <= end_date)]

    print('切割后的原始300数据是',data_filtered)

    # 选择要保留的列
    columns_to_keep = ['CLOSE', 'INDUSTRY', 'MARKET_CAP', 'NEXT_RET']

    # 使用loc选择这些列并保留索引列
    filtered_data = data_filtered.loc[:, columns_to_keep]
    print('filtered_data',filtered_data)

    # 载入行业
    loader = StockDataLoaderV2(start_date=start_date, end_date=end_date)
    indmap = loader.load_industry_mapping()

    ctx = easy_factor_test(
        factor=fac_filtered,
        stock_data=filtered_data,
        industry_mapping=indmap,
        use_preprocessing=False,
    )
    print(ctx)
    ctx.show()

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
from small_test import give_serial_data

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
    # 载入因子
    data1: pd.DataFrame = pd.read_hdf('C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf')
    print(data1.NEXT_RET)

    # # Step 1: 计算每个日期每个行业的平均收益率
    # industry_daily_mean_returns = data1.groupby(['date', 'INDUSTRY'])['NEXT_RET'].mean()
    # print('industry_daily_mean_returns',industry_daily_mean_returns)
    # # Step 2: 从每只股票的收益率中减去其行业的当天平均收益率
    # def subtract_daily_industry_mean(row):
    #     return industry_daily_mean_returns.loc[row.name[0], row['INDUSTRY']]
    # # 使用`apply`函数并按行(axis=1)应用上面的函数
    # data1['NEXT_RET'] = data1['NEXT_RET'] - data1.apply(subtract_daily_industry_mean, axis=1)
    # print('截面中性化完的收益率',data1.NEXT_RET)

    # 对'NEXT_RET'列进行+1然后取对数操作
    # data1['NEXT_RET'] = np.log(data1['NEXT_RET'] + 1)
    data = data1
    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()
    logger.info('因子已载入。')
    print(data.columns)

    # 载入行业
    loader = StockDataLoaderV2(start_date=start_date, end_date=end_date)
    indmap = loader.load_industry_mapping()


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


    # 先定义GRU模型参数
    input_size = None  # 这里稍后会在循环中被赋值
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    batch_size = 256
    num_epochs = 1

    # 开始统计模型运行时间
    start_time = time.time()

    fac = pd.Series()
    for year in range(2009, 2010):  # 从2010开始滚动，到2022
        logger.info(f"Training model for {year}...")

        # 定义滚动窗口
        train_start = date(year, 1, 1)
        train_end = date(year + 8, 12, 31)
        valid_start = date(year + 9, 1, 1)
        valid_end = date(year + 9, 12, 31)
        test_start = date(year + 10, 1, 1)
        if year == 2012:
            test_end = date(2022, 6, 1)
        else:
            test_end = date(year + 10, 12, 31)

        # 根据滚动窗口划分数据集
        train_data_df = data[(dates >= train_start) & (dates < train_end)]
        valid_data_df = data[(dates >= valid_start) & (dates < valid_end)]
        test_data_df = data[(dates >= test_start) & (dates < test_end)]

        X_train, y_train, factor_cols = get_data_Xy(train_data_df, 'train')
        X_valid, y_valid, _ = get_data_Xy(valid_data_df, 'valid', factor_cols)
        X_test, y_test, _ = get_data_Xy(test_data_df, 'test', factor_cols)

        print('特征列是',factor_cols)
        print('特征数量是',len(factor_cols))
        print('标签列是',y_train)

        input_size = len(factor_cols)  # 根据训练数据更新input_size

        # 转换为 PyTorch tensor
        # new_X_train = give_serial_data(X_train)
        X_train_tensor = torch.FloatTensor(X_train.values).unsqueeze(1)  # [batch, seq_len, input_size]
        # 这里需要改变X_train_tensor，将间隔记录下来

        # new_y_train = give_serial_data(y_train)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_valid_tensor = torch.FloatTensor(X_valid.values).unsqueeze(1)
        y_valid_tensor = torch.FloatTensor(y_valid.values)

        batch_size = 256
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        # 初始化新的模型实例
        model = GRUModel(input_size, hidden_size, num_layers, dropout)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第10和20个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        # 训练模型
        for epoch in range(num_epochs):
            training_loss = 0
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                # 每10个epoch，打印当前学习率
            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}, Training Loss {training_loss / len(train_loader)},Current Learning Rate: {current_lr:.6f}")

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in valid_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    valid_loss += loss.item()
            print(f"Year: {year}, Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss / len(valid_loader)}")

            # 在测试集上进行预测
            X_test_tensor = torch.FloatTensor(X_test.values).unsqueeze(1)
            with torch.no_grad():
                model.eval()
                y_test_pred = model(X_test_tensor)

            # 同样的处理，合并预测结果到 fac
            predictions = pd.Series(y_test_pred.numpy(), index=X_test.index, name=f'Alpha158_GRU_{year}')
            if not predictions.empty:
                if fac.empty:
                    fac = predictions
                else:
                    fac = pd.concat([fac, predictions])
            scheduler.step()

    # 在循环结束后释放其它的大数据结构
    del data1
    gc.collect()

    # 结束时间并计算运行时间
    end_time = time.time()
    elapsed_time_minutes = (end_time - start_time) / 60  # 这会给出时间差，以分钟为单位

    print(f"模型运行时间为: {elapsed_time_minutes:.2f} 分钟")

    # fac[:] = np.exp(fac) - 1
    index_df = pd.DataFrame(fac.index.tolist(), columns=['date', 'stock'])
    start_date_filter = datetime(2019, 1, 1).date()
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
    start_date = datetime.strptime('2019-01-01', '%Y-%m-%d').date()
    end_date = datetime.strptime('2023-06-01', '%Y-%m-%d').date()

    # 使用多级索引的日期进行筛选
    data_filtered =data[(data.index.get_level_values('date') >= start_date) & (data.index.get_level_values('date') <= end_date)]

    print('切割后的原始300数据是',data_filtered)

    # 选择要保留的列
    columns_to_keep = ['CLOSE', 'INDUSTRY', 'MARKET_CAP', 'NEXT_RET']

    # 使用loc选择这些列并保留索引列
    filtered_data = data_filtered.loc[:, columns_to_keep]
    print('filtered_data',filtered_data)


    ctx = easy_factor_test(
        factor=fac_filtered,
        stock_data=filtered_data,
        industry_mapping=indmap,
        use_preprocessing=False,
    )
    print(ctx)
    ctx.show()

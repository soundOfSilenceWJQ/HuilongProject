import os
from datetime import date, datetime
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
from tqdm import tqdm

from get_small_time_span import get_snippets
import pickle

from util import expo, log, get_data_Xy



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
    base_path = 'C:\\Users\ipwx\Desktop\\testing\\'
    # 数据准备
    # 载入因子
    data: pd.DataFrame = pd.read_hdf("C:/Users/ipwx/Desktop/朱/_Alpha158_Financial01_Barra_HT_proceed.hdf")

    data['NEXT_RET'] = np.log(data['NEXT_RET'] + 1)

    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()

    pred_data = pd.Series(index=data.index, name='PRED_NEXT_RET')

    for year in range(2010, 2011):  # 从2009开始滚动，到2022

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
        X_valid, y_valid, _ = get_data_Xy(valid_data_df, 'valid', factor_cols)
        X_test, y_test, _ = get_data_Xy(test_data_df, 'test', factor_cols)
        # 得到snippets
        # 如果没有找到相应文件，就重新生成
        if os.path.exists(base_path + '\X_train_tensor' + str(year) + '.pt') == False:
            X_train_tensor = get_snippets(X_train, 3)
            y_train_tensor = get_snippets(y_train, 3)
            y_train_tensor = y_train_tensor[:, -1]
            torch.save(X_train_tensor, base_path + '\X_train_tensor' + str(year) + '.pt')
            torch.save(y_train_tensor, base_path + '\y_train_tensor' + str(year) + '.pt')

        else:
            X_train_tensor = torch.load(base_path + '\X_train_tensor' + str(year) + '.pt')
            y_train_tensor = torch.load(base_path + '\y_train_tensor' + str(year) + '.pt')

        if os.path.exists(base_path + '\X_valid_tensor.pt') == False:
            X_valid_tensor = get_snippets(X_valid, 3)
            y_valid_tensor = get_snippets(y_valid, 3)
            y_valid_tensor = y_valid_tensor[:, -1]
            torch.save(X_valid_tensor, base_path + '\X_valid_tensor' + str(year) + '.pt')
            torch.save(y_valid_tensor, base_path + '\y_valid_tensor'  + str(year) + '.pt')
        else:
            X_valid_tensor = torch.load(base_path + '\X_valid_tensor'  + str(year) + '.pt')
            y_valid_tensor = torch.load(base_path + '\y_valid_tensor'  + str(year) + '.pt')

        if os.path.exists(base_path + '\X_test_tensor' + str(year) + '.pt') == False:
            X_test_tensor = get_snippets(X_test, 3)
            y_test_tensor, index_info = get_snippets(y_test, 3, True)

            y_test_tensor = y_test_tensor[:, -1]
            torch.save(X_test_tensor, base_path + '\X_test_tensor' + str(year) + '.pt')
            torch.save(y_test_tensor, base_path + '\y_test_tensor' + str(year) + '.pt')
            with open(base_path + 'index_info' + str(year) + '.pkl', 'wb') as f:
                pickle.dump(index_info, f)

        else:
            X_test_tensor = torch.load(base_path + '\X_test_tensor' + str(year) + '.pt')
            y_test_tensor = torch.load(base_path + '\y_test_tensor' + str(year) + '.pt')
            with open(base_path + 'index_info' + str(year) + '.pkl', 'rb') as f:
                index_info = pickle.load(f)

        # 定义GRU模型
        batch_size = 256
        input_size = len(factor_cols)  # 根据训练数据更新input_size
        hidden_size = 256
        num_layers = 2
        dropout = 0.2
        num_epochs = 1

        model = GRUModel(input_size, hidden_size, num_layers, dropout)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第10和20个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        # 开始统计模型运行时间
        start_time = time.time()

        logger.info(f"Training model for {year}...")

        for epoch in range(num_epochs):
            for i in tqdm(range(0, len(X_train_tensor), batch_size), desc=f'Epoch {epoch} train'):
                optimizer.zero_grad()
                batch_inputs = X_train_tensor[i:i + batch_size]
                batch_targets = y_train_tensor[i:i + batch_size]

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

                loss.backward()
                optimizer.step()
                # print('batch mae: ', torch.mean(torch.abs(outputs - batch_targets)))
                # print('batch targets mean:', torch.mean(batch_targets))
                # print('batch outputs mean:', torch.mean(outputs))

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item():.4f}")

            # 衡量在验证集上的性能
            with torch.no_grad():
                valid_loss = 0
                for i in tqdm(range(0, len(X_valid_tensor), batch_size), desc=f'Epoch {epoch} valid'):
                    # batch_x要是序列数据（前几个月的因子值）
                    batch_inputs = X_valid_tensor[i:i + batch_size]
                    batch_targets = y_valid_tensor[i:i + batch_size]
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    valid_loss += loss.item()
                print(f"Validation Loss: {valid_loss / len(X_valid_tensor)}")
                y_valid_pred = model(X_valid_tensor)
                print("mae:", torch.mean(torch.abs(expo(y_valid_pred) - expo(y_valid_tensor))))
                print("mean of y_valid_tensor:", torch.mean(expo(y_valid_tensor)))
                print("mean of y_valid_pred:", torch.mean(expo(y_valid_pred)))
            #
            scheduler.step()

        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60  # 这会给出时间差，以分钟为单位

        print(f"模型运行时间为: {elapsed_time_minutes:.2f} 分钟")

        # 测试模型
        with torch.no_grad():
            model.eval()
            y_train_pred = model(X_train_tensor)
            print("total train mae:", torch.mean(torch.abs(torch.exp(y_train_pred) - torch.exp(y_train_tensor))))

        with torch.no_grad():
            model.eval()
            y_test_pred = model(X_test_tensor)
            print("test mae:", torch.mean(torch.abs(torch.exp(y_test_pred) - torch.exp(y_test_tensor))))

        y_test_pred = expo(y_test_pred)
        y_test_pred = y_test_pred.cpu().numpy()
        # 遍历y_test_pred，将其填入new_data中
        for i in range(len(y_test_pred)):
            pred_data.loc[(index_info[i][0], index_info[i][1])] = y_test_pred[i]

        # pred_data.to_csv('C:\\Users\ipwx\Desktop\\testing\\pred_data.csv')

    columns_to_keep = ['CLOSE', 'INDUSTRY', 'MARKET_CAP', 'NEXT_RET']
    merged_data = pd.concat([data.loc[:, columns_to_keep], pred_data], axis=1, join='inner')
    filtered_data = merged_data.dropna(how='any')

    # 载入行业
    loader = StockDataLoaderV2(start_date=start_date, end_date=end_date)
    indmap = loader.load_industry_mapping()

    ctx = easy_factor_test(
        factor=pred_data,
        stock_data=filtered_data,
        industry_mapping=indmap,
        use_preprocessing=False,
    )
    print(ctx)
    ctx.show()



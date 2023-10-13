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

from data_preparation import get_snippets, load_snippets
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
    # 一些训练参数：
    class config:
        TN = 3
        training_time_span = 10
        valid_time_span = 0
        testing_time_span = 1
        batch_size = 256
        input_size = 246  # 根据训练数据更新input_size
        hidden_size = 256
        num_layers = 2
        dropout = 0.2
        num_epochs = 1
        base_path = 'C:\\Users\\ipwx\\Desktop\\testing\\'
    # 数据准备
    # 载入因子
    data: pd.DataFrame = pd.read_hdf(config.base_path + "_Alpha158_Financial01_Barra_HT_proceed.hdf")

    a, b, c = load_snippets([2009], 'C:\\Users\\ipwx\\Desktop\\testing\\')

    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()

    pred_data = pd.Series(index=data.index, name='PRED_NEXT_RET')

    for year in range(2009, 2010):  # 从2009开始滚动，到2022
        # 定义滚动窗口
        train_year_list = [year + i for i in range(config.training_time_span)]
        valid_year_list = [year + config.training_time_span + i for i in range(config.valid_time_span)]
        test_year_list = [year + config.training_time_span + config.valid_time_span + i for i in range(config.testing_time_span)]
        train_X, train_y, _ = load_snippets(train_year_list, config.base_path, config.TN)
        valid_X, valid_y, _ = load_snippets(valid_year_list, config.base_path, config.TN)
        test_X, test_y, index_info = load_snippets(test_year_list, config.base_path, config.TN)

        model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.dropout)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第10和20个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

        # 开始统计模型运行时间
        start_time = time.time()

        logger.info(f"Training model for {year}...")

        for epoch in range(config.num_epochs):
            model.train()
            running_loss = 0
            for i in tqdm(range(0, len(train_X), config.batch_size), desc=f'Epoch {epoch} train'):
                optimizer.zero_grad()
                batch_inputs = train_X[i:i + config.batch_size]
                batch_targets = train_y[i:i + config.batch_size]

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # print('batch mae: ', torch.mean(torch.abs(outputs - batch_targets)))
                # print('batch targets mean:', torch.mean(batch_targets))
                # print('batch outputs mean:', torch.mean(outputs))
            # y_train_pred = model(train_X)
            # print("mae:", torch.mean(torch.abs(expo(y_train_pred) - expo(y_valid_tensor))))
            # print("mean of train_y:", torch.mean(expo(train_y)))
            # print("mean of y_train_pred:", torch.mean(expo(y_train_pred)))

            print(f"Epoch {epoch + 1}/{config.num_epochs}, Training Loss: {running_loss / config.batch_size:.4f}")

            # # 衡量在验证集上的性能
            # with torch.no_grad():
            #     valid_loss = 0
            #     for i in tqdm(range(0, len(X_valid_tensor), batch_size), desc=f'Epoch {epoch} valid'):
            #         # batch_x要是序列数据（前几个月的因子值）
            #         batch_inputs = X_valid_tensor[i:i + batch_size]
            #         batch_targets = y_valid_tensor[i:i + batch_size]
            #         outputs = model(batch_inputs)
            #         loss = criterion(outputs, batch_targets)
            #         valid_loss += loss.item()
            #     print(f"Validation Loss: {valid_loss / len(X_valid_tensor)}")
            #     y_valid_pred = model(X_valid_tensor)
            #     print("mae:", torch.mean(torch.abs(expo(y_valid_pred) - expo(y_valid_tensor))))
            #     print("mean of y_valid_tensor:", torch.mean(expo(y_valid_tensor)))
            #     print("mean of y_valid_pred:", torch.mean(expo(y_valid_pred)))
            #
            scheduler.step()

        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60

        print(f"模型运行时间为: {elapsed_time_minutes:.2f} 分钟")

        # 测试模型
        with torch.no_grad():
            model.eval()
            y_train_pred = model(train_X)
            print("total train mae:", torch.mean(torch.abs(expo(y_train_pred) - expo(train_y))))
            print("train target mean:", torch.mean(expo(train_y)))
            print("train pred mean:", torch.mean(expo(y_train_pred)))

        with torch.no_grad():
            model.eval()
            test_y_pred = model(test_X)
            print("test mae:", torch.mean(torch.abs(expo(test_y_pred) - expo(test_y))))
            print("test target mean:", torch.mean(expo(test_y)))
            print("test pred mean:", torch.mean(expo(test_y_pred)))

        y_test_pred = expo(test_y_pred)
        y_test_pred = y_test_pred.cpu().numpy()
        # 遍历y_test_pred，将其填入new_data中
        for i in range(len(y_test_pred)):
            pred_data.loc[(index_info[i][0], index_info[i][1])] = y_test_pred[i]



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



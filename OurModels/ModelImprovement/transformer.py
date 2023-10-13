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

from data_preparation import get_snippets
import pickle

from util import expo, log, get_data_Xy


# Transformer模型定义
class StockPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(StockPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.transformer(src, src)
        return self.fc(output)

if __name__ == '__main__':
    base_path = 'C:\\Users\ipwx\Desktop\\testing\\'
    # 数据准备
    # 载入因子
    data: pd.DataFrame = pd.read_hdf(base_path + "_Alpha158_Financial01_Barra_HT_proceed.hdf")

    data['NEXT_RET'] = np.log(data['NEXT_RET'] + 1)

    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()

    pred_data = pd.Series(index=data.index, name='PRED_NEXT_RET')

    for year in range(2009, 2010):  # 从2009开始滚动，到2022

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
            torch.save(X_valid_tensor, base_path + '\X_valid_tensor.pt')
            torch.save(y_valid_tensor, base_path + '\y_valid_tensor.pt')
        else:
            X_valid_tensor = torch.load(base_path + '\X_valid_tensor.pt')
            y_valid_tensor = torch.load(base_path + '\y_valid_tensor.pt')

        if os.path.exists(base_path + '\X_test_tensor' + str(year) + '.pt') == False:
            X_test_tensor = get_snippets(X_test, 3)
            y_test_tensor, index_info = get_snippets(y_test, 3, True)

            y_test_tensor = y_test_tensor[:, -1]
            torch.save(X_test_tensor, base_path + '\X_test_tensor' + str(year) + '.pt')
            torch.save(y_test_tensor, base_path + '\y_test_tensor' + str(year) + '.pt')
            with open(base_path + 'index_info.pkl', 'wb') as f:
                pickle.dump(index_info, f)

        else:
            X_test_tensor = torch.load(base_path + '\X_test_tensor' + str(year) + '.pt')
            y_test_tensor = torch.load(base_path + '\y_test_tensor' + str(year) + '.pt')
            with open(base_path + 'index_info.pkl', 'rb') as f:
                index_info = pickle.load(f)

        # 创建模型实例
        model = StockPredictor(d_model=factors, nhead=factors, num_encoder_layers=3, num_decoder_layers=3)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        num_epochs = 100
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(data).squeeze(-1)
            loss = criterion(outputs[:, :-1], returns[:, 1:])  # 预测下一个时间点的收益率
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ...接下来可以评估模型在验证集和测试集上的性能

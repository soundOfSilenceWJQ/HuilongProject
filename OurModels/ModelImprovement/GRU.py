import os
from datetime import date, datetime
import pandas as pd
import numpy as np
from loguru import logger
from quant_stock.backtest import easy_factor_test
from quant_stock.core import StockDataLoaderV2
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import time
import gc
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from data_preparation import load_snippets

from util import expo, log, filter_by_date


# 模型定义
class MLPModel(nn.Module):
    def __init__(self, input_size=158, output_size=1):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64,32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

class GRU(nn.Module):
    def __init__(self, input_size=158, hidden_size=64, target_size=1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,# 传入我们上面定义的参数
            hidden_size=hidden_size,# 传入我们上面定义的参数
            batch_first=True,# 为什么设置为True上面解释过了
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 32), # 加入线性层的原因是，GRU的输出，参考官网为(batch_size, seq_len, hidden_size)
            nn.LeakyReLU(),             # 这边的多层全连接，根据自己的输出自己定义就好，
            nn.Linear(32, 16),          # 我们需要将其最后打成（batch_size, output_size）比如单值预测，这个output_size就是1，
            nn.LeakyReLU(),             # 这边我们等于targets
            nn.Linear(16, target_size)      # 这边输出的（batch_size, targets）且这个targets是上面一个模块已经定义好了
        )

    def forward(self, input):
        output, h_n = self.gru(input, None)# output:(batch_size, seq_len, hidden_size)，h0可以直接None
        # print(output.shape)
        output = output[:, -1, :]# output:(batch_size, hidden_size)
        output = self.mlp(output)# 进过一个多层感知机，也就是全连接层，output:(batch_size, output_size)
        return output

class LSTM(nn.Module):
    def __init__(self, input_size=158, hidden_size=64, target_size=1, drop_out=0.15):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 传入我们上面定义的参数
            hidden_size=hidden_size,  # 传入我们上面定义的参数
            batch_first=True,  # 为什么设置为True上面解释过了
            dropout=drop_out
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 32), # 加入线性层的原因是，GRU的输出，参考官网为(batch_size, seq_len, hidden_size)
            nn.LeakyReLU(),             # 这边的多层全连接，根据自己的输出自己定义就好，
            nn.Linear(32, 16),          # 我们需要将其最后打成（batch_size, output_size）比如单值预测，这个output_size就是1，
            nn.LeakyReLU(),             # 这边我们等于targets
            nn.Linear(16, target_size)      # 这边输出的（batch_size, targets）且这个targets是上面一个模块已经定义好了
        )

    def forward(self, input):
        output, h_n = self.lstm(input, None)  # output:(batch_size, seq_len, hidden_size)，h0可以直接None
        # print(output.shape)
        output = output[:, -1, :]  # output:(batch_size, hidden_size)
        output = self.mlp(output)  # 进过一个多层感知机，也就是全连接层，output:(batch_size, output_size)
        return output


if __name__ == '__main__':
    # 一些训练参数：
    class TrainConfig:
        TN = 2  # 1代表只用当期数据，2代表用当期和前一期数据，以此类推
        train_begin_year = 2009  # 从2009开始滚动
        training_time_span = 5  # 用于训练的年份数
        valid_time_span = 0  # 用于验证的年份数
        batch_size = 256
        input_size = 158  # 模型输入向量维度，即每个截面的因子数量
        num_epochs = 10  # 训练epoch数
        base_path = 'C:\\Users\\ipwx\\Desktop\\testing\\'
        rolling_window = 2022 - train_begin_year - training_time_span - valid_time_span + 1  # 滚动次数，至少是1

    class ModelConfig:
        input_size = 158
        hidden_size = 64
        num_layers = 2
        dropout = 0.15  # 模型训练参数，dropout rate

    # 数据准备
    data: pd.DataFrame = pd.read_hdf(TrainConfig.base_path + "_Alpha158_Financial01_Barra_HT_proceed.hdf")
    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()
    pred_data = pd.Series(index=data.index, name='PRED_NEXT_RET')

    for year in range(TrainConfig.train_begin_year, TrainConfig.train_begin_year + TrainConfig.rolling_window):
        # 定义滚动窗口
        train_year_list = [year + i for i in range(TrainConfig.training_time_span)]
        valid_year_list = [year + TrainConfig.training_time_span + i for i in range(TrainConfig.valid_time_span)]
        test_year_list = [year + TrainConfig.training_time_span + TrainConfig.valid_time_span]  # 测试只用1年的数据
        train_X, train_y, _ = load_snippets(train_year_list, TrainConfig.base_path, TrainConfig.TN)
        valid_X, valid_y, _ = load_snippets(valid_year_list, TrainConfig.base_path, TrainConfig.TN)
        test_X, test_y, index_info = load_snippets(test_year_list, TrainConfig.base_path, TrainConfig.TN)

        train_dataset = TensorDataset(train_X, train_y)
        valid_dataset = TensorDataset(valid_X, valid_y)
        test_dataset = TensorDataset(test_X, test_y)
        train_loader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size, shuffle=True)
        if TrainConfig.valid_time_span > 0:
            valid_loader = DataLoader(valid_dataset, batch_size=TrainConfig.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=TrainConfig.batch_size, shuffle=True)

        model = LSTM(ModelConfig.input_size, ModelConfig.hidden_size, 1, ModelConfig.dropout)
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第5和10个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[int(TrainConfig.num_epochs / 2), int(TrainConfig.num_epochs / 4 * 3)], gamma=0.1)

        # 开始统计模型运行时间
        start_time = time.time()

        logger.info(f"Training model for {year}...")

        for epoch in range(TrainConfig.num_epochs):
            training_loss = 0
            model.train()
            for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1} train'):
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.reshape(-1, 1))
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            print(f"Training Loss: {training_loss / len(train_loader):.6f}")

            # 在验证集上进行测试
            if TrainConfig.valid_time_span > 0:
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in tqdm(valid_loader, desc=f'Epoch {epoch + 1} valid'):
                        # batch_x要是序列数据（前几个月的因子值）
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y.reshape(-1, 1))
                        valid_loss += loss.item()
                    print(f"Validation Loss: {valid_loss / len(valid_X) * TrainConfig.batch_size:.6f}")
            scheduler.step()

        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60

        print(f"模型运行时间为: {elapsed_time_minutes:.2f} 分钟")

        # 在测试集上进行评估
        with torch.no_grad():
            model.eval()
            test_y_pred = model(test_X)     # 这种写法有问题，应该一个一个batch的预测
            test_mse_loss = 0
            test_mae_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in tqdm(test_loader):
                    outputs = model(batch_x)
                    test_mse_loss += criterion(outputs, batch_y.reshape(-1, 1)).item()
                    test_mae_loss += criterion2(outputs, batch_y.reshape(-1, 1)).item()
            test_y_pred = test_y_pred.cpu().numpy()
            print("test mse:", test_mse_loss / len(test_loader))
            print("test mae:", test_mae_loss / len(test_loader))
            print("test_y mean:", torch.mean(test_y))
            print("test_y_pred mean:", np.mean(test_y_pred))

        # index_info_year_len = 0
        # 遍历y_test_pred，将其填入new_data中
        # test_begin_year = test_year_list[0]
        # begin_token = 0
        # end_token = len(index_info[0])  # 第一年项目数
        # for test_year in test_year_list:
        #     for i in range(begin_token, end_token):
        #         pred_data.loc[(index_info[test_year - test_begin_year][i - begin_token][0],
        #                        index_info[test_year - test_begin_year][i - begin_token][1])] = test_y_pred[i]
        #     begin_token = end_token
        #     end_token = begin_token + len(index_info[test_year - test_begin_year])
        for i in range(0, len(test_y_pred)):
            pred_data.loc[(index_info[0][i][0], index_info[0][i][1])] = test_y_pred[i]

    columns_to_keep = ['CLOSE', 'INDUSTRY', 'MARKET_CAP', 'NEXT_RET']
    merged_data = pd.concat([data.loc[:, columns_to_keep], pred_data], axis=1, join='inner')

    # 对merged_data和pred_data进行日期过滤
    test_begin_year = TrainConfig.train_begin_year + TrainConfig.training_time_span + TrainConfig.valid_time_span
    test_end_year = test_begin_year + TrainConfig.rolling_window - 1
    start_date_filter = datetime(test_begin_year, 1, 1).date()
    end_date_filter = datetime(test_end_year, 12, 31).date()
    merged_data = filter_by_date(merged_data, start_date_filter, end_date_filter)
    pred_data = filter_by_date(pred_data, start_date_filter, end_date_filter)

    # 载入行业
    loader = StockDataLoaderV2(start_date=start_date, end_date=end_date)
    ind_map = loader.load_industry_mapping()

    ctx = easy_factor_test(
        factor=pred_data,
        stock_data=merged_data.loc[:, columns_to_keep],
        industry_mapping=ind_map,
        use_preprocessing=False,
    )
    print(ctx)
    ctx.show()

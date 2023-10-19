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
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)  # 假设输出是一个数值

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # GRU运算
        output, h_0 = self.gru(x, h_0)

        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 转换维度，用于输出
        output = output.reshape(timestep, batch_size, -1)

        # 我们只需要返回最后一个时间片的数据即可
        return output[-1]


if __name__ == '__main__':
    # 一些训练参数：
    class config:
        TN = 2  # 1代表只用当期数据，2代表用当期和前一期数据，以此类推
        train_begin_year = 2009  # 从2009开始滚动
        training_time_span = 5  # 用于训练的年份数
        valid_time_span = 0  # 用于验证的年份数
        batch_size = 256
        input_size = 158  # 模型输入向量维度，即每个截面的因子数量
        hidden_size = 64  # 模型隐藏层大小
        num_layers = 2  # gru隐藏层层数
        dropout = 0.2  # 模型训练参数，dropout rate
        num_epochs = 1  # 训练epoch数
        base_path = 'C:\\Users\\ipwx\\Desktop\\testing\\'
        rolling_window = 1  # 滚动次数，至少是1


    # 数据准备
    data: pd.DataFrame = pd.read_hdf(config.base_path + "_Alpha158_Financial01_Barra_HT_proceed.hdf")
    dates = data.index.get_level_values('date')
    start_date, end_date = dates.min(), dates.max()
    pred_data = pd.Series(index=data.index, name='PRED_NEXT_RET')

    for year in range(config.train_begin_year, config.train_begin_year + config.rolling_window):
        # 定义滚动窗口
        train_year_list = [year + i for i in range(config.training_time_span)]
        valid_year_list = [year + config.training_time_span + i for i in range(config.valid_time_span)]
        test_year_list = [year + config.training_time_span + config.valid_time_span]  # 测试只用1年的数据
        train_X, train_y, _ = load_snippets(train_year_list, config.base_path, config.TN)
        valid_X, valid_y, _ = load_snippets(valid_year_list, config.base_path, config.TN)
        test_X, test_y, index_info = load_snippets(test_year_list, config.base_path, config.TN)

        train_dataset = TensorDataset(train_X, train_y)
        valid_dataset = TensorDataset(valid_X, valid_y)
        test_dataset = TensorDataset(test_X, test_y)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        if config.valid_time_span > 0:
            valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        model = GRUModel(config.input_size, config.hidden_size, config.num_layers, config.dropout)
        criterion = nn.MSELoss()
        criterion2 = nn.L1Loss()
        optimizer = Adam(model.parameters(), lr=0.01)  # 初始化时的学习率设置为0.01

        # 定义当达到第5和10个epoch时降低学习率
        scheduler = MultiStepLR(optimizer, milestones=[int(config.num_epochs / 2), int(config.num_epochs / 4 * 3)], gamma=0.1)

        # 开始统计模型运行时间
        start_time = time.time()

        logger.info(f"Training model for {year}...")

        for epoch in range(config.num_epochs):
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
            if config.valid_time_span > 0:
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in tqdm(valid_loader, desc=f'Epoch {epoch + 1} valid'):
                        # batch_x要是序列数据（前几个月的因子值）
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y.reshape(-1, 1))
                        valid_loss += loss.item()
                    print(f"Validation Loss: {valid_loss / len(valid_X) * config.batch_size:.6f}")
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
    test_begin_year = config.train_begin_year + config.training_time_span + config.valid_time_span
    test_end_year = test_begin_year + config.rolling_window - 1
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

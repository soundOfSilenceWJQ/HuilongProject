import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """MLP只能用来做TN=1的测试"""
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
    def __init__(self, input_size=158, hidden_size=64, target_size=1, dropout=0.0):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 32),     # 加入线性层的原因是，GRU的输出，参考官网为(batch_size, seq_len, hidden_size)
            nn.LeakyReLU(),             # 这边的多层全连接，根据自己的输出自己定义就好，
            nn.Linear(32, 16),          # 我们需要将其最后打成（batch_size, output_size）比如单值预测，这个output_size就是1，
            nn.LeakyReLU(),             # 这边我们等于targets
            nn.Linear(16, target_size)      # 这边输出的（batch_size, targets）且这个targets是上面一个模块已经定义好了
        )

    def forward(self, input):
        output, h_n = self.gru(input, None)  # output:(batch_size, seq_len, hidden_size)，h0可以直接None
        output = output[:, -1, :]  # output:(batch_size, hidden_size)
        output = self.mlp(output)  # 进过一个多层感知机，也就是全连接层，output:(batch_size, output_size)
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


# class Transformer(nn.Module):
#     def __init__(self, d_model=158, nhead=2, num_encoder_layers=1, num_decoder_layers=1):
#         super(Transformer, self).__init__()
#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=True)
#         self.fc = nn.Linear(d_model, 1)
#
#     def forward(self, src):
#         output = self.transformer(src, src)
#         return self.fc(output)

class Transformer(nn.Module):
    def __init__(self, input_dim=158, d_model=128, nhead=4, num_encoder_layers=1, output_dim=1):
        super(Transformer, self).__init__()
        # self.embedding = nn.Embedding(input_dim, d_model)
        self.feature_layer = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, src):
        # src = self.embedding(src) * math.sqrt(self.d_model)
        output = self.feature_layer(src)
        output = self.transformer_encoder(output)
        output = self.fc(output)
        return output

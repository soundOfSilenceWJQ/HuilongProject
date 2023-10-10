import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generating Dummy Stock Price Data
num_days = 3650
num_features = 246
stock_feature = torch.rand(num_days, num_features)
stock_revenue = torch.rand(num_days, 1)

# Preprocessing the Data
input_seq_len = 10
output_seq_len = 1
num_samples = num_days - input_seq_len - output_seq_len + 1

# 将stock_feature和stock_revenue转换为滚动训练数据
# print(stock_feature[0:0+input_seq_len])
src_data = []
tgt_data = []

# 遍历stock_feature
for i in range(num_samples):
    src_data.append(stock_feature[i:i+input_seq_len].numpy().tolist())
    tgt_data.append(stock_feature[i:i+input_seq_len].numpy().tolist())

# 将src_data和tgt_data转换为tensor
src_data = torch.tensor(src_data).float()
tgt_data = torch.tensor(tgt_data).float()


# src_data = torch.tensor([stock_feature[i:i+input_seq_len] for i in range(num_samples)]).float()
# tgt_data = torch.tensor([stock_revenue[i+input_seq_len:i+input_seq_len+output_seq_len] for i in range(num_samples)]).float()

# Creating a Custom Transformer Model
class StockPriceTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super(StockPriceTransformer, self).__init__()
        self.input_linear = nn.Linear(num_features, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_linear(src)
        tgt = self.input_linear(tgt)
        output = self.transformer(src, tgt)
        output = self.output_linear(output)
        return output

d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

model = StockPriceTransformer(d_model, nhead, num_layers, dropout=dropout)

# Training the Model
epochs = 20
lr = 0.001
batch_size = 100

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i in range(0, num_samples, batch_size):
        src_batch = src_data[i:i+batch_size].transpose(0, 1)
        tgt_batch = tgt_data[i:i+batch_size].transpose(0, 1)

        optimizer.zero_grad()
        output = model(src_batch, tgt_batch[:-1])
        loss = criterion(output, tgt_batch[1:])
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Predicting the Next 5 Days of Stock Prices
src = torch.tensor(stock_feature[-input_seq_len:]).unsqueeze(-2).float()
tgt = torch.zeros(output_seq_len, 1).unsqueeze(-2).float()

for i in range(output_seq_len):
    output = model(src, tgt)
    new_tgt_point = output[-1].item()
    tgt[i][0][0] = new_tgt_point

    new_src_point_tensor = torch.tensor([stock_feature[-output_seq_len+i+1]]).unsqueeze(-2).float()
    src[0:-1] = src[1:]
    src[-1] = new_src_point_tensor

predicted_stock_prices_change_rate= (tgt.squeeze() - stock_revenue[-output_seq_len:])/stock_revenue[-output_seq_len:]
print(f"Predicted stock prices change rate for the next {output_seq_len} days: {predicted_stock_prices_change_rate}")
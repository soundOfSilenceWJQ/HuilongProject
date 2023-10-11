import torch
import torch.nn as nn
import torch.optim as optim

# 1. 数据预处理
# 假设你的数据是一个(时间长度 x 因子数量)的张量
# 这里我们模拟一些数据
data_len = 1000
factor_num = 5
data = torch.randn(data_len, factor_num)
returns = torch.randn(data_len, 1)

seq_length = 3
inputs, targets = [], []

for i in range(data_len - seq_length):
    inputs.append(data[i:i + seq_length])
    targets.append(returns[i + seq_length])

inputs = torch.stack(inputs, 0)
targets = torch.stack(targets, 0)


# 2. 定义GRU模型
class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


model = StockGRU(input_size=factor_num, hidden_size=20, num_layers=2, output_size=1)

# 3. 训练模型
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_targets = targets[i:i + batch_size]

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('batch mae:', torch.mean(torch.abs(outputs - batch_targets)))
        print('batch targets mean:', torch.mean(batch_targets))
        print('batch outputs mean:', torch.mean(outputs))

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

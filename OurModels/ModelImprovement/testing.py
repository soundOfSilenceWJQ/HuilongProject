import torch
import torch.nn as nn
import torch.optim as optim

# 假设你的数据是以下形状：
stocks, time_length, factors = 1000, 3, 246
data = torch.randn(stocks, time_length, factors)
returns = torch.randn(stocks, time_length)

# Transformer模型定义
class StockPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(StockPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.transformer(src, src)
        return self.fc(output)

# 创建模型实例
model = StockPredictor(d_model=factors, nhead=factors, num_encoder_layers=3, num_decoder_layers=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 2
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(data).squeeze(-1)
    loss = criterion(outputs[:, :-1], returns[:, 1:])  # 预测下一个时间点的收益率
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ...接下来可以评估模型在验证集和测试集上的性能

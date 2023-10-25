import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ListNet模型定义
class ListNet(nn.Module):
    def __init__(self, input_dim):
        super(ListNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        score = self.fc3(out)
        return score

# 自定义数据集类
class StockDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'target': self.targets[idx]}
        return sample


# 训练ListNet模型
def train_listnet(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            data, target = batch['data'], batch['target']
            score = model(data)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")


# 示例用法
if __name__ == "__main__":
    # 假设你有一个包含因子值的数据集和对应的排名标签
    data = [[[1, 1], [2, 2], [3, 3]], [[1, 1], [3, 3], [2, 2]], [[2, 2], [1, 1], [3, 3]], [[2, 2], [3, 3], [1, 1]],
            [[3, 3], [1, 1], [2, 2]], [[3, 3], [2, 2], [1, 1]]]
    targets = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    # 将data和targets转化为torch向量
    data = torch.tensor(data)
    targets = torch.tensor(targets)

    input_dim = 3  # 输入特征维度
    model = ListNet(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 创建自定义数据集和数据加载器
    dataset = StockDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_epochs = 100
    train_listnet(model, dataloader, optimizer, criterion, num_epochs)

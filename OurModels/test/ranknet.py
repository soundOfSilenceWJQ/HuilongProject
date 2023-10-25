import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RankNetDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


class RankNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RankNetModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        o1 = self.fc(x1)
        o2 = self.fc(x2)
        return o1, o2


def ranknet_loss(o1, o2, label, epsilon=1e-6):
    """
    Compute the RankNet loss.
    label should be either -1, 0 or 1.
    """
    diff = o1 - o2
    return torch.mean(torch.log(1.0 + torch.exp(-label * diff)) + epsilon * (diff ** 2))


# 假设数据
# X1和X2分别为两组文档的特征
# X1 = torch.randn(1000, 32)
# X2 = torch.randn(1000, 32)
# 将X1, X2随机填充全1或全0向量
X1 = torch.empty(1000, 32)
X2 = torch.empty(1000, 32)
y = torch.empty(1000)
ones = torch.ones(32)
zeros = torch.zeros(32)
for i in range(0, 1000):
    if random.random() < 0.5:
        X1[i] = ones
        X2[i] = zeros
        y[i] = 1
    else:
        X1[i] = zeros
        X2[i] = ones
        y[i] = -1

# y为标签，1表示X1的文档排名高于X2，-1表示相反，0表示两者相等
# y = torch.randint(-1, 2, (1000,))

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1.numpy(), X2.numpy(), y.numpy(),
                                                                         test_size=0.2, random_state=42)

train_dataset = RankNetDataset(X1_train, X2_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = RankNetDataset(X1_test, X2_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32)

model = RankNetModel(input_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (input1, input2, label) in enumerate(train_loader):
        optimizer.zero_grad()
        o1, o2 = model(input1.float(), input2.float())
        loss = ranknet_loss(o1.view(-1), o2.view(-1), label.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (i + 1):.4f}")

print("Training complete!")

# 测试模型
with torch.no_grad():
    model.eval()
    testing_loss = 0
    for i, (input1, input2, label) in enumerate(test_loader):
        o1, o2 = model(input1.float(), input2.float())
        loss = ranknet_loss(o1.view(-1), o2.view(-1), label.float())
        testing_loss += loss.item()
    print(f"Testing Loss: {testing_loss / (i + 1):.4f}")

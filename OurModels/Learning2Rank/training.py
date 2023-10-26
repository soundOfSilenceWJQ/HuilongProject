import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import models


class RankNetDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def ranknet_loss(o1, o2, label, epsilon=1e-6):
    """
    Compute the RankNet loss.
    label should be either -1, 0 or 1.
    """
    diff = o1 - o2
    return torch.mean(torch.log(1.0 + torch.exp(-label * diff)) + epsilon * (diff ** 2))


if __name__ == "__main__":
    batch_size = 256
    base_path = "C:\\Users\\ipwx\\Desktop\\testing\\Ranking\\"
    X1 = torch.load(base_path + "X1_tensor.pt")
    X2 = torch.load(base_path + "X2_tensor.pt")
    y = torch.load(base_path + "y_tensor.pt")
    y = (y + 1) / 2

    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1.numpy(), X2.numpy(), y.numpy(),
                                                                             test_size=0.2, random_state=42)

    train_dataset = RankNetDataset(X1_train, X2_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = RankNetDataset(X1_test, X2_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = models.RankNetModel(input_dim=158)
    bceloss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct_num = 0
        total_count = 0
        for i, (input1, input2, label) in enumerate(train_loader):
            optimizer.zero_grad()
            prob = model(input1.float(), input2.float())

            loss = bceloss(prob.squeeze(), label)
            # o1, o2 = model(input1.float(), input2.float())
            # loss = ranknet_loss(o1.squeeze(), o2.squeeze(), label.float())
            loss.backward()
            optimizer.step()
            predicted_order = (prob > 0.5).float()
            diff = predicted_order - label
            # 统计diff中0的数量
            train_correct_num += (diff == 0).sum().item()
            total_count += len(label)
            running_loss += loss.item()
        print(f"Training epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / (i + 1):.4f}, Accuracy: {train_correct_num / total_count}")
        # 测试模型
        with torch.no_grad():
            model.eval()
            testing_loss = 0
            testing_correct_num = 0
            total_count = 0
            for i, (input1, input2, label) in enumerate(test_loader):
                prob = model(input1.float(), input2.float())

                loss = bceloss(prob.squeeze(), label)
                predicted_order = (prob > 0.5).float()
                diff = predicted_order - label
                # 统计diff中0的数量
                testing_correct_num += (diff == 0).sum().item()
                total_count += len(label)
                testing_loss += loss.item()
            print(f"Valid Loss: {testing_loss / (i + 1):.4f}, Accuracy: {testing_correct_num / total_count}")
    print("Training complete!")

    # 将模型进行本地存储
    torch.save(model.state_dict(), 'model_weights.pth')





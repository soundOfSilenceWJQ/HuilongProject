import torch
import models
import pandas as pd

if __name__ == '__main__':
    model = models.RankNetModel(input_dim=158)  # 请确保MyModel类的结构与你保存时的模型结构相同
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # 设置为评估模式，对于有Dropout和BatchNorm是必要的


    def compare(tensor1, tensor2):
        with torch.no_grad():
            prob = model(tensor1.float(), tensor2.float())
            if prob > 0.5:
                return 1
            elif prob < 0.5:
                return -1
            else:
                return 0

    # 将某几年的数据提取出来做测试
    data = pd.read_hdf("")

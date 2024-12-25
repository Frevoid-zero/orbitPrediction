import sys

import torch.nn as nn
import torch

class WeightedMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight  # 可以传递一个权重参数

    def forward(self, input, target):
        # print(input.size())
        # print(target.size())
        # sys.exit(0)
        loss = torch.mean((input - target) ** 2)  # 普通的 MSE
        return self.weight * loss  # 返回加权的 MSE 损失

if __name__ == "__main__":
    criterion = WeightedMSELoss(1)
    pred = torch.zeros((3,4,1))
    target = torch.zeros((3,4,8))
    for i in range(4):
        for j in range(3):
            pred[j][i][0] = j
    for i in range(4):
        for j in range(3):
            target[j][i][0] = i
    sum = 0
    for i in range(4):
        for j in range(3):
            sum += (pred[j][i][0]-target[j][i][0])*(pred[j][i][0]-target[j][i][0])
    sum/=12
    print(sum)
    loss = criterion(pred.squeeze(-1), target[:, :, 0])
    print(loss)
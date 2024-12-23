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
    pass
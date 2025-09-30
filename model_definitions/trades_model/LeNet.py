import torch
from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入维度是16*4*4，输出维度是120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 第二个全连接层：输入维度是120，输出维度是84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet1(nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = self.fc1(x)
        return x


class LeNet4(nn.Module):
    def __init__(self):
        super(LeNet4, self).__init__()

        # 第一层卷积 + 池化
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入1通道，输出6通道，卷积核大小为5
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 使用2x2的平均池化

        # 第二层卷积 + 池化
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入6通道，输出16通道，卷积核大小为5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 使用2x2的平均池化

        # 全连接层
        self.fc1 = nn.Linear(16 *4 * 4, 84)  # 16个4x4的特征图，展平后输入到全连接层
        self.fc2 = nn.Linear(84, 10)  # 84个神经元到10个输出，代表分类结果

    def forward(self, x):
        # 卷积层 + 池化层
        x = torch.relu(self.conv1(x))  # 卷积 + 激活
        x = self.pool1(x)  # 池化
        x = torch.relu(self.conv2(x))  # 卷积 + 激活
        x = self.pool2(x)  # 池化

        # 展平
        x = x.view(-1, 16 * 4 * 4)  # 展平为一维张量

        # 全连接层
        x = torch.relu(self.fc1(x))  # 激活
        x = self.fc2(x)  # 输出层

        return x

class LeNet5_SVHN(nn.Module):
    def __init__(self):
        super(LeNet5_SVHN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # 输入通道 3（RGB），输出通道 6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 输入通道 6，输出通道 16
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入大小 16*5*5，输出 120
        self.fc2 = nn.Linear(120, 84)          # 输入 120，输出 84
        self.fc3 = nn.Linear(84, 10)           # 输入 84，输出 10（10 类）

    def forward(self, x):
        # 卷积层 + 池化层 + 激活函数
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 输出大小: (6, 14, 14)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # 输出大小: (16, 5, 5)
        # 展平
        x = x.view(-1, 16 * 5 * 5)  # 展平为一维向量
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x

# 实例化模型
net = Net()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器，使用SGD，加上momentum和weight decay
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# 为简单起见，假设我们的训练数据是随机生成的
# 真实情况下，您需要加载您的数据集
train_X = np.load('../Assignment1-Dataset/train_data.npy').astype(np.float32)
train_y = np.load('../Assignment1-Dataset/train_label.npy').astype(np.int64)

# 将 numpy 数组转换为 torch 张量
inputs = torch.from_numpy(train_X)
targets = torch.from_numpy(train_y)

# 创建 TensorDataset
dataset = TensorDataset(inputs, targets)

# 创建 DataLoader
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模式
net.train()

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()   # 清零梯度缓存
        outputs = net(inputs)   # 前向传播
        loss = criterion(outputs, labels.view(-1))  # 计算损失
        loss.backward()     # 反向传播
        optimizer.step()    # 更新权重
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/(i+1)}")

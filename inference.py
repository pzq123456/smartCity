from matplotlib import pyplot as plt
import numpy as np
import torch
from utils import MSE,monthName
from dataset import MyDataset
import tqdm

from model import RNN_LSTM  # Gives easier dataset managment by creating mini batches etc.
from sklearn.preprocessing import MinMaxScaler # 数据规范化

# 数据规范化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 5
hidden_size = 256
num_layers = 2
pridict_lenth = 120
sequence_length = 120
learning_rate = 0.001
batch_size = 2
num_epochs = 10

def save_path(dir, name):
    return dir + name + '.pth'

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

# Load Data
test_idx = np.loadtxt('data/CRU/processed/test_idx.csv', delimiter=',', dtype=np.int32)
# 创建数据集
test_dataset = MyDataset(test_idx, 'data/CRU/processed/', transformX=scaler_X.fit_transform, transformY=scaler_y.fit_transform)


# inference
# load_model(model, optimizer, save_path('model/', 'model'))
model_path = save_path('model/', 'model')
# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN_LSTM(input_size, hidden_size, num_layers, pridict_lenth).to(device)
model.load_state_dict(torch.load(model_path)['model_state_dict']) # 加载模型
model.eval() # 设置为评估模式

MSEs = [] # 保存每个样本的 MSE
inference = [] # 保存每个样本的预测值
idx = []

for i in tqdm.tqdm(range(len(test_dataset))):
    data, label = test_dataset[i]
    data = data.unsqueeze(0)
    data = data.to(device=device)
    label = label.to(device=device)
    with torch.no_grad():
        scores = model(data)
        scores = scaler_y.inverse_transform(scores.cpu().numpy())
        label = scaler_y.inverse_transform([label.cpu().numpy()])
        scores = scores.squeeze()
        label = label.squeeze()
        inference.append(scores)
        MSEs.append(MSE(scores, label))
        idx.append(test_idx[i]) # 保存样本的索引
    # break # 只测试一个样本

print("save result")

# 将 MSE, inference, idx 组织成一个 csv 文件
MSEs = np.array(MSEs)
inference = np.array(inference)
idx = np.array(idx)
# 表结构：idx, MSE, 2011-1, 2011-2, ..., 2011-12, 2012-1, 2012-2, ..., 2020-12 其中 2011-1 表示 2011 年 1 月的预测值
monthNames = [monthName(i) for i in range(120)]

# 生成表头
header = ['idx', 'MSE']
header.extend(monthNames)
# 生成表内容
content = []
for i in range(len(idx)):
    row = [idx[i], MSEs[i]]
    row.extend(inference[i])
    content.append(row)
# 保存为 csv 文件
np.savetxt('result.csv', content, delimiter=',', fmt='%s', header=','.join(header))










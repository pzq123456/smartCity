from matplotlib import pyplot as plt
import numpy as np
import torch

from dataset import MyDataset
# from torch.utils.data import (
#     DataLoader,
# )

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
data,label = test_dataset[10]
data = data.unsqueeze(0)
data = data.to(device=device)
label = label.to(device=device)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN_LSTM(input_size, hidden_size, num_layers, pridict_lenth).to(device)

model.load_state_dict(torch.load(model_path)['model_state_dict']) # 加载模型
model.eval() # 设置为评估模式

with torch.no_grad():
    scores = model(data)
    scores = scaler_y.inverse_transform(scores.cpu().numpy())
    label = scaler_y.inverse_transform([label.cpu().numpy()])
    print(scores)
    print(label)
    # 保存为 csv
    np.savetxt('scores.csv', scores, delimiter=',')
    np.savetxt('label.csv', label, delimiter=',')

    plt.plot(scores[0], label='predict')
    plt.plot(label[0], label='real')
    plt.legend()
    plt.show()

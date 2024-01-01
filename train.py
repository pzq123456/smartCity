# Imports
import numpy as np
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm
from dataset import MyDataset # For a nice progress bar!
from model import RNN_LSTM  
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # 数据规范化
# 数据规范化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def save_path(dir, name):
    return dir + name + '.pth'

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def plot_loss(losses):
    plt.plot(losses, label='loss')
    plt.legend()
    plt.show()

def check_accuracy(loader, model,error=0.5):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            scores = scaler_y.inverse_transform(scores.cpu().numpy())
            num_correct += np.sum(np.abs(scores - scaler_y.inverse_transform(y.cpu().numpy())) < error)
            num_samples += x.shape[0]
    
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
    )
    model.train()
    return float(num_correct)/float(num_samples)*100




# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 5
hidden_size = 256
num_layers = 2
pridict_lenth = 120
sequence_length = 120
learning_rate = 0.005
batch_size = 4
num_epochs = 10

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN_LSTM(input_size, hidden_size, num_layers, pridict_lenth).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Load Data
test_idx = np.loadtxt('data/CRU/processed/test_idx.csv', delimiter=',', dtype=np.int32)
train_idx = np.loadtxt('data/CRU/processed/train_idx.csv', delimiter=',', dtype=np.int32)
# 创建数据集
# test_dataset = MyDataset(test_idx, 'data/CRU/processed/', transformX=scaler_X.fit_transform, transformY=scaler_y.fit_transform)
# train_dataset = MyDataset(train_idx, 'data/CRU/processed/', transformX=scaler_X.fit_transform, transformY=scaler_y.fit_transform)
test_dataset = MyDataset(test_idx, 'data/CRU/processed/')
train_dataset = MyDataset(train_idx, 'data/CRU/processed/')
# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

min_loss = 1000000
losses = []
accs = []
# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent update step/adam step
        optimizer.step()
    # check_accuracy(test_loader, model, error=0.1)

# save_model(model, optimizer, save_path('model/', 'model'))
plot_loss(losses)



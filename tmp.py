import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 假设有训练数据 X_train 和标签 y_train
# 这里只是一个示例，实际数据可能需要更多预处理

# 数据规范化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# 假设 X_train 是一个形状为 (样本数, 特征数) 的张量
# y_train 是一个形状为 (样本数, 1) 的张量
X_train = torch.rand(100, 2)  # 100个样本，2个特征
y_train = torch.rand(100, 1)  # 对应的标签

X_train_normalized = torch.FloatTensor(scaler_X.fit_transform(X_train.numpy()))
y_train_normalized = torch.FloatTensor(scaler_y.fit_transform(y_train.numpy()))

# 划分数据集
X_train_norm, X_val_norm, y_train_norm, y_val_norm = train_test_split(X_train_normalized, y_train_normalized, test_size=0.2, random_state=42)

# 定义模型和其他训练过程...

# 训练模型...

# 在测试集上进行预测
# 假设 X_test 是一个形状为 (测试样本数, 特征数) 的张量
X_test = torch.rand(20, 2)

# 数据规范化
X_test_normalized = torch.FloatTensor(scaler_X.transform(X_test.numpy()))

# 模型预测
with torch.no_grad():
    model.eval()
    y_pred_normalized = model(X_test_normalized)

# 逆向规范化预测结果
y_pred = torch.FloatTensor(scaler_y.inverse_transform(y_pred_normalized.numpy()))

# 打印逆向规范化后的预测结果
print("逆向规范化后的预测结果:")
print(y_pred)
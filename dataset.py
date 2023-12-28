# import torch
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def train_test_idx(total, train_ratio=0.8, startByOne=True):
    '''
    传入数据集的总数
    返回一个 list 包含所有数据的索引 乱序
    '''
    if startByOne: # [1,2,3,4,5,6,7,8,9,10]
        idx = np.arange(1, total+1)
    else: # [0,1,2,3,4,5,6,7,8,9]
        idx = np.arange(total)

    np.random.shuffle(idx)
    train_idx = idx[:int(total*train_ratio)]
    test_idx = idx[int(total*train_ratio):]
    return train_idx, test_idx

# define the dataset class
class MyDataset(Dataset):
    '''
    所有数据以 1.csv, 2.csv, 3.csv, ... , 116.csv 的形式存储在 data/ 目录下
    传入的 list 为乱序抽取后的数据集 train 占 80% test 占 20%
    传入的 dir 为数据集目录
    '''
    def __init__(self, list, dir, transform=None):
        self.list = list # [1,2,3,4,5,6,7,8,9,10]
        self.dir = dir # 'data/'
        self.transform = transform
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        # 读取数据
        data = np.loadtxt(self.dir + str(self.list[idx]) + '.csv', delimiter=',', dtype=np.float32)

        # 转换成 tensor
        data = torch.from_numpy(data)
        # 提取 data[3] 作为 label 也就是 降水量
        label = data[3]
        # 删除 data[3] 作为 data
        data = torch.cat((data[:3], data[4:]))

        return data, label


if __name__ == '__main__':
    data = np.loadtxt('smartCity/data/CRU/processed/1.csv', delimiter=',', dtype=np.float32)
    data = torch.from_numpy(data)
    data = torch.cat((data[:3], data[4:]))
    print(data.shape)
# 数据预处理
import netCDF4
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm # 进度条
import pandas as pd


dataType = ['cld','dtr','pet','pre','tmp','vap'] # 数据类型用于生成文件名
inPath = "data/CRU/" # 数据路径
outPath = "data/CRU/processed/" # 输出路径
year = [2011,2020] # 数据年份

def getFileName(year,dataType):
    '''
    year: 数据年份
    dataType: 数据类型
    return: 文件名
    '''
    return 'cru_ts4.06.'+str(year[0])+'.'+str(year[1])+'.'+dataType+'.dat.nc'

# 辅助函数
def getLocation(oriLat,oriLon):
    '''
    oriLat: 原始纬度
    oriLon: 原始经度
    return: 转换后的纬度和经度(0.5*0.5)
    '''
    lat = HelpF1(oriLat,0.75,0.25)
    lon = HelpF1(oriLon,0.75,0.25)
    return lat,lon

def getLatIndex(lat):
    '''
    lat: 转换后的纬度
    return: 纬度索引(nc 文件中的纬度索引)
    '''
    return int((lat+89.75)/0.5)

def getLonIndex(lon):
    '''
    lon: 转换后的经度
    return: 经度索引(nc 文件中的经度索引)
    '''
    return int((lon+179.75)/0.5)

def getLocIndex(lat,lon):
    '''
    lat: 转换后的纬度
    lon: 转换后的经度
    return: 纬度索引和经度索引(nc 文件中的纬度索引和经度索引)
    '''
    lat,lon = getLocation(lat,lon)
    latIndex = getLatIndex(lat)
    lonIndex = getLonIndex(lon)
    return latIndex,lonIndex

def HelpF1(x,top,button):
    '''
    x: 原始数据
    top: 上限
    button: 下限
    return: 转换后的数据
    '''
    R,I =math.modf(x)
    gap1 = abs(R - top)
    gap2 = abs(R - button)
    if gap1 >= gap2 :
        return I+button
    else:
        return I+top

def readData(inPath,year,Type):
    '''
    inPath: 数据路径
    year: 数据年份
    Type: 数据类型
    return: 数据
    '''
    fileName = getFileName(year,Type)
    file = netCDF4.Dataset(inPath+fileName)
    data = file.variables[Type][:]
    file.close()
    return data

def getDataDict(year,dataType):
    '''
    将数据类型作为键，数据作为值，生成字典
    year: 数据年份
    dataType: 数据类型
    return: 数据字典
    '''
    dataDict = {}
    for i in dataType:
        dataDict[i] = readData(inPath,year,i)
    return dataDict

# 接受 dataDict 字典、location 位置 查找 location 位置的数据
def getData(dataDict,location):
    '''
    dataDict: 数据字典
    location: 位置
    return: 数据
    '''
    lat,lon = getLocation(location[1],location[0])
    latIndex,lonIndex = getLocIndex(lat,lon)
    data = {}
    for i in dataDict:
        data[i] = dataDict[i][:,latIndex,lonIndex]
    return data

# 将字典拼接成矩阵
def dict2matrix(dataDict):
    '''
    dataDict: 数据字典
    return: 数据矩阵
    '''
    matrix = []
    for i in dataDict:
        matrix.append(dataDict[i])
    return np.array(matrix)

def getMetaData(path):
    '''
    path: csv 文件路径
    return: csv 文件数据
    '''
    # 读取 csv 文件 并处理为经纬度列表
    # 使用 pandas 读取 csv 文件
    data = pd.read_csv(path)
    return data

def LoadData(path):
    '''
    path: csv 文件路径
    return: 数据矩阵
    '''
    # 读取 csv 文件 并处理为经纬度列表
    # 使用 pandas 读取 csv 文件
    data = pd.read_csv(path)
    return data



if __name__ == "__main__":

    metadata = getMetaData('county.csv')
    dataDict = getDataDict(year,dataType)

    # 主循环
    for i in tqdm.tqdm(metadata.values):
        # print("正在处理：",i[0],i[1])
        location = [i[2],i[3]]
        data = getData(dataDict,location)
        # 设置精度
        np.set_printoptions(precision=4)
        matrix = dict2matrix(data)
        np.savetxt(outPath+str(i[0])+'.csv',matrix,delimiter=',')
    
    print("数据处理完成！"+outPath)




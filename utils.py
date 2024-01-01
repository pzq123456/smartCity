import pandas as pd
def MSE(list1, list2):
    '''
    Mean Squared Error
    '''
    return sum([(x - y) ** 2 for x, y in zip(list1, list2)]) / len(list1)

def monthName(idx,startYear=2011, starOne=False):
    '''
    根据 idx 和 startYear 返回月份的字符串
    '''
    # year = startYear + idx // 12
    # month = idx % 12 + 1
    # return str(year) + '-' + str(month)

    # 若索引从 1 开始，则需要将 idx 减 1
    if starOne:
        idx -= 1
    year = startYear + idx // 12
    month = idx % 12 + 1
    return str(year) + '-' + str(month)

def getLocation(idx,path):
    '''
    读取 path 中的 csv 文件，返回第 idx 个位置的经纬度
    ID,name,lon,lat
    1,济南,117,36.65
    2,历城,117.07,36.69
    3,长清,116.73,36.55
    4,章丘,117.53,36.72
    5,青岛,120.33,36.07
    6,崂山,120.42,36.15
    '''
    # 将 idx 与 ID 对应
    df = pd.read_csv(path)
    # 首先拿到 ID 对应的索引
    idx = df[df['ID']==idx].index[0]
    # 再拿到经纬度
    lon = df.iloc[idx]['lon']
    lat = df.iloc[idx]['lat']
    return lon,lat

def getLocName(idx,path):
    '''
    读取 path 中的 csv 文件，返回第 idx 个位置的名称
    ID,name,lon,lat
    1,济南,117,36.65
    2,历城,117.07,36.69
    3,长清,116.73,36.55
    4,章丘,117.53,36.72
    5,青岛,120.33,36.07
    6,崂山,120.42,36.15
    '''
    # 将 idx 与 ID 对应
    df = pd.read_csv(path)
    # 首先拿到 ID 对应的索引
    idx = df[df['ID']==idx].index[0]
    # 再拿到经纬度
    name = df.iloc[idx]['name']
    return name

def addLocInfo(path,metaPath):
    '''
    给 path 中的 csv 文件添加经纬度信息
    '''
    df = pd.read_csv(path)
    # 获取当前 idx 列表
    idxList = df['idx'].tolist()
    # print(idxList)
    # 获取经纬度信息
    lonList = []
    latList = []
    for idx in idxList:
        lon,lat = getLocation(idx,metaPath)
        lonList.append(lon)
        latList.append(lat)
    # 添加经纬度信息
    df['lon'] = lonList
    df['lat'] = latList
    # 保存
    df.to_csv(path,index=False)

def addLocName(path,metaPath):
    '''
    给 path 中的 csv 文件添加经纬度信息
    '''
    df = pd.read_csv(path)
    # 获取当前 idx 列表
    idxList = df['idx'].tolist()
    # print(idxList)
    # 获取经纬度信息
    nameList = []
    for idx in idxList:
        name = getLocName(idx,metaPath)
        nameList.append(name)
    # 添加经纬度信息
    df['name'] = nameList
    # 保存
    df.to_csv(path,index=False)


path = 'result.csv'
metaPath = 'county.csv'
addLocName(path,metaPath)

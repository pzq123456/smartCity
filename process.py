# 数据预处理
import netCDF4
import numpy as np
import math
import matplotlib.pyplot as plt

dataType = ['cld','dtr','pet','pre','tmp','vap'] # 数据类型用于生成文件名
inPath = "/data/CRU/" # 数据路径
outPath = "/data/CRU/processed/" # 输出路径
year = [2011,2020] # 数据年份

def getFileName(year,dataType):
    '''
    year: 数据年份
    dataType: 数据类型
    return: 文件名
    '''
    return 'cru_ts4.06.'+str(year[0])+'.'+str(year[1])+'.'+dataType+'.dat.nc'

def getPath(inPath,year,dataType):
    '''
    inPath: 数据路径
    year: 数据年份
    dataType: 数据类型
    return: 完整路径
    '''
    return inPath+getFileName(year,dataType)

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

def readData(inPath,year,dataType):
    '''
    inPath: 数据路径
    year: 数据年份
    dataType: 数据类型
    return: 数据
    '''
    f = netCDF4.Dataset(getPath(inPath,year,dataType))
    data = f.variables[dataType]
    # 以键值对的形式存储数据 键为 dataType
    return data






# # 读取数据
# def readData():
#     # windows
#     # f1 = netCDF4.Dataset('data/cru_ts4.06.2001.2010.pre.dat.nc')
#     # f2 = netCDF4.Dataset('data/cru_ts4.06.2001.2010.tmp.dat.nc')
#     # f3 = netCDF4.Dataset('data/cru_ts4.06.2001.2010.tmx.dat.nc')
#     # f4 = netCDF4.Dataset('data/cru_ts4.06.2011.2020.tmn.dat.nc')
#     # linux
#     f1 = netCDF4.Dataset('DTZT/data/cru_ts4.06.2001.2010.pre.dat.nc')
#     f2 = netCDF4.Dataset('DTZT/data/cru_ts4.06.2001.2010.tmp.dat.nc')
#     f3 = netCDF4.Dataset('DTZT/data/cru_ts4.06.2001.2010.tmx.dat.nc')
#     f4 = netCDF4.Dataset('DTZT/data/cru_ts4.06.2011.2020.tmn.dat.nc')

#     pre = f1.variables['pre'] # 逐月降水量 10年
#     tmp = f2.variables['tmp'] # 月均温 
#     tmx = f3.variables['tmx'] # 月最高温
#     tmn = f4.variables['tmn'] # 月最低温
#     lat = f1.variables['lat'] # 纬度
#     lon = f1.variables['lon'] # 经度
#     return pre,tmp,tmx,tmn,lat,lon


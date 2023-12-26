# 智慧城市实验

## 数据
1. 气象数据收集
   1. cld : 
   2. dtr : diurnal temperature range 是日较差，即最高温和最低温度的差值。
   3. frs : 
   4. pet : 潜在蒸发量又称潜在蒸发散量（Potential Evapotranspiration，即PET），是指充分供水下垫面（即充分湿润表面或开阔水体）蒸发/蒸腾到空气中的水量，又称可能蒸发散量或蒸发能力。
   5. pre : precipitation 是降水量，即降水的总量。
   6. tmn : mean of minimum temperature 是最低温度的平均值。
   7. tmp : mean of temperature 是温度的平均值。
   8. tmx : mean of maximum temperature 是最高温度的平均值。
   9.  vap : vapor pressure 是水汽压。
   10. wet : wet day frequency 是湿日频率，即年度湿日数。
2. 地形数据收集
使用 GEE 收集地形数据，包括 DEM、坡度、坡向等。首先获取山东省的行政区划，然后上传至 GEE 平台。山东省的行政区划使用高德地图提供的 JSON 数据转化为 Shape 文件，然后上传至 GEE 平台。
![](imgs/山东行政区划.jpg)

## Reference
1. [全国省市县矢量边界提取kml,shp,svg格式下载](https://dx3377.com/map/bound)
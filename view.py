import numpy as np
import matplotlib.pyplot as plt
# path smartCity/data/CRU/processed/1.csv
if __name__ == '__main__':
    # 1.读取csv文件
    data = np.loadtxt('smartCity/data/CRU/processed/1.csv', delimiter=',') # 不能跳过第一行
    # print(data.shape)
    # # 6*120 6个特征，120个时间步 cld,dtr,pet,pre,tmp,vap 云量，日辐射，潜在蒸散发，降水，温度，水汽压
    # # add title "济南市 2011年1月-2020年12月 气象数据"
    # # add x label "时间"
    # # add y label "气象数据"
    # # 2.绘制图像
    # plt.xlabel("time")
    # plt.ylabel("features")
    features = ['cld','dtr','pet','pre','tmp','vap']
    # for i in range(6):
    #     plt.plot(data[i], label=features[i])
    # plt.legend()
    # plt.show()

    # 120 个月数据求均值
    # 3.求均值
    avg = np.mean(data, axis=1)
    # print(avg.shape)
    # 计算各特征均值之间的相关性在用 heatmap 可视化
    heatmap = np.corrcoef(data)

    # print(heatmap.shape)
    # 4.绘制图像
    plt.xlabel("features")
    plt.ylabel("features")
    plt.imshow(heatmap, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(6), features)
    plt.yticks(np.arange(6), features)
    plt.show()


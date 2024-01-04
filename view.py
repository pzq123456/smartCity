from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# path smartCity/data/CRU/processed/1.csv

def heatMap(dir,idx,savedir):
    # use idx and dir to get the data path
    path = dir + str(idx) + '.csv'
    data = np.loadtxt(path, delimiter=',')  # 不能跳过第一行
    heatmap = np.corrcoef(data)
    features = ['cld', 'dtr', 'pet', 'pre', 'tmp', 'vap']

    plt.xlabel("features")
    plt.ylabel("features")
    plt.imshow(heatmap, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(6), features)
    plt.yticks(np.arange(6), features)
    if savedir:
        plt.savefig(savedir + str(idx) + '.png')
    # plt.show()



def render_custom_color_strip(data, custom_colors):
    # 创建一个图形对象
    fig, ax = plt.subplots(figsize=(10, 1))

    # 创建一个自定义色带
    cmap = ListedColormap(custom_colors)

    # 根据数据设置归一化范围
    norm = plt.Normalize(min(data), max(data))

    # 创建色带对象
    color_strip = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # 渲染色带
    color_strip.set_array([])  # 设置空数组，以便正确渲染色带
    plt.colorbar(color_strip, orientation='horizontal', ax=ax)

    # 标注最大值、中值和最小值
    max_value = max(data)
    min_value = min(data)
    median_value = np.median(data)

    ax.text(min_value, 1.5, f'Min: {min_value:.2f}', color='black', ha='center', va='center', fontweight='bold')
    ax.text(median_value, 1.5, f'Median: {median_value:.2f}', color='black', ha='center', va='center', fontweight='bold')
    ax.text(max_value, 1.5, f'Max: {max_value:.2f}', color='black', ha='center', va='center', fontweight='bold')

    # 隐藏坐标轴
    ax.set_yticks([])
    ax.set_xticks([])

    plt.show()


def generate_color_strip(start_color, end_color, num_colors):
    # 解析初始颜色和末尾颜色
    start_rgb = [float(val) / 255 for val in start_color.split(',')]
    end_rgb = [float(val) / 255 for val in end_color.split(',')]

    # 创建线性分段的色带
    cdict = {'red':   [(0, start_rgb[0], start_rgb[0]),
                       (1, end_rgb[0], end_rgb[0])],

             'green': [(0, start_rgb[1], start_rgb[1]),
                       (1, end_rgb[1], end_rgb[1])],

             'blue':  [(0, start_rgb[2], start_rgb[2]),
                       (1, end_rgb[2], end_rgb[2])]
            }

    cmap = LinearSegmentedColormap('custom_cmap', cdict, N=num_colors)
    
    # 生成色带
    colors = [cmap(i) for i in range(num_colors)]
    return colors



def plot_r2(y_true, y_pred,savedir=None):
    """
    绘制R方图

    参数:
    - y_true: 实际观测值
    - y_pred: 模型预测值
    """
    # 计算R方
    r2 = r2_score(y_true, y_pred)

    # 绘制散点图
    plt.scatter(y_true, y_pred, color='blue', label='Actual vs. Predicted')

    # 绘制对角线
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', label='Perfect Fit')

    # 设置图表标题和轴标签
    plt.title(f'R-squared: {r2:.4f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    # 显示图例
    plt.legend()
    if savedir:
        plt.savefig(savedir)
    # 显示图表
    # plt.show()

if __name__ == '__main__':

    # 测试 r2_score
    # 读取 result1.csv 及 labels1.csv 其中存储的是 6 个样本的预测值和真实值
    result = np.loadtxt('result1.csv', delimiter=',',)
    labels = np.loadtxt('labels1.csv', delimiter=',',)
    for i in range(6):
        savedir = 'data/R2/'
        savedir+='r2_'+str(i)+'.png'
        plot_r2(labels[i], result[i],savedir)



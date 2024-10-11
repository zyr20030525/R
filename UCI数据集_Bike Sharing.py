from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
import numpy as np

#指定中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 提取数据集
bike_sharing = fetch_ucirepo(id=275)

# 数据（转换为 pandas 的 DataFrame）
data = bike_sharing.data.features
targets = bike_sharing.data.targets

# 排除日期列
data_no_date = data.drop(columns=['dteday'])

# 散点图
sns.scatterplot(data=data_no_date, x='temp', y='atemp')
plt.title('温度与调整后温度散点图')
plt.show()

# 气泡图
sns.scatterplot(data=data_no_date, x='temp', y='hum', size='windspeed')
plt.title('温度、湿度与风速气泡图')
plt.show()

# 竖向柱状图（以季节为例）
sns.countplot(data=data_no_date, x='season')
plt.title('季节竖向柱状图')
plt.show()

# 横向柱状图（以天气状况为例）
sns.countplot(data=data_no_date, y='weathersit')
plt.title('天气状况横向柱状图')
plt.show()

# 极坐标柱状图（以小时为例）
hours = data_no_date['hr'].value_counts().sort_index()
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, len(hours))
width = (2 * np.pi) / len(hours)
bars = ax.bar(theta, hours, width=width, bottom=0)
ax.set_xticks(theta)
ax.set_xticklabels(range(0, 24))
plt.title('小时极坐标柱状图')
plt.show()

# 有数据信息的饼状图（以工作日为例）
weekday_counts = data_no_date['weekday'].value_counts()
plt.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%')
plt.title('工作日饼状图')
plt.show()

# 收缩显示的饼状图（突出显示某一个工作日，比如 0）
explode = [0.1 if i == 0 else 0 for i in range(len(weekday_counts))]
plt.pie(weekday_counts, labels=weekday_counts.index, autopct='%1.1f%%', explode=explode)
plt.title('突出显示的工作日饼状图')
plt.show()

# 立体线形图（以温度和调整后温度的变化趋势为例）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data_no_date['temp'], data_no_date['atemp'], np.arange(len(data_no_date)))
plt.title('温度与调整后温度立体线形图')
plt.show()

# 立体散点图（以温度和调整后温度为例）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_no_date['temp'], data_no_date['atemp'], np.random.rand(len(data_no_date)))
plt.title('温度与调整后温度立体散点图')
plt.show()

# 立体柱状图（以季节和温度为例）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for season in data_no_date['season'].unique():
    season_data = data_no_date[data_no_date['season'] == season]['temp']
    hist, bins = np.histogram(season_data, bins=10)
    xs = np.linspace(min(bins), max(bins), len(hist))
    ax.bar(xs, hist, zs=season, zdir='y', alpha=0.8)
plt.title('季节与温度立体柱状图')
plt.show()

# 热力图（计算相关性矩阵并绘制）
corr_matrix = data_no_date.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title('相关性热力图')
plt.show()

# 箱线图（以季节和温度为例）
sns.boxplot(data=data_no_date, x='season', y='temp')
plt.title('季节与温度箱线图')
plt.show()

# 小提琴图（以季节和温度为例）
sns.violinplot(data=data_no_date, x='season', y='temp')
plt.title('季节与温度小提琴图')
plt.show()

# 不同特征组合图（例如温度、湿度和风速的关系）
sns.pairplot(data_no_date[['temp', 'hum', 'windspeed']])
plt.suptitle('温度、湿度与风速成对图')
plt.show()

# 线性回归模型图（假设以温度和调整后温度进行线性回归）
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X_temp = data_no_date[['temp']]
y_atemp = data_no_date['atemp']
model.fit(X_temp, y_atemp)
y_pred = model.predict(X_temp)
plt.scatter(X_temp, y_atemp, color='blue', label='实际值')
plt.plot(X_temp, y_pred, color='red', label='预测值')
plt.title('温度与调整后温度线性回归模型图')
plt.xlabel('温度')
plt.ylabel('调整后温度')
plt.legend()
plt.show()

# 聚类模型图（以 KMeans 聚类为例，假设对温度和调整后温度进行聚类）
kmeans = KMeans(n_clusters=3)
X_cluster = data_no_date[['temp', 'atemp']]
kmeans.fit(X_cluster)
labels = kmeans.labels_
plt.scatter(X_cluster['temp'], X_cluster['atemp'], c=labels)
plt.title('温度与调整后温度聚类模型图')
plt.xlabel('温度')
plt.ylabel('调整后温度')
plt.show()

# 决策树模型图（假设以温度、湿度和风速预测某个目标）
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
dtree = DecisionTreeRegressor(max_depth=3)
X_tree = data_no_date[['temp', 'hum', 'windspeed']]
y_tree = targets['cnt']
dtree.fit(X_tree, y_tree)
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=['温度', '湿度', '风速'], filled=True)
plt.title('决策树模型图')
plt.show()

# 多棵决策树模型图（随机森林）
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_tree, y_tree)
# 这里可以根据需要进行可视化，例如展示特征重要性
feature_importances = rf.feature_importances_
plt.bar(X_tree.columns, feature_importances)
plt.title('随机森林特征重要性')
plt.show()

# 相关性网络图
corr_matrix = data_no_date.corr()
G = nx.Graph()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.5:  # 设置相关性阈值
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('相关性网络图')
plt.show()

# 圆盘状显示的网络图
pos = nx.circular_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('圆盘状网络图')
plt.show()

# 含属性及边的颜色及粗度表达的网络图
colors = ['r' if abs(labels[e]) > 0.7 else 'b' for e in G.edges()]
widths = [abs(labels[e]) * 5 for e in G.edges()]
nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=10, edge_color=colors, width=widths)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('含属性及边的颜色及粗度表达的网络图')
plt.show()

# 环形布局的网络图
pos = nx.kamada_kawai_layout(G)  # 环形布局算法之一
nx.draw_networkx(G, pos, with_labels=True, node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('环形布局网络图')
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import networkx as nx
import os

#指定中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
day_data = pd.read_csv('day.csv')
hour_data = pd.read_csv('hour.csv')

# 只选择数值型列进行相关性计算
numeric_columns = day_data.select_dtypes(include=[np.number]).columns
corr_matrix = day_data[numeric_columns].corr()

# 基本统计信息
print("Day数据基本统计信息：")
print(day_data.describe())
print("Hour数据基本统计信息：")
print(hour_data.describe())

# 检查并创建保存图片的目录（如果不存在）
save_directory = 'saved_images'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 散点图
plt.scatter(day_data['temp'], day_data['cnt'])
plt.xlabel('温度')
plt.ylabel('租赁数量')
plt.title('Day数据 - 温度与租赁数量的关系')
plt.savefig(os.path.join(save_directory, 'scatter_temp_cnt.png'))
plt.close()

# 泡状图（以温度和风速对租赁数量的影响为例，使用大小表示租赁数量）
plt.scatter(day_data['temp'], day_data['windspeed'], s=day_data['cnt'] * 0.1)
plt.xlabel('温度')
plt.ylabel('风速')
plt.title('Day数据 - 温度、风速与租赁数量的关系（泡状图）')
plt.savefig(os.path.join(save_directory, 'bubble_temp_windspeed_cnt.png'))
plt.close()

# 竖向柱状图（以季节为例）
season_counts = day_data['season'].value_counts()
plt.bar(season_counts.index, season_counts.values)
plt.xlabel('季节')
plt.ylabel('数量')
plt.title('Day数据 - 季节分布（竖向柱状图）')
plt.savefig(os.path.join(save_directory, 'vertical_bar_season.png'))
plt.close()

# 横向柱状图（以月份为例）
month_counts = day_data['mnth'].value_counts()
plt.barh(month_counts.index, month_counts.values)
plt.xlabel('数量')
plt.ylabel('月份')
plt.title('Day数据 - 月份分布（横向柱状图）')
plt.savefig(os.path.join(save_directory, 'horizontal_bar_month.png'))
plt.close()

# 极坐标柱状图（以工作日和非工作日为例）
workingday_counts = day_data['workingday'].value_counts()
theta = [0, np.pi]
radii = workingday_counts.values
width = np.pi / 2  # 调整宽度以适应两个刻度位置
ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=0.0)
ax.set_xticks([0, np.pi])  # 设置两个刻度位置
ax.set_xticklabels(['非工作日', '工作日'])
plt.title('Day数据 - 工作日分布（极坐标柱状图）')
plt.savefig(os.path.join(save_directory, 'polar_bar_workingday.png'))
plt.close()

# 有数据信息的饼状图（以天气情况为例）
weather_counts = day_data['weathersit'].value_counts()
plt.pie(weather_counts.values, labels=weather_counts.index, autopct='%1.1f%%')
plt.title('Day数据 - 天气情况分布（饼状图）')
plt.savefig(os.path.join(save_directory, 'pie_weather.png'))
plt.close()

# 收缩显示的饼状图（以年份为例）
year_counts = day_data['yr'].value_counts()
plt.pie(year_counts.values, labels=year_counts.index, autopct='%1.1f%%', explode=(0.1, 0))
plt.title('Day数据 - 年份分布（收缩显示的饼状图）')
plt.savefig(os.path.join(save_directory, 'shrunk_pie_year.png'))
plt.close()

# 立体线形图（以温度和风速随时间的变化为例，假设'instant'为时间顺序）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(day_data['instant'], day_data['temp'], day_data['windspeed'])
ax.set_xlabel('时间')
ax.set_ylabel('温度')
ax.set_zlabel('风速')
plt.title('Day数据 - 温度和风速随时间的变化（立体线形图）')
plt.savefig(os.path.join(save_directory, '3d_line_temp_windspeed_time.png'))
plt.close()

# 立体散点图（以温度、风速和租赁数量为例）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(day_data['temp'], day_data['windspeed'], day_data['cnt'])
ax.set_xlabel('温度')
ax.set_ylabel('风速')
ax.set_zlabel('租赁数量')
plt.title('Day数据 - 温度、风速和租赁数量的关系（立体散点图）')
plt.savefig(os.path.join(save_directory, '3d_scatter_temp_windspeed_cnt.png'))
plt.close()

# 立体柱状图（以季节和年份为例）
year_season_matrix = day_data.groupby(['yr', 'season'])['cnt'].sum().unstack()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, year in enumerate(year_season_matrix.columns):
    for j, season in enumerate(year_season_matrix.index):
        ax.bar3d(i, j, 0, 1, 1, year_season_matrix[year][season])
# 设置固定数量的刻度
ax.set_xticks(range(len(year_season_matrix.columns)))
ax.set_yticks(range(len(year_season_matrix.index)))
# 设置刻度标签
ax.set_xticklabels(year_season_matrix.columns)
ax.set_yticklabels(year_season_matrix.index)
ax.set_xlabel('年份')
ax.set_ylabel('季节')
ax.set_zlabel('租赁数量')
plt.title('Day数据 - 季节和年份的租赁数量分布（立体柱状图）')
plt.savefig(os.path.join(save_directory, '3d_bar_year_season_cnt.png'))
plt.close()

# 热力图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Day数据 - 特征相关性热力图')
plt.savefig(os.path.join(save_directory, 'heatmap_corr.png'))
plt.close()

# 箱线图
sns.boxplot(data=day_data[['temp', 'windspeed', 'cnt']])
plt.title('Day数据 - 温度、风速和租赁数量箱线图')
plt.savefig(os.path.join(save_directory, 'boxplot_temp_windspeed_cnt.png'))
plt.close()

# 小提琴图
sns.violinplot(data=day_data[['temp', 'windspeed', 'cnt']])
plt.title('Day数据 - 温度、风速和租赁数量小提琴图')
plt.savefig(os.path.join(save_directory, 'violinplot_temp_windspeed_cnt.png'))
plt.close()

# 不同特征组合图（以温度和季节为例）
sns.scatterplot(data=day_data, x='temp', y='cnt', hue='season')
plt.title('Day数据 - 温度和租赁数量按季节区分')
plt.savefig(os.path.join(save_directory, 'scatter_temp_cnt_season.png'))
plt.close()

# 相嵌散点分布的逻辑回归模型图
X = day_data[['temp', 'windspeed']]
X.columns = ['temp_feature', 'windspeed_feature']  # 给 X 指定特征名称
y = day_data['cnt']
logreg = LogisticRegression()
logreg.fit(X, y)
temp_range = np.linspace(day_data['temp'].min(), day_data['temp'].max(), 100)
windspeed_range = np.linspace(day_data['windspeed'].min(), day_data['windspeed'].max(), 100)
temp_grid, windspeed_grid = np.meshgrid(temp_range, windspeed_range)
X_grid = np.c_[temp_grid.ravel(), windspeed_grid.ravel()]
y_grid = logreg.predict(X_grid)
y_grid = y_grid.reshape(temp_grid.shape)
plt.contourf(temp_grid, windspeed_grid, y_grid, alpha=0.3)
plt.scatter(day_data['temp'], day_data['windspeed'], c=day_data['cnt'], cmap='viridis')
plt.xlabel('温度')
plt.ylabel('风速')
plt.title('Day数据 - 逻辑回归模型预测与散点分布')
plt.savefig(os.path.join(save_directory, 'logistic_regression.png'))
plt.close()

# 线性回归模型图
linreg = LinearRegression()
linreg.fit(X, y)
y_pred = linreg.predict(X)
plt.scatter(day_data['temp'], day_data['cnt'])
plt.plot(day_data['temp'], y_pred, color='red')
plt.xlabel('温度')
plt.ylabel('租赁数量')
plt.title('Day数据 - 线性回归模型预测')
plt.savefig(os.path.join(save_directory, 'linear_regression.png'))
plt.close()

# 聚类模型图
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
plt.scatter(day_data['temp'], day_data['windspeed'], c=labels, cmap='viridis')
plt.xlabel('温度')
plt.ylabel('风速')
plt.title('Day数据 - KMeans 聚类结果')
plt.savefig(os.path.join(save_directory, 'kmeans_clustering.png'))
plt.close()

# 决策树模型图
dtree = DecisionTreeRegressor()
dtree.fit(X, y)
y_pred_tree = dtree.predict(X)
plt.scatter(day_data['temp'], day_data['cnt'])
plt.plot(day_data['temp'], y_pred_tree, color='green')
plt.xlabel('温度')
plt.ylabel('租赁数量')
plt.title('Day数据 - 决策树回归模型预测')
plt.savefig(os.path.join(save_directory, 'decision_tree_regression.png'))
plt.close()

# 多棵决策树模型图（随机森林）
rf = RandomForestRegressor()
rf.fit(X, y)
y_pred_rf = rf.predict(X)
plt.scatter(day_data['temp'], day_data['cnt'])
plt.plot(day_data['temp'], y_pred_rf, color='purple')
plt.xlabel('温度')
plt.ylabel('租赁数量')
plt.title('Day数据 - 随机森林回归模型预测')
plt.savefig(os.path.join(save_directory, 'random_forest_regression.png'))
plt.close()

# 相关性网络图
G = nx.Graph()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Day数据 - 相关性网络图')
plt.savefig(os.path.join(save_directory, 'correlation_network_graph.png'))
plt.close()

# 圆盘状显示的网络图
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Day数据 - 圆盘状相关性网络图')
plt.savefig(os.path.join(save_directory, 'circular_correlation_network_graph.png'))
plt.close()

# 含属性及边的颜色及粗度表达的网络图
edge_colors = []
edge_widths = []
for u, v, data in G.edges(data=True):
    edge_colors.append('red' if data['weight'] > 0 else 'blue')
    edge_widths.append(abs(data['weight']) * 2)
nx.draw(G, pos, with_labels=True, node_color='orange', node_size=500, font_size=10, edge_color=edge_colors, width=edge_widths)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Day数据 - 属性及边的颜色及粗度表达的网络图')
plt.savefig(os.path.join(save_directory, 'colored_weighted_network_graph.png'))
plt.close()

# 环形布局的网络图
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_color='pink', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Day数据 - 环形布局的网络图')
plt.savefig(os.path.join(save_directory, 'circular_network_graph.png'))
plt.close()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 创建数据
data = {
    "Model": ["1D-CNN", "RF", "LGBM", "XGB"],
    "R2": [0.9779, 0.9436, 0.9721, 0.987],
    "RMSE": [0.1072, 0.1546, 0.1087, 0.0742],
    "MAE": [0.0769, 0.1138, 0.0799, 0.0599],
}

df = pd.DataFrame(data)

# 计算均值和标准差
mean_RMSE = np.mean(df["RMSE"])
mean_MAE = np.mean(df["MAE"])
std_RMSE = np.std(df["RMSE"])
std_MAE = np.std(df["MAE"])

# 计算半径和角度
radius = np.sqrt((df["RMSE"] - mean_RMSE) ** 2 + (df["MAE"] - mean_MAE) ** 2)
theta = np.arctan2(df["MAE"] - mean_MAE, df["RMSE"] - mean_RMSE)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制泰勒图
for i in range(len(radius)):
    ax.scatter(radius[i] * np.cos(theta[i]), radius[i] * np.sin(theta[i]), s=200, edgecolors="k", linewidths=1.5, alpha=0.8)
    ax.text(radius[i] * np.cos(theta[i]) + 0.002, radius[i] * np.sin(theta[i]) - 0.002, df["Model"][i], fontsize=12, fontweight='bold')

# 添加标准差圆圈
for std in range(1, 4):
    circle = plt.Circle((0, 0), std * np.sqrt((std_RMSE ** 2 + std_MAE ** 2) / 2), color='k', fill=False, linestyle='--', linewidth=1)
    ax.add_artist(circle)

# 添加标签和标题
ax.set_xlabel("RMSE", fontsize=14, fontweight='bold')
ax.set_ylabel("MAE", fontsize=14, fontweight='bold')
ax.set_title("美化后的泰勒图", fontsize=18, fontweight='bold')

# 设置轴范围和刻度
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_xticks(np.arange(-0.5, 1.6, 0.5))
ax.set_yticks(np.arange(-0.5, 1.6, 0.5))

# 显示图像
plt.show()


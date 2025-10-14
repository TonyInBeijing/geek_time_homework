'''
Author: TonyInBeijing
Date: 2025-10-14 21:47:26
LastEditors: TonyInBeijing
LastEditTime: 2025-10-15 00:02:31
FilePath: /geektime/homework/week1/R4_user_vectorization_and_visualization.py
Description: 用户画像向量化和可视化

'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sentence_transformers import SentenceTransformer
import random

FILE_PATH = "user_profiles.csv"
MODEL_NAME = "BAAI/bge-m3"
CAT_COLS = ["性别", "所在城市", "消费水平"]
NUM_COLS = ["年龄", "最近活跃天数"]
COLOR_COL = "消费水平"

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS',
                                   'Heiti TC', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv(FILE_PATH)

print("加载 BGE-m3 嵌入模型...")
model = SentenceTransformer(MODEL_NAME)
print("对类别特征编码...")
cat_vectors = [
    model.encode(data[col].astype(str).tolist())
    for col in CAT_COLS
]

num_vectors = data[NUM_COLS].values
user_matrix = np.hstack(cat_vectors + [num_vectors])
print(f"特征拼接后shape: {user_matrix.shape}")

scaler = StandardScaler()
user_matrix_std = scaler.fit_transform(user_matrix)

# --- 3. PCA三维降维 ---
pca = PCA(n_components=3)
user_3d = pca.fit_transform(user_matrix_std)
exp_var_3d = pca.explained_variance_ratio_
print(f"\n--- PCA降至3维 ---")
for i, var in enumerate(exp_var_3d, 1):
    print(f"主成分{i}解释的方差：{var:.2%}")
print(f"累计解释的方差：{np.sum(exp_var_3d):.2%}")

# --- 4. 3D静态可视化（Matplotlib） ---
def plot_3d_matplotlib(X_3d, df, exp_var, color_col, n_label=20):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    color_map = {
        k: c for k, c in zip(sorted(df[color_col].unique()), plt.cm.tab10.colors)
    }
    colors = df[color_col].map(color_map)
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=colors, alpha=0.5, s=40)
    # 随机点标签
    idxs = random.sample(range(len(df)), min(n_label, len(df)))
    for i in idxs:
        label = "-".join(str(df.loc[i, col]) for col in CAT_COLS + NUM_COLS)
        ax.text(X_3d[i, 0], X_3d[i, 1], X_3d[i, 2], label, fontsize=7, alpha=0.7)
    # 图例
    for level, color in color_map.items():
        ax.scatter([], [], [], color=color, label=level)
    ax.set_title("用户画像PCA三维可视化", fontsize=15)
    ax.set_xlabel(f"主成分1 ({exp_var[0]:.1%})")
    ax.set_ylabel(f"主成分2 ({exp_var[1]:.1%})")
    ax.set_zlabel(f"主成分3 ({exp_var[2]:.1%})")
    ax.legend(title=color_col)
    plt.tight_layout()
    plt.show()

plot_3d_matplotlib(user_3d, data, exp_var_3d, COLOR_COL, n_label=20)

# 可视化
df_plot = data.copy()
df_plot["PC1"] = user_3d[:, 0]
df_plot["PC2"] = user_3d[:, 1]
df_plot["PC3"] = user_3d[:, 2]
fig3d = px.scatter_3d(
    df_plot, x="PC1", y="PC2", z="PC3", color=COLOR_COL,
    hover_data={col: True for col in CAT_COLS + NUM_COLS},
    title="用户画像PCA三维交互可视化"
)
fig3d.update_traces(marker=dict(size=6, opacity=0.7))
fig3d.show()


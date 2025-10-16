'''
Author: TonyInBeijing
Date: 2025-10-15 19:05:01
LastEditors: TonyInBeijing
LastEditTime: 2025-10-16 21:49:44
FilePath: /geektime/homework/week2/R2_kmeans_user_clustering.py
Description: 作业二：基于 K-Means 的用户分群与画像洞察

'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist

FILE_PATH = "../week1/user_profiles.csv"
MODEL_NAME = "BAAI/bge-m3"
CAT_COLS = ["性别", "所在城市", "消费水平"]
NUM_COLS = ["年龄", "最近活跃天数"]
N_CLUSTERS_RANGE = range(2, 11)

data = pd.read_csv(FILE_PATH)
model = SentenceTransformer(MODEL_NAME)

cat_vectors = []
for col in CAT_COLS:
    vectors = model.encode(data[col].astype(str).tolist())
    cat_vectors.append(vectors)

num_vectors = data[NUM_COLS].values
user_matrix = np.hstack(cat_vectors + [num_vectors])
print(f"特征拼接后形状: {user_matrix.shape}")

scaler = StandardScaler()
user_matrix_standard = scaler.fit_transform(user_matrix)

k = 3

final_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(user_matrix_standard)
centroids = final_kmeans.cluster_centers_


# 找出每个簇中离质心最近的真实用户
representative_indices = []   # 每个簇的代表用户索引
for cid in range(k):
    # 取该簇所有样本的索引
    idxs = np.where(cluster_labels == cid)[0]
    if len(idxs) == 0:
        continue
    # 取出这些样本在标准化空间的向量
    cluster_vectors = user_matrix_standard[idxs]
    # 计算这些样本到簇质心 cid 的距离（欧氏）
    dists = cdist(cluster_vectors, centroids[cid].reshape(1, -1), metric='euclidean').reshape(-1)
    # 找到最小距离对应的样本（真实索引）
    min_local_idx = np.argmin(dists)
    rep_idx = idxs[min_local_idx]
    representative_indices.append(rep_idx)
    # 打印代表用户信息（完整行）
    print(f"\n=== 簇 {cid} 的代表用户（样本索引 {rep_idx}） ===")
    print(data.iloc[rep_idx].to_dict())





data_clustered = data.copy()
data_clustered["聚类标签"] = cluster_labels
print(data_clustered["聚类标签"].value_counts())

cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

for cluster_id, count in cluster_counts.items():
    print(f"  聚类 {cluster_id}: {count} 用户")

print("\n进行PCA降维...")
pca = PCA(n_components=3)
user_3d = pca.fit_transform(user_matrix_standard)
centroids_3d = pca.transform(centroids)

exp_var = pca.explained_variance_ratio_
print(f"PCA前3个主成分解释的方差比例: {exp_var}")
print(f"累计解释方差: {np.sum(exp_var):.2%}")

# --- 7. 交互式3D可视化 ---
print("\n生成交互式3D聚类可视化...")
df_plot = data_clustered.copy()
df_plot['PC1'] = user_3d[:, 0]
df_plot['PC2'] = user_3d[:, 1]
df_plot['PC3'] = user_3d[:, 2]

# 用户点
fig3d = px.scatter_3d(
    df_plot, x='PC1', y='PC2', z='PC3',
    color='聚类标签',
    hover_data=CAT_COLS + NUM_COLS,
    title='K-means聚类结果交互式可视化',
    color_discrete_sequence=px.colors.qualitative.Set1
)

# 添加聚类中心点
centroids_df = pd.DataFrame({
    'PC1': centroids_3d[:, 0],
    'PC2': centroids_3d[:, 1],
    'PC3': centroids_3d[:, 2],
    '聚类中心': [f'中心{i}' for i in range(k)]
})

fig3d.add_trace(go.Scatter3d(
    x=centroids_df['PC1'],
    y=centroids_df['PC2'],
    z=centroids_df['PC3'],
    mode='markers+text',
    marker=dict(size=15, color='red', symbol='x'),
    text=centroids_df['聚类中心'],
    textposition='top center',
    name='聚类中心',
    showlegend=True
))

fig3d.show()

print("\n🎯 交互式3D可视化已生成！")
print("   • 可以旋转、缩放、点击查看详细信息")
print("   • 红色X标记为各聚类中心点")
print("   • 不同颜色代表不同聚类")

# --- 8. 保存结果 ---
# 保存聚类结果
data_clustered.to_csv('user_clustering_results.csv', index=False)
print(f"\n📁 聚类结果已保存到 user_clustering_results.csv")

# 计算最终聚类的轮廓系数
final_silhouette_score = silhouette_score(user_matrix_standard, cluster_labels)

print(f"\n=== ✅ K-means聚类分析完成 ===")
print(f"📊 聚类数量: {k}")
print(f"🎯 轮廓系数: {final_silhouette_score:.3f}")
print(f"📈 PCA累计解释方差: {np.sum(exp_var):.2%}")

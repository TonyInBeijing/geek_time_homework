'''
Author: TonyInBeijing
Date: 2025-10-14 00:14:36
LastEditors: TonyInBeijing
LastEditTime: 2025-10-14 00:24:15
FilePath: /geektime/homework/week1/R2_split_and_analyze_consumption.py
Description: 使用简单流出法抽取数据

'''
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
FILE_PATH = "user_profiles.csv"
data = pd.read_csv(FILE_PATH)

# 消费水平分布（占比）
golden_dist = data["消费水平"].value_counts(normalize=True).sort_index()
print("消费水平分布（占比）:")
for level, ratio in golden_dist.items():
    print(f"消费水平 {level}: {ratio:.2%}")

# 随机划分训练集和测试集
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, shuffle=True)

# 统计分布
train_dist = train_data["消费水平"].value_counts(normalize=True).sort_index()
for level, ratio in train_dist.items():
    print(f"训练集 - 消费水平 {level}: {ratio:.2%}")
test_dist = test_data["消费水平"].value_counts(normalize=True).sort_index()
for level, ratio in test_dist.items():
    print(f"测试集 - 消费水平 {level}: {ratio:.2%}")

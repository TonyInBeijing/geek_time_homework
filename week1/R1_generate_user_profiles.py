'''
Author: TonyInBeijing
Date: 2025-10-13 22:02:48
LastEditors: TonyInBeijing
LastEditTime: 2025-10-14 00:16:23
FilePath: /geektime/homework/week1/R1_generate_user_profiles.py
Description: 随机生成用户数据

'''
import pandas as pd
import numpy as np

USER_NUM = 1000
SEED = 2025

np.random.seed(SEED)  # 设置随机种子

sex_choices = ["男", "女"]
sex_probs = [0.48, 0.52]

city_choices = ['北京', '上海', '广州', '深圳', '其他']
city_probs = [0.18, 0.17, 0.15, 0.15, 0.35]

level_choices = ['高', '中', '低']
level_probs = [0.2, 0.5, 0.3]


# 随机创建样本
sex_sample = np.random.choice(sex_choices, USER_NUM, p=sex_probs)
city_sample = np.random.choice(city_choices, USER_NUM, p=city_probs)
level_sample = np.random.choice(level_choices, USER_NUM, p=level_probs)

# 创建正态分布的年龄样本
age_sample = np.clip(
    np.random.normal(loc=35, scale=8, size=USER_NUM),
    18, 60
).astype(int)

# 创建指数分布的活跃天数
active_days_sample = np.clip(
    np.random.exponential(scale=7, size=USER_NUM),
    1, 30
).astype(int)

# 创建表格数据
user_data = pd.DataFrame({
    '性别': sex_sample,
    '所在城市': city_sample,
    '消费水平': level_sample,
    '年龄': age_sample,
    '最近活跃天数': active_days_sample
})

user_data.to_csv("user_profiles.csv",index=False,encoding="utf-8-sig")

print("用户画像数据已生成并保存到 user_profiles.csv")

'''
Author: TonyInBeijing
Date: 2025-10-15 19:56:41
LastEditors: TonyInBeijing
LastEditTime: 2025-10-15 20:09:47
FilePath: /geektime/homework/week2/R1_logistic_regression.py
Description: 作业一：基于语义理解的文本分类器

'''

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

MODEL_NAME = "BAAI/bge-m3"
POSITIVE_TEXTS = [
    "这家餐厅的菜真是太好吃了，服务也特别贴心！",
    "今天的天气真不错，心情一下子变好了。",
    "电影拍得很感人，我都看哭了。",
    "客服态度非常好，帮我解决了所有问题。",
    "这次购物体验非常愉快，物流也很快！",
    "产品质量超出预期，下次还会再买。",
    "朋友送的礼物特别用心，真的很开心。",
    "比赛虽然紧张，但结果很满意！",
    "音乐节现场气氛太棒了！",
    "新手机手感很好，功能也很强大。",
    "今天遇到很多好事，真是幸运的一天！",
    "这款咖啡香气浓郁，喝完整个人都精神了。"
]

NEGATIVE_TEXTS = [
    "这家餐厅的菜太咸了，还等了一个小时才上菜。",
    "今天一整天都在下雨，真烦。",
    "电影剧情太无聊了，差点睡着。",
    "客服一点都不耐心，问题没解决还被挂断电话。",
    "这次购物真糟糕，质量太差了。",
    "包裹到的时候已经坏了。",
    "礼物包装随便，完全没有心意。",
    "比赛输了，还被裁判误判，气死了！",
    "音乐节人太多了，体验非常差。",
    "手机买回来第二天就出问题了。",
    "今天真倒霉，什么事都不顺。",
    "咖啡太苦了，一点也不好喝。"
]

TEST_TEXTS = [
    "我很喜欢这次的旅行，风景太美了！",
    "真失望，酒店服务态度非常差。",
    "产品包装精美，客服也很热情。",
    "这次购物体验太糟糕了，再也不会来了。"
]

texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
labels = [1] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS)

# 使用 bge-m3 模型进行文本嵌入
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, normalize_embeddings=True)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42)

# 训练逻辑回归分类器
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)

# 评估模型
y_pred = clf.predict(x_test)

test_emb = model.encode(TEST_TEXTS, normalize_embeddings=True)
pred = clf.predict(test_emb)

for t, p in zip(TEST_TEXTS, pred):
    print(f"文本: {t} => 预测类别: {'积极' if p == 1 else '消极'}")

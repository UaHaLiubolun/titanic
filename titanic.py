import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head(3)
train.info()
train.describe()

sns.set(context="paper", font="monospace")
sns.set(style="white")
f, ax = plt.subplots(figsize=(10, 6))
train_corr = train.drop("PassengerId", axis=1).corr()
sns.heatmap(train_corr, ax=ax, vmax=.9, square=True)
ax.set_xticklabels(train_corr.index, size=15)
ax.set_yticklabels(train_corr.columns[::-1], size=15)
ax.set_title('train feature corr', fontsize=20)
plt.show()

# 创建一个子图
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
sns.distplot(train.Age.fillna(-20), rug=True, color='b', ax=axes[0])
ax0 = axes[0]
ax0.set_title('age distribution')
ax0.set_xlabel("")

ax1 = axes[1]
ax1.set_title('age survived distribution')
k1 = sns.distplot(train[train.Survived==0].Age.fillna(-20), hist=False, color='r', ax=axes[1], label='dead')
k2 = sns.distplot(train[train.Survived==1].Age.fillna(-20), hist=False, color='g', ax=axes[1], label='alive')
ax1.set_xlabel('')

ax1.legend(fontsize=16)

f_1, ax_1 = plt.subplots(figsize=(8, 3))
ax.set_title("Sex Age dist", size=20)
sns.distplot(train[train.Sex=='female'].dropna().Age, hist=False, color='pink', label='female')
sns.distplot(train[train.Sex=='male'].dropna().Age, hist=False, color='blue', label='male')
ax_1.legend(fontsize=15)

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
plt.show()


f_1, ax_1 = plt.subplots(figsize=(8, 3))
ax.set_title("Sex Age dist", size=20)
sns.distplot(train[train.Sex=='female'].dropna().Age, hist=False, color='pink', label='female')
sns.distplot(train[train.Sex=='male'].dropna().Age, hist=False, color='blue', label='male')
ax_1.legend(fontsize=15)
plt.show()


f_2, ax_2 = plt.subplots(figsize=(8, 3))
ax_2.set_title('Pclass Age dist', size=20)
sns.distplot(train[train.Pclass==1].dropna().Age, hist=False, color='pink', label='P1')
sns.distplot(train[train.Pclass==2].dropna().Age, hist=False, color='blue', label='P2')
sns.distplot(train[train.Pclass==3].dropna().Age, hist=False, color='g', label='P3')
ax_2.legend(fontsize=15)
plt.show()


y_dead = train[train.Survived==0].groupby('Pclass')['Survived'].count()
y_alive = train[train.Survived==1].groupby('Pclass')['Survived'].count()
pos = [1, 2, 3]
ax_3 = plt.figure(figsize=(8, 4)).add_subplot(111)
ax_3.bar(pos, y_dead, color='r', alpha=0.6, label='dead')
ax_3.bar(pos, y_alive, color='g', alpha=0.6, bottom=y_dead,  label='alive')
ax_3.legend(fontsize=16, loc='best')
ax_3.set_xticks(pos)
ax_3.set_xticklabels(['Pclass%d'%(i) for i in range(1, 4)], size=15)
ax_3.set_title('Pclass Survived count', size=20)
plt.show()


pos_1 = range(0, 6)
age_list = []
for Pclass_ in range(1, 4):
    for Survived_ in range(0, 2):
        age_list.append(train[(train.Pclass == Pclass_) & (train.Survived == Survived_)].Age.fillna(-20).values)
f_4, ax_4 = plt.subplots(3, 1, figsize=(10, 6))
i_Pclass = 1
for ax_ in ax_4:
    sns.distplot(age_list[i_Pclass * 2 - 2], hist=False, ax=ax_, label='Pclass%d, survived:0'%(i_Pclass), color='r')
    sns.distplot(age_list[i_Pclass * 2 - 1], hist=False, ax=ax_, label='Pclass%d, survived:1'%(i_Pclass), color='g')
    i_Pclass+=1
    ax_.set_xlabel('age', size=15)
    ax_.legend(fontsize=15)
plt.show()


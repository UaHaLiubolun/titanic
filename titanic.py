import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve


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

train.Sex.value_counts()
train.groupby("Sex")['Survived'].mean()

ax_5 = plt.figure(figsize=(10, 4)).add_subplot(111)
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train.dropna(), split=True)
ax_5.set_xlabel('Sex', size=20)
ax_5.set_xticklabels(['Female', 'male'], size=18)
ax_5.set_ylabel('Age', size=20)
ax_5.legend(fontsize=25, loc='bes')
plt.show()

label = []
for sex_i in ['female', 'male']:
    for pclass_i in range(1, 4):
        label.append('sex:%s,Pclass:%d' % (sex_i, pclass_i))
pos_2 = range(6)
f_5 = plt.figure(figsize=(16, 4))
ax_6 = f_5.add_subplot(111)
ax_6.bar(pos_2,
         train[train['Survived'] == 0].groupby(['Sex', 'Pclass'])['Survived'].count().values,
         color='r',
         alpha=0.5,
         align='center',
         tick_label=label,
         label='dead')
ax_6.bar(pos,
        train[train['Survived'] == 1].groupby(['Sex','Pclass'])['Survived'].count().values,
        bottom=train[train['Survived'] == 0].groupby(['Sex','Pclass'])['Survived'].count().values,
        color='g',
        alpha=0.5,
        align='center',
        tick_label=label,
        label='alive')
ax_6.tick_params(labelsize=15)
ax_6.set_title('sex_pclass_survived', size=30)
ax_6.legend(fontsize=15,loc='best')
plt.show()

train.Embarked.fillna('S', inplace=True)

# 年龄离散化
def age_map(x):
    if x<10:
        return '10-'
    if x<60:
        return '%d-%d' % (x // 5 * 5, x // 5 * 5 + 5)
    elif x>=60:
        return '60+'
    else:
        return 'Null'
train['Age_map'] = train['Age'].apply(lambda x: age_map(x))
test['Age_map'] = test['Age'].apply(lambda x: age_map(x))
train.groupby('Age_map')['Survived'].agg(['count', 'mean'])
test[test.Fare.isnull()]
test.loc[test.Fare.isnull(), 'Fare'] = test[(test.Pclass == 1) & (test.Embarked == 'S') & (test.Sex == 'male')].dropna().Fare.mean()

scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(train['Fare'].values.reshape(-1, 1))
train.Fare = fare_scale_param.transform(train['Fare'].values.reshape(-1, 1))
test.Fare = fare_scale_param.transform(test['Fare'].values.reshape(-1, 1))

train_x = pd.concat([train[['SibSp', 'Parch', 'Fare']], pd.get_dummies(train[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])],axis=1)
train_y = train.Survived
test_x = pd.concat([test[['SibSp', 'Parch', 'Fare']], pd.get_dummies(test[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Age_map']])],axis=1)

base_line_model = LogisticRegression()
param = {'penalty': ['l1', 'l2'],
         'C': [0.1, 0.5, 1.0, 5.0]}
grd = GridSearchCV(estimator=base_line_model, param_grid=param, cv=5, n_jobs=3)
grd.fit(train_x, train_y)


def plot_learning_curve(clf, title, x, y, ylim=None, cv=None, n_jobs=3, train_sizes=np.linspace(.05, 1., 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        clf, x, y, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax = plt.figure().add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel(u"train_num_of_samples")
    ax.set_ylabel(u"score")

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                    alpha=0.1, color="b")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                    alpha=0.1, color="r")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"testCV score")

    ax.legend(loc="best")

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(grd, u"learning_rate", train_x, train_y)

gender_submission = pd.DataFrame({'PassengerId': test.iloc[:, 0], 'Survived': grd.predict(test_x)})
gender_submission.to_csv('gender_submission1.csv', index=None)

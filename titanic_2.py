import pandas as pd

titanic = pd.read_csv("train.csv")

# 年龄缺失用均值填补
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
# 性别用 0 1 替换
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
# 用最多的填补
titanic['Embarked'] = titanic['Embarked'].fillna('S')

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

# 用到的特征
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic['Survived'].iloc[train]
    alg.fit(train_predictors, train_target)
    test_prediction = alg.predict(titanic[predictors].iloc[test, :])
    test_prediction
    predictions.append(test_prediction)

import numpy as np
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accury = sum(predictions[predictions == titanic['Survived']]) / len(predictions)

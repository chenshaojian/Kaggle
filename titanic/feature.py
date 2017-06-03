#-*-coding:utf-8
'''
版本号:v1.0
处理缺失数据和数据的数值化
'''
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
import sys


def set_missing_Age(df, rfr=None):
    # 选取数值型特征值作为训练特征值
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 训练数据和测试数据区分

    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()

    # X,y区分

    train_x, train_y = know_age[:, 1:], know_age[:, 0]
    test_x = unknow_age[:, 1:]

    # model
    if (rfr == None):
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(train_x, train_y)

    #predict
    test_y = rfr.predict(test_x)

    #补全
    #data[data.Age.isnull()][['Age']]
    df.loc[df.Age.isnull(), 'Age'] = test_y

    return df, rfr


def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(), 'Cabin'] = 1
    df.loc[df.Cabin.isnull(), 'Cabin'] = 0
    return df


def set_Sex_type(df):
    df.loc[df.Sex == 'female', 'Sex'] = 0
    df.loc[df.Sex == 'male', 'Sex'] = 1
    return df


def set_Embarked_type(df):
    df.loc[df.Embarked.isnull(), 'Embarked'] = 0
    df.loc[df.Embarked == 'S', 'Embarked'] = 1
    df.loc[df.Embarked == 'C', 'Embarked'] = 2
    df.loc[df.Embarked == 'Q', 'Embarked'] = 3
    return df


def dummies_type(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df


def set_Scaling_data(df):
    #Age,Fare归一化处理
    scaler = preprocessing.StandardScaler()
    age_scale = scaler.fit(df['Age'])
    df['Age'] = scaler.fit_transform(df['Age'], age_scale)
    #由于测试集里出现Fare缺失补充数据
    df.loc[df.Fare.isnull(), 'Fare'] = round(df[df.Fare.notnull()]['Fare'].mean(), 1)
    fare_scale = scaler.fit(df['Fare'])
    df['Fare'] = scaler.fit_transform(df['Fare'], fare_scale)
    return df


def model(train_x, train_y):
    train_x = np.array(train_x, dtype=int)
    train_y = np.array(train_y, dtype=int)
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(train_x, train_y)
    return clf


def main():
    train_df = pd.read_csv("./train.csv")
    test_df = pd.read_csv("./test.csv")

    train_df, rfr = set_missing_Age(train_df)
    train_df = set_Cabin_type(train_df)
    train_df = set_Sex_type(train_df)
    train_df = set_Embarked_type(train_df)
    train_df = set_Scaling_data(train_df)
    train_df = train_df.filter(regex='Survived|Age.*|SibSp|Parch|Fare.*|Cabin.*|Embarked.*|Sex.*|Pclass.*')
    train_data = train_df.as_matrix()
    train_x, train_y = train_data[:, 1:], train_data[:, 0]
    test_df, rfr = set_missing_Age(test_df, rfr)
    test_df = set_Cabin_type(test_df)
    test_df = set_Sex_type(test_df)
    test_df = set_Embarked_type(test_df)
    test_df = set_Scaling_data(test_df)
    test_df = test_df.filter(regex='Age.*|SibSp|Parch|Fare.*|Cabin.*|Embarked.*|Sex.*|Pclass.*')
    test_data = test_df.as_matrix()
    test_x = test_data
    print train_x.shape, test_x.shape
    clf = model(train_x, train_y)
    test_x = np.array(test_x, dtype=int)
    pred = clf.predict(test_x)
    return pred


if __name__ == '__main__':
    main()

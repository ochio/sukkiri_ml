import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('./datafiles/Survived.csv')
# print(df['Survived'].value_counts())
# print(df.isnull().sum())
# print(df.shape)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
x = df[col]
t = df['Survived']


def learn(x,t,depth=3):
	x_train, x_test, y_train, y_test = train_test_split(x,t,test_size=0.2, random_state=0)
	model = tree.DecisionTreeClassifier(max_depth=depth, random_state=0 , class_weight='balanced')
	model.fit(x_train, y_train)

	score = model.score(X = x_train, y = y_train)
	score2 = model.score(X = x_test, y = y_test)
	return round(score,3), round(score2,3), model

# for j in range(1,15):
# 	train_score, test_score, model = learn(x, t, depth = j)
# 	sentence = '訓練データの正解率{}'
# 	sentence2 = 'テストデータの正解率{}'
# 	total_sentence = '深さ{}:' + sentence + sentence2
# 	print(total_sentence.format(j, train_score, test_score))

df2 = pd.read_csv('./datafiles/Survived.csv')
# print(df2['Age'].mean())
# print(df2['Age'].median())
# print(df2.groupby('Survived').mean()['Age'])
# print(df2.groupby('Pclass').mean()['Age'])
# print(pd.pivot_table(df2, index = 'Survived', columns = 'Pclass', values = 'Age'))

is_null = df2['Age'].isnull()

df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 0) & (is_null), 'Age'] = 43
df2.loc[(df2['Pclass'] == 1) & (df2['Survived'] == 1) & (is_null), 'Age'] = 35

df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 0) & (is_null), 'Age'] = 33
df2.loc[(df2['Pclass'] == 2) & (df2['Survived'] == 1) & (is_null), 'Age'] = 25

df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 0) & (is_null), 'Age'] = 26
df2.loc[(df2['Pclass'] == 3) & (df2['Survived'] == 1) & (is_null), 'Age'] = 20

col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
x = df2[col]
t = df2['Survived']

male = pd.get_dummies(df2['Sex'], drop_first=True)
# pd.get_dummies(df2['Embarked'], drop_first=True)
# print(male)

# train_score, test_score, model = learn(x,t)

# for j in range(1,15):
# 	s1,s2,m = learn(x, t, depth = j)
# 	sentence = '訓練データの正解率{}'
# 	sentence2 = 'テストデータの正解率{}'
# 	total_sentence = '深さ{}:' + sentence + sentence2
# 	print(total_sentence.format(j, s1, s2))


# sex = df2.groupby('Sex').mean()
# sex['Survived'].plot(kind='bar')
# plt.show()

x_temp = pd.concat([x,male], axis=1)
x_new = x_temp.drop('Sex', axis=1)

# for j in range(1,6):
# 	s1,s2,m = learn(x_new, t, depth = j)
# 	total_sentence = '深さ{}:訓練データの正解率{}::テストデータの正解率{}'
# 	print(total_sentence.format(j, s1, s2))

s1,s2,model=learn(x_new, t, depth=5)

import pickle
with open('survived.pkl', 'wb') as f:
	pickle.dump(model, f)

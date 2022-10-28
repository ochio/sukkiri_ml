from cmath import nan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('./datafiles/Bank.csv')
# print(df.shape)

str_col_name = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']
str_df = df[str_col_name]
str_df2 = pd.get_dummies(str_df, drop_first=True)

num_df = df.drop(str_col_name, axis=1)
df2 = pd.concat([num_df, str_df2,str_df], axis=1)

# print(df2.columns)

train_val, test = train_test_split(df2, test_size=0.1, random_state=9)
# print(train_val.head())

a = train_val.isnull().sum()
# print(a[a>0])

train_val2 = train_val.fillna(train_val.median())

# print(train_val2['y'].value_counts())

t = train_val2['y']
x = train_val2.drop(str_col_name, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state=13)

model = tree.DecisionTreeClassifier(random_state=3, max_depth=3, class_weight='balanced')

model.fit(x_train, y_train)
# print(model.score(x_val, y_val))

def learn(x,t,i):
	x_train, x_val, y_train, y_val = train_test_split(x, t, test_size=0.2, random_state=13)
	datas = [x_train, x_val, y_train, y_val]
	model = tree.DecisionTreeClassifier(random_state=i, max_depth=i, class_weight='balanced')
	model.fit(x_train, y_train)
	train_score = model.score(x_train, y_train)

	val_score = model.score(x_val, y_val)
	return train_score, val_score, model, datas

# for i in range(1, 20):
# 	s1,s2,model,datas = learn(x,t,i)
# 	print(i,s1,s2)

model = tree.DecisionTreeClassifier(max_depth=11, random_state=11)
model.fit(x,t)
test2 = test.copy()
test2 = test2.fillna(train_val.median())


test_y = test2['y']
test_x = test2.drop(str_col_name, axis=1)
test_x = test_x.drop(['id', 'y', 'day'], axis=1)
# print(model.score(test_x, test_y))

a = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
# print(a[0:9])
# print(str_df.columns)

# for name in str_df.columns:
# 	print(train_val.groupby(name)['y'].mean())
# 	print('#####')



# print(pd.pivot_table(train_val, index="housing", columns="loan", values="duration"))
# print(pd.pivot_table(train_val, index="housing", columns="contact", values="duration"))
# print(pd.pivot_table(train_val, index="loan", columns="contact", values="duration"))

def nan_fill(train_val):
	isnull = train_val['duration'].isnull()

	train_val2 = train_val.copy()
	train_val2.loc[(isnull) & (train_val2['housing'] == 'yes') & (train_val2['loan'] == "yes"), 'duration'] = 439
	train_val2.loc[(isnull) & (train_val2['housing'] == 'yes') & (train_val2['loan'] == "no"), 'duration'] = 332

	train_val2.loc[(isnull) & (train_val2['housing'] == 'no') & (train_val2['loan'] == "yes"), 'duration'] = 301
	train_val2.loc[(isnull) & (train_val2['housing'] == 'no') & (train_val2['loan'] == "no"), 'duration'] = 237

	return train_val2
train_val2 = nan_fill(train_val)

# print(train_val2.groupby('y')['duration'].median())
# print(train_val2.groupby('y')['amount'].median())
# print(train_val2.groupby('y')['campaign'].median())
# print(train_val2.groupby('y')['age'].median())

t = train_val2['y']
x = train_val2.drop(str_col_name, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

# for i in range(1,20):
# 	s1,s2,model,datas = learn(x,t,i)
# 	print(i,s1,s2)

s1,s2,model,datas = learn(x,t,10)

def syuuki(model, datas, flag=False):
	if flag:
		pre = model.predict(datas[0])
		y_val = datas[2]
	else:
		pre = model.predict(datas[1])
		y_val=datas[3]
	data = {
		"pred": pre,
		"true": y_val
	}

	tmp = pd.DataFrame(data)
	return tmp, pd.pivot_table(tmp, index="true", columns="pred", values="true", aggfunc=len)
tmp,a = syuuki(model, datas, False)
# print(a)

false = tmp.loc[(tmp['pred']==1)&(tmp['true']==0)].index
true = tmp.loc[(tmp['pred']==0)&(tmp['true']==0)].index
true_df = train_val2.loc[true]
false_df = train_val2.loc[false]
pd.concat([true_df.mean()["age":], false_df.mean()["age":]], axis=1).plot(kind="bar")
# plt.show()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
tmp2 = train_val2.drop(str_col_name, axis=1)
sc_data = sc.fit_transform(tmp2)
sc_df = pd.DataFrame(sc_data, columns=tmp2.columns, index=tmp2.index)

true_df=sc_df.loc[true]
false_df=sc_df.loc[false]
temp2 = pd.concat([false_df.mean()["age":], true_df.mean()["age":]], axis=1)
temp2.plot(kind="bar")
# plt.show()

model_tree = tree.DecisionTreeClassifier(max_depth=10, random_state=10, class_weight="balanced")
model_tree.fit(x,t)

test2 = nan_fill(test)

t = test2['y']
x = test2.drop(str_col_name, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)
print(model_tree.score(x,t))
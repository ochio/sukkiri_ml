import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datafiles/Boston.csv')
# print(df.head(2))

# print(df['CRIME'].value_counts())
crime = pd.get_dummies(df['CRIME'], drop_first=True)
df2 = pd.concat([df, crime], axis = 1)
df2 = df2.drop('CRIME', axis=1)
# print(df2)

train_val, test = train_test_split(df2, test_size=0.2, random_state=0)

# print(train_val.isnull().sum())
train_val_mean = train_val.mean()
train_val2 = train_val.fillna(train_val_mean)

colname = train_val2.columns

# for name in colname:
# 	train_val2.plot(kind = 'scatter', x = name, y = 'PRICE')
# plt.show()

out_line1 = train_val2[(train_val2['RM'] < 6) & (train_val2['PRICE'] > 40)].index
out_line2 = train_val2[(train_val2['PTRATIO'] > 18) & (train_val2['PRICE'] > 40)].index

# print(out_line1, out_line2)

train_val3 = train_val2.drop([76], axis=0)

col = ['INDUS','NOX', 'RM', 'PTRATIO', 'LSTAT', 'PRICE']
train_val4 = train_val3[col]
# print(train_val4.head(3))
# print(train_val4.corr()['PRICE'])

train_cor = train_val4.corr()['PRICE']
abs_cor = train_cor.map(abs)
# print(abs_cor.sort_values(ascending=False))

col = ['RM', 'LSTAT', 'PTRATIO']
x = train_val4[col]
t = train_val4[['PRICE']]
x_train, x_val, y_train, y_val = train_test_split(x,t, test_size=0.2, random_state=0)

sc_model_x = StandardScaler()
sc_model_x.fit(x_train)

sc_x = sc_model_x.transform(x_train)
# print(sc_x)

tmp_df = pd.DataFrame(sc_x, columns = x_train.columns)
# print(tmp_df.mean())
# print(tmp_df.std())

sc_model_y = StandardScaler()
sc_model_y.fit(y_train)

sc_y = sc_model_y.transform(y_train)

model = LinearRegression()
model.fit(sc_x, sc_y)
# print(model.score(x_val, y_val))
sc_x_val = sc_model_x.transform(x_val)
sc_y_val = sc_model_y.transform(y_val)

# print(model.score(sc_x_val, sc_y_val))

def learn(x,t):
	x_train, x_val, y_train, y_val = train_test_split(x,t,test_size=0.2, random_state=0)

	sc_model_x = StandardScaler()
	sc_model_y = StandardScaler()
	sc_model_x.fit(x_train)
	sc_x_train = sc_model_x.transform(x_train)
	sc_model_y.fit(y_train)
	sc_y_train = sc_model_y.transform(y_train)
	model = LinearRegression()
	model.fit(sc_x_train, sc_y_train)

	sc_x_val = sc_model_x.transform(x_val)
	sc_y_val = sc_model_y.transform(y_val)

	train_score = model.score(sc_x_train, sc_y_train)
	val_score = model.score(sc_x_val, sc_y_val)

	return train_score, val_score

x = train_val3.loc[:, ['RM', 'LSTAT', 'PTRATIO', 'INDUS']]
t = train_val3[['PRICE']]

s1,s2 = learn(x,t)
# print(s1,s2)

x['RM2'] = x['RM'] ** 2
x = x.drop('INDUS', axis = 1)
s1,s2 = learn(x,t)
# print(s1,s2)

x['LSTAT2'] = x['LSTAT'] ** 2
s1,s2 = learn(x,t)
# print(s1,s2)

x['PTRATIO2'] = x['PTRATIO'] ** 2
s1,s2 = learn(x,t)
# print(s1,s2)

x['RM * LSTAT'] = x['RM'] * x['LSTAT']
s1,s2 = learn(x,t)
# print(s1,s2)

sc_model_x2 = StandardScaler()
sc_model_x2.fit(x)
sc_x = sc_model_x2.transform(x)

sc_model_y2 = StandardScaler()
sc_model_y2.fit(t)
sc_y = sc_model_y2.transform(t)
model = LinearRegression()
model.fit(sc_x, sc_y)

test2 = test.fillna(train_val.mean())
x_test = test2.loc[:,  ['RM', 'LSTAT', 'PTRATIO']]
y_test = test2[['PRICE']]

x_test['RM2'] = x_test['RM'] ** 2
x_test['LSTAT2'] = x_test['LSTAT'] ** 2
x_test['PTRATIO2'] = x_test['PTRATIO'] ** 2
x_test['RM * LSTAT'] = x_test['RM'] * x_test['LSTAT']

sc_x_test = sc_model_x2.transform(x_test)
sc_y_test = sc_model_y2.transform(y_test)

# print(model.score(sc_x_test, sc_y_test))

import pickle

with open('boston.pkl', 'wb') as f:
	pickle.dump(model, f)

with open('boston_scx.pkl', 'wb') as f:
	pickle.dump(sc_model_x2, f)

with open('boston_scx.pkl', 'wb') as f:
	pickle.dump(sc_model_y2, f)
	
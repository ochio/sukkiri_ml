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
x = train_val[col]
t = train_val[['PRICE']]
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
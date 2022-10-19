import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./datafiles/cinema.csv')

df2 = df.fillna(df.mean())

no = df2[(df2['SNS2'] > 1000) & (df2['sales'] < 8500)].index
df3 = df2.drop(no, axis=0)

x = df3.loc[:, 'SNS1': 'original']

t = df3["sales"]

x_train, x_test, y_train, y_test = train_test_split(x,t, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(x_train, y_train)

new = [[150, 700, 300, 0]]

model.predict(new)
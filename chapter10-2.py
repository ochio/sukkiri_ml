import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./datafiles/bike.tsv', sep='\t')
# print(df.head(3))

weather = pd.read_csv('./datafiles/weather.csv', encoding='shift-jis')
# print(weather)

temp = pd.read_json('./datafiles/temp.json').T
# print(temp.head(2))

df2 = df.merge(weather, how="inner", on="weather_id")
# print(df2.head(2))
# print(df2.groupby('weather').mean()['cnt'])

df3 = df2.merge(temp, how="left", on="dteday")
# print(df3[df3['dteday'] == '2011-07-20'])

df3['atemp'] = df3['atemp'].astype(float)
df3['atemp'] = df3['atemp'].interpolate()
# df3['atemp'].loc[220:240].plot(kind='line')
# plt.show()

# ---------------------------
iris_df = pd.read_csv('./datafiles/iris.csv')
non_df = iris_df.dropna()

x = non_df.loc[:, 'がく片幅':'花弁幅']
t = non_df['がく片長さ']
model = LinearRegression()
model.fit(x,t)

condition = iris_df['がく片長さ'].isnull()
non_data = iris_df.loc[condition]

x = non_data.loc[:, 'がく片幅':'花弁幅']
pred = model.predict(x)
print(pred)

iris_df.loc[condition, 'がく片長さ'] = pred
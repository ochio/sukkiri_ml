import pandas as pd

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
print(df3[df3['dteday'] == '2011-07-20'])
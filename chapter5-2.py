import pandas as pd

df = pd.read_csv('./datafiles/iris.csv')

colmean = df.mean()

df2 = df.fillna(colmean)

xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']

x = df2[xcol]
t = df2['種類']
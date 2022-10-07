import pandas as pd

df = pd.read_csv("./datafiles/KvsT.csv")

xcol = ['身長', '体重', '年代']

x = df[xcol]

t = df['派閥']

print(t)
import pandas as pd

d = {
	'データベースの試験得点': [70, 72, 75, 80],
	'ネットワークの試験得点': [80, 85, 79, 92]
}

df = pd.DataFrame(data=d)

df.index = ['一郎', '二郎', '三郎', '四郎']


df2 = pd.read_csv("./datafiles/ex1.csv")

print(df2.index)
print(df2.columns)
print(df2[['x0', 'x2']])
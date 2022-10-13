from sklearn import tree
import pandas as pd

df = pd.read_csv('./datafiles/iris.csv')

colmean = df.mean(numeric_only=True)

df2 = df.fillna(colmean)

xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']

x = df2[xcol]
t = df2['種類']

model = tree.DecisionTreeClassifier(max_depth=2,random_state=0)

model.fit(x,t)

print(model.score(x,t))
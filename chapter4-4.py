from sklearn import tree
import pandas as pd

df = pd.read_csv("./datafiles/KvsT.csv")

xcol = ['身長', '体重', '年代']

x = df[xcol]

t = df['派閥']

model = tree.DecisionTreeClassifier(random_state = 0)

print(type(x), type(t))
model.fit(x,t)

taro = [[170, 70, 20]]

matsuda = [172, 65, 20]
asagi = [158, 48, 20]
new_data = [matsuda, asagi]

print(model.predict(new_data))
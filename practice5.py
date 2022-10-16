import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('./datafiles/ex2.csv')
print(df.head(3))
print(df.shape)
print(df["target"].value_counts())
print(df.isnull().sum())

df2=df.fillna(df.median())

x = df2[['x0', "x1", "x2", "x3"]]
t = df2["target"]

x_train, x_test, y_train, y_test = train_test_split(x,t, test_size = 0.2, random_state=0)

model = tree.DecisionTreeClassifier(max_depth=3,random_state=0)

model.fit(x_train.values,y_train)
model.score(x_test, y_test)

newdata = pd.DataFrame([[1.56, 0.23, -1.1, -2.8]])

print(model.predict(newdata))

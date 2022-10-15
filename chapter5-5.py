from cmath import pi
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('./datafiles/iris.csv')

colmean = df.mean(numeric_only=True)

df2 = df.fillna(colmean)

xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']

x = df2[xcol]
t = df2['種類']

x_train, x_test, y_train, y_test = train_test_split(x,t, test_size = 0.3, random_state=0)

model = tree.DecisionTreeClassifier(max_depth=2,random_state=0)

model.fit(x_train,y_train)
model.score(x_test, y_test)

x_train.columns = ['gaku_nagasa', 'gaku_haba', 'kaben_nagasa', 'kaben_haba']

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=x_train.columns, filled=True)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('./datafiles/Bank.csv')

job = pd.get_dummies(df['job'], drop_first=True)
marital = pd.get_dummies(df['marital'], drop_first=True)
education = pd.get_dummies(df['education'], drop_first=True)
default = pd.get_dummies(df['default'], drop_first=True)
housing = pd.get_dummies(df['housing'], drop_first=True)
loan = pd.get_dummies(df['loan'], drop_first=True)
contact = pd.get_dummies(df['contact'], drop_first=True)
month = pd.get_dummies(df['month'], drop_first=True)

df2 = pd.concat([df,job,marital,education,default,housing,loan,contact, month] , axis = 1)
df2 = df2.drop(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month'], axis=1)

train_val, test = train_test_split(df2, test_size=0.2, random_state=0)
train_val_mean = train_val.median()
train_val2 = train_val.fillna(train_val_mean)

col = ['age', 'campaign', 'duration']

x = train_val2[col]
t = train_val2['y']

x_train, x_val, y_train, y_val = train_test_split(x,t, test_size=0.2, random_state=0)

model = tree.DecisionTreeClassifier(max_depth=3, random_state=0 , class_weight='balanced')
model.fit(x_train, y_train)

# print(model.score(x_val, y_val))

def learn(x,t,i):
	x_train, x_val, y_train, y_val = train_test_split(x,t, test_size=0.2, random_state=0)
	data = [x_train, x_val, y_train, y_val]

	model = tree.DecisionTreeClassifier(max_depth=i, random_state=i , class_weight='balanced')
	model.fit(x_train, y_train)

	train_score = model.score(x_train, y_train)
	val_score = model.score(x_val, y_val)
	return train_score, val_score,model, data
	
# for i in range(1,20):
# 	s1,s2,model,datas = learn(x,t,i)
# 	print(i,s1,s2)

model = tree.DecisionTreeClassifier(max_depth=11, random_state=11)
model.fit(x,t)
test2 = test.copy()
test2 = test2.fillna(test2.median())

# print(test2)

test_x = test2[col]
test_y = test2['y']
# print(test_y)
# print(model.score(test_x, test_y))

a = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
# print(a)

for name in col:
	print(train_val.groupby[name]['y'])

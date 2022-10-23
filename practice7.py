import pandas as pd

df = pd.read_csv('./datafiles/ex4.csv')
# print(df['sex'].mean())

print(df.groupby('class').mean()['score'])
print(pd.pivot_table(df, index='class', columns='sex', values='score'))

# col = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
# x = df[col]

dept = pd.get_dummies(df['dept_id'], drop_first=True)
df2 = pd.concat([df,dept], axis = 1)
df2 = df2.drop('dept_id', axis=1)

print(df2)
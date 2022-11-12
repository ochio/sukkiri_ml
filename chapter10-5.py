import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datafiles/Bank.csv')
# print(df.columns)

str_col_name=['job','default','marital','education','housing','loan','contact','month']
str_df=df[str_col_name]
# print(str_df.head())
str_df2=pd.get_dummies(str_df, drop_first=True)
# print(str_df2.columns)
num_df = df.drop(str_col_name,axis=1)
df2=pd.concat([num_df,str_df2,str_df], axis=1)
# print(df2.columns)

train_val, test = train_test_split(df2, test_size=0.1, random_state=9)
# print(train_val.head())
is_nan=train_val.isnull().sum()
# print(is_nan[is_nan>0])

# print(train_val.corr())
train_val.corr(numeric_only=True)['duration'].map(abs).sort_values(ascending=False)

num_df=train_val.drop(str_col_name, axis=1)
num_df=num_df.drop('id', axis=1)
num_df2=num_df.dropna()
mcd2=MinCovDet(random_state=0, support_fraction=0.7)
mcd2.fit(num_df2)
# print(mcd2)
dis=mcd2.mahalanobis(num_df2)
dis=pd.Series(dis)
# dis.plot(kind="box")
# plt.show()

no=dis[dis>300000].index
no=num_df2.iloc[no[0]:(no[0]+1), :].index
# print(no)
train_val2=train_val.drop(no)

train_val2.corr(numeric_only=True)['duration'].map(abs).sort_values(ascending=False)

not_nan_df=train_val2.dropna()
temp_t=not_nan_df['duration']
temp_x = not_nan_df[['housing_yes','loan_yes','age','marital_single' ,'job_student']]

model_liner=LinearRegression()
a,b,c,d=train_test_split(temp_x,temp_t,random_state=0,test_size=0.2)

model_liner.fit(a,c)
# print(model_liner.score(a,c), model_liner.score(b,d))

is_null=train_val2['duration'].isnull()
non_x=train_val2.loc[is_null, ['housing_yes','loan_yes','age','marital_single','job_student']]
pred_d = model_liner.predict(non_x)
train_val2.loc[is_null, 'duration'] = pred_d

train_val2.loc[train_val['y'] == 0, "duration"].plot(kind="hist")
train_val2.loc[train_val['y'] == 1, "duration"].plot(kind="hist", alpha=0.4)
# plt.show()

def learn(x,t,i):
    x_train,x_val,y_train,y_val = train_test_split(x,t,test_size=0.2, random_state=13)
    
    datas=[x_train,x_val,y_train, y_val]
    model = tree.DecisionTreeClassifier(random_state=i, max_depth=i, class_weight='balanced')
    model.fit(x_train, y_train)
    train_score=model.score(x_train,y_train)
    val_score=model.score(x_val,y_val)
    return train_score, val_score, model, datas

t = train_val2['y']
x = train_val2.drop(str_col_name, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)

for i in range(1,15):
    s1,s2,model,datas=learn(x,t,i)
    # print(i,s1,s2)
    
test2 = test.copy()
isnull=test2['duration'].isnull()
model_tree=tree.DecisionTreeClassifier(random_state=10,max_depth=10,class_weight='balanced')
if isnull.sum() > 0:
    temp_x=test2.loc[isnull,['housing_yes','loan_yes','age','marital_single','job_student']]
    pred_d=model_liner.predict(temp_x)
    test2.loc[isnull, 'duration'] =pred_d
x_test=test2.drop(str_col_name, axis=1)
x_test=x_test.drop(['id', 'y', 'day'], axis=1)
y_test=test['y']

# print(model.score(x_test,y_test))

s1,s2,model,datas=learn(x,t,9)

def syuukei(model,datas,flag=False):
    if flag:
        pre=model.predict(datas[0])
        y_val=datas[2]
    else:
        pre=model.predict(datas[1])
        y_val=datas[3]
    data={
        "pred":pre,
        "true":y_val
		}
    tmp=pd.DataFrame(data)
    return tmp, pd.pivot_table(tmp,index="true", columns="pred", values="true", aggfunc=len)
tmp,a=syuukei(model,datas,False)
# print(a)


sc = StandardScaler()
tmp2 = train_val2.drop(str_col_name, axis=1)
sc_data = sc.fit_transform(tmp2)
sc_df = pd.DataFrame(sc_data, columns=tmp2.columns, index=tmp2.index)
# print(df)

pre = model.predict(sc_df.drop(["id", "day", "y"], axis=1))
target = tmp2["y"]
true = (pre == target)
false = (pre != target)

true_df=sc_df.loc[true]
false_df=sc_df.loc[false]
# print(true_df)

temp2=pd.concat([false_df.mean()["age":], true_df.mean()["age":]], axis=1)
temp2.plot(kind="bar")
# plt.show()

# print(train_val2.groupby('loan')['y'].mean())
# print(train_val2.groupby('housing')['y'].mean())

train_val3=train_val2.copy()
train_val3["du*hon"]=train_val3["duration"] * train_val3["housing_yes"]
train_val3["du*loan"]=train_val3["duration"] * train_val3["loan_yes"]
train_val3["du*age"]=train_val3["duration"] * train_val3["age"]

t = train_val3["y"]

monthcol=['month_aug',
       'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
       'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep']
jobcol=['job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
       'job_self-employed', 'job_services', 'job_student', 'job_technician',
       'job_unemployed', 'job_unknown']
x = train_val3.drop(str_col_name, axis=1)
x = x.drop(jobcol, axis=1)

x = x.drop(monthcol, axis=1)
x = x.drop(['id', 'y', 'day'], axis=1)
# print(x.columns)

for i in range(5,15):
    s1,s2,model,datas=learn(x,t,i)
    # print(i,s1,s2)

s1,s2,model,datas=learn(x,t,9)
tmp,a=syuukei(model,datas,False)
# print(a)

pd.Series(model.feature_importances_, index=x.columns)

i=9
model=tree.DecisionTreeClassifier(random_state=i, max_depth=i, class_weight="balanced")
model.fit(x,t)

test2 = test.copy()
isnull=test['duration'].isnull()
if isnull.sum()>0:
    temp_x=test2.loc[isnull, ['housing_yes','loan_yes','age','marital_single','job_student']]
    pred_d=model_liner.predict(temp_x)
    test2.loc[isnull, 'duration'] = pred_d

test2["du*hon"]=test2["duration"]*test2["housing_yes"]
test2["du*loam"]=test2["duration"]*test2["loan_yes"]
test2["du*age"]=test2["duration"]*test2["age"]

x_test = test2.drop(str_col_name, axis=1)
x_test = x_test.drop(jobcol, axis=1)
x_test = x_test.drop(monthcol, axis=1)
x_test = x_test.drop(['id', 'y', 'day'], axis=1)
y_test = test['y']

print(x_test.columns)
print(model.score(x_test, y_test))





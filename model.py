import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pickle

dt=pd.read_csv("admins.csv")
college=np.unique(dt['College'])
clg_id=[]
for i in range(len(college)):
    clg_id.append(i+1)
dt['College_id']=dt['College'].replace(college,clg_id)

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

x = dt.iloc[:, 4].values
y = dt.iloc[:, 6].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0)

y_col=y_test
x_col=x_test

x_train= x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

prediction = lm.predict(x_test)

pickle.dump(lm, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl','rb'))
pred = model.predict([[5516]])
pr=pd.DataFrame()
pred=int(pred)
pr["id"]=[pred]
clg_name=pr["id"].replace(clg_id,college)
print(clg_name.to_string(index=False))
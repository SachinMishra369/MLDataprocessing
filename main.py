# https://github.com/yaroslav-v/android-maps-zone
#importing libraries
import pandas as pd
import numpy as np
data=pd.read_csv("data.csv")
#dePendent variable 
x=data.iloc[:,:-1].values
#indePendent variable 
y=data.iloc[:,-1].values

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x[:,1:3]=imputer.fit_transform(x[:,1:3])
# imputer.fit()
#encoding categorical data such as name and string values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
#encoding dePendent variable
from sklearn.preprocessing import LabelEncoder
cty=LabelEncoder()
y=cty.fit_transform(y)
print(y)
#spliting the data into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
print(x_train)

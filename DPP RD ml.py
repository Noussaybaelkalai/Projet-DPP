#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras


# In[2]:


pip install tensorflow


# In[3]:


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report
import datetime as dt
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.read_csv("C:/Users/HP/Downloads/dataa.csv")
data


# In[5]:


pd.set_option('display.max_rows',data.shape[0]+1)


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


sns.set(rc={'figure.figsize': (30, 10)})
sns.boxplot(data=data.select_dtypes(include='number'))


# In[9]:


sns.pairplot(data)


# In[10]:


f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, ax=ax)


# In[11]:


from pandas.plotting import scatter_matrix
scatter_matrix(data.iloc[:,1:8], alpha=1, figsize=(6,6), diagonal = 'kde',color="red")
plt.show()


# In[12]:


#Superposiiton des histogrammes
data.plot(kind='hist',alpha=0.5,bins=20) 
plt.show()


# In[13]:


#Histogrammes séparés
data.hist(bins=20,color="red") 
plt.show()


# In[14]:


#Plot pour open et close
data
data.plot(kind="scatter", x="Open" , y="Close",
               c='Volume',s=100,cmap="viridis")
plt.show()


# In[15]:


print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.shape)


# In[16]:


data.isnull().sum().sort_values(ascending=False)


# In[17]:


#on remplace les valeurs mqt par les valeurs qui se précède 
data.fillna(method='bfill',inplace=True)
data.isnull().sum().any()


# In[18]:


data.isnull().sum().sort_values(ascending=False)


# In[19]:


data=data.drop(['Date','Unnamed: 0'],axis=1)


# In[20]:


print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.shape)


# In[22]:


sns.set(rc={'figure.figsize': (30, 10)})
sns.boxplot(data=data.select_dtypes(include='number'))


# In[23]:


sns.pairplot(data)


# In[24]:


f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, ax=ax)


# In[25]:


from pandas.plotting import scatter_matrix
scatter_matrix(data.iloc[:,1:7], alpha = 1, figsize = (6, 6), diagonal = 'kde',color="red")
plt.show() # On voit que les variables discrètes sont éliminées


# In[26]:


#Superposiiton des histogrammes
data.plot(kind='hist',alpha=0.5,bins=20) 
plt.show()


# In[27]:


#Histogrammes séparés
data.hist(bins=20,color="red") 
plt.show()


# In[28]:


#Plot pour open et close
data
data.plot(kind="scatter", x="Open" , y="Close",
               c='Volume',s=100,cmap="viridis")
plt.show()


# In[21]:


x= data.drop('Close',axis=1)
y=data['Close']


# In[18]:


x.head()


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


scaler= StandardScaler()
scaler.fit(x)


# In[21]:


scaled_data=scaler.transform(x)
scaled_data


# In[22]:


from sklearn.decomposition import PCA


# In[23]:


pca=PCA(n_components=2,svd_solver='full')
pca.fit(scaled_data)


# In[24]:


x_pca=pca.transform(scaled_data)


# In[25]:


scaled_data.shape


# In[26]:


x_pca.shape


# In[27]:


x_pca


# In[28]:


plt.figure(figsize=(16,8))
plt.scatter(x_pca[:,0],x_pca[:,1],c=data['Close'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


# In[29]:


data.var()


# In[30]:


# feature selection using varianceThreshold
# from sklearn.feature_selection import VarianceThreshold
# sel = VarianceThreshold(threshold=(0.3))
# sel_fit=sel.fit_transform(data)


# In[31]:


data


# In[48]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train_sel,y_test_sel=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape,x_test.shape


# In[49]:


x_train.corr()


# In[50]:


import seaborn as sns
# Using pearson correlation
plt.figure(figsize=(16,8))
cor=x_train.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.CMRmap_r)
plt.show()


# In[51]:


def correlation(dataset,threshold):
    col_corr=set()  #  set of all the names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[52]:


corr_features=correlation(x_train,0.9)
len(corr_features)


# In[53]:


corr_features


# In[54]:


# x_train.drop(corr_features,axis=1)


# In[ ]:





# In[55]:


from sklearn.feature_selection import mutual_info_regression
# determine the mutual information 
mutual_info=mutual_info_regression(x_train,y_train_sel)
mutual_info


# In[56]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False)


# In[57]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(12,8))


# In[58]:


from sklearn.feature_selection import SelectKBest
# selecting the top 3 
selected_top_col=SelectKBest(mutual_info_regression,k=3)
selected_top_col.fit(x_train,y_train_sel)
x_train.columns[selected_top_col.get_support()]


# In[59]:


data_red=pd.DataFrame(selected_top_col.fit_transform(x_train,y_train_sel))
data_red.shape


# In[60]:


data_red


# In[61]:


sns.pairplot(data_red)


# In[62]:


sns.boxplot(data=data_red)


# In[63]:


f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(data_red.corr(), annot=True, ax=ax)


# In[64]:


plt.plot(data['Close'])


# In[65]:


ma100=data['Close'].rolling(100).mean()
ma100


# In[66]:


plt.figure(figsize=(16,8))
plt.plot(data['Close'])
plt.plot(ma100)


# In[67]:


ma200=data['Close'].rolling(200).mean()
ma200


# In[68]:


plt.figure(figsize=(16,8))
plt.plot(data['Close'])
plt.plot(ma100)
plt.plot(ma200)


# In[ ]:





# In[69]:


# create a new dataframe with only close column
data=data.filter(['Close'])

# convert dataframe to a numpy array
dataset=data.values

# get the number of rows to train the model on
training_data_len= math.ceil(len(dataset)*.8)


# In[70]:


training_data_len


# In[71]:


# Scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_r=scaler.fit_transform(dataset)


# In[72]:


scaled_data_r.shape


# In[73]:


# Create the trainig dataset
# Create the scaled trainig dataset
train_data=scaled_data_r[0:training_data_len,:]
# Split the data into x_train and y_train datasets
x_train=[]
y_train=[]


for i in range(60 ,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=90:
        print(x_train)
        print(y_train)
        print()


# In[74]:


# Convert x_train and y_train to a numpy array
x_train,y_train=np.array(x_train),np.array(y_train)
x_train.shape


# In[75]:


#  Reshape the data
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[76]:


# Build the lSTM model
model= Sequential()
model.add(LSTM(50, return_sequences=True ,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False ))
model.add(Dense(25))
model.add(Dense(1))


# In[77]:


# Compile the model
model.compile(optimizer='adam' ,loss='mean_squared_error')


# In[78]:


model.fit(x_train,y_train,batch_size=1, epochs=1)


# In[79]:


# Create the testing dataset
test_data=scaled_data_r[training_data_len-60:,:]
# Create the dataset x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60 ,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[80]:


# Convert the data to a numpy array
x_test=np.array(x_test)


# In[81]:


# Reshape the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[82]:


# Get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[83]:


predictions.shape


# In[84]:


# Get the root mean squared error
rmse = np.sqrt((np.mean(predictions -y_test)**2))
rmse


# In[85]:


train= data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions']=predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()


# In[86]:


# Show the valid and predictions prices
valid


# In[87]:


from sklearn.model_selection import train_test_split
x_train0,x_test0,y_train0,y_test0=train_test_split(scaled_data,y,test_size=0.3,random_state=0)
x_train0.shape,x_test0.shape


# In[88]:


from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x_pca,y,test_size=0.3,random_state=0)
x_train1.shape,x_test1.shape


# In[89]:


from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline


# In[90]:


# SVR applied on the data with all its features
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(x_train0, y_train0)
regr.score(x_train0, y_train0)
   


# In[93]:


#SVR applied on the reduced Data using PCA
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(x_train1, y_train1)
regr.score(x_train1, y_train1)


# In[94]:


#SVR applied on the reduced Data using feature selection methods
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(data_red, y_train_sel)
regr.score(data_red, y_train_sel)


# In[ ]:





# In[ ]:





import matplotlib.pyplot as plt
import pandas as pd
import yfinance  as yf
import pandas_datareader as pdr
import numpy as np
import seaborn as sns
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
from keras.models import load_model
import streamlit as st

data=pd.read_csv("C:/Users/HP/Downloads/dataa.csv")

import io
def info(df):
    buffer = io.StringIO()
    data.info(verbose=True,buf=buffer,null_counts=None)
    s = buffer.getvalue()
    st.text(s)
    return

def datainfo():
    st.header('Présentation du jeu de données')
    st.header('Jeu de données')
    st.write(data)
    st.header('Description du jeu de données')
    st.write(data.describe())
    st.header('Informations sur le jeu de données')
    info(data)

print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.shape)

data.isnull().sum().sort_values(ascending=False)


#on remplace les valeurs mqt par les valeurs qui se répètent
data.fillna(data.mean(),inplace=True)
data.isnull().sum().any()


data=data.drop(['Date','Unnamed: 0'],axis=1)

print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.shape)

def visualisation():
    # Visulazition
    st.subheader('Closing price vs Time chart')
    fig = plt.figure(figsize=(16, 8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close price USD', fontsize=18)
    plt.plot(data['Close'])
    st.pyplot(fig)

    # Visulazition
    st.subheader('Closing price vs Time chart 100MA')
    fig = plt.figure(figsize=(16, 8))
    ma100 = data['Close'].rolling(100).mean()
    plt.plot(ma100)
    plt.plot(data['Close'])
    plt.legend(['ma100', 'Close'], loc='lower right')
    st.pyplot(fig)

    # Visulazition
    st.subheader('Closing price vs Time chart 100MA  AND 200MA')
    fig = plt.figure(figsize=(16, 8))
    ma100 = data['Close'].rolling(100).mean()
    ma200 = data['Close'].rolling(200).mean()
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(data['Close'])
    plt.legend(['ma100', 'ma200', 'Close'], loc='lower right')
    st.pyplot(fig)

x= data.drop('Close',axis=1)
y=data['Close']





scaler= StandardScaler()
scaler.fit(x)

scaled_data=scaler.transform(x)


from sklearn.decomposition import PCA

pca=PCA(n_components=2,svd_solver='full')
pca.fit(scaled_data)

x_pca=pca.transform(scaled_data)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train_sel,y_test=train_test_split(x,y,test_size=0.3,random_state=0)




def correlation(dataset,threshold):
    col_corr=set()  #  set of all the names of correlated columns
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

corr_features=correlation(x_train,0.9)


from sklearn.feature_selection import mutual_info_regression
# determine the mutual information
mutual_info=mutual_info_regression(x_train,y_train_sel)


mutual_info = pd.Series(mutual_info)
mutual_info.index = x_train.columns
mutual_info.sort_values(ascending=False)

from sklearn.feature_selection import SelectKBest
# selecting the top 3
selected_top_col=SelectKBest(mutual_info_regression,k=3)
selected_top_col.fit(x_train,y_train_sel)


data_red=pd.DataFrame(selected_top_col.fit_transform(x_train,y_train_sel))






from sklearn.model_selection import train_test_split
x_train0,x_test0,y_train0,y_test0=train_test_split(scaled_data,y,test_size=0.3,random_state=0)

x_train1,x_test1,y_train1,y_test1=train_test_split(x_pca,y,test_size=0.3,random_state=0)
def reg_data():
  # SVR applied on the data with all its features
  regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
  regr.fit(x_train0, y_train0)
  st.write("L'algorithme SVR donne le résultat avec une précision de : ")
  st.write(regr.score(x_train0, y_train0))
  return

def reg_data_pca():
  #SVR applied on the reduced Data using PCA
  regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
  regr.fit(x_train1, y_train1)
  st.write("L'algorithme SVR  appliqué a une dataset de 2 dimension donne le résultat avec une précision de : ")
  st.write(regr.score(x_train1, y_train1))
  return

def reg_data_sel():
   #SVR applied on the reduced Data using feature selection methods
   regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
   regr.fit(data_red, y_train_sel)
   st.write("L'algorithme SVR appliqé a une dataset réduit a 3 dimension donne le résultat avec une précision de : ")
   st.write(regr.score(data_red, y_train_sel))
   return


def prediction_LSTM():
    # create a new dataframe with only close column
    data_red = data.filter(['Close'])

    # convert dataframe to a numpy array
    dataset = data_red.values

    # get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_r = scaler.fit_transform(dataset)

    # Create the trainig dataset
    # Create the scaled trainig dataset
    train_data = scaled_data_r[0:training_data_len, :]
    # Split the data into x_train and y_train datasets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 100:
            print(x_train)
            print(y_train)
            print()

    # Convert x_train and y_train to a numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)

    #  Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the lSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing dataset
    test_data = scaled_data_r[training_data_len - 60:, :]
    # Create the dataset x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # Get the root mean squared error
    rmse = np.sqrt((np.mean(predictions - y_test) ** 2))
    st.write('Erreur quadratique moyenne')
    st.write(rmse)

    # Plot the data
    train = data_red[:training_data_len]
    valid = data_red[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    st.subheader('Predictions vs Close vs valid ')
    fig = plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    st.pyplot(fig)
def user_input():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    Data=st.sidebar.button('RY.TO',key=1, help=None, on_click=datainfo, args=None, kwargs=None)
    Visualisation=st.sidebar.button('Visualisation',key=1, help=None, on_click=visualisation, args=None, kwargs=None)
    SVC = st.sidebar.button('SVR ', key=5, help=None, on_click=reg_data, args=None, kwargs=None)
    SVC_pca = st.sidebar.button('SVR-PCA ', key=5, help=None, on_click=reg_data_pca, args=None, kwargs=None)
    SVC_sel= st.sidebar.button('SVR-Feauture-selection ', key=5, help=None, on_click=reg_data_sel, args=None, kwargs=None)
    Prediction = st.sidebar.button('LSTM', key=1, help=None, on_click=prediction_LSTM, args=None, kwargs=None)
    dataf=pd.DataFrame(data)
    return dataf
data=user_input()

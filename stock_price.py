import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
import tensorflow as tf
from tensorflow.contrib import rnn
import yfinance as yf
from sklearn import preprocessing
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from operator import itemgetter
from sklearn.metrics import mean_squared_error, r2_score, f1_score, explained_variance_score 
from math import sqrt
import matplotlib.pyplot as plt2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import figure


tf.reset_default_graph()

def get_data(comp_name,start_date, end_date):

  '''Function to get the historical data of a company, using the yfinance api
     Arguments(str): comp_name - Ticker name of the stock
     Returns dataframe of all data - inclusive of High, Low, Close, Open prices
     for all dates specified.
  '''
  stock=yf.Ticker(comp_name)
  data=stock.history(start=start_date, end=end_date)
  req_data = data.drop(['Dividends', 'Stock Splits', 'Volume'],axis=1)
  return req_data

def normalize(data):

    '''Function to normalize the dataset using MinMaxScaler
     Argument(s): dataframe containing data which is not normalized
     Returns normalized dataframe- inclusive of High, Low, Close, Open prices
     for all dates.
  '''
    scaler = MinMaxScaler()
    x=data.values
    x_scaled = scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
 
    return data

def split_data(data):

  '''Function to split the data into test and train after reshaping it into
      (batchsize, sequence_length, no_of_columns/features).
     Argument(s): dataframe containing data which is normalized
     Returns train and test sets
  '''
  ip = data

  #Target is all data shifted one row upwards - the last row is NAN
  goal = data.shift(-1)

  #Consider the data except the last row as it will be the day predicted
  ip = ip[:-2].values
  goal = goal[:-2].values

  #Ensure the data is reshaped correctly
  r1 = (ip.shape[0])%(SEQLEN*4) 
  r2 = (goal.shape[0])%(SEQLEN*4)
  ip = ip[:-r1]
  goal = goal[:-r2]

  #Reshape the data into (batch_size, seq_len, features)
  X = np.reshape(ip, ((ip.shape[0]*ip.shape[1])//(SEQLEN*4), SEQLEN,4))
  Y = np.reshape(goal, ((goal.shape[0]*goal.shape[1])//(SEQLEN*4), SEQLEN,4))

  #Using the inbuilt sklearn module to perform the test-train split
  xT, xTe, yT, yTe = train_test_split(X, Y, test_size = 0.2, random_state = 25)

  return xT, xTe, yT, yTe

def train_input():

    '''This function converts the training data into corresponding tensors
      so it can be used by the RNN model Estimator as an input function
      for training.
    '''

    x_t = tf.convert_to_tensor(x_train, np.float64)
    y_t = tf.convert_to_tensor(y_train, np.float64)
 
    return x_t, y_t
 
def test_ip():

  '''This function converts the test data into corresponding tensors
      so it can be used by the RNN model Estimator as an input function 
      for evaluating the model.
  '''
  
  dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  dataset = dataset.repeat(1)
  dataset = dataset.batch(128)
  samples, labels = dataset.make_one_shot_iterator().get_next()
  return samples, labels

def create_cell_for_layer(units, flag):

  '''Function to create GRU, LSTM cells for a given layer of the Deep Network
     Argument(s): 
        units(int): No. of units for the cell
        flag(int): Indicates which type of layer to create :
                  1: Comprises of LSTM Cells
                  2: Comprises of GRU Cells
                  3: Comprises of a combination
                  Default : Combination
     Returns created_cell for the layer
  '''
  if flag == 1:

    cell=rnn.BasicLSTMCell(units)
    cell=tf.contrib.rnn.DropoutWrapper(cell, dropout)
  elif flag == 2: 

    cell = tf.nn.rnn_cell.GRUCell(units)
    cell=tf.contrib.rnn.DropoutWrapper(cell, dropout)
  else:
    cell=rnn.BasicLSTMCell(units)
    cell = tf.nn.rnn_cell.GRUCell(units)
    cell=tf.contrib.rnn.DropoutWrapper(cell, dropout)

  

  return cell   

def RNN_stock_pred(features, labels, mode):

    '''This function defines the model for the problem at hand. A multi-layered
       RNN architecture is created, comprised of either LSTM, GRU or a combo
       of both cells. Estimator is used to return the predicted values.
    '''
    #Batch Size depending on the input it receives
    b_size = tf.shape(features)[0]
    
    #Multi layered architecture constructed, state_is_tuple is either true or false depending on whether we are using GRU(False) or LSTM(True)
    cell = tf.nn.rnn_cell.MultiRNNCell([create_cell_for_layer(units, choice) for _ in range(2)], state_is_tuple=pick_between[choice-1])
    output, states = tf.nn.dynamic_rnn(cell, features, dtype=tf.float64)
    
    output = tf.reshape(output, [b_size * SEQLEN, units])
    reduced_op = tf.layers.dense(output, 4)  # Yr l[BATCHSIZE*SEQLEN, 1]
    reduced_op = tf.reshape(reduced_op, [b_size, SEQLEN, feat])  # Yr [BATCHSIZE, SEQLEN, 1]
    final = reduced_op[:, -1, :]  # Last output Yout [BATCHSIZE, 1]  #getting the last batch

    loss = train_op = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.losses.mean_squared_error(reduced_op, labels)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = tf.contrib.training.create_train_op(loss, optimizer)
 
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"res": final},
        loss=loss,
        train_op=train_op
    )

def disp_graph(predicted, actual):

  '''Function to dispay the graph plotted between actual values vs 
    predicted values the model gives out.
     Argument(s):[numpy arrays] of predicted and actual values
  '''
  #figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
  plt2.rcParams["figure.figsize"] = (10,10)
  plt2.plot(predicted,color='blue', label='Predicted Values')
  plt2.plot(actual,color='red', label='Test')
  plt2.legend(loc='upper right')
  plt2.ylabel("Normalized Prices")
  plt2.title("Google (GOOGL) predictions")
  plt2.show()

def disp_scatter(predicted, actual):

  '''Function to dispay the scatter plot between actual values vs 
    predicted values the model gives out.
     Argument(s):[numpy arrays] of predicted and actual values
  '''
  plt2.plot(predicted, 'go', label='Predicted Values')
  plt2.plot(actual,color='red', label='y_test')
  plt2.legend(loc='upper right')
  plt2.show()

def train_and_estimate():

  '''Function to train the model using tensorflow High Level API - Estimator
     The name of the model function is passed as a paramater to the estimator
     along with the training configuration which is saved in our temp_op
     directory. The model is trained for <epochs>, and then performs predictions 
     using the estimator.predict() function which takes the test_ip function as 
     input. 
     Returns: List of close prices for actual, predicted values
  '''
  training_config = tf.estimator.RunConfig(model_dir="./temp_op")
  estimator = tf.estimator.Estimator(model_fn=RNN_stock_pred, config=training_config)
  estimator.train(input_fn=train_input, steps = epochs)

  #Results are predicted on test_input
  results = estimator.predict(test_ip)

  res_final = [res["res"] for res in results]
  predict = np.array(res_final)
  actual = y_test[:, -1]

  #Consider the last column of the predicted and actual values : Close
  predict_close = predict[:, -1]
  actual_close = actual[:,-1]
  return predict_close, actual_close

def calculate_results(pred, y_test):

  '''Function to calculate the Mean Absolute Percentage Error using the 
     mathematical formula, as well as R squared which is a good measure
     of how well a regression type model performs. 
     Argument(s):[numpy arrays] of predicted and actual values
     Returns: Mean Absolute Percentage Error (MAPE)
  '''
  
  mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
  r2= r2_score(y_test,pred)
  mse= mean_squared_error(y_test,pred)
  ex= explained_variance_score(y_test, pred)
  print("\n\n\n\n")
  print("######### RESULTS #########")
  print("\n")
  print("Mean Absolute Percentage Error:", mape)
  print("R squared:", r2)
  print("Mean Squared Error:", mse)
  print("Explained Variance Score:", ex)

  return mape

#HYPERPARAMETERS

learning_rate = 0.02 #Learning Rate for Adam Optimizer 
SEQLEN = 20         #Sequence Length
feat = 4            #Open, Close, High, Low
units = 80          #NO of GRU, LSTM Units
dropout = 0.8       #Droput rate
pick_between = ["True", "False", "False"]
epochs = 10         #No. of Epochs required for training

#REQUIRED - can be modified as per requirement 
stock_name = "MSFT"    #Ticker Name for Stock
start = "2015-01-01"   #Start Date for dataset
end = "2019-01-01"     #End date for dataset
choice = 3             # 1 - LSTM, 2- GRU, 3 - COMBO

#Retrieve the raw data set using get_data() which uses the yfinance api
raw_data = get_data(stock_name, start, end)

#Normalize the dataset using normalize()
norm_data =normalize(raw_data)

#Split the data into test and train set
x_train, x_test, y_train, y_test= split_data(norm_data)

#Train the model, and retrieve predicted vs actual values
predict, actual = train_and_estimate()

#Calculate Mean Absolute Percentage Error
calculate_results(predict, actual)

#Visualization of predcited vs actual results
disp_graph(predict, actual)
disp_scatter(predict,actual)

'''caler = MinMaxScaler(feature_range=(0, 1))
obj = scaler.fit(predict)
#obj = scaler.fit(testY)
testPredict = obj.inverse_transform(predict)
#testY = obj1.inverse_transform([testY])
'''
#Denormalized = ($normalizedValue * ($max-$min) + $min)
#testpred = scaler.inverse_transform(predict)

print(y_test.shape)


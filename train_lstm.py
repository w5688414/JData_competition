import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from data_cleaning_06 import *
import numpy as np
from keras.callbacks import ModelCheckpoint 

data_X=load_data("./cache/get_train_test_set_06_7_-9_-9_X_2000.pkl")
data_Y=load_data("./cache/get_train_test_set_06_7_-9_-9_Y_2000.pkl")

# print(data_X)
data_X=np.array(data_X)
# print(data_X.shape)
data_X=data_X[:,:,7:]
data_Y=np.array(data_Y)
print(data_X.shape)
print(data_Y.shape)
X_train=data_X
print(data_X[0])

# training_set=pd.read_csv('item_table.csv')   #reading csv file
# print(training_set.head())			   #print first five rows

# training_set_x=training_set.iloc[:,2:10].values
# training_set_y=training_set.iloc[:,10:11].values
# # 将整型变为float
# dataset_x = training_set_x.astype('float32')
# dataset_y = training_set_y.astype('float32')

# # print(dataset)
# sc = MinMaxScaler()			   #scaling using normalisation 
# data_X = sc.fit_transform(data_X)
# print(data_X.shape)
# X_train_norm = (X_train - X_train.min())/(X_train.max()-X_train.min())
# print(X_train_norm[0])
# scaler_x = MinMaxScaler ( feature_range =( 0, 1))
# x = scaler_x.fit_transform (dataset_x)
 
 
# scaler_y =MinMaxScaler ( feature_range =( 0, 1))
# y = scaler_y.fit_transform (dataset_y)

# train_end=int(len(y)*0.7)

# x_train = x [0: train_end,]
# x_test = x[ train_end +1:len(x),]    
# y_train = y [0: train_end] 
# y_test = y[ train_end +1:len(y)] 

model = Sequential ()
model.add (LSTM (30 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(data_X.shape[1], data_X.shape[2]) ))
# model.add(Dropout(0.2))
model.add(Dense(64,activation='tanh'))
model.add (Dense(1, activation = 'linear'))
print(model.summary())
epochs = 10
checkpointer = ModelCheckpoint(filepath='weights.best.hdf5', verbose=1, save_best_only=True)
print('Start training...')

model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop")   
model.fit (data_X, data_Y, batch_size =16, nb_epoch =epochs, validation_split=0.3,callbacks=[checkpointer],shuffle = False)
# model.save('my_model.h5')   
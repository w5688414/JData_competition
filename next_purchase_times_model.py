import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import numpy as np
from keras.callbacks import ModelCheckpoint 
from util import DataLoader, Features
from SBBTree_ONLINE import SBBTree
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta

# test code
Data = DataLoader(
				FILE_jdata_sku_basic_info='./data_ori/jdata_sku_basic_info.csv',
				FILE_jdata_user_action='./data_ori/jdata_user_action.csv',
				FILE_jdata_user_basic_info='./data_ori/jdata_user_basic_info.csv',
				FILE_jdata_user_comment_score='./data_ori/jdata_user_comment_score.csv',
				FILE_jdata_user_order='./data_ori/jdata_user_order.csv'
			)

# train data
TrainFeatures = Features(
						DataLoader=Data,
						PredMonthBegin = datetime(2017, 4, 1),
						PredMonthEnd = datetime(2017, 4, 30),
						FeatureMonthList = [(datetime(2017, 3, 1), datetime(2017, 3, 31), 1),\
									(datetime(2017, 1, 1), datetime(2017, 3, 31), 3),\
									(datetime(2016, 10, 1), datetime(2017, 3, 31), 6)],
						MakeLabel = True
					)

# pred data
PredFeatures = Features(
					DataLoader=Data,
					PredMonthBegin = datetime(2017, 5, 1),
					PredMonthEnd = datetime(2017, 5, 31),
					FeatureMonthList = [(datetime(2017, 4, 1), datetime(2017, 4, 30), 1),\
									(datetime(2017, 2, 1), datetime(2017, 4, 30), 3),\
									(datetime(2016, 11, 1), datetime(2017, 4, 30), 6)],
					MakeLabel = False
				)

train_features = TrainFeatures.TrainColumns
cols = TrainFeatures.IDColumns + TrainFeatures.LabelColumns + train_features

train_label_FirstTime = 'Label_30_101_BuyNum'
X = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime]>0][train_features].fillna(0).values   #缺省值补0
y = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime]>0][train_label_FirstTime].values 

# print(X[0])
# print(y.shape)

scaler_x = MinMaxScaler ( feature_range =( 0, 1))
x = scaler_x.fit_transform (X)
x=x.reshape(x.shape[0], 1, x.shape[1])

model = Sequential ()
model.add (LSTM (30 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(1, x.shape[2]) ))
model.add(Dropout(0.3))
model.add(Dense(64,activation='tanh'))
model.add (Dense(1, activation = 'linear'))
print(model.summary())
epochs = 100
checkpointer = ModelCheckpoint(filepath='weights.purchase_times.hdf5', verbose=1, save_best_only=True)
print('Start training...')

model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop")   
model.fit (x, y, batch_size =16, nb_epoch =epochs, validation_split=0.3,callbacks=[checkpointer],shuffle = False)
# model.save('my_model.h5')   
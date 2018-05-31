import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
import numpy as np
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import TensorBoard
from util import DataLoader, Features
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


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

train_label_FirstTime = 'Label_30_101_FirstTime'
X = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime]>0][train_features].fillna(0).values   #缺省值补0
y = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime]>0][train_label_FirstTime].values 

# print(X[0])
# print(y.shape)

scaler_x = MinMaxScaler ( feature_range =( 0, 1))
x = scaler_x.fit_transform (X)
x=x.reshape(x.shape[0], 1, x.shape[1])

# model = Sequential ()
# model.add (LSTM (128 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(1, x.shape[2]) ))
# # model.add(Dropout(0.2))
# model.add(Dropout(0.3))
# model.add (Dense(1, activation = 'linear'))
# print(model.summary())
# epochs = 50
# model_names = ('./trained_models/' +
#                'purchase_date.{epoch:02d}-{val_loss:.4f}.hdf5')
# checkpointer = ModelCheckpoint(filepath=model_names, verbose=1, save_best_only=True)
# tensorboard_logs =TensorBoard(log_dir='./keras_logs', write_graph=True)
# print('Start training...')

# model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop")   
# model.fit (x, y, batch_size =16, nb_epoch =epochs, validation_split=0.3,callbacks=[checkpointer,tensorboard_logs],shuffle = False)

def lstm_model():
	model = Sequential ()
	model.add (LSTM (256 , activation = 'tanh', inner_activation = 'hard_sigmoid' , input_shape =(1, x.shape[2]) ))
	# model.add(Dropout(0.2))
	model.add(Dropout(0.3))
	model.add (Dense(1, activation = 'linear'))
	
	return model

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=57, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

model=lstm_model()
print(model.summary())
epochs = 50
model_names = ('./trained_models/' +
               'purchase_date.{epoch:02d}-{val_loss:.4f}.hdf5')
checkpointer = ModelCheckpoint(filepath=model_names, verbose=1, save_best_only=True)
tensorboard_logs =TensorBoard(log_dir='./keras_logs', write_graph=True)
print('Start training...')

model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop")   
model.fit (x, y, batch_size =16, nb_epoch =epochs, validation_split=0.3,callbacks=[checkpointer,tensorboard_logs],shuffle = False)
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# # fix random seed
# seed = 7
# np.random.seed(seed)
# kfold= KFold(n_splits =10, random_state=seed)
# results = cross_val_score(estimator, X, y, cv=kfold)

# print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))
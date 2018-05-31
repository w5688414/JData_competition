#coding=utf-8
from __future__ import print_function
import os
import pandas as pd
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, load_model  
from keras.callbacks import ModelCheckpoint 
from sklearn.preprocessing import MinMaxScaler
import cv2, numpy as np
from datetime import datetime, timedelta
from util import DataLoader, Features

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
train_features = TrainFeatures.TrainColumns
X_pred = PredFeatures.data_BuyOrNot_FirstTime[train_features].fillna(0).values
scaler_x = MinMaxScaler ( feature_range =( 0, 1))
x = scaler_x.fit_transform (X_pred)
x=x.reshape(x.shape[0], 1, x.shape[1])

# train 下个月购买次数预测 回归模型

model = load_model("./trained_models/purchase_times.44-0.3226.hdf5")
model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop") 
train_label_BuyNum = 'Label_30_101_BuyNum'
PredFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum] = model.predict(x)
# pred=model.predict(x)

# train 当月首次购买时间预测 回归模型
model = load_model("./trained_models/purchase_date.49-6.3767.hdf5")
model.compile (loss ="mean_absolute_error" , optimizer = "rmsprop") 
train_label_FirstTime = 'Label_30_101_FirstTime'
PredFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] = model.predict(x)

# submit
columns = ['user_id'] + [train_label_BuyNum] + [train_label_FirstTime]
out_submit = PredFeatures.data_BuyOrNot_FirstTime[columns].sort_values(['Label_30_101_BuyNum'],ascending=False)
out_submit[train_label_FirstTime] = out_submit[train_label_FirstTime].map(lambda day: datetime(2017, 5, 1)+timedelta(days=int(day+0.49-1)))

out_submit = out_submit[['user_id']+[train_label_FirstTime]]
out_submit.columns = ['user_id','pred_date']
out_submit.head(50000).to_csv('./predict.csv',index=False,header=True)
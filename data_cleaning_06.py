#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2018/5/18
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import os
import pickle
from datetime import timedelta
from datetime import datetime
import pandas as pd
import numpy as np

# 定义文件名
JDATA_USER_ACTION = 'data_ori/jdata_user_action.csv'
JDATA_SKU_BASIC_INFO = 'data_ori/jdata_sku_basic_info.csv'
JDATA_USER_BASIC_INFO = 'data_ori/jdata_user_basic_info.csv'
JDATA_USER_COMMENT_SCORE = 'data_ori/jdata_user_comment_score.csv'
JDATA_USER_ORDER = 'data_ori/jdata_user_order.csv'
ITEM_ORDER_COMMENT_FILE = "clean_data/item_order_comment.csv"
ITEM_ACTION_FILE = "clean_data/item_action.csv"


def load_data(dump_path):
    '''
    加载数据
    :param dump_path:
    :return:
    '''
    return pickle.load(open(dump_path, 'rb'))


def dump_data(obj, dump_path):
    '''
    存储数据
    :param obj:
    :param dump_path:
    :return:
    '''
    pickle.dump(obj, open(dump_path, 'wb+'))


def is_exist(dump_path):
    '''
    资源是否存在
    :param dump_path:
    :return:
    '''
    return os.path.exists(dump_path)


def get_from_fname(fname):
    '''
    通过名字获取数据
    :param fname:
    :return:
    '''
    df_item = pd.read_csv(fname, header=0)
    return df_item


def get_all_sku_06():
    '''
    获取所有商品信息
    :return:
    '''
    dump_path = './cache/get_all_sku_06.pkl'
    if is_exist(dump_path):
        sku = load_data(dump_path)
    else:
        sku = get_from_fname(JDATA_SKU_BASIC_INFO)
        para_2_one_hot = pd.get_dummies(sku['para_2'], prefix='para_2')
        para_3_one_hot = pd.get_dummies(sku['para_3'], prefix='para_3')
        sku = pd.concat([sku, para_2_one_hot, para_3_one_hot], axis=1)
        del sku['para_2']
        del sku['para_3']
        dump_data(sku, dump_path)
    return sku


def get_all_user_06():
    '''
    获取所有用户信息
    :return:
    '''
    dump_path = './cache/get_all_user_06.pkl'
    if is_exist(dump_path):
        user = load_data(dump_path)
    else:
        user = get_from_fname(JDATA_USER_BASIC_INFO)
        dump_data(user, dump_path)
    return user


def get_all_order_06():
    '''
    获取所有订单信息
    :return:
    '''
    dump_path = './cache/get_all_order_06.pkl'
    if is_exist(dump_path):
        order = load_data(dump_path)
    else:
        order = get_from_fname(JDATA_USER_ORDER)
        user = get_all_user_06()
        sku = get_all_sku_06()
        order = pd.merge(order, user, how='left', on='user_id')
        order = pd.merge(order, sku, how='left', on='sku_id')
        order = order[(order['age'] != 1) & (order['age'] != 6)]
        order = order[(order['cate'] == 101) & (order['cate'] != 71) & (order['cate'] != 30)]
        order['a_type'] = 3
        order['a_num'] = 1
        order.rename(columns={'o_date': 'date'}, inplace=True)
        order.rename(columns={'o_sku_num': 'num'}, inplace=True)
        order = order[['user_id', 'date', 'a_type', 'sku_id', 'num']]
        print(order.shape)
        dump_data(order, dump_path)
    return order


def get_all_action_06():
    '''
   将下单行为追加到行为表中,获取全部行为信息
   :return:
   '''
    dump_path = './cache/get_all_action_06.pkl'
    if is_exist(dump_path):
        actions = load_data(dump_path)
    else:
        actions = get_from_fname(JDATA_USER_ACTION)
        sku = get_all_sku_06()
        user = get_all_user_06()
        actions = actions.groupby(['user_id', 'a_date', 'a_type', 'sku_id'], as_index=False).sum()
        actions = pd.merge(actions, sku, how='left', on='sku_id')
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = actions[(actions['age'] != 1) & (actions['age'] != 6)]
        actions = actions[(actions['cate'] == 101) & (actions['cate'] != 71) & (actions['cate'] != 30)]
        actions.rename(columns={'a_date': 'date'}, inplace=True)
        actions.rename(columns={'a_num': 'num'}, inplace=True)
        actions = actions[['user_id', 'date', 'a_type', 'sku_id', 'num']]
        print(actions.shape)
        dump_data(actions, dump_path)
    return actions


def get_order_by_cate_age_sex():
    pass


def get_all_action_order_06():
    dump_path = './cache/get_all_action_order_06.pkl'
    if is_exist(dump_path):
        action_order = load_data(dump_path)
    else:
        user = get_all_user_06()
        sku = get_all_sku_06()
        action = get_all_action_06()
        order = get_all_order_06()
        action_order = pd.concat([action, order], axis=0)
        action_order = pd.merge(action_order, sku, how='left', on='sku_id')
        action_order = pd.merge(action_order, user, how='left', on='user_id')
        # action_order = action_order[(action_order['age'] != 1) & (action_order['age'] != 6)]
        # action_order = action_order[
        #     (action_order['cate'] == 101) & (action_order['cate'] != 71) & (action_order['cate'] != 30)]
        action_order = action_order.sort_values(by='date', ascending=True)
        dump_data(action_order, dump_path)
    return action_order


def get_user_by_dim(age, lv):
    user = get_all_user_06()
    if age != -9:
        user = user[user['age'] == age]
    if lv != -9:
        user = user[user['user_lv_cd'] == lv]
    return user


def get_usr_order_action(user_id):
    all_action_order = get_all_action_order_06()
    action_order = all_action_order[all_action_order['user_id'] == user_id]
    return action_order


def get_train_test_set_06(step_size=7, age=-9, lv=-9):
    dump_path_x = './cache/get_train_test_set_06_%s_%s_%s_X.pkl' % (step_size, age, lv)
    dump_path_y = './cache/get_train_test_set_06_%s_%s_%s_Y.pkl' % (step_size, age, lv)

    # 通过维度筛选出的用户
    if is_exist(dump_path_x) & is_exist(dump_path_y):
        X = load_data(dump_path_x)
        Y = load_data(dump_path_y)
    else:
        dim_user = get_user_by_dim(age, lv)
        X = []
        Y = []
        for id, u in enumerate(dim_user.values):
            dump_path_x_cache = './cache/get_train_test_set_06_%s_%s_%s_X_%s.pkl' % (step_size, age, lv, id)
            dump_path_y_cache = './cache/get_train_test_set_06_%s_%s_%s_Y_%s.pkl' % (step_size, age, lv, id)
            if is_exist(dump_path_x_cache) & is_exist(dump_path_y_cache):
                X = load_data(dump_path_x_cache)
                Y = load_data(dump_path_y_cache)
            else:
                # 用户U所有的行为和订单
                user_order_action = get_usr_order_action(u[0])
                user_order_action = user_order_action.sort_values(by=['a_type', 'date'], ascending=[False, False])
                # 用户U所有行为中的下单行为
                user_order = user_order_action[user_order_action['a_type'] == 3]
                # 用户所有行为中的点击和关注行为
                # 注：只要的点击行为，没有要关注行为
                user_action = user_order_action[user_order_action['a_type'] == 1]

                for j, o in enumerate(user_order.values):
                    # 下单时间
                    o_date = o[1]
                    # 产品sku
                    o_sku = o[3]
                    o_days = datetime.strptime(o_date, '%Y-%m-%d') - timedelta(days=step_size)
                    o_days = o_days.strftime('%Y-%m-%d')
                    # 下单时间前step_size天的行为
                    user_action_o = user_action[(user_action['date'] > o_days) & (user_action['date'] <= o_date)]
                    user_action_o = user_action_o[user_action_o['sku_id'] == o_sku]
                    # 遍历前七天，生成数据，为空填充0
                    x = []
                    for i in range(step_size):
                        day = datetime.strptime(o_days, '%Y-%m-%d') + timedelta(days=i + 1)
                        day = day.strftime('%Y-%m-%d')
                        # 行为当天
                        day_action = user_action_o[user_action_o['date'] == day]
                        # 如果行为不为空
                        c = o.copy()
                        o = np.array(o)
                        if day_action.empty:
                            c[0] = o[0]
                            c[1] = day
                            c[2] = 0
                            c[3] = o[3]
                            c[4] = 0
                            c[5] = 0
                            c[6] = 0
                            c[7] = 0
                            c[8] = 0
                            c[9] = 0
                            c[10] = 0
                            c[11] = 0
                            c[12] = 0
                            c[13] = 0
                            c[14] = 0
                            c[15] = 0
                            c[16] = 0
                            c[17] = 0
                            c[18] = 0
                            c[19] = 0
                            c[20] = 0
                            c[21] = 0
                            c[22] = 0
                            c[23] = 0
                            c[24] = 0
                            c[25] = 0
                            x.append(c)
                        else:
                            day_action = np.array(day_action)
                            c[0] = day_action[0][0]
                            c[1] = day_action[0][1]
                            c[2] = day_action[0][2]
                            c[3] = day_action[0][3]
                            c[4] = day_action[0][4]
                            c[5] = day_action[0][5]
                            c[6] = day_action[0][6]
                            c[7] = day_action[0][7]
                            c[8] = day_action[0][8]
                            c[9] = day_action[0][9]
                            c[10] = day_action[0][10]
                            c[11] = day_action[0][11]
                            c[12] = day_action[0][12]
                            c[13] = day_action[0][13]
                            c[14] = day_action[0][14]
                            c[15] = day_action[0][15]
                            c[16] = day_action[0][16]
                            c[17] = day_action[0][17]
                            c[18] = day_action[0][18]
                            c[19] = day_action[0][19]
                            c[20] = day_action[0][20]
                            c[21] = day_action[0][21]
                            c[22] = day_action[0][22]
                            c[23] = day_action[0][23]
                            c[24] = day_action[0][24]
                            c[25] = day_action[0][25]
                            x.append(c)
                        X.append(x)
                    x_c = x.copy()
                    x_c = pd.DataFrame(x_c)
                    x_c = x_c.groupby(by=0, as_index=False).sum()
                    x_action_num = x_c[4][0]
                    y_o_num = o[4]
                    if x_action_num == 0:
                        buy_rate = 1
                    else:
                        buy_rate = y_o_num / x_action_num
                    Y.append(buy_rate)
                    if id % 100 == 0:
                        dump_data(X, dump_path_x_cache)
                        dump_data(Y, dump_path_y_cache)
            print('id ：%s' % (id + 1), 'X ：%s' % (len(X)), 'Y ：%s' % (len(Y)))
        dump_data(X, dump_path_x)
        dump_data(Y, dump_path_y)
        return X, Y


def get_handle_order_06():
    order = get_all_order_06()
    order = order.groupby(['o_date', 'cate'], as_index=False).mean()
    order = order[order['cate'] == 101]
    del order['user_id']
    del order['sku_id']
    del order['o_id']
    del order['o_date']
    del order['cate']
    return order


if __name__ == '__main__':
    get_train_test_set_06(7, -9, -9)
    # get_usr_order_action()

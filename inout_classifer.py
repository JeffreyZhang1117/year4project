import os
import warnings
import pandas as pd
import numpy as np
from os import path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import data_process as dp
import draw_map as dm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, plot_confusion_matrix,accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
warnings. filterwarnings("ignore")


def min_max_norm(data, col):
    target_col = data[col]
    max_num = max(target_col.dropna())
    min_num = min(target_col.dropna())
    std = (target_col - min_num) / (max_num - min_num)
    data[col] = std

    return data


def smooth_label_indoor(data, window_size=10, percentage=1):
    i = 0
    while i < (len(data) - window_size - 1):
        if sum(data.iloc[i:i + window_size + 1]['pred_label'] == 0) > percentage:
            data.iloc[i:i + window_size + 1]['pred_label'] = 0
        i += window_size
    return data


def smooth_label_com(data, window_size=10, percentage=1):
    i = 0
    while i < (len(data) - window_size - 1):
        if sum(data.iloc[i:i + window_size + 1]['pred_label'] == 1) > percentage:
            data.iloc[i:i + window_size + 1]['pred_label'] = 1
        i += window_size
    return data  # add time-series features


def add_time_feature_train(data, t=10):
    try:
        data.sort_values(by=['phoneTimestamp'], inplace=True)
    except:
        data.sort_values(by=['timestamp'], inplace=True)
    time_data = data.iloc[0:2]
    time_data['avg_dist'] = 0
    time_data['same_land'] = 0
    time_data['same_road'] = 0
    for i in range(len(data) - t):
        if (len(data[i:i + t + 1]['file label'].unique()) == 1):
            temp = data.iloc[i]
            if ((len(data[i:i + t + 1]['close_road_idx'].unique()) == 1) and data.iloc[i]['close_road_idx'] != 0):
                temp['same_road'] = 1
            else:
                temp['same_road'] = 0
            if ((len(data[i:i + t + 1]['close_land_idx'].unique()) == 1) and data.iloc[i]['close_land_idx'] != 0):
                temp['same_land'] = 1
            else:
                temp['same_land'] = 0
            temp['avg_dist'] = data[i:i + t + 1]['gps_dist'].mean()
            time_data = time_data.append(temp)

    time_data = time_data.iloc[2:]
    time_data['highway'] = time_data['highway'].astype('category')
    time_data['highway_encode'] = time_data['highway'].cat.codes
    time_data['landuse'] = time_data['landuse'].astype('category')
    time_data['landuse_encode'] = time_data['landuse'].cat.codes

    return time_data


def add_time_feature(data, t=10):
    try:
        data.sort_values(by=['phoneTimestamp'], inplace=True)
    except:
        data.sort_values(by=['timestamp'], inplace=True)
    time_data = data.iloc[0:2]
    time_data['avg_dist'] = 0
    time_data['same_land'] = 0
    time_data['same_road'] = 0
    for i in range(len(data) - t):
        temp = data.iloc[i]
        if ((len(data[i:i + t + 1]['close_road_idx'].unique()) == 1) and data.iloc[i]['close_road_idx'] != 0):
            temp['same_road'] = 1
        else:
            temp['same_road'] = 0
        if ((len(data[i:i + t + 1]['close_land_idx'].unique()) == 1) and data.iloc[i]['close_land_idx'] != 0):
            temp['same_land'] = 1
        else:
            temp['same_land'] = 0
        temp['avg_dist'] = data[i:i + t + 1]['gps_dist'].mean()
        time_data = time_data.append(temp)

    time_data = time_data.iloc[2:]
    time_data['highway'] = time_data['highway'].astype('category')
    time_data['highway_encode'] = time_data['highway'].cat.codes
    time_data['landuse'] = time_data['landuse'].astype('category')
    time_data['landuse_encode'] = time_data['landuse'].cat.codes

    return time_data

def create_traindata ():
    #edi data
    data_path = os.path.join(os.getcwd(), 'new_data_set_with_roads_distance_land.csv')
    edi_data = pd.read_csv(data_path, index_col=0)
    edi_data = edi_data[edi_data['i/o'] != 1]
    edi_data.loc[edi_data['i/o'].isin([2, 1]), 'i/o'] = 1  # commuting
    edi_data.sort_values(by=['timestamp'], inplace=True)

    #london data
    test_london = pd.read_csv(os.path.join(os.getcwd(), "all_data_new_select_with_roads_distance_land_processed.csv"))
    merge_cols = ['pm1', 'pm2_5', 'pm10', 'temperature', 'humidity', 'bin0', 'bin1', 'bin2',
                  'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10',
                  'bin11', 'bin12', 'bin13', 'bin14', 'bin15', 'gpsLongitude',
                  'gpsLatitude', 'gpsAccuracy', 'file label', 'timestamp', 'highway', 'distance', 'landuse', 'gps_dist',
                  'gps_dist_std', 'i/o', 'geometry', 'close_road_idx', 'close_land_idx']
    london = test_london[merge_cols]
    edin = edi_data[merge_cols]

    #merge london and edin data as training set
    edin_london_data = pd.concat([london, edin])
    # add time_feature
    all_train_time = add_time_feature_train(edin_london_data, t=37)

    # seperate train x and train y
    chosen_columns = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8',
                      'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15',
                      'highway_encode', 'distance', 'landuse_encode',
                      'avg_dist', 'same_road', 'same_land']

    x = all_train_time[chosen_columns]
    y = all_train_time['i/o']

    return x,y,test_london

def add_osm_to_inhale(inhalecode:str):
    inhale_path = path.join(os.getcwd(),inhalecode + "_airspeck_personal_manual_raw.csv")
    full_data = pd.read_csv(inhale_path)
    inhal_road = dp.load_highway('inhale')
    inhal_land = dp.load_landuse('inhale')

    inhalp = dp.add_road_feature(inhal_road, full_data)
    inhalp = dp.add_land_feature(inhal_land, inhalp)
    inhalp = dp.distance_euclidean(inhalp)
    inhalp = dp.calculate_std(inhalp, 'gps_dist', k=10)

    bins = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8',
            'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15']
    inhalp[bins] = inhalp[bins].div(inhalp[bins].sum(axis=1), axis=0)
    for b in bins:
        inhalp[b].fillna(0, inplace=True)

    return inhalp


#indoor and outdoor model
def inout_model(x,y):
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0, max_depth=10))
    clf.fit(x, y)
    return clf

#commuting model
def commuting_model(london_data):
    new_test = add_time_feature(london_data, t=37)
    new_test = new_test[new_test['i/o'] == 1]

    chosen_columns = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8',
                      'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15',
                      'highway_encode', 'distance', 'landuse_encode',
                      'avg_dist', 'same_road', 'same_land']

    X = new_test[chosen_columns]
    y = new_test['activity'].astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0, max_depth=10))
    clf.fit(X_train, y_train)

    return clf

# manual adjust data
def manual_adjust(data):
    data.loc[data['landuse'] == 'outland', 'pred_label'] = 1
    data.loc[data['highway'].isin(['tertiary', 'unclassified', 'primary_secondary']), 'pred_label'] = 1

    return data

def Apply_on_inhale (rawdata_code:str):
    #get trainning data
    train_x,train_y, london_data = create_traindata()
    print(1)
    #train clf model
    clf_io = inout_model(train_x,train_y)
    print(2)

    #process raw data
    process_data = add_osm_to_inhale(rawdata_code)
    test_inhale_io = add_time_feature(process_data, t=37)

    # choose useful columns
    chosen_columns = ['bin0', 'bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8',
                      'bin9', 'bin10', 'bin11', 'bin12', 'bin13', 'bin14', 'bin15',
                      'highway_encode', 'distance', 'landuse_encode',
                      'avg_dist', 'same_road', 'same_land']
    test_x = test_inhale_io[chosen_columns]
    # predict indoor/ outdoor
    y_test_preds = clf_io.predict(test_x)
    test_inhale_io['pred_label'] = y_test_preds

    print(3)
    #train commute model
    clf_commut = commuting_model(london_data)

    # do commuting prdiction
    inhal_io = test_inhale_io[test_inhale_io['pred_label'] == 1]
    test_inhale_commut = add_time_feature(inhal_io, t=37)
    commuting_x = test_inhale_commut[chosen_columns]
    y_test_preds = clf_commut.predict(commuting_x)
    test_inhale_commut['commuting_label'] = y_test_preds
    test_inhale_commut = smooth_label_indoor(test_inhale_commut, window_size=10, percentage=5)
    test_inhale_commut = smooth_label_com(test_inhale_commut, window_size=10, percentage=10)
    test_inhale_commut = manual_adjust(test_inhale_commut)
    print(4)

    #save two prediction file into folder
    file1 = test_inhale_io.to_csv(path.join(os.getcwd(), rawdata_code+"_airspeck_personal_manual_inout.csv"))
    file2 = test_inhale_commut.to_csv(path.join(os.getcwd(), rawdata_code+"_airspeck_personal_manual_commuting.csv"))
    return test_inhale_io, test_inhale_commut




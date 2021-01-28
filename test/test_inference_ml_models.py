import pytest
import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import date, timedelta
from dateutil.parser import parse
from contextlib import contextmanager

from s.inference_ml_models import (get_days_difference_today, fetch_weather_forecast, get_predicted_temperature_at_time, 
                                    get_predicted_rain_at_time, bool_to_int, 
                                   generate_features_for_date_time, perform_numeric_nn_prediction,
                                   deserialize_object_given_path, load_ensemble_models_from_disk, 
                                   load_numeric_nn_model_scal_enc, get_number_days_predictions,
                                   perform_predictions_ensemble_next_days, perform_predictions_numeric_nn_next_days,
                                   generate_input_rows_df, load_use_models_for_inference)

from s.functions import(perform_models_ensemble_prediction, check_list_files_exist)

from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from test.test_functions import (is_int_valid_in_range)

#####CONFIG OPTIONS BEGIN
#We will test for the Bolzano and Bassa Atesina weather station
INDEX_WEATHER_STATION = 1

#The number of days for which we would like to get predictions (EX: 5)
NUM_DAYS_PREDICTIONS = get_number_days_predictions(INDEX_WEATHER_STATION)

LANG = "it"

MODELS_PATH = "./d/Stations_Models_2018"

SITI_CODSITO = "00000002"

#####CONFIG OPTIONS END


@contextmanager
def not_raises(Exception):
    try:
        yield
    except Exception as err:
        raise AssertionError(
            "Exception {0} raised !".format(
                repr(err)
            )
        )


#Test for 'get_number_days_prediction'
def test_get_number_days_predictions():
    
    #let's test the method 'get_number_days_predictions'
    num_days_preds = get_number_days_predictions(INDEX_WEATHER_STATION)
    
    assert(isinstance(num_days_preds, int))
    
    assert(num_days_preds >= 0 and num_days_preds <= 6)

#Test for 'get_days_difference_today
def test_get_days_difference_today():
    
    #Check for dates in the future
    for i in range(0, NUM_DAYS_PREDICTIONS + 1):
        date_check =  date.today() + timedelta(days=i)
        date_time_check = str(date_check) + " 00:00:00"
        date_time_parsed = parse(date_time_check)
        
        days_difference = get_days_difference_today(date_time_parsed)
        
        assert(days_difference == i)
        
    #Check for dates in the past
    for i in range(0, NUM_DAYS_PREDICTIONS + 1):
        date_check =  date.today() - timedelta(days=i)
        date_time_check = str(date_check) + " 00:00:00"
        date_time_parsed = parse(date_time_check)
        
        days_difference = get_days_difference_today(date_time_parsed)
        
        assert(days_difference == -i)
        
#Test for 'fetch_weather_forecast'
def test_fetch_weather_forecast():
    #let's see if the weather webservice of the province of Bolzano does indeed work for all stations
    #that we are considering
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    unique_weather_indices = list(np.unique(df_tratte["WEATHER_INDEX"]))
    
    list_languages = ["it", "de"]
    
    for lang in list_languages:
        for i in range(0, NUM_DAYS_PREDICTIONS + 1):
            for weather_index in unique_weather_indices:
                #let's perform a GET request
                weather_forecast_station = fetch_weather_forecast(weather_index, i, lang)
                #Check if weather_forecast_station was successfully gotten
                assert(weather_forecast_station != None)
                #Check if the keys we are using are part of the weather forecast keys
                list_weather_keys = ["temperatureMax", "temperatureMin", "rainFrom", "rainTo", "rainTimespan1", "rainTimespan2"]
               
                assert(all(weather_forecast_station[x] != None for x in list_weather_keys))
                
                assert(all(x in  list(weather_forecast_station.keys()) for x in list_weather_keys))
                
                #if within 3 days, we also need to check for rainTimespan3 and rainTimespan4 to be present in the weather forecasts
                if i < 2:
                    list_accurate_weather_keys = ["rainTimespan3", "rainTimespan4"]
                    assert(all(x in  list(weather_forecast_station.keys()) for x in list_accurate_weather_keys))

            
#Test for 'get_predicted_temperature_at_time'
def test_get_predicted_temperature_at_time():
    
    #Let's loop through all next 5 days for which we want a prediction
    for d in range(0, NUM_DAYS_PREDICTIONS + 1):
        date_predict =  date.today() + timedelta(days=d)
        weather_forecast = fetch_weather_forecast(INDEX_WEATHER_STATION, d, LANG)
        assert(weather_forecast != None)
        #let's loop through all the hours for which we want a prediction
        for h in range(0, 24):
            date_time_predict = str(date_predict) + " " + str(h) + ":00:00"
            date_time_parsed = parse(date_time_predict)
            time_parsed = date_time_parsed.time()
            #check if we do indeed have a weather forecast
            temperature_predicted = get_predicted_temperature_at_time(time_parsed, weather_forecast)
            #if the temperature predicted a valid float?
            assert(type(temperature_predicted) == np.float64)
            
    
#Test for "get_rain_log_normal_distribution"        
def test_get_predicted_rain_at_time():
    
    #Let's loop through all next 5 days for which we want a prediction
    for d in range(0, NUM_DAYS_PREDICTIONS + 1):
        date_predict =  date.today() + timedelta(days=d)
        weather_forecast = fetch_weather_forecast(INDEX_WEATHER_STATION, d, LANG)
        assert(weather_forecast != None)

        #let's loop through all the hours for which we want a prediction
        for h in range(0, 24):
            date_time_predict = str(date_predict) + " " + str(h) + ":00:00"
            date_time_parsed = parse(date_time_predict)
            time_parsed = date_time_parsed.time()            
            #check if we do indeed have a weather forecast
            
            #let's try to get the quantity of rain in mm
            precipitation_time_mm = get_predicted_rain_at_time(time_parsed, weather_forecast, d)
            assert(precipitation_time_mm >= 0)

def test_bool_to_int():
    assert(bool_to_int(True) == 1)
    
    assert(bool_to_int(False) == 0)

#Test for "generate_input_features_given_date_time"
def test_generate_input_features_given_date_time():
    
    #Firstly generate a data frame containing the input features
    for d in range(0, NUM_DAYS_PREDICTIONS + 1):
        date_predict = date.today() +  timedelta(days=d)
        weather_forecast = fetch_weather_forecast(INDEX_WEATHER_STATION, d, LANG)

        for h in range(0, 24):
            #The date for which we want a prediction
            date_time_predict = str(date_predict) + " " + str(h) + ":00:00"
            
            input_row_X = generate_features_for_date_time(INDEX_WEATHER_STATION, date_time_predict, weather_forecast) 
            
            #let's check if the input features have been successfully generated
            features_list_check = ["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]
            assert(all(feature_name in features_list_check for feature_name in list(input_row_X.columns)))
            
            #Let's check if the input features have valid types
            #NB: Niederschlag can be either int64 (if 0) or float64 (if > 0)
            assert(type(input_row_X["HOUR"][0]) == np.int64 and type(input_row_X["WEEK_DAY"][0]) == np.int64 and
                   type(input_row_X["TEMPERATURE"][0]) == np.float64 and 
                   (type(input_row_X["NIEDERSCHLAG"][0]) == np.float64 or type(input_row_X["NIEDERSCHLAG"][0]) == np.int64)
                    )
            
            #Hour generated is in a valid range
            assert(is_int_valid_in_range(input_row_X["HOUR"].iloc[0], 0, 25) == True)
            
            #WEEK_DAY is in a valid range
            assert(is_int_valid_in_range(input_row_X["WEEK_DAY"].iloc[0], 0, 8) == True)

            #Niederschlag is non-negative
            assert(input_row_X["NIEDERSCHLAG"].iloc[0] >= 0)

            #Temperature exists            
            assert(input_row_X["NIEDERSCHLAG"].iloc[0] != None)

def test_deserializaze_object_given_path():
    #Let's see if we can indeed load up an existing model
    
    #Numerical nn load up test
    list_numeric_traffic_labels = ["COUNT_1", "COUNT_2"]
    
    for traffic_num_level_label in list_numeric_traffic_labels:
        with not_raises(Exception):
            #Load up a valid model, no exception
            nn_numeric_model = deserialize_object_given_path(MODELS_PATH, "nn_numeric", traffic_num_level_label, SITI_CODSITO)
            
            assert(nn_numeric_model != None)
            assert(type(nn_numeric_model) == Sequential)
            
        #Let's load up an invalid model
        with pytest.raises(Exception):
            nn_numeric_model = deserialize_object_given_path(MODELS_PATH, "dummy_nn_numeric", traffic_num_level_label, SITI_CODSITO)
    
        #Numerical scaler load up test
        with not_raises(Exception):
            #Load up a valid model, no exception
            scaler_niederschlag = deserialize_object_given_path(MODELS_PATH, "scaler_niederschlag", traffic_num_level_label, SITI_CODSITO)
            
            assert(scaler_niederschlag != None)
            assert(type(scaler_niederschlag) == StandardScaler)
            
        
def test_load_ensemble_models_from_disk():
    list_traffic_labels = ["TRAFFIC_1", "TRAFFIC_2"]
    
    #Let's load up the ensemble models for both labels
    for traffic_level_label in list_traffic_labels:
        list_models = load_ensemble_models_from_disk(traffic_level_label, SITI_CODSITO, MODELS_PATH)
        
        assert(list_models != None)
        
        assert(len(list_models) == 3)
        
        assert(type(list_models) == list)
        #Let's check if the loaded up models are of the right type
        assert(type(list_models[0]) == RandomForestClassifier)
        assert(type(list_models[1]) == KNeighborsClassifier)
        assert(type(list_models[2]) == DecisionTreeClassifier) 
        
def test_load_numeric_nn_model_scal_end():
    list_numeric_traffic_labels = ["COUNT_1", "COUNT_2"]
    
    for traffic_num_level_label in list_numeric_traffic_labels:
        list_objs_loaded = load_numeric_nn_model_scal_enc(traffic_num_level_label, SITI_CODSITO, MODELS_PATH)
        
        #Let's see if the objects have been successfully loaded and the correct number of objs was loaded
        assert(list_objs_loaded != None)
        assert(len(list_objs_loaded) == 6)
        
        #Let's now check the types of the objects
        assert(type(list_objs_loaded[0]) == StandardScaler)
        assert(type(list_objs_loaded[1]) == StandardScaler)
        assert(type(list_objs_loaded[2]) == OneHotEncoder)
        assert(type(list_objs_loaded[3]) == OneHotEncoder)
        assert(type(list_objs_loaded[4]) == MinMaxScaler)
        assert(type(list_objs_loaded[5]) == Sequential)


def test_generate_input_rows_df():
    #Let's generate the input rows that will be passed to the neural networks for prediction purposes
    input_rows_df = generate_input_rows_df(NUM_DAYS_PREDICTIONS, INDEX_WEATHER_STATION)
    
    assert(len(input_rows_df) > 0)
    
    features_list_check = ["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY", "DAY", "MONTH", "YEAR"]
     
    #Let's check if correct columns generated
    assert(all(feature_name in features_list_check for feature_name in list(input_rows_df.columns)))
    
    #let's check if correct number of columns
    assert(len(input_rows_df.columns) == len(features_list_check))
    
    #let's check if correct number of rows
    assert(len(input_rows_df) == (NUM_DAYS_PREDICTIONS + 1) * 24)
    
def test_perform_models_ensemble_prediction():
    #Let's load up  the models
    input_rows_df = generate_input_rows_df(NUM_DAYS_PREDICTIONS, INDEX_WEATHER_STATION)
    
    input_rows_X = input_rows_df[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    
    list_traffic_labels = ["TRAFFIC_1", "TRAFFIC_2"]
    list_model_names = ["Optimized Decision Tree", "Simple Random Forest","Best KNN"]
    
    for traffic_level_label in list_traffic_labels:
        #Let's load up the ensemble models for both labels
        list_models = load_ensemble_models_from_disk(traffic_level_label, SITI_CODSITO, MODELS_PATH)    
        assert(len(list_models) == 3)
        assert(list_models != None)
        #Predictions using Random Forest model
        y_labels_rf = list_models[0].predict(input_rows_X)
        #Predictions using KNN classifier
        y_labels_knn = list_models[1].predict(input_rows_X)
        #Predictions using Decision Tree
        y_labels_dec_tree = list_models[2].predict(input_rows_X)
        
        #Let's check if the predictions obtained are as many as the rows in the input data frame and host values in range [1,4]
        assert(len(y_labels_rf) == len(input_rows_X))
        assert(all(is_int_valid_in_range(y_label, 1, 5) for y_label in y_labels_rf))

        assert(len(y_labels_knn) == len(input_rows_X))
        assert(all(is_int_valid_in_range(y_label, 1, 5) for y_label in y_labels_knn))

        assert(len(y_labels_dec_tree) == len(input_rows_X))
        assert(all(is_int_valid_in_range(y_label, 1, 5) for y_label in y_labels_dec_tree))
        
        #Let's perform ensemble predictions 
        y_labels_ensemble = perform_models_ensemble_prediction(input_rows_X, list_models, list_model_names)
        
        assert(y_labels_ensemble != None)
        assert(len(y_labels_ensemble) == len(input_rows_X))
        assert(all(is_int_valid_in_range(y_label, 1, 5) for y_label in y_labels_ensemble))


def test_perform_predictions_ensemble_next_days():
    input_rows_df = generate_input_rows_df(NUM_DAYS_PREDICTIONS, INDEX_WEATHER_STATION)
    
    input_rows_X = input_rows_df[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]    
    
    #Let's load up the models for traffic_1 and traffic_2
    list_models_traffic_1 = load_ensemble_models_from_disk("TRAFFIC_1", SITI_CODSITO, MODELS_PATH)    
    list_models_traffic_2 = load_ensemble_models_from_disk("TRAFFIC_2", SITI_CODSITO, MODELS_PATH)    
    
    y_predicted_traffic_1, y_predicted_traffic_2 = perform_predictions_ensemble_next_days(list_models_traffic_1, list_models_traffic_2, input_rows_X)
    
    #Let's check if a proper number of predictions has been generated 
    assert(len(y_predicted_traffic_1) == len(input_rows_X) and len(y_predicted_traffic_2) == len(input_rows_X))
    
    #Let's check if correct values have been indeed generated
    assert(all(is_int_valid_in_range(y_pred_label_1, 1, 5) for y_pred_label_1 in y_predicted_traffic_1))
    assert(all(is_int_valid_in_range(y_pred_label_2, 1, 5) for y_pred_label_2 in y_predicted_traffic_2))

    
def test_perform_numeric_nn_predictions():
    input_rows_df = generate_input_rows_df(NUM_DAYS_PREDICTIONS, INDEX_WEATHER_STATION)
    
    input_rows_X = input_rows_df[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]  
    
    list_num_traffic_labels = ["COUNT_1", "COUNT_2"]
    
    
    for traffic_num_label in list_num_traffic_labels:
        #let's load up the objects for performing a prediction with a neural network
        list_objects_count_1 = load_numeric_nn_model_scal_enc(traffic_num_label, SITI_CODSITO, models_path=MODELS_PATH)
        assert(list_objects_count_1 != None)
        #Let's use these objects for performing predictions
        y_num_pred_count_1 = perform_numeric_nn_prediction(list_objects_count_1, input_rows_X)
        #Let's check if there are as many y values predicted generated as there are input row containing features
        assert(len(y_num_pred_count_1) == len(input_rows_X))
        
        #Let's check if the predicted y values are >= 0
        assert(all(y_num_pred > 0 for y_num_pred in y_num_pred_count_1))
        
        #Let's check if the predicted values are indeed integers
        assert(all(type(y_num_pred) == int for y_num_pred in y_num_pred_count_1))
    

def test_perform_predictions_numeric_nn_next_days():
    input_rows_df = generate_input_rows_df(NUM_DAYS_PREDICTIONS, INDEX_WEATHER_STATION)
    
    input_rows_X = input_rows_df[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]  
    
    list_objects_count_1 = load_numeric_nn_model_scal_enc("COUNT_1", SITI_CODSITO, models_path=MODELS_PATH)
    list_objects_count_2 = load_numeric_nn_model_scal_enc("COUNT_2", SITI_CODSITO, models_path=MODELS_PATH)
    
    assert(len(list_objects_count_1) == len(list_objects_count_2))

    y_num_pred_count_1, y_num_pred_count_2 = perform_predictions_numeric_nn_next_days(list_objects_count_1, list_objects_count_2, input_rows_X)

    #Let's see if the proper output is generated from this method
    assert(len(y_num_pred_count_1) == len(y_num_pred_count_2) and len(y_num_pred_count_1) == len(input_rows_X) and len(y_num_pred_count_2) == len(input_rows_X))
    
    #Predicted values are non-negative...
    assert(all(y_num_pred > 0 for y_num_pred in y_num_pred_count_1))
    assert(all(y_num_pred > 0 for y_num_pred in y_num_pred_count_2))
    
    #Predicted values are integers...
    assert(all(type(y_num_pred) == int for y_num_pred in y_num_pred_count_1))
    assert(all(type(y_num_pred) == int for y_num_pred in y_num_pred_count_2))


def test_load_use_models_for_inference():
    path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv"
    df_tratte_all = pd.read_csv(path_metadata, sep=",",  encoding = "ISO-8859-1")
    
    #We got one single 'tratta' 
    df_tratta_single = df_tratte_all[df_tratte_all["SITI_CODSITO"] == 2]
    
    #Let's use this single tratta to perform inference
    df_tratta_single.to_csv("./d/data_frame_single_tratta_test.csv", index=False, header=True)
    
    dir_predictions_test = "./d/Stations_Predictions_Test"
    
    #if the folder does not already exist, then create it
    if not os.path.isdir(dir_predictions_test):
        os.mkdir(dir_predictions_test)
    
    #Let's generate some predictions to the newly created folder
    load_use_models_for_inference(folder_models="./d/Stations_Models_2018", folder_predictions=dir_predictions_test,
                        folder_past="./d/Stations_Past_2018", path_metadata="./d/data_frame_single_tratta_test.csv")
    
    #let's check if the predictions have been correctly generated
    with not_raises(Exception):
        check_list_files_exist([dir_predictions_test + "/00000002_predictions.csv"])
        
    #Let's remove all the files in the newly generated directory
    for file in listdir(dir_predictions_test):
        os.remove(os.path.join(dir_predictions_test, file))
        
    #And let's remove the directory created
    os.rmdir(dir_predictions_test)
    

    
    
    

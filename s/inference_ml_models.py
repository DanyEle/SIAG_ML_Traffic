# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:36:05 2019

@author: dgadler

This file is used to load the models trained via "Train_ML_Models" and use 
them for inference to produce predictions for the next 5 upcoming days to the disk 
"""
from .functions import (get_siti_codsito_given_index, load_df_traffic_station, display_info_about_station, 
                       output_predictions_to_folder, check_list_folders_exist, check_list_files_exist,
                       perform_models_ensemble_prediction, preprocess_X_data_nn)
import pandas as pd
from dateutil.parser import parse
from datetime import date, timedelta
import datetime
import requests
import statistics
import numpy as np
import random
from math import log
from joblib import load


#Input: - 
#Output: - number_days, an integer integer representing the number of days for which 
#we are going to perform weather predictions.
#Presently, we get the maximum number of days for which predictions exist from the Province of Bolzano's webservice
#0 = today (6.02.2020), 5 = 6 days from today (11.02.2020)
def get_number_days_predictions(index_weather_station):
    date_today = date.today()
    #let's perform a dummy request to the weather 
    url_weather_provinz = "http://daten.buergernetz.bz.it/services/weather/district/" + str(index_weather_station) + "/bulletin"
    # sending get request and saving the response as response object 
    r = requests.get(url = url_weather_provinz)
    data_received = r.json()
    
    last_weather_forecast = data_received["forecasts"][len(data_received["forecasts"]) - 1]
    
    #let's get the date of the last weather forecast
    date_last_weather_forecast = datetime.datetime.strptime(last_weather_forecast["date"], '%Y-%m-%dT%H:%M:%S').date()
    
    #let's get the number of days between the last date forecast and today's date.
    number_days = (date_last_weather_forecast - date_today).days
    
    return(number_days)




#Input: date_time_parsed: the input date & time for which we want a prediction
#Output: the number of days of difference between today and the date_time_parsed
def get_days_difference_today(date_time_parsed):
    #Just get the date for the prediction
    date_parsed = date_time_parsed.date()
    date_today = date.today()
    days_difference = (date_parsed - date_today).days
    #print(days_difference)
    return(days_difference)
    
#Input: - weather_station_index: INT [1-7]
#       - days_difference: INT, the number of days of difference between heute and the day where
#       the weather forecast is seeked
#       - language: "it" | "de"
#Output: the weather forecast as returned from the webservice of the provincia di Bolzano
def fetch_weather_forecast(weather_station_index, days_difference, language):
    parameters = {"format" : "json", "lang" : language}
    url_weather_provinz = "http://daten.buergernetz.bz.it/services/weather/district/" + str(weather_station_index) + "/bulletin"
    # sending get request and saving the response as response object 
    r = requests.get(url = url_weather_provinz, params = parameters)
    data_received = r.json()
    #Now get the piece of weather we need, depending on the days_difference
    #print(days_difference)
    weather_forecast_fetched = data_received["forecasts"][days_difference]
    
    return(weather_forecast_fetched)
    
#Input: time_parsed: the time where the temperature is seeked from the date passed
#       fetched_weather_forecast: the body of the request performed to the webservice
#Output: the predicted temperature at the time desired
def get_predicted_temperature_at_time(time_parsed, fetched_weather_forecast):
    #Get max temperature for the date requested
    max_temperature = fetched_weather_forecast["temperatureMax"]
    min_temperature = fetched_weather_forecast["temperatureMin"]
    #Mean temperature for the date requested
    avg_temperature = (max_temperature + min_temperature) / 2
    #Standard deviation for the date requested
    sequence_temperatures = [ i for i in range(int(min_temperature), int(max_temperature) + 1)]
    std_temperature = statistics.stdev(sequence_temperatures)
    #Generate as many numbers (12) in the range, so that they are distributed around the AVG. and follow the STD of 
    #the sequence [min_temperature, ..., max_temperature]
    #Values for the first part of the day (midnight to noon)
    random_temperature_1 = np.random.normal(avg_temperature, std_temperature, size=12)
    random_temperature_1_sorted = sorted(random_temperature_1)
    #Values for the second part of the day(noon to midnight)
    random_temperature_2 = np.random.normal(avg_temperature, std_temperature, size=12)
    random_temperature_2_sorted = sorted(random_temperature_2, reverse=True)
    #Temperature values successfully generated, with the highest temperature lying at about 12/13
    #0--> midnight (00:01)
    #1 --> 1 AM
    #17 --> 5 PM
    random_temperature_vals_merged = random_temperature_1_sorted + random_temperature_2_sorted   
    #Take the temperature corresponding to the time's hour queried.EX: 17:30 --> Get the temperature at time 17.
    temperature_predicted_at_time = random_temperature_vals_merged[ int(time_parsed.hour)]
    
    return(temperature_predicted_at_time)
    

#Input: rain_from_mm: the minimum level of rainfall on the input date
#		rain_to_mm: the maximum level of rainfall on the input date
def get_rain_log_normal_distribution(rain_from_mm, rain_to_mm):
    mean_rain_mm = (rain_from_mm + rain_to_mm) / 2
    
    #https://journals.ametsoc.org/doi/abs/10.1175/1520-0450%281994%29033%3C1486%3AAPDMFR%3E2.0.CO%3B2
    #A Probability Distribution Model for Rain Rate --> Log normal distribution suggested
    precipitation_mm = np.random.lognormal(mean=log(mean_rain_mm), sigma=0.5, size=None) * 0.01
    
    #Once again: let's roll a dice: 50% chance that it rains
    dice_roll = random.randrange(0, 11)
    
    if dice_roll <= 5:
       return(precipitation_mm)
    else:
    	return(0)


#Input: - time_parsed: the time passed as input
#       - fetched_weather_forecast: the body response of the webservice to the query for weather
#		- days_difference: the number of days of difference between the current date and the date for which a prediction is requested
#Output: the amount of rain in mm that is predicted to fall on the date passed as input at the time requested
def get_predicted_rain_at_time(time_parsed, fetched_weather_forecast, days_difference):
    
    #The following attributes could be potentially considered as well
    """
    reliability = fetched_weather_forecast["reliability"]    
    storms = fetched_weather_forecast["storms"]
    freeze = fetched_weather_forecast["freeze"]
    """
    rain_from_mm = fetched_weather_forecast["rainFrom"]
    rain_to_mm = fetched_weather_forecast["rainTo"]

    hour_parsed = time_parsed.hour
    #The probability that it will rain at the time specified
    probability_rain_time = -1
    #Asking for a prediction more than 3 days from today, then::
    #rain_timespan1 is the rain forecasted from 00:00 to 12:00
    #rain_timespan2 is the rain forecasted from 12:01 to 23:59
    
    #fetched weather does not have rainTimespan3 and 4:
    if (('rainTimespan3' not in fetched_weather_forecast.keys()) and ('rainTimespan4' not in fetched_weather_forecast.keys())):
         if hour_parsed >= 0 and hour_parsed <= 12:
            probability_rain_time =  fetched_weather_forecast["rainTimespan1"]
        #Time interval 2
         elif hour_parsed > 12 and hour_parsed <= 24:
            probability_rain_time =  fetched_weather_forecast["rainTimespan2"]
            
    elif 'rainTimespan1' in fetched_weather_forecast.keys() and 'rainTimespan2' in fetched_weather_forecast.keys() and 'rainTimespan3' in fetched_weather_forecast.keys() and 'rainTimespan4' in fetched_weather_forecast.keys():
        #Time interval 1:
        if hour_parsed >= 0 and hour_parsed <= 6:
            probability_rain_time =  fetched_weather_forecast["rainTimespan1"]
        #Time interval 2
        elif hour_parsed > 6 and hour_parsed <= 12:
            probability_rain_time =  fetched_weather_forecast["rainTimespan2"]
        #Time interval 3
        elif hour_parsed > 12 and hour_parsed <= 18:
            probability_rain_time = fetched_weather_forecast["rainTimespan3"]
        #Time interval 4
        elif hour_parsed > 18 and hour_parsed <= 24:
            probability_rain_time = fetched_weather_forecast["rainTimespan4"]
            
    #probability that it will rain at the time specified is 0, so there won't be any rain at that time
    if probability_rain_time == 0:
        precipitation_time_mm = 0
    #There is some chance that it will rain indeed
    else:
    	#Michael's approach
    	#precipitation_time_mm = get_rain_gamma_distribution(hour_parsed)
    	#My approach - use a log-normal distribution in the fromTo to the upTo interval
    	precipitation_time_mm = get_rain_log_normal_distribution(rain_from_mm, rain_to_mm)
      
    return(precipitation_time_mm)            
      
#Input: boolean_value: True | False
#Output: 0 if boolean_value is False
#        1 if boolean_value is True
def bool_to_int(boolean_value):
    return (int(boolean_value == True))

#Input:- weather_station_index:  #https://it.wikipedia.org/wiki/Burgraviato#/media/File:Comunità_comprensoriali_Alto_Adige.svg
    #1 --> Bolzano, Überetsch and Unterland / Bolzano, Oltradige e Bassa Atesina
    #2 --> Burggrafenamt - Meran and surroundings / Burgraviato - Merano e dintorni
    #3 --> Vinschgau / Val Venosta
    #4 --> Eisacktal and Sarntal / Val d'Isarco e Val Sarentino
    #5--> Wipptal - Sterzing and surroundings / Alta Val d'Isarco - Vipiteno e dintorni
    #6 --> Pustertal / Val Pusteria
    #7 --> Ladinia - Dolomites / Ladinia - Dolomiti (Val Gardena?)            
# - date_time_pred_input:  the date and time in STRING format. For example: "2019-11-13 18:30:00"
# - fetched_weather_forecast: the daily weather forecast obtained for the day
#Output: a data frame containing the features to be fed to the predictive model
def generate_features_for_date_time(weather_station_index, date_time_pred_input, fetched_weather_forecast):
    date_time_parsed = parse(date_time_pred_input)
    #Get number of days of difference between today and date_time_pred_input
    days_difference = get_days_difference_today(date_time_parsed)
    
    time_parsed = date_time_parsed.time()
    #Temperature
    temperature_predicted = get_predicted_temperature_at_time(time_parsed, fetched_weather_forecast)
   
    #Niederschlag / precipitazioni
    #Let's now get the level of precipitazioni for the date and time provided
    rain_predicted = get_predicted_rain_at_time(time_parsed, fetched_weather_forecast, days_difference) 
    
    #The day of the week
    week_day = date_time_parsed.weekday()
    
    #The hour for which we want a prediction
    hour_input = date_time_parsed.hour
    
    #Let's now put together all the values into a single data frame
    data_frame_input_rows = pd.DataFrame({"TEMPERATURE" : [temperature_predicted],
                                          "NIEDERSCHLAG" : [rain_predicted],
                                          "HOUR" : [hour_input],
                                          "WEEK_DAY" : [week_day]
                                            })
    
    return(data_frame_input_rows)
    
    
    

def deserialize_object_given_path(models_path, model_name, traffic_level_label, siti_codsito):
    full_name_model_path = models_path + "/" + model_name + "_" + traffic_level_label + "_" + siti_codsito + ".pck"
    
    model_loaded = load(open(full_name_model_path, 'rb'))
    
    return(model_loaded)


#Input: traffic_level_label = "TRAFFIC_1" | "TRAFFIC_2"
#       siti_codsito: the code of the current station
#       models_path: the path where the models are stored
#Output: - list_models: a list of models de-serialized from disk
def load_ensemble_models_from_disk(traffic_level_label, siti_codsito, models_path):
    simple_rf_model = deserialize_object_given_path(models_path, "simple_rf", traffic_level_label, siti_codsito)
    kn_model = deserialize_object_given_path(models_path, "kn_model", traffic_level_label, siti_codsito)
    opt_dec_tree = deserialize_object_given_path(models_path, "opt_dec_tree", traffic_level_label, siti_codsito)

    list_models = [simple_rf_model, kn_model, opt_dec_tree]
    
    return(list_models)


#Input: same as "load_ensemble_models_from_disk"
#Output: - list_objs: the objects required for predictions to be performed using the serialized Neural Network
def load_numeric_nn_model_scal_enc(traffic_level_label, siti_codsito, models_path):
    scaler_temperature = deserialize_object_given_path(models_path, "scaler_temperature", traffic_level_label, siti_codsito)
    scaler_niederschlag = deserialize_object_given_path(models_path, "scaler_niederschlag", traffic_level_label, siti_codsito)
    encoder_week_day = deserialize_object_given_path(models_path, "encoder_week_day", traffic_level_label, siti_codsito)
    encoder_hours = deserialize_object_given_path(models_path, "encoder_hours", traffic_level_label, siti_codsito)
    scaler_number_vehicles = deserialize_object_given_path(models_path, "scaler_number_vehicles", traffic_level_label, siti_codsito)
    
    nn_numeric_model = deserialize_object_given_path(models_path, "nn_numeric", traffic_level_label, siti_codsito)
    
    list_objs = [scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours, scaler_number_vehicles, nn_numeric_model]
    
    return(list_objs)

#Input: number_days_predictions, the number of days for which we want to generate features
#       index_weather_station, the index of the weather station to be passed
#Output: df_input_rows, data frame having the following columns:
#       "TEMPERATURE", "NIEDERSCHLAG", "HOUR", WEEK_DAY", "DAYS", "MONTHS", "YEARS"
def generate_input_rows_df(number_days_predictions, index_weather_station):
    date_today = date.today()
    
     #This data frame will host the rows to be passed as input to the models
    df_input_rows = pd.DataFrame({"TEMPERATURE" : [], "NIEDERSCHLAG" : [], "HOUR" : [],  "WEEK_DAY" : [], "DAY" : [], "MONTH" : [], "YEAR" : [], })
    
    #Firstly generate a data frame containing the input features
    for date_index in range(0, (number_days_predictions + 1)):
        fetched_weather_forecast = fetch_weather_forecast(index_weather_station, date_index, language="de")
        #The date for which we want a prediction
        date_predict = date_today +  timedelta(days=date_index)
        #let's iterate over the different hours of the following days and perform predictions for all of them 
        for h in range(0, 24):
            time_predict = str(h) + ":00:00"
            
            input_date_time = str(date_predict) + " " + str(time_predict)

            input_row_X = generate_features_for_date_time(index_weather_station, input_date_time, fetched_weather_forecast)
            
            input_date_time_parsed = parse(input_date_time)
            
            day_date_input = input_date_time_parsed.day
            month_date_input = input_date_time_parsed.month
            year_date_input = input_date_time_parsed.year
            
            input_row_X["DAY"] = day_date_input
            input_row_X["MONTH"] = month_date_input
            input_row_X["YEAR"] = year_date_input
            
            df_input_rows = df_input_rows.append(input_row_X, sort=False)
            
    return(df_input_rows)
    
    


#Input: - list_models_traffic_1: the list of all models generated for the 
#       - index_weather_station: the index of the weather station for which features are to be generated
#       - df_input_rows_X: the input features in a data frame format to be passed as input to the models
#Output:- y_labels_pred_traffic_1, the categorical y values predicted for traffic 1
#       - y_labels_pred_traffic_2, the categorical y values predicted for traffic 2
def perform_predictions_ensemble_next_days(list_models_traffic_1, list_models_traffic_2, df_input_rows_X):
    list_model_names = ["Optimized Decision Tree", "Simple Random Forest","Best KNN"]
    
    #now let's pass every single row as input to the models and perform ensemble predictions
    y_labels_pred_traffic_1 = perform_models_ensemble_prediction(df_input_rows_X, list_models_traffic_1, list_model_names)
    y_labels_pred_traffic_2 = perform_models_ensemble_prediction(df_input_rows_X, list_models_traffic_2, list_model_names)
    
    return(y_labels_pred_traffic_1, y_labels_pred_traffic_2)


#Input: - list_objects: the de-serialized objects loaded from disk
#       - df_input_rows_X: the features given as input for the current model
#Output: the number of vehicles (integer) predicted for the features given as input
def perform_numeric_nn_prediction(list_nn_objects, df_input_rows_X):
    scaler_temperature = list_nn_objects[0]
    scaler_niederschlag = list_nn_objects[1]
    encoder_week_day = list_nn_objects[2]
    encoder_hours = list_nn_objects[3]
    scaler_number_vehicles = list_nn_objects[4]
    seq_nn_model = list_nn_objects[5]
    
    X_input_data = preprocess_X_data_nn(df_input_rows_X, scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours)
    
    #Actually perform predictions using the model loaded in a scaled manner
    y_predicted_nums = seq_nn_model.predict(X_input_data)
    
    #Un-scale the outputs produced from the model
    y_predicted_nums_unscaled = scaler_number_vehicles.inverse_transform(y_predicted_nums).flatten("C")
    
    #Convert every single element in the array to an integer
    y_predicted_nums_unscaled = [int(y_pred_val) for y_pred_val in y_predicted_nums_unscaled]
    
    #Let's now apply the max function to avoid negative values
    
    y_predicted_nums_max = [ max(y_pred_val, 0) for y_pred_val in y_predicted_nums_unscaled]
    
    return(y_predicted_nums_max)


#Input: -list_objects_count_1: the objects loaded from disk for the count 1 models
#       -list_objects_count_2: the objects loaded from disk for the count 2 models
#       -df_input_rows_X: a data frame containing the data to be passed to the model for performing a prediction
#Output:-y_values_count_1, the numeric integer y values predicted for count 1
#        -y_values_count_2, the numeric integer y values predicted for count 2
def perform_predictions_numeric_nn_next_days(list_objects_count_1, list_objects_count_2,  df_input_rows_X):
    y_values_count_1 = perform_numeric_nn_prediction(list_objects_count_1, df_input_rows_X)
    y_values_count_2 = perform_numeric_nn_prediction(list_objects_count_2, df_input_rows_X)
            
    return(y_values_count_1, y_values_count_2)
    

def load_use_models_for_inference(folder_models="./d/Stations_Models_2018", folder_predictions="./d/Stations_Predictions",
                        folder_past="./d/Stations_Past_2018", path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv"):
    
    check_list_folders_exist([folder_models, folder_predictions, folder_past])
    check_list_files_exist([path_metadata])

    #Let's load up the directions and the weather
    df_tratte = pd.read_csv(path_metadata, sep=",",  encoding = "ISO-8859-1")

    #let's iterate over all the stations (tratte)
    for i in range(0,  len(df_tratte)):       
        #let's get the corresponding SITI_CODSITO and the corresponding input_data_Frame
        siti_codsito = get_siti_codsito_given_index(i, df_tratte)   
        input_data_frame = load_df_traffic_station(siti_codsito, input_path=folder_past)
        
        if input_data_frame is not None:
            display_info_about_station(i, siti_codsito, 1, df_tratte)
            
            #Let's load up the models generated for TRAFFIC_1 and TRAFFIC_2
            list_models_traffic_1 = load_ensemble_models_from_disk("TRAFFIC_1", siti_codsito, models_path=folder_models)
            list_models_traffic_2 = load_ensemble_models_from_disk("TRAFFIC_2", siti_codsito, models_path=folder_models)
            
            index_weather_station =  df_tratte["WEATHER_INDEX"][i]
            number_days_predictions = get_number_days_predictions(index_weather_station)

            #Generate the rows to be passed as input to the categorical and numerical models
            df_input_rows = generate_input_rows_df(number_days_predictions, index_weather_station)
            df_input_rows_X = df_input_rows[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
            
            y_labels_pred_traffic_1, y_labels_pred_traffic_2 = perform_predictions_ensemble_next_days(list_models_traffic_1, list_models_traffic_2, df_input_rows_X)
            
            #Let's load up the models and scalers generated for COUNT_1 and COUNT_2
            list_objects_count_1 = load_numeric_nn_model_scal_enc("COUNT_1", siti_codsito, models_path=folder_models)
            list_objects_count_2 = load_numeric_nn_model_scal_enc("COUNT_2", siti_codsito, models_path=folder_models)
            y_values_pred_count_1, y_values_pred_count_2 = perform_predictions_numeric_nn_next_days(list_objects_count_1, list_objects_count_2, df_input_rows_X)
            
            
            #Finally, let's format the output, putting together categorical and numerical outputs
            df_predictions_out = pd.DataFrame({"TRAFFIC_1" : y_labels_pred_traffic_1, "TRAFFIC_2" : y_labels_pred_traffic_1,
                                               "DAY" : df_input_rows["DAY"], "MONTH" : df_input_rows["MONTH"],
                                               "YEAR": df_input_rows["YEAR"], "HOUR" : df_input_rows["HOUR"],
                                               "COUNT_1" : y_values_pred_count_1, "COUNT_2" :y_values_pred_count_2 })
            

            #And let's output it...
            output_predictions_to_folder(df_predictions_out, siti_codsito, output_folder=folder_predictions)
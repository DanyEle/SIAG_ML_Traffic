# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:01:19 2019

@author: dgadler

This file contains methods that are used by The Train_ML_Models and Inference_ML_Models
files for loading and pre-processing data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import date
import datetime
import numpy as np
from collections import Counter
import os

from os import listdir
from os.path import isfile, join

#Input: - list_folders: a list of folder paths to be checked for their existence.
#       If the folders passed are relative, their existence is checked for in the
#       current directory.
#Output: - an error message is raised in case one folder in the list of folders passed does not exist
def check_list_folders_exist(list_folders):
    current_dir = os.getcwd()
    for folder in list_folders:
        if not os.path.isdir(folder):
            raise Exception("Folder [" + folder + "] does not exist in " + str(current_dir))
    #No exception raised...
    print("Input folders " + str(list_folders) +  " exist!")
            
            
#Input: - list_files: a list of files to be check for their existence. The files passed can be 
#       in either relative or absolute paths
#Output: - an error message is raised in case one file in the list of files does not exist
def check_list_files_exist(list_files):
    current_dir = os.getcwd()

    for file in list_files:
        if not os.path.isfile(file):
            raise Exception("File [" + file + "] does not exist in " + str(current_dir))
            
    print("Input files " + str(list_files) +  " exist!")


#Input: - i, an index lying in range (0, len(df_tratte))
#       - df_tratte, a data frame containing the tratte loaded from the disk
#Output: - SITI_CODSITO: the SITI_CODSITO corresponding to the tratta
def get_siti_codsito_given_index(i, df_tratte):
    tratta_row = df_tratte.iloc[i]
    
    siti_codsito = tratta_row["SITI_CODSITO"]
    
    if siti_codsito < 10:
        siti_codsito = "0000000" + str(siti_codsito)
    else:
        siti_codsito = "000000" + str(siti_codsito)
    
    return(siti_codsito)
    
    
    
#Input: - sito_codsito, an index integer in the range [00000002, 00000076] corresponding to the index of the traffic stations' data
#       - input_path: the path where files are hosted
#Output: - past_data_frame: a pandas data frame containing the input dataset in a 
#        None if the input data frame has no rows
def load_df_traffic_station(siti_codsito, input_path="./Stations_Past_2018"):
        station_file_path = str(input_path + "/" + siti_codsito + ".csv")
        past_data_frame = pd.read_csv(station_file_path)
        
        if len(past_data_frame) > 1:
            return(past_data_frame)
        
        #An exception would be raised if the file does not exist
    
#Input: - past_data_frame
#Output: - past_data_frame_extra  having extra columns MONTH, SEASON, HOLIDAY_IT, HOLIDAY_GER, HOUR, WEEKDAY
def preprocess_data_frame(past_data_frame):
    past_data_frame_extra = past_data_frame
    
    print("Adding information about weekday and hour")
    
    
    list_week_days = []
    list_hours = []
    
    #let's iterate over all dates in the rows of the input data frame
    for i in range(0, len(past_data_frame)):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(past_data_frame)))
        past_date_time_raw = past_data_frame.iloc[i, ]["DATE_HOURS"]
        past_date_hour_parsed = datetime.datetime.strptime(past_date_time_raw, '%y%m%d_%H')  
        past_date_parsed = past_date_hour_parsed.date()
        
        list_hours.append(past_date_hour_parsed.hour)
        list_week_days.append(past_date_parsed.weekday())

    
    past_data_frame_extra["WEEK_DAY"] = list_week_days
    past_data_frame_extra["HOUR"] = list_hours

    return(past_data_frame_extra)
    
            
            
#Input: X_data, a data frame having columns "TEMPERATURE", "NIEDERSCHLAG", "WEEK_DAY", "HOUR"
#Output: X_data_concat: the data frame pre-processed and ready to be passed to the NN, following train/test holdout
#       Categorical variables are turned into tensors and numerical variables are transformed into a range
#X_data_concat[0][0:23] --> one-hot encoded hours
#X_data_concat[0][24:31] --> one-hot encoded week days
#X_data_concat[0][32] --> scaled niederschlag
#X_data_concat[0][33] --> scaled temperature
def preprocess_X_data_nn(X_data, scaler_temperature=None, scaler_niederschlag=None, encoder_week_day=None, encoder_hours=None):        
    #Encode the categorical variables
    X_encoded_hours = encoder_hours.transform(X_data[["HOUR"]]).toarray() 
    X_encoded_week_day = encoder_week_day.transform(X_data[["WEEK_DAY"]]).toarray() 

    #Transform the numerical variables
    X_transformed_niederschlag = scaler_niederschlag.transform(X_data[["NIEDERSCHLAG"]])
    X_transformed_temperature = scaler_temperature.transform(X_data[["TEMPERATURE"]])

    #Let's now concatenate niederschlag and temperature column-wise
    X_data_concat =  np.hstack((X_encoded_hours, X_encoded_week_day,X_transformed_niederschlag,X_transformed_temperature))
    
    return(X_data_concat)
            
#Input: X_data
#Output: scaler_temperature, scaler_niederschlag, encoder_hours, encoder_week_day
#The scalers and encoders fitted to the input data
def fit_scalers_encoders(X_data):
     scaler_temperature = StandardScaler()
     scaler_niederschlag = StandardScaler()
     encoder_week_day = OneHotEncoder(categories="auto")
     encoder_hours = OneHotEncoder(categories="auto")
        
     scaler_temperature.fit(X_data[["TEMPERATURE"]])
     scaler_niederschlag.fit(X_data[["NIEDERSCHLAG"]])
     encoder_hours.fit( X_data[["HOUR"]])
     encoder_week_day.fit(X_data[["WEEK_DAY"]])
     
     return(scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours)


#Input: index_weather_station: the index corresponding to one of the 7 weather stations
#       used for prediction purposes in Südtirol/Alto Adige
#Output: the name of the weather station corresponding to the index            
def weather_index_int_to_name(index_weather_station, lang="it"):
    if lang == "de":
        switcher = {
                1: "Bozen, Überetsch und Unterland",
                2: "Burggrafenamt - Meran und Umgebung",
                3: "Vinschgau",
                4: "Eisacktal und Sarntal",
                5: "Wipptal - Sterzing und Umgebung",
                6: "Pustertal",
                7: "Ladinia - Dolomiten"
            }
    elif lang == "it":
         switcher = {
                1: "Bolzano, Oltradige e Bassa Atesina",
                2: "Burgraviato - Merano e dintorni",
                3: "Val Venosta",
                4: "Val d'Isarco e Val Sarentino",
                5: "Alta Val d'Isarco - Vipiteno e dintorni",
                6: "Val Pusteria",
                7: "Ladinia - Dolomiti"
            }
    
    weather_station_name = switcher.get(index_weather_station, "Invalid Weather Station Index. It must be within [1-7].")
    
    return(weather_station_name)
    
def display_info_about_station(i, siti_codsito, l, df_tratte):
    print("Loading up dataset  " + str(siti_codsito) + ".csv")
    
    index_weather_station = df_tratte["WEATHER_INDEX"][i]
    #From direction 1 to direction 2
    traffic_sensor_location_name = df_tratte["ABITATO_SITO_I"][i]
    weather_station_name = weather_index_int_to_name(index_weather_station, "it")

    if(l == 1):
        print("From " + df_tratte["DESCRIZIONE_I_1"][i] + " to " + df_tratte["DESCRIZIONE_I_2"][i])
    if(l == 2):
        print("From " + df_tratte["DESCRIZIONE_I_2"][i] + " to " + df_tratte["DESCRIZIONE_I_1"][i])
    
    print("Sensor location: " + str(traffic_sensor_location_name))
    
    print("Closest Weather station for predictions:" + weather_station_name)


#Input: - df_predictions, a data frame containing predictions to be output
#       - siti_codsito, the code of a sito
#       - output_folder: the path of the folder to which the data frame should be output
#Output: the df_predictions data frame is output to the output_folder/siti_codsito + "_predictions.csv" file
def output_predictions_to_folder(df_predictions, siti_codsito,output_folder="./Stations_Predictions_Data_2018"):
    file_name_output = siti_codsito + "_predictions.csv"
    
    full_path_output = output_folder + "/" +  file_name_output
    
     #Let's output the df_predictions to a file that will be parsed by the Flask webservice
    df_predictions.to_csv(full_path_output, index=False)
    
#Input: - X_data: the input data for which we want the corresponding labels to be predicted
#       - list_models: the list of models generated via process_data_generate_list_models
#Output: the most frequent y predictions using all models, row-wise on data_frame_y_predictions
def perform_models_ensemble_prediction(X_data, list_models, list_model_names):
    #Data frame hosting all models' predictions
    data_frame_y_predictions = pd.DataFrame()

    #Iterate over the  considered models and perform a prediction with each model
    for m in range(0, len(list_models)):
        y_predictions = list_models[m].predict(X_data)
        
        #if predictions are not in a list, then turn them into a list
        if isinstance(y_predictions, list) != "list":
            y_predictions = list(y_predictions)
            
        data_frame_y_predictions[list_model_names[m]] = y_predictions
        
    #Let's now find the most likely value in each row: that's our y prediction for that very row
    list_y_predictions_ensemble = []
    for row_index in range(0, len(y_predictions)):
        y_pred_most_frequent = Counter(data_frame_y_predictions.iloc[row_index]).most_common(1)[0][0]
        list_y_predictions_ensemble.append(y_pred_most_frequent)
    
    return(list_y_predictions_ensemble)
    
    
#Input: - list_strings: a list of strings
#Output:- list_strings is concatenated into a single string, where each string is separated
#         by means of a ":"
def list_strings_to_string(list_strings):  
    # initialize an empty string 
    str_return = ""  
    # traverse in the string   
    for string in list_strings:  
        str_return += string + ":"
    # return string   
    return str_return 



#########################METHODS USED IN THE 'app.py' file#################

#Input: list_dataora, a list of dataora strings, with each string having the following format:
#YYMMDD_HH. EX: "180101_00"
#Output: list_day, list_month, list_year, list_hours all of them in numeric integer format
#This function converts the input list, containining date & time in string format into four 
#numeric lists, each one containing the day, month, year and hour in numeric integer format, respectively
def list_dataora_to_numeric(list_dataora):
    list_days = []
    list_months = []
    list_years = []
    list_hours = []
    
    for i in range(0, len(list_dataora)):
        date_time_str = list_dataora[i]
        date_time_obj = datetime.datetime.strptime(date_time_str, '%y%m%d_%H')  
        #Let's extract the day, month, year and hour in numeric format and add them all to respective lists
        date_day = date_time_obj.day
        date_month = date_time_obj.month
        date_year = date_time_obj.year
        date_hour = date_time_obj.hour
        
        list_days.append(date_day)
        list_months.append(date_month)
        list_years.append(date_year)
        list_hours.append(date_hour)
        
    return(list_days, list_months, list_years, list_hours)



#Input: the df_tratte, loaded from the respective station
#Output: a list of lists containing the past data for the upcoming days
                #list_past_all_stations[i][0] --> TRAFFIC_1 at that hour,day,month and year for station i 
                #list_past_all_stations[i][1] --> TRAFFIC_2 at that hour,day,month and year for station i 
                #list_past_all_stations[i][2] --> the days for which past traffic is computed for station i 
                #list_past_all_stations[i][3] --> the months for which past traffic is computed for station i 
                #list_past_all_stations[i][4] --> the years for which past traffic is computed for station i 
                #list_past_all_stations[i][5] --> the hours for which past traffic is computed for station i 
                #list_past_all_stations[i][6] --> COUNT_1, the number of vehicles sensed in the past for station i in direction 1 
                #list_past_all_stations[i][7] --> COUNT_2, the number of vehicles sensed in the past for station i in direction 2 
def load_list_traffic_past_2018(folder_past="../d/Stations_Past_2018"):
    #will host a list of lists for all stations
    list_past_all_stations = []
    
    for i in range(2, 76): #2, 76
        
        if i < 10:
            station_file_name = "0000000" + str(i)
        else:
            station_file_name = "000000" + str(i)
        
        station_file_path = str(folder_past + "/" + station_file_name + ".csv")
        past_data_frame = pd.read_csv(station_file_path)
        
        if len(past_data_frame) > 1:
            
            print("Loading station [" + str(i) + "] " + str(len(past_data_frame)) + " observations.")
            
            list_days, list_months, list_years, list_hours = list_dataora_to_numeric(past_data_frame["DATE_HOURS"])
            past_data_frame["DAY"] = list_days
            past_data_frame["MONTH"] = list_months
            past_data_frame["YEAR"] = list_years
            past_data_frame["HOUR"] = list_hours
    
            past_data_frame_numeric = past_data_frame.drop(columns=["DATE_HOURS", "SCODE", "NIEDERSCHLAG", "TEMPERATURE"])     
            
            #Let's change the order of columns to reflect the same order of the predictions
            past_data_frame_numeric = past_data_frame_numeric[["TRAFFIC_1", "TRAFFIC_2", "DAY", "MONTH", "YEAR", "HOUR", "COUNT_1", "COUNT_2"]]
            
            #Let's now convert the data frame to a list of lists contained in list_past_data_frame
            list_past_data_frame = []
            #iterate over the columns of the data frame to product a list of lists
            for column_index in range(0, len(past_data_frame_numeric.columns)):
                list_past_data_frame.append(list(past_data_frame_numeric.iloc[:, column_index]))
                    
            list_past_all_stations.append(list_past_data_frame)         
         
    return(list_past_all_stations)


#Input: the df_tratte loaded from disk
#Output: a list of dictionaries (JSON format-like) ready to be plotted in Google Maps
#        containing in each entry of the list the following information:
#        list_markers_direzioni[i]["icon"] --> string, the transparent default icon
#        list_markers_direzioni[i]["lat"] --> float, the latitude of the marker
#        list_markers_direzioni[i]["lng"] --> float, the longitude of the marker
#        list_markers_direzioni[i]["infobox] --> string, the content of the infobox popping up with the marker
def produce_markers_from_tratte(df_tratte):
    list_markers_direzioni = []
    
    print("Loading markers from file...")
    
    for i in range(0, len(df_tratte)):
        #Let's get the full row for that direction
        tratta_row = df_tratte.iloc[i]
        
        icon_marker  = (r"./static/imgs/transparent_logo_resized.png")

        
        lat_marker = tratta_row["latitude"]
        lng_marker = tratta_row["longitude"]
        
        sensor_location_IT = tratta_row["ABITATO_SITO_I"]
        sensor_location_DE = tratta_row["ABITATO_SITO_D"]
        
        street_name_IT = tratta_row["FW_NAME_CUSTOM_I"]
        street_name_DE = tratta_row["FW_NAME_CUSTOM_D"]                
        
        direction_1_IT = tratta_row["DESCRIZIONE_I_1"]
        direction_1_DE = tratta_row["DESCRIZIONE_D_1"]
        direction_2_IT = tratta_row["DESCRIZIONE_I_2"]
        direction_2_DE = tratta_row["DESCRIZIONE_D_2"]
        
        weather_station_IT = tratta_row["NAME_I"]
        weather_station_DE = tratta_row["NAME_D"]
        
        
        infobox_marker = (
          " <p style=\"font-size:20px\"> <b> Location: </b>  " + sensor_location_IT + " / " + sensor_location_DE + "<br>"     
        + " <b> Street name:  </b> " +  street_name_IT  + " / " + street_name_DE + " <br> " 
        + " <b> Closest Weather station: </b> " + weather_station_IT + " / " + weather_station_DE  + " <br> "
        + " <b>Direction 1: </b> From: " + direction_2_IT + " / " + direction_2_DE + " To: "  + direction_1_IT + " / " + direction_1_DE + " <br>"
        + " <b>Direction 2: </b> From: " + direction_1_IT + " / " + direction_1_DE + " To: " + direction_2_IT + " / " + direction_2_DE + " <br>"
        + " <center><a href=./heatmap?station_id=" + str(i) + "> View Traffic Heatmap </a> </p> </center>")
        
        #Finally, put all the elements together into a dictionary
        marker_created = { 'icon' : icon_marker, 'lat' : lat_marker, 'lng' : lng_marker, 'infobox' : infobox_marker }
        #Add the marker to a list
        list_markers_direzioni.append(marker_created)
        
    return(list_markers_direzioni)
    
    
#Input: the path where predictions lie
#Output: a list of lists containing the predictions for the upcoming days
                #list_predictions_all_stations[i][0] --> TRAFFIC_1 at that hour,day,month and year  for station i 
                #list_predictions_all_stations[i][1] --> TRAFFIC_2 at that hour,day,month and year  for station i 
                #list_predictions_all_stations[i][2] --> the days for which predictions are computed for station i 
                #list_predictions_all_stations[i][3] --> the months for which predictions are computed for station i 
                #list_predictions_all_stations[i][4] --> the years for which predictions are computed for station i 
                #list_predictions_all_stations[i][5] --> the hours for which predictions are computed for station i 
                #list_predictions_all_stations[i][6] --> COUNT_1, the number of vehicles predicted for station i in direction 1 
                #list_predictions_all_stations[i][7] --> COUNT_2, the number of vehicles predicted for station i in direction 2 
#In the present function, we load up all the prediction files contained in 'Stations_Predictions' and turn them into a list of lists
def load_list_traffic_predictions(folder_predictions="../d/Stations_Predictions"):
    list_predictions_all_stations = []
    
    #let's load up all the files in the input_folder
    list_files_folder = [f for f in listdir(folder_predictions) if isfile(join(folder_predictions, f))]
    
    #For every single file containing predictions of a station...
    for file_predictions_station in list_files_folder:
        file_full_path = folder_predictions + "/" + file_predictions_station
        
        print("Loading up " + file_predictions_station)
        
        #Let's load up the content of the file
        df_predictions_station = pd.read_csv(file_full_path)
        
        list_predictions_data_frame = []
        #iterate over the columns of the data frame and turn the data frame into a list of lists
        for column_index in range(0, len(df_predictions_station.columns)):
            list_predictions_data_frame.append(list(df_predictions_station.iloc[:, column_index]))
        
        list_predictions_all_stations.append(list_predictions_data_frame)
        
    return(list_predictions_all_stations)

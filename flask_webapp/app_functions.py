from config_ml import (STORAGE_ACCOUNT_NAME,
                       STORAGE_ACCOUNT_KEY,
                       CONTAINER_INPUT_NAME,
                       CONTAINER_OUTPUT_NAME)

import datetime
import pandas as pd

from azure.storage.blob import BlockBlobService
from io import StringIO



def get_siti_codsito_given_index(i, df_tratte):
    tratta_row = df_tratte.iloc[i]
    
    siti_codsito = tratta_row["SITI_CODSITO"]
    
    if siti_codsito < 10:
        siti_codsito = "0000000" + str(siti_codsito)
    else:
        siti_codsito = "000000" + str(siti_codsito)
    
    return(siti_codsito)
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
    return (str_return )


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
def load_list_traffic_predictions(df_tratte):
    list_predictions_all_stations = []
    
    print("Loading predictions")
    for i in range(0,  len(df_tratte)):       
        print(str(i) + "/" + str(len(df_tratte)))
        #let's get the corresponding SITI_CODSITO and the corresponding input_data_Frame
        siti_codsito = get_siti_codsito_given_index(i, df_tratte)   
          
        #let's load up the CSV from Azure Blob storage
        blob_service_input = BlockBlobService(account_name=STORAGE_ACCOUNT_NAME, account_key=STORAGE_ACCOUNT_KEY)
        blob_string_input = blob_service_input.get_blob_to_text(container_name=CONTAINER_OUTPUT_NAME, blob_name=siti_codsito + ".csv").content
        df_predictions_station = pd.read_csv(StringIO(blob_string_input))
        
        if df_predictions_station is not None and len(df_predictions_station) > 1:
            print("Loading station [" + str(i) + "] " + str(len(df_predictions_station)) + " predictions.")
        
            list_predictions_data_frame = []
            #iterate over the columns of the data frame and turn the data frame into a list of lists
            for column_index in range(0, len(df_predictions_station.columns)):
                list_predictions_data_frame.append(list(df_predictions_station.iloc[:, column_index]))
        
        list_predictions_all_stations.append(list_predictions_data_frame)
        
    return(list_predictions_all_stations)


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
def load_list_traffic_past_2018(df_tratte):
    #will host a list of lists for all stations
    list_past_all_stations = []
    
    for i in range(0,  len(df_tratte)):       
        #let's get the corresponding SITI_CODSITO and the corresponding input_data_Frame
        siti_codsito = get_siti_codsito_given_index(i, df_tratte)   
          
        #let's load up the CSV from Azure Blob storage
        blob_service_input = BlockBlobService(account_name=STORAGE_ACCOUNT_NAME, account_key=STORAGE_ACCOUNT_KEY)
        blob_string_input = blob_service_input.get_blob_to_text(container_name=CONTAINER_INPUT_NAME,blob_name=siti_codsito + ".csv").content
        input_data_frame = pd.read_csv(StringIO(blob_string_input))
        
        if input_data_frame is not None and len(input_data_frame) > 1:
            
            print("Loading station [" + str(i) + "] " + str(len(input_data_frame)) + " 2018 past observations.")
            
            list_days, list_months, list_years, list_hours = list_dataora_to_numeric(input_data_frame["DATE_HOURS"])
            input_data_frame["DAY"] = list_days
            input_data_frame["MONTH"] = list_months
            input_data_frame["YEAR"] = list_years
            input_data_frame["HOUR"] = list_hours
    
            past_data_frame_numeric = input_data_frame.drop(columns=["DATE_HOURS", "SCODE", "NIEDERSCHLAG", "TEMPERATURE"])     
            
            #Let's change the order of columns to reflect the same order of the predictions
            past_data_frame_numeric = past_data_frame_numeric[["TRAFFIC_1", "TRAFFIC_2", "DAY", "MONTH", "YEAR", "HOUR", "COUNT_1", "COUNT_2"]]
            
            #Let's now convert the data frame to a list of lists contained in list_past_data_frame
            list_past_data_frame = []
            #iterate over the columns of the data frame to product a list of lists
            for column_index in range(0, len(past_data_frame_numeric.columns)):
                list_past_data_frame.append(list(past_data_frame_numeric.iloc[:, column_index]))
                    
            list_past_all_stations.append(list_past_data_frame)         
         
    return(list_past_all_stations)
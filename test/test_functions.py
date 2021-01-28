#Let's import the source files
import pandas as pd
import pytest
import os
from contextlib import contextmanager

from s.functions import (check_list_files_exist, check_list_folders_exist, get_siti_codsito_given_index,
                         load_df_traffic_station, preprocess_data_frame, fit_scalers_encoders, 
                         preprocess_X_data_nn, weather_index_int_to_name, output_predictions_to_folder,
                         list_strings_to_string, list_dataora_to_numeric, load_list_traffic_past_2018,
                         display_info_about_station, produce_markers_from_tratte, load_list_traffic_predictions)


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

#Input: - list_all_stations: (list) list_predictions_all_stations | list_past_all_stations
#       - column_index : (integer) the index of the column to be checked 
#       - lower_bound (integer), the value for which we want to check every single number in the list of lists to be within
#       - upper_bound (integer)
#Output: AssertionError if any value in any list of the list of lists is found not be in the range [lower_bound, upper_bound]
def check_all_nums_columns_in_range(list_all_stations, column_index, lower_bound, upper_bound):
    #Let's get all lists at position 'column_index' in all stations
    list_list_numbers = [[number for number in list_station[column_index]] for list_station in list_all_stations]
    
    #Let's flatten the lists of lists resulting
    flattened_list_numbers = [val for sublist in list_list_numbers for val in sublist]
    
    assert(all(is_int_valid_in_range(number, lower_bound, upper_bound) == True for number in flattened_list_numbers))

#Input: - list_all_stations: (list) list_predictions_all_stations | list_past_all_stations
#       - column_index : (integer) the index of the column to be checked 
#Output: AssertionError if any value in any list of the list of lists is found not to be positive
def check_all_nums_columns_positive(list_all_stations, column_index):
    #Let's get all lists at position 'column_index' in all stations
    list_list_numbers = [[number for number in list_station[column_index]] for list_station in list_all_stations]
    
    #Let's flatten the lists of lists resulting
    flattened_list_numbers = [val for sublist in list_list_numbers for val in sublist]
    
    def test_funct(number):
        print(number)
        return number >= 0
    
    assert(all(test_funct(number)  for number in flattened_list_numbers))
        
#Input: - integer, a numeric integer value
#       - lower_bound, a numeric integer 
#       - upper_bound, a numeric integer
#True if integer in [lower_bound, upper_bound)
#False if integer is not in [lower_bound, upper_bound)
def is_int_valid_in_range(integer, lower_bound, upper_bound):
    if integer in list(range(lower_bound, upper_bound)):
        return True
    else:
        return False


#Check if the 'check_list_folders_exist' method does indeed work
def test_check_list_folders_exist():
    #Let's check if an exception is raised when checking for a non-existing folder 
    with pytest.raises(Exception):
        check_list_folders_exist(["./d/MyFolder", "./d/InvalidFolder"])
    
    #Let's check if no exception is raised when checking for the folders that should exist
    with not_raises(Exception):
        check_list_folders_exist(["./d/Stations_Predictions", "./d/Stations_Past_2018", "./d/Stations_Models_2018"])
        
    
#Check if the 'check_list_files_exist' method does indeed work
def test_check_files_exist():
    #Let's check if an exception is raised when checking for a non-existing file
    with pytest.raises(Exception):
        check_list_files_exist(["./d/DummyFile.dmp", "./d/DummyFile2.dmb"])
    
    #Let's check if no exception is raised when checking for an existing folder
    with not_raises(Exception):
        check_list_files_exist(["./d/data_frame_tratte_meteo_suedtirol_fixed.csv"])
        
#TODO: Check if all the dependencies are satisfied in terms of datasets and models
        
#Test for 'get_siti_codsito_given_index' method 
def test_get_siti_codsito_given_index():
    
    #load up the df_tratte
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")

    assert(df_tratte is not None)
    
    #single-digit siti_codsito, first one in the folder
    assert(get_siti_codsito_given_index(0, df_tratte) == "00000002")
    
    #two-digit siti_codsito
    assert(get_siti_codsito_given_index(11, df_tratte) == "00000013")

#Test for loading up a data frame containing a 2018 dataset
def test_load_df_traffic_station():
    
    ###Test for loading up a valid dataset
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    
    #Let's get the first siti_codsito
    siti_codsito_init = get_siti_codsito_given_index(0, df_tratte)
    
    input_data_frame = load_df_traffic_station(siti_codsito_init, input_path="./d/Stations_Past_2018")
    
    #Check if a data frame was indeed loaded up
    assert(type(input_data_frame) == pd.core.frame.DataFrame)
    
    #Let's see if the loaded dataframe has more than 1 row
    assert(len(input_data_frame) > 1)
    
    ###Test for loading up a file in a non-existing folder
    with pytest.raises(FileNotFoundError):
        input_data_frame = load_df_traffic_station(siti_codsito_init, input_path="./d/Stations_Dummy_2018")

#Test 
def test_preprocess_data_frame():
    
    #let's see if we can correctly preprocess a data frame and add meta-data information to it
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    
    #Let's get the first siti_codsito
    siti_codsito_init = get_siti_codsito_given_index(0, df_tratte)
    
    input_data_frame = load_df_traffic_station(siti_codsito_init, input_path="./d/Stations_Past_2018")
    
    preprocessed_data_frame = preprocess_data_frame(input_data_frame)
    
    display_info_about_station(0, "000000002", 1, df_tratte)
    
    display_info_about_station(0, "000000002", 2, df_tratte)

    
    #the preprocessed data frame has more than 1 row
    assert(len(preprocessed_data_frame) > 1)
    
    #the number of rows in the pre-processed data frame is the same as in the original loaded data frame
    assert(len(preprocessed_data_frame) == len(input_data_frame))
    
    #Let's check if the columns "WEEK_DAY" and "HOUR" do exist in the preprocessed data frame
    assert("WEEK_DAY" in preprocessed_data_frame.columns)
    
    assert("HOUR" in preprocessed_data_frame.columns)

#Test for 'fit_scalers_encoders and 
#for 'preprocess_X_data_nn'
def test_fit_scalers_encoders():
    
    #let's load up one dummy dataset
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    #Let's get the first siti_codsito
    siti_codsito_init = get_siti_codsito_given_index(0, df_tratte)
    input_data_frame = load_df_traffic_station(siti_codsito_init, input_path="./d/Stations_Past_2018")
    
    traffic_X_preprocessed = preprocess_data_frame(input_data_frame)
    
    #Let's get the X data only
    traffic_X_df = traffic_X_preprocessed[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    
    #create scalers and encoders for X data
    scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours = fit_scalers_encoders(traffic_X_df)
    
    assert(scaler_temperature != None and scaler_niederschlag != None and encoder_week_day != None and encoder_hours != None )
    
    #let's apply the scaler to the X data
    
    X_data_concat = preprocess_X_data_nn(traffic_X_df, scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours)
    
    #Let's check if the X data preprocessed has the same length as the input X df
    assert(len(X_data_concat) == len(traffic_X_df))
    
    #Let's check if the scalers have been successfully applied, namely if:
    
    #HOURS are categorical --> one-hot encoded. So only one value must be 1 among them
    assert(sum(X_data_concat[0][0:24]) == 1)
    
    #Let's check if the hour has been correctly encoded, namely midnight is correctly encoded to [1,0,0,......0]
    assert(X_data_concat[0][0] == 1 and traffic_X_df.iloc[0]["HOUR"] == 0)
    
    #WEEK_DAY are categorical --> one-hot encoded. So only one value must be 1 among them
    assert(sum(X_data_concat[0][24:31]) == 1)
    
    #Let's check if the WEEK_DAY has been correctly encoded, namely monday is correctly encoded to [1,0,0,......0]
    assert(X_data_concat[0][24] == 1 and traffic_X_df.iloc[0]["WEEK_DAY"] == 0)
    
    #NIEDERSCHLAG is numeric --> Check if it has been Z-scaled properly to a known value
    assert(X_data_concat[0][31] < -0.144774 and X_data_concat[0][31] > -0.144775)
    
    #TEMPERATURE is numeric --> check if it has been Z-scaled properly to a known value
    assert(X_data_concat[0][32] < -1.403439 and X_data_concat[0][32] > -1.40344)

#Test for weather_index_int_to_name
def test_weather_index_int_to_name():
    
    #let's test italian names
    assert(weather_index_int_to_name(1, "it") == "Bolzano, Oltradige e Bassa Atesina")
    
    assert(weather_index_int_to_name(7, "it") == "Ladinia - Dolomiti")
    
    #Let's test german names
    assert(weather_index_int_to_name(1, "de") == "Bozen, Überetsch und Unterland")
    
    assert(weather_index_int_to_name(7, "de") == "Ladinia - Dolomiten")
    
    #let's see if an exception is indeed raised if an invalid language is passed
    with pytest.raises(Exception):
        weather_index_int_to_name(7, "CN")

        
#Test for output_predictions_to_folder
def test_output_predictions_to_folder():
    dummy_df_output = pd.DataFrame({"A": [1,2,3], "B" : [1,2,3]})
    
    #let's output the df to the current folder
    output_predictions_to_folder(dummy_df_output, "test", output_folder="./d/")
    
    #Let's see if the file was successfully created
    with not_raises(Exception):
        check_list_files_exist(["./d/test_predictions.csv"])
        
    #let's clean up and remove the output file now
    os.remove("./d/test_predictions.csv")
    
#Test for list_to_string
def test_list_to_string():
    #Let's see if the method works for integers
    test_list_strings = ["a", "b", "c"]
    
    assert("a:b:c:" == list_strings_to_string(test_list_strings))
    

        
#Test for list_dataora_to_numeric
def test_list_dataora_to_numeric():
    
    #Input: integer, an integer 
    #Output: - True if integer lies in [lower_bound , upper_bound), where the upper bound is not included
    #        - False if integer does not lie in [lower_bound, upper_bound]
    
    #let's try to load up a dummy input dataset for past 2018
    station_file_name = "00000002.csv"
    folder_past="./d/Stations_Past_2018"
    
    station_file_path = str(folder_past + "/" + station_file_name)
    past_data_frame = pd.read_csv(station_file_path)
    
    #let's check if the input data frame has been successfully loaded
    assert(len(past_data_frame) > 1)
    
    list_days, list_months, list_years, list_hours = list_dataora_to_numeric(past_data_frame["DATE_HOURS"])

    #let's see if the same number of elements exists in the new lists
    assert(len(list_days) == len(past_data_frame) and len(list_months) == len(past_data_frame) and len(list_years) == len(past_data_frame) and len(list_hours) == len(past_data_frame))

    #let's see if there is a valid number of elements
    assert(len(list_days) > 1 and len(list_months) > 1 and len(list_years) > 1 and len(list_hours) > 1)

    #let's check if all the years are 2018
    assert(all(year == 2018 for year in list_years))
    
    #let's check if the days generated are valid
    assert(all(is_int_valid_in_range(day, 0, 32) == True for day in list_days))
    
    #Let's check if the months generated are valid
    assert(all(is_int_valid_in_range(month, 1, 13) == True for month in list_months))
    
    #Let's check if the hours generated are valid
    assert(all(is_int_valid_in_range(month, 0, 25) == True for month in list_months))


#test for load_list_traffic_past_2018
def test_load_list_traffic_past_2018():
    
    #Let's see what happens if we pass an invalid input folder
    with pytest.raises(FileNotFoundError):
        load_list_traffic_past_2018("./d/Invalid_Folder")

    #Let's see if past stations are indeed loaded when a valid folder is passed
    list_past_all_stations = load_list_traffic_past_2018("./d/Stations_Past_2018")

    #A correct number of files was loaded?
    assert(len(list_past_all_stations) == 71)
    
    #Let's check how many columns every single loaded data frame has
    assert(len(list_past_station) == 8 for list_past_station in list_past_all_stations)
    
    #Let's see if the correct number of elements is present in a loaded data frame
    assert(len(list_past_all_stations[0][0]) == 8736) 
    assert(len(list_past_all_stations[0][1]) == 8736) 
    assert(len(list_past_all_stations[0][2]) == 8736) 
    assert(len(list_past_all_stations[0][3]) == 8736) 
    assert(len(list_past_all_stations[0][4]) == 8736) 
    assert(len(list_past_all_stations[0][5]) == 8736) 
    assert(len(list_past_all_stations[0][6]) == 8736) 
    assert(len(list_past_all_stations[0][7]) == 8736) 
    
    #TRAFFIC_1 of all stations
    check_all_nums_columns_in_range(list_past_all_stations, 0, 0, 5)
    
    #TRAFFIC_2 of all stations
    check_all_nums_columns_in_range(list_past_all_stations, 1, 0, 5)
    
    #DAYS of all stations
    check_all_nums_columns_in_range(list_past_all_stations, 2, 1, 32)

    #months
    check_all_nums_columns_in_range(list_past_all_stations, 3, 1, 13)

    #years for all stations
    check_all_nums_columns_in_range(list_past_all_stations, 4, 2000, 3000)

    #hours for all stations    
    check_all_nums_columns_in_range(list_past_all_stations, 5, 0, 24)
    
    #COUNT_1 --> Let's see if all of them are positive
    check_all_nums_columns_positive(list_past_all_stations, 6)
    
    #COUNT_2
    check_all_nums_columns_positive(list_past_all_stations, 7)
    
def test_produce_markers_from_direzioni():
    
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    
    list_markers_direzioni = produce_markers_from_tratte(df_tratte)
    
    assert(len(df_tratte) == len(list_markers_direzioni))
    
    #Let's see if the tratte are contained in 'list_markers_direzioni'
    for i in range(0, len(df_tratte)):
        #Let's get all the data corresponding to this 'tratta'
        tratta_row = df_tratte.iloc[i]
        
        #Let's now get all the content of the marker generated for this 'tratta'
        marker_tratta = list_markers_direzioni[i]
        
        #icon set correctly in marker?
        assert("/static/imgs/transparent_logo_resized.png" in marker_tratta["icon"])
        #longitude set correctly in marker?
        assert(tratta_row["longitude"] == marker_tratta["lng"])
        #latitude set correctly?
        assert(tratta_row["latitude"] == marker_tratta["lat"])
        #Sensor location in the proper infobox?
        assert(tratta_row["ABITATO_SITO_I"] in marker_tratta["infobox"])
        assert(tratta_row["ABITATO_SITO_D"] in marker_tratta["infobox"])
        #Starting and ending sensor position is in the infobox?
        assert(tratta_row["DESCRIZIONE_I_1"] in marker_tratta["infobox"])
        assert(tratta_row["DESCRIZIONE_D_1"] in marker_tratta["infobox"])
        assert(tratta_row["DESCRIZIONE_I_2"] in marker_tratta["infobox"])
        assert(tratta_row["DESCRIZIONE_D_2"] in marker_tratta["infobox"])
        #Street name in the infobox?
        assert(tratta_row["FW_NAME_CUSTOM_I"] in marker_tratta["infobox"])
        assert(tratta_row["FW_NAME_CUSTOM_D"] in marker_tratta["infobox"])
        #Station index set correctly?
        assert(str(i) in marker_tratta["infobox"])
    

def test_load_list_traffic_predictions():
    
    #Let's try to load up predictions from an invalid folder
    with pytest.raises(FileNotFoundError):
        load_list_traffic_past_2018("./d/Invalid_Folder")
    
    #Let's load up the traffic predictions
    list_predictions_all_stations = load_list_traffic_predictions(folder_predictions="./d/Stations_Predictions")
    
    #Let's check if a correct number of columns is present for every single station
    assert(len(list_predictions_station) == 8 for list_predictions_station in list_predictions_all_stations)
    
    #Let's check if the TRAFFIC_1 and TRAFFIC_2 columns host valid values
    check_all_nums_columns_in_range(list_predictions_all_stations, 0, 0, 5)
    
    #TRAFFIC_2 of all stations
    check_all_nums_columns_in_range(list_predictions_all_stations, 1, 0, 5)
    
    #DAYS of all stations
    check_all_nums_columns_in_range(list_predictions_all_stations, 2, 1, 32)

    #months
    check_all_nums_columns_in_range(list_predictions_all_stations, 3, 1, 13)

    #years for all stations
    check_all_nums_columns_in_range(list_predictions_all_stations, 4, 2000, 3000)

    #hours for all stations    
    check_all_nums_columns_in_range(list_predictions_all_stations, 5, 0, 24)
    
    #COUNT_1 --> Let's see if all of them are positive
    check_all_nums_columns_positive(list_predictions_all_stations, 6)
    
    #COUNT_2 --> Let's see if all of them are positive indeed
    check_all_nums_columns_positive(list_predictions_all_stations, 7)
    

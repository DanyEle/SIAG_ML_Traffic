import os
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from contextlib import contextmanager

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.models  import Sequential
from keras.layers import Dense 
from keras.optimizers import  Adam

from collections import Counter


from s.train_ml_models import (compute_train_test_accuracy, simple_decision_tree_model,
            optimized_decision_tree_model, simple_random_forest_model,
             knn_model, plot_train_valid_loss, random_resample_from_categories, 
             serialize_object, train_models_categorical_output,
            train_nn_model_numerical_output, compute_percentage_error,init_train_models,
            compute_accuracy_models_ensemble)

from s.functions import (get_siti_codsito_given_index, load_df_traffic_station, preprocess_data_frame,
                         check_list_files_exist, fit_scalers_encoders, preprocess_X_data_nn)


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

def load_input_data_frame(traffic_level_label):
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")

    #let's load up the dataset for ora , i.e., the first dataset existing
    siti_codsito = get_siti_codsito_given_index(0, df_tratte)   
    input_data_frame = load_df_traffic_station(siti_codsito, input_path="./d/Stations_Past_2018")
    input_data_frame_processed = preprocess_data_frame(input_data_frame)
    input_data_frame = input_data_frame_processed.dropna()
    
    assert(len(input_data_frame) > 0)
    
    #Let's check if there are indeed no NAs
    assert(sum(len(input_data_frame) - input_data_frame.count()) == 0)
    
    traffic_y_df = input_data_frame[[traffic_level_label]]
    traffic_X_df = input_data_frame[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    #Our data suffers from data unbalancing. Let's re-balance it via random resampling
    traffic_X_df_balanced, traffic_y_df_balanced = random_resample_from_categories(traffic_X_df, traffic_y_df, traffic_level_label)
    
    return(traffic_X_df, traffic_y_df, traffic_X_df_balanced, traffic_y_df_balanced)
    
    
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


#Let's try to create models on the dataset of ora
def test_random_resampling():
    
    #let's perform random resampling for TRAFFIC_1 and TRAFFIC_2
    list_traffic_level_labels = ["TRAFFIC_1", "TRAFFIC_2"]
    
    for traffic_level_label in list_traffic_level_labels:
        
        print(traffic_level_label)
        
        traffic_X_df, traffic_y_df, traffic_X_df_balanced, traffic_y_df_balanced = load_input_data_frame(traffic_level_label)
     
        #let's check if the count of class-wise Y labels before resampling is different from the count after resampling
        assert(Counter(traffic_y_df_balanced[traffic_level_label]) != Counter(traffic_y_df[traffic_level_label]))
        
        #let's check if the X and Y data re-balanced is different from the X data before re-balancing

        assert(not(traffic_y_df_balanced.equals(traffic_y_df)))
        assert(not(traffic_X_df_balanced.equals(traffic_X_df)))
        
        #let's see if there is still the same number of rows in the y data
        assert(len(traffic_y_df_balanced[traffic_level_label]) == len(traffic_y_df[traffic_level_label]))
        #and same number of rows in the re-balanced X data
        assert(len(traffic_X_df_balanced) == len(traffic_X_df))
    
#Test for the train_test_split() method
def test_categorical_model_creation():
    traffic_level_label = "TRAFFIC_1"
    
    traffic_X_df, traffic_y_df, traffic_X_df_balanced, traffic_y_df_balanced = load_input_data_frame("TRAFFIC_1")

    X_train, X_test, y_train, y_test = train_test_split(traffic_X_df_balanced, traffic_y_df_balanced, test_size=0.2,random_state=101)

    #Successful decision tree created?
    simple_dec_tree = simple_decision_tree_model(X_train, X_test, y_train, y_test)
    
    assert(not(simple_dec_tree == None))
    assert(type(simple_dec_tree) == DecisionTreeClassifier)
    
    #Let's test the method 'compute_train_test_accuracy' method
    acc_train_simp_dec_tree, acc_test_simp_dec_tree = compute_train_test_accuracy(simple_dec_tree, X_train, X_test, y_train, y_test, "Simple Dec Tree")
    assert(acc_train_simp_dec_tree > 0)
    assert(acc_test_simp_dec_tree > 0)
    
    #Let's test the 'optimized_decision_tree_model' method
    optimized_dec_tree, acc_test_opt_dec_tree = optimized_decision_tree_model(X_train, X_test, y_train, y_test, simple_dec_tree)
    assert(type(optimized_dec_tree) == DecisionTreeClassifier)
    assert(not(optimized_dec_tree == None))
    #Let's test the simple random forest model creation
    #Successful simple random forest creation
    simple_rf_model, acc_test_simple_rf_model = simple_random_forest_model(X_train, X_test, y_train, y_test)
    assert(type(simple_rf_model) == RandomForestClassifier)

    assert(not(optimized_dec_tree == None))
    assert(acc_test_simple_rf_model > 0)
    
    #Let's now test for KNN model creation
    kn_model, acc_test_knn_model = knn_model(X_train, X_test, y_train, y_test, max_n=5)
    assert(type(kn_model) == KNeighborsClassifier)
    assert(not(kn_model == None))
    assert(acc_test_knn_model > 0)
    
    #Let's test model serialization
    serialize_object("./d/", simple_rf_model, "simple_rf", traffic_level_label, "00000002")
    serialize_object("./d/", kn_model, "kn_model", traffic_level_label, "00000002")
    serialize_object("./d/", optimized_dec_tree, "opt_dec_tree", traffic_level_label, "00000002")
    
    #let's check if the models have indeed been written to disk
    with not_raises(Exception):
        check_list_files_exist(["./d/kn_model_TRAFFIC_1_00000002.pck", "./d/opt_dec_tree_TRAFFIC_1_00000002.pck",
                                "./d/simple_rf_TRAFFIC_1_00000002.pck"])
    
    #in that case, let's remove the models from the disk
    os.remove("./d/kn_model_TRAFFIC_1_00000002.pck")
    os.remove("./d/opt_dec_tree_TRAFFIC_1_00000002.pck")
    os.remove("./d/simple_rf_TRAFFIC_1_00000002.pck")
    
    #let's test accuracy computation
    list_models = [optimized_dec_tree, simple_rf_model, kn_model]
    
    compute_accuracy_models_ensemble(list_models, traffic_y_df, X_train, X_test, y_train, y_test)
    
#Test for "train_models_categorical_output", which uses the methods listed in 'test_categorical_method_creation'
#for TRAFFIC_1 and for TRAFFIC_2
def test_train_models_categorical_output():
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")

    #let's load up the dataset for ora , i.e., the first dataset existing
    siti_codsito = get_siti_codsito_given_index(0, df_tratte)   
    input_data_frame = load_df_traffic_station(siti_codsito, input_path="./d/Stations_Past_2018")
    
    input_data_frame_preprocessed = preprocess_data_frame(input_data_frame)
    
    list_traffic_labels = ["TRAFFIC_1", "TRAFFIC_2"]
    
    for traffic_level_label in list_traffic_labels:
        list_models = train_models_categorical_output(input_data_frame_preprocessed, traffic_level_label, "00000002", "./d/")
        
        assert(type(list_models[0]) == DecisionTreeClassifier)
        assert(type(list_models[1]) == RandomForestClassifier)
        assert(type(list_models[2]) == KNeighborsClassifier)
        
        #let's check if the models have been indeed successfully created
        assert(not(list_models == None))
        assert(len(list_models) == 3)
        
        #let's delete the created models
        with not_raises(Exception):
            check_list_files_exist(["./d/kn_model_" + traffic_level_label + "_00000002.pck",
                                    "./d/opt_dec_tree_" + traffic_level_label + "_00000002.pck",
                                    "./d/simple_rf_" + traffic_level_label + "_00000002.pck"])
    
        os.remove("./d/kn_model_" + traffic_level_label + "_00000002.pck")
        os.remove("./d/opt_dec_tree_" + traffic_level_label  + "_00000002.pck")
        os.remove("./d/simple_rf_" + traffic_level_label + "_00000002.pck")
        

#Test for the scalers and encoders methods for numeric NN
def test_nn_numeric_scalers_encoders():
    traffic_num_label = "COUNT_1"
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    #let's load up the dataset for ora , i.e., the first dataset existing
    siti_codsito = get_siti_codsito_given_index(0, df_tratte)   
    
    input_data_frame = load_df_traffic_station(siti_codsito, input_path="./d/Stations_Past_2018")
    input_data_frame_processed = preprocess_data_frame(input_data_frame)
        
    traffic_y_df_numeric = input_data_frame_processed[[traffic_num_label]]
    traffic_X_df = input_data_frame_processed[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    
    #let's check that all values in the y labels are indeed numeric
    assert(all(isinstance(y_label, int) for y_label in traffic_y_df_numeric[traffic_num_label]))
    #let's check that all values in the y labels are > 0
    assert(all(y_label >= 0 for y_label in traffic_y_df_numeric[traffic_num_label]))
    
    assert(len(traffic_X_df) == len(input_data_frame) and len(traffic_y_df_numeric) == len(input_data_frame))

    #Test if the 'fit_scalers_encoders' method does indeed work
    scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours = fit_scalers_encoders(traffic_X_df)
    
    #let's see if the scalers have been successfully generated
    scaler_number_vehicles = MinMaxScaler()
    scaler_number_vehicles.fit(traffic_y_df_numeric)
    
    assert(scaler_temperature != None and scaler_niederschlag != None and encoder_week_day != None and encoder_hours != None and scaler_number_vehicles != None)
    assert(type(scaler_temperature) == StandardScaler and type(scaler_niederschlag) == StandardScaler and type(encoder_week_day) == OneHotEncoder and type(encoder_hours) == OneHotEncoder and type(scaler_number_vehicles) == MinMaxScaler)
    
    #Now let's use the scalers and encoders for scaling & encoding the input data
    traffic_y_df_scaled_balanced = scaler_number_vehicles.transform(traffic_y_df_numeric).flatten("C")
    
    #Let's check if the shape of the scaled y labels is correct
    assert(traffic_y_df_scaled_balanced.shape[0] == len(input_data_frame) and len(traffic_y_df_scaled_balanced) == len(input_data_frame))
    assert(traffic_y_df_scaled_balanced.ndim == 1)
    
    #Let's see if the values in the scaled y labels are indeed between 0 and 1
    assert(all(0 <= y_label <= 1 for y_label in  traffic_y_df_scaled_balanced) == True)
    
def test_nn_numeric_model_creation():
    #Data loading - Ora Nord.
    traffic_num_label = "COUNT_1"
    
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    siti_codsito = get_siti_codsito_given_index(0, df_tratte)   
    input_data_frame = load_df_traffic_station(siti_codsito, input_path="./d/Stations_Past_2018")

    input_data_frame_processed = preprocess_data_frame(input_data_frame)
    traffic_X_df = input_data_frame_processed[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    traffic_y_df_numeric = input_data_frame[[traffic_num_label]]
    
    #Scalers and encoders    
    scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours = fit_scalers_encoders(traffic_X_df)
    scaler_number_vehicles = MinMaxScaler()
    scaler_number_vehicles.fit(traffic_y_df_numeric)
    traffic_y_df_scaled_balanced = scaler_number_vehicles.transform(traffic_y_df_numeric).flatten("C")
    traffic_X_df_processed = preprocess_X_data_nn(traffic_X_df, scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours)
    
    #Holdout 80-20
    X_train, X_test, y_train, y_test = train_test_split(traffic_X_df_processed, traffic_y_df_scaled_balanced, test_size=0.2,random_state=101)    
    
    #Neural Network model generation
    nn_model = Sequential()
    nn_model.add(Dense(100, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
    nn_model.add(Dense(100, activation="relu"))
    nn_model.add(Dense(100, activation="relu"))
    nn_model.add(Dense(1, kernel_initializer='normal') )
    # Compile model
    nn_model.compile(loss='mse', optimizer=Adam())
    run_hist = nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1) 
    
    plot_train_valid_loss(run_hist)
    
    #Let's check if the NN model has been successfully generated and fitted
    
    assert(nn_model != None)
    assert(type(nn_model) == Sequential)
    #Number of times for which the model has been trained
    assert(len(run_hist.history["val_loss"]) == 10 and len(run_hist.history["loss"]) == 10)

    acc_train = round(run_hist.history["loss"][-1], 3)
    acc_test = round(run_hist.history["val_loss"][-1], 3)
    
    #Test 
    assert(acc_train > 0)
    assert(acc_test > 0)
        
    mean_diff_perc_rounded_train = compute_percentage_error(nn_model, X_train, y_train, scaler_number_vehicles, "Training")
    mean_diff_perc_rounded_test = compute_percentage_error(nn_model, X_test, y_test, scaler_number_vehicles, "Test")
    
    assert(mean_diff_perc_rounded_test > 0 and mean_diff_perc_rounded_train > 0)

    
def test_train_nn_numerical_output():
    traffic_num_label = "COUNT_1"
    
    df_tratte = pd.read_csv("./d/data_frame_tratte_meteo_suedtirol_fixed.csv", sep=",",  encoding = "ISO-8859-1")
    siti_codsito = get_siti_codsito_given_index(0, df_tratte)   
    input_data_frame = load_df_traffic_station(siti_codsito, input_path="./d/Stations_Past_2018")

    input_data_frame_processed = preprocess_data_frame(input_data_frame)
    
    train_nn_model_numerical_output(input_data_frame_processed, traffic_num_label, siti_codsito, "./d")
    
    #Let's check if the scalers, encoders and the numeric NN model have been successfully dumped to disk
    with not_raises(Exception):
        check_list_files_exist(["./d/scaler_temperature_COUNT_1_00000002.pck", "./d/scaler_niederschlag_COUNT_1_00000002.pck",
                            "./d/encoder_week_day_COUNT_1_00000002.pck", "./d/encoder_hours_COUNT_1_00000002.pck",
                            "./d/scaler_number_vehicles_COUNT_1_00000002.pck", "./d/nn_numeric_COUNT_1_00000002.pck" ])
    
    #let's remove the generated scalers, encoders and the NN model
    os.remove("./d/scaler_temperature_COUNT_1_00000002.pck")
    os.remove("./d/scaler_niederschlag_COUNT_1_00000002.pck")
    os.remove("./d/encoder_week_day_COUNT_1_00000002.pck")
    os.remove("./d/encoder_hours_COUNT_1_00000002.pck")
    os.remove("./d/scaler_number_vehicles_COUNT_1_00000002.pck")
    os.remove("./d/nn_numeric_COUNT_1_00000002.pck")
    
def test_init_train_models():
    #Let's try to load up some train models for one single stations
    path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv"
    df_tratte_all = pd.read_csv(path_metadata, sep=",",  encoding = "ISO-8859-1")
    
    #We got one single 'tratta' 
    df_tratta_single = df_tratte_all[df_tratte_all["SITI_CODSITO"] == 2]
    
    #Let's save this tratta to the disk
    df_tratta_single.to_csv("./d/data_frame_single_tratta_test.csv", index=False, header=True)
    
    #Let's now create a dummy folder where the resulting models will be hosted 
    dir_models_test = "./d/Stations_Models_Test"
    #if the folder does not already exist, then create it
    if not os.path.isdir(dir_models_test):
        os.mkdir(dir_models_test)
    
    #Let's now generate the models into the folders specified
    init_train_models(folder_models=dir_models_test, folder_past="./d/Stations_Past_2018",
                        path_metadata="./d/data_frame_single_tratta_test.csv")
    
    
    list_num_traffic_labels = ["COUNT_1", "COUNT_2"]
    #Objects for the COUNT_1 and COUNT_2 labels
    for num_traffic_label in list_num_traffic_labels:
        with not_raises(Exception):
            check_list_files_exist([ dir_models_test + "/scaler_temperature_"+ num_traffic_label + "_00000002.pck", 
                                     dir_models_test + "/scaler_niederschlag_" + num_traffic_label + "_00000002.pck",
                                     dir_models_test + "/encoder_week_day_" + num_traffic_label + "_00000002.pck",
                                     dir_models_test + "/encoder_hours_" + num_traffic_label + "_00000002.pck",
                                     dir_models_test + "/scaler_number_vehicles_" + num_traffic_label + "_00000002.pck",
                                     dir_models_test + "/nn_numeric_" + num_traffic_label +  "_00000002.pck"])
    
    #Objects for the TRAFFIC_1 and TRAFFIC_2 labels
    list_traffic_labels = ["TRAFFIC_1", "TRAFFIC_2"]
    
    for traffic_level_label in list_traffic_labels:
        #let's delete the created models
        with not_raises(Exception):
            check_list_files_exist([dir_models_test + "/kn_model_" + traffic_level_label + "_00000002.pck",
                                    dir_models_test + "/opt_dec_tree_" + traffic_level_label + "_00000002.pck",
                                    dir_models_test + "/simple_rf_" + traffic_level_label + "_00000002.pck"])

    #Clean up of resources created
    os.remove("./d/data_frame_single_tratta_test.csv")
    
    #Let's remove all the content that was created in the folder passed as input
    for file in listdir(dir_models_test):
        os.remove(os.path.join(dir_models_test, file))
    
    os.rmdir(dir_models_test)

    

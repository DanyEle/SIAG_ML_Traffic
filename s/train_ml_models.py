#!/usr/bin/env python
# coding: utf-8


"""
Created on Thu Oct 24 12:36:05 2019

@author: dgadler

This file is used to train Machine Learning models and serialize them to disk
for later usage for inference purposes

Files' Usage Order:
Train_ML_Models (useds Data_Loading) --> Inference_ML_Models
"""

#Custom dependencies upon loading from my own files
from .functions import (preprocess_X_data_nn, get_siti_codsito_given_index, 
load_df_traffic_station, display_info_about_station, fit_scalers_encoders,
check_list_folders_exist, check_list_files_exist, perform_models_ensemble_prediction,
preprocess_data_frame)


#First of all, load up dependency packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from keras.models  import Sequential
from keras.layers import Dense 
from keras.optimizers import  Adam
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
warnings.simplefilter("ignore")
import datetime
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

#Plotting decision trees
#import pydotplus 
#import graphviz
#from keras.utils import plot_model, model_to_dot
#from IPython.display import SVG
#from IPython.display import Image
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


#####################################
######MODELS CREATION AND TRAINING###
#####################################



#Input: - traffic_X_df, the input traffic_X_df data as a data frame
#       - traffic_y_df, the labels to be predicted
#       - traffic_level_label:  "TRAFFIC_1" | "TRAFFIC_2"
#Output: - a data frame of the input X data frame, rebalanced in an equal proportion 
#           as          the y labels
#        - a data frame of the input Y labels, rebalanced based on the 
#This method performs data re-balancing for the input X data frame and the input Y labels
#such that the all Y labels are balanced (EX: 4 labels: 1,2,3,4 --> both for X and Y, we consider
#approx. 25% data points from label 1 , 25% data points from label 2, 25% data points from label 3,
#25% data points fro label 4. This re-sampling is applied both to X data and Y data
def random_resample_from_categories(traffic_X_df, traffic_y_df, traffic_level_label): 
    print("Performing random resampling based on the categories")
    #Idea: let's iterate over all the data points and take a random data point from each category.
    normal_traffic_X_df = traffic_X_df[traffic_y_df[traffic_level_label] == 1]
    moderate_traffic_X_df = traffic_X_df[traffic_y_df[traffic_level_label] == 2]
    high_traffic_X_df = traffic_X_df[traffic_y_df[traffic_level_label] == 3]
    extreme_traffic_X_df = traffic_X_df[traffic_y_df[traffic_level_label] == 4]

    normal_traffic_y_df = traffic_y_df[traffic_y_df[traffic_level_label] == 1]
    moderate_traffic_y_df = traffic_y_df[traffic_y_df[traffic_level_label] == 2]
    high_traffic_y_df = traffic_y_df[traffic_y_df[traffic_level_label] == 3]
    extreme_traffic_y_df = traffic_y_df[traffic_y_df[traffic_level_label] ==  4]
    
    new_traffic_X_df = []
    new_traffic_y_df = []
    
    #Let's iterate over all the data points
    for i in range(0, len(traffic_X_df)):
        #let's roll a dice...
        random_val = random.randrange(0, 4)
        #if 0--> Take a random value from normal
        if random_val == 0:
            index_data_point = random.randrange(0, len(normal_traffic_X_df))
            new_traffic_X_df.append(normal_traffic_X_df.iloc[index_data_point])
            new_traffic_y_df.append(normal_traffic_y_df.iloc[index_data_point])

        #if 1 --> Take a random value moderate
        if random_val == 1:
            index_data_point = random.randrange(0, len(moderate_traffic_X_df))
            new_traffic_X_df.append(moderate_traffic_X_df.iloc[index_data_point])
            new_traffic_y_df.append(moderate_traffic_y_df.iloc[index_data_point])
        #if 2 --> Take a random value from high
        if random_val == 2:
            index_data_point = random.randrange(0, len(high_traffic_X_df))
            new_traffic_X_df.append(high_traffic_X_df.iloc[index_data_point])
            new_traffic_y_df.append(high_traffic_y_df.iloc[index_data_point])
         #if 3 --> Take a random value from extreme
        if random_val == 3:
            index_data_point = random.randrange(0, len(extreme_traffic_X_df))
            new_traffic_X_df.append(extreme_traffic_X_df.iloc[index_data_point])
            new_traffic_y_df.append(extreme_traffic_y_df.iloc[index_data_point])
        
    return(pd.DataFrame(new_traffic_X_df), pd.DataFrame(new_traffic_y_df))

        
def compute_train_test_accuracy(model, X_train, X_test, y_train, y_test, model_name):
    y_predicted_train = model.predict(X_train) 
    y_predicted_test = model.predict(X_test)
    
    acc_train = round(accuracy_score(y_train,y_predicted_train), 3)
    acc_test = round(accuracy_score(y_test,y_predicted_test), 3)
    
    
    print(model_name + ": Train accuracy is " + str(acc_train))
    print(model_name + ": Test accuracy is " + str(acc_test))
    
    return(acc_train, acc_test)
   
    
def simple_decision_tree_model(X_train, X_test, y_train, y_test):    
    #Simple decision tree - no optimization
    dec_tree = DecisionTreeClassifier(criterion='gini', max_depth=2, 
                             min_samples_split=2, min_samples_leaf=1)
    dec_tree.fit(X_train, y_train)
    
#    dot_data = tree.export_graphviz(dec_tree, out_file=None) 
#    graph = graphviz.Source(dot_data) 
#    graph.render("iris") 

    #print("Features' importance:")

    attributes_consider_X = list(X_train.columns.values)
    for col, imp in zip(attributes_consider_X, dec_tree.feature_importances_):
        print(col, imp) 
        
    #acc_train_simple, acc_test_simple = compute_train_test_accuracy(dec_tree, X_train, X_test, y_train, y_test, "Simple Decision Tree")
    return(dec_tree)
    


#Optimization techniques with Grid Search and Randomized Search over all parameters
def optimize_model(model, optimization_method, param_list, X, y):
    if(optimization_method == 1):
        grid_search = GridSearchCV(model, param_grid=param_list)
        grid_search.fit(X, y)
        clf = grid_search.best_estimator_
        #report(grid_search.cv_results_, n_top=3)
        return(clf)
        
        
    elif(optimization_method == 2):
        random_search = RandomizedSearchCV(model, param_distributions=param_list) #n_iter=100
        random_search.fit(X, y)
        clf = random_search.best_estimator_
        #report(random_search.cv_results_, n_top=3)
        return(clf)
    

def optimized_decision_tree_model(X_train, X_test, y_train, y_test, simp_dec_tree):
    #Optimized Decision Tree
    param_list_grid = {'min_samples_split': [2, 5, 10, 20, 30, 40, 50, 100],
                      'min_samples_leaf': [1, 5, 10, 20, 30, 40, 50, 100],
                      'max_depth': [None] + list(np.arange(2, 5)),
                     }
   
    dec_tree_optimized_grid = optimize_model(simp_dec_tree, 1, param_list_grid, X_train, y_train)
#    dot_data = tree.export_graphviz(dec_tree_optimized_grid, out_file=None) 
#    graph = graphviz.Source(dot_data) 
#    graph.render("iris") 
    print("Features' importance:")
    attributes_consider_X = list(X_train.columns.values)
    for col, imp in zip(attributes_consider_X, dec_tree_optimized_grid.feature_importances_):
        print(col, round(imp, 3)) 
 
    acc_train_opt, acc_test_opt = compute_train_test_accuracy(dec_tree_optimized_grid, X_train, X_test, y_train, y_test, "Optimized Decision Tree")
    
    return(dec_tree_optimized_grid, acc_test_opt)
    
    

def simple_random_forest_model(X_train, X_test, y_train, y_test):        
    #Random Forest classifier, using 200 decision trees
    simple_rf_model = RandomForestClassifier(n_estimators=200)
    #we train both with X input data and Y output data
    simple_rf_model.fit(X_train, y_train)
    
    attributes_consider_X = list(X_train.columns.values)
    for col, imp in zip(attributes_consider_X, simple_rf_model.feature_importances_):
        print(col, imp) 
    
    acc_train_simple, acc_test_simple = compute_train_test_accuracy(simple_rf_model , X_train, X_test, y_train, y_test, "Simple Random Forest")

    return(simple_rf_model , acc_test_simple)    
    
    
#K Nearest Neighbors
def knn_model(X_train, X_test, y_train, y_test, max_n=100):
    knn_test_accuracies = []  
    knn_train_accuracies = []
    knn_models = []
    
    for i in range(1, max_n): #100
        #print("n = " + str(i))
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(X_train, y_train) 

        y_predicted_train = knn_model.predict(X_train) 
        y_predicted_test = knn_model.predict(X_test)
        
        train_accuracy = format(accuracy_score(y_train,y_predicted_train))
        test_accuracy = format(accuracy_score(y_test,y_predicted_test))
                
        knn_train_accuracies.append(train_accuracy)
        knn_test_accuracies.append(test_accuracy)
        knn_models.append(knn_model)
        
    index_best_knn_accuracy = (np.argmax(knn_test_accuracies))
    
    best_knn_train_accuracy = round(float(knn_test_accuracies[index_best_knn_accuracy]), 3)
    best_knn_test_accuracy = round(float(knn_test_accuracies[index_best_knn_accuracy]), 3)
    best_knn_model = knn_models[index_best_knn_accuracy]         
    
    print("Best KNN for n = ",str(index_best_knn_accuracy + 1))

    print('Best KNN train accuracy = ', best_knn_train_accuracy)
    print('Best KNN test accuracy = ', best_knn_test_accuracy)
    
    return(best_knn_model, best_knn_test_accuracy)

####Data pre-processing required for the neural network###
    

def plot_train_valid_loss(keras_model):
    #plot accuracy of second deep learning model
    fig, ax = plt.subplots()
    ax.plot(keras_model.history["loss"],'r', marker='.', label="Train Loss")
    ax.plot(keras_model.history["val_loss"],'b', marker='.', label="Test Loss")
    ax.legend()
    

def serialize_object(models_path, model_input, model_name, traffic_level_label, siti_codsito):
    full_name_model_path = models_path + "/" + model_name + "_" + traffic_level_label + "_" + siti_codsito + ".pck"
    dump(model_input, open(full_name_model_path, 'wb'))
    
    
 
#Input:- list_models: the list of models generated
#     - input_data_frame: the input data frame containing the data loaded from the file
#Output: the accuracy of the ensemble of models in 'list_models' on the whole data frame
def compute_accuracy_models_ensemble(list_models, traffic_y_df, X_train, X_test, y_train, y_test):
    list_model_names = ["Simple Decision Tree", "Optimized Decision Tree", "Simple Random Forest", "Optimized Random Forest",
                        "Best KNN", "Naive Bayes", "SVM", "Neural Network"]
    
    y_train_predictions_ensemble = perform_models_ensemble_prediction(X_train, list_models, list_model_names)
    accuracy_ensemble_train = round(accuracy_score(y_train,y_train_predictions_ensemble), 3)
    
    y_test_predictions_ensemble = perform_models_ensemble_prediction(X_test, list_models, list_model_names)
    accuracy_ensemble_test = round(accuracy_score(y_test,y_test_predictions_ensemble), 3)
    
    print('Ensemble of models: Train accuracy is ' + str(accuracy_ensemble_train))
    print('Ensemble of models: Test accuracy is ' + str(accuracy_ensemble_test))
        
#Input: - input_data_frame: the data loaded from a single dataset Stations_Data/station_data_frame_i_l.csv
#       - traffic_level_label: "TRAFFIC_1" | "TRAFFIC_2", the level for the y predictions to be considered
#       - siti_codsito: the codsito for the current sito
#       - models_path: the path to which the models should be exported to
#Output: - y_labels_predicted: the list of predicted labels for the input data frame
#        - list_models: a list of models generated
#        - list_acc_test_models: a list of the test accuracies for the different models generated
#This method takes the input data, preprocesses it and generates many different models as output to the path 'models_path'
def train_models_categorical_output(input_data_frame_processed, traffic_level_label, siti_codsito, models_path):
    print(traffic_level_label)
    #let's remove all the rows where at least one element is missing (i.e., it is NA)
    input_data_frame_processed = input_data_frame_processed.dropna()
    
    #the y to predict is TRAFFIC_Y. Use traffic_y_df[traffic_level_label].value_counts() to get the count of distinct values.
    traffic_y_df = input_data_frame_processed[[traffic_level_label]]
    
    traffic_X_df = input_data_frame_processed[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]
    
    #Our data suffers from data unbalancing. Let's re-balance it via random resampling
    traffic_X_df_balanced, traffic_y_df_balanced = random_resample_from_categories(traffic_X_df, traffic_y_df, traffic_level_label)
    
    #Let's now split the dataset by holdout, using 80% of the dataset for training and last 20% for test taken randomly
    X_train, X_test, y_train, y_test = train_test_split(traffic_X_df_balanced, traffic_y_df_balanced, test_size=0.2,random_state=101)
    #And apply downsampling to the different data sets
    ####Predictive models begin
    
    #Simple Decision tree
    simple_dec_tree = simple_decision_tree_model(X_train, X_test, y_train, y_test)
    #Optimized Decision tree
    optimized_dec_tree, acc_test_opt_dec_tree = optimized_decision_tree_model(X_train, X_test, y_train, y_test, simple_dec_tree)
    #Serialize the optimized decision tree
    serialize_object(models_path, optimized_dec_tree, "opt_dec_tree", traffic_level_label, siti_codsito)
    
    #Simple Random Forest
    simple_rf_model, acc_test_simple_rf_model = simple_random_forest_model(X_train, X_test, y_train, y_test)
    #Serialize the Simple Random Forest model
    serialize_object(models_path, simple_rf_model, "simple_rf", traffic_level_label, siti_codsito)

    #KNN Model
    kn_model, acc_test_knn_model = knn_model(X_train, X_test, y_train, y_test, max_n=5)
    serialize_object(models_path, kn_model, "kn_model", traffic_level_label, siti_codsito)
    
    list_models = [optimized_dec_tree, simple_rf_model, kn_model]

    #Potentially, we can use a model for prediction with the following line:
    compute_accuracy_models_ensemble(list_models, traffic_y_df, X_train, X_test, y_train, y_test)
    
    return(list_models)
    
    
#See comment of train_models_categorical_output for a description of the inputs
# -traffic_cat_label: the label of a categorical variable (like TRAFFIC_1)
# -traffic_num_label: the label of a numerical variable (like COUNT_1)
def train_nn_model_numerical_output(input_data_frame_processed, traffic_num_label, siti_codsito, models_path):
    traffic_y_df_numeric = input_data_frame_processed[[traffic_num_label]]
    
    traffic_X_df = input_data_frame_processed[["TEMPERATURE", "NIEDERSCHLAG", "HOUR", "WEEK_DAY"]]

    #Let's fit the scalers and encoders and return them
    scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours = fit_scalers_encoders(traffic_X_df)
    scaler_number_vehicles = MinMaxScaler()
    scaler_number_vehicles.fit(traffic_y_df_numeric)
    
    #Now let's use the scalers and encoders for scaling & encoding the input data
    traffic_y_df_scaled_balanced = scaler_number_vehicles.transform(traffic_y_df_numeric).flatten("C")
    traffic_X_df_processed = preprocess_X_data_nn(traffic_X_df, scaler_temperature, scaler_niederschlag, encoder_week_day, encoder_hours)
    
    #Holdout 80-20
    X_train, X_test, y_train, y_test = train_test_split(traffic_X_df_processed, traffic_y_df_scaled_balanced, test_size=0.2,random_state=101)    
    
    #FULLY CONNECTED NN LAYERS
    nn_model = Sequential()
    nn_model.add(Dense(100, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu'))
    nn_model.add(Dense(100, activation="relu"))
    nn_model.add(Dense(100, activation="relu"))
    nn_model.add(Dense(1, kernel_initializer='normal') )
    # Compile model
    nn_model.compile(loss='mse', optimizer=Adam())
    run_hist = nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1) 
    #plot_train_valid_loss(run_hist)

    acc_train = round(run_hist.history["loss"][-1], 3)
    acc_test = round(run_hist.history["val_loss"][-1], 3)
    
    print('Neural Network: Train loss is ' + str(acc_train))
    print('Neural Network: Test loss is ' + str(acc_test))
    
    compute_percentage_error(nn_model, X_train, y_train, scaler_number_vehicles, "Training")
    compute_percentage_error(nn_model, X_test, y_test, scaler_number_vehicles, "Test")

    #Let's serialize the scalers and encoders
    serialize_object(models_path, scaler_temperature, "scaler_temperature", traffic_num_label, siti_codsito)
    serialize_object(models_path, scaler_niederschlag, "scaler_niederschlag", traffic_num_label, siti_codsito)
    serialize_object(models_path, encoder_week_day, "encoder_week_day", traffic_num_label, siti_codsito)
    serialize_object(models_path, encoder_hours, "encoder_hours", traffic_num_label, siti_codsito)
    serialize_object(models_path, scaler_number_vehicles, "scaler_number_vehicles", traffic_num_label, siti_codsito)
    
    #Let's serialize the Neural Network model
    serialize_object(models_path, nn_model, "nn_numeric", traffic_num_label, siti_codsito)

    
#Input: model: a neural network model
#       X_data: X_train or X_test, features used as input to the Neural Network
#       scaler_number_vehicles: the scaler used for the number of vehicles
#       training_type: "Training" | "Test"
#Output: the mean difference in percentage between the actual value and the predicted value for all input data points
def compute_percentage_error(model, X_data, y_data, scaler_number_vehicles, training_type):
    
    #First thing first: let's unscale the actual y and the predicted y
    y_actual_unscaled = scaler_number_vehicles.inverse_transform(y_data.reshape(-1, 1))
    list_y_actual_unscaled = list(y_actual_unscaled.flatten("C"))
    
    y_predicted_scaled = model.predict(X_data).flatten("C")
    y_predicted_unscaled = scaler_number_vehicles.inverse_transform(y_predicted_scaled.reshape(-1, 1))
    list_y_predicted_unscaled = list(y_predicted_unscaled.flatten("C"))
    
    #Will contain the difference in percentage between list_y_actual and list_y_predicted
    list_mean_percentages = []
    
    for i in range(0, len(list_y_actual_unscaled)):
        y_actual = list_y_actual_unscaled[i]
        y_predicted = list_y_predicted_unscaled[i]
        
        y_diff_perc = (abs(y_actual - y_predicted) / ((y_actual + y_predicted) / 2)) * 100
        
        list_mean_percentages.append(round(y_diff_perc, 3))
            
    mean_diff_perc = sum(list_mean_percentages) / len(list_mean_percentages)
    
    mean_diff_perc_rounded = round(mean_diff_perc, 3)
    
    print(training_type + ": mean difference is " + str(mean_diff_perc_rounded) + "%")
    
    return(mean_diff_perc_rounded)
    

#Input: None
#Output: 00000002_predictions.csv in the folder given, namely the predictions for station with that SITI_CODSITO
#        Note that this output is only produced for data frames that are valid (NB: At least 1 row of data)
def init_train_models(folder_models="./d/Stations_Models_2018", folder_past="./d/Stations_Past_2018",
                        path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv"):
        
    check_list_folders_exist([folder_models, folder_past])
    check_list_files_exist([path_metadata])
    
    #Let's load up the metadata, containing the station, the closest weather station to the station and the directions
    df_tratte = pd.read_csv(path_metadata, sep=",",  encoding = "ISO-8859-1")

    #let's iterate over all the stations (tratte)
    for i in range(0, len(df_tratte)):
        #let's get the corresponding SITI_CODSITO
        siti_codsito = get_siti_codsito_given_index(i, df_tratte)   
        input_data_frame = load_df_traffic_station(siti_codsito, input_path=folder_past)
        
        if input_data_frame is not None:
            #Let's add some further information to the input data frame
            input_data_frame_processed = preprocess_data_frame(input_data_frame)
        
            display_info_about_station(i, siti_codsito, 1, df_tratte)
            #Generate models for TRAFFIC_1 and TRAFFIC_2 target attributes
            train_models_categorical_output(input_data_frame_processed, "TRAFFIC_1", siti_codsito, models_path=folder_models)
            train_models_categorical_output(input_data_frame_processed, "TRAFFIC_2", siti_codsito, models_path=folder_models)
            
            #Generate models for COUNT_1 and COUNT_2 targets attributes
            train_nn_model_numerical_output(input_data_frame_processed, "COUNT_1", siti_codsito, models_path=folder_models)
            train_nn_model_numerical_output(input_data_frame_processed, "COUNT_2", siti_codsito, models_path=folder_models)

        print("--------------------------")
 

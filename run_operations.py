import sys


if len(sys.argv) == 2: 
    
    #passing 'run_operations.py train'
    if sys.argv[1] == "train":
        from s.train_ml_models import init_train_models
        init_train_models(folder_models="./d/Stations_Models_2018", folder_past="./d/Stations_Past_2018",
                          path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv")
    
    elif sys.argv[1] == "inference":
        from s.inference_ml_models import load_use_models_for_inference
        load_use_models_for_inference(folder_models="./d/Stations_Models_2018", folder_predictions="./d/Stations_Predictions", folder_past="./d/Stations_Past_2018",
                        path_metadata="./d/data_frame_tratte_meteo_suedtirol_fixed.csv")
    else:
        print("Usage: python run_operations.py train|inference")
else:
        print("Usage: python run_operations.py train|inference")

        

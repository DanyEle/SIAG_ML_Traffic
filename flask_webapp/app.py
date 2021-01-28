

"""
Created on Wed Oct 30 08:12:30 2019

@author: dgadler

File used to start up the Flask frontend 

"""

from app_functions import (list_strings_to_string, produce_markers_from_tratte,
                           load_list_traffic_predictions, load_list_traffic_past_2018)

from flask import Flask, render_template, request
from flask_googlemaps import GoogleMaps, Map
import pandas as pd

#####MAIN INIT --- GLOBAL VARIABLES (read-only) BEGIN####
df_tratte = pd.read_csv("./data_frame_metadata_tratte.csv", sep=",",  encoding = "ISO-8859-1")
#Let's load up the markers with the default transparent logo
markers_direzioni = produce_markers_from_tratte(df_tratte)
#Let's load up past data for 2018
list_past_all_stations = load_list_traffic_past_2018(df_tratte)

#the list_predictions_all_stations will be updated within the function invoked by the scheduler every day at 10:05
list_predictions_all_stations = load_list_traffic_predictions(df_tratte)

app = Flask(__name__, template_folder="templates", static_folder="static")

#Todo: create an API key
GoogleMaps(
    app
    #key=""
)


#Accessing the index of the webservice
@app.route('/')
def fullmap():
    fullmap = Map(
        identifier="fullmap",
        varname="fullmap",
        style=(
            "height:90%;"
            "width:100%;"
            "position:absolute;"
            "z-index:200;"
        ),
        lat=46.480667, 
        lng=11.327051,
        markers=markers_direzioni,
        # maptype = "TERRAIN",
        zoom="9.5"
    )
          
    #Let's pass parameters over to the template!
    return render_template(
        'index.html',
        fullmap=fullmap,
        list_past_all_stations=list_past_all_stations,
        list_predictions_all_stations=list_predictions_all_stations,
        GOOGLEMAPS_KEY=request.args.get('apikey')
    )
    
#Accessing the heatmap of the webservice
@app.route('/heatmap')
def heatmap():          
    #Let's pass parameters over to the template!
    return render_template(
        'heatmap.html',
        list_names_stations_IT=list_strings_to_string(list(df_tratte["ABITATO_SITO_I"])),
        list_names_stations_DE=list_strings_to_string(list(df_tratte["ABITATO_SITO_D"])),
        list_direzione_1_IT=list_strings_to_string(list(df_tratte["DESCRIZIONE_I_1"])),
        list_direzione_1_DE=list_strings_to_string(list(df_tratte["DESCRIZIONE_D_1"])),
        list_direzione_2_IT=list_strings_to_string(list(df_tratte["DESCRIZIONE_I_2"])),
        list_direzione_2_DE=list_strings_to_string(list(df_tratte["DESCRIZIONE_D_2"])),
        list_predictions_all_stations=list_predictions_all_stations,
    ) 
    
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


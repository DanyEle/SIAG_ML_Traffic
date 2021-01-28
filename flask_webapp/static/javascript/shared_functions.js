//GLOBAL VARIABLES: None


//Input: list_all_stations: contains a list of lists for every single station
//list_all_stations[i][0] --> Traffic levels 1 of station i
//list_all_stations[i][1] --> Traffic levels 2 of station i
//list_all_stations[i][2] --> Days of station i
//list_all_stations[i][3] --> MONTHs of station i
//list_all_stations[i][4] --> YEARs of station i
//list_all_stations[i][5] --> HOURs of station i
//list_all_stations[i][6] --> Count_1 of station i
//list_all_stations[i][7] --> Count_2 of station i
//Output: list_dicts_all_stations: a list of dictionaries, where we have
//list_dicts_all_stations[i] --> the dictionary of dates and traffic levels of station i
//                                such dictionary contains as key the date and
//                                as value a list of the traffic levels for the two directions
//EX: list_dicts_all_stations[i]["1-1-2018:05:00"] => [traffic_1, traffic_2] at the date and hour specified
function initialize_dicts_all_stations(list_all_stations)
{
    //our list of dictionaries
    var list_dicts_all_stations = [];

    //let's iterate over all the stations
    for(var i = 0; i < list_all_stations.length; i++)
    {
        //contains the data (lists) of the ith-station
        var station_i_past_lists = list_all_stations[i];

        //let's get all the traffic level (1) for station i
        var list_traffic_levels_1_station_i = station_i_past_lists[0];
        //and let's get all the traffic level(2) for station i
        var list_traffic_levels_2_station_i = station_i_past_lists[1];
        //let's get all the days for station i
        var list_days_station_i = station_i_past_lists[2];
        //and all months
        var list_months_station_i = station_i_past_lists[3];
        //and all years
        var list_years_station_i = station_i_past_lists[4];
        //and all hours
        var list_hours_station_i = station_i_past_lists[5];
        //and all count_1
        var list_count_1_station_i = station_i_past_lists[6];
        //and all count_2
        var list_count_2_station_i = station_i_past_lists[7];


        //for every single station, let's add one dictionary!
        list_dicts_all_stations.push({})

        //populate the dictionary of a station by iterating over all observations of a single station
        for(var j = 0; j < list_days_station_i.length; j++)
        {
            //let's get all the single different values for a single observation
            var day_obs_j = list_days_station_i[j];
            var month_obs_j = list_months_station_i[j];
            var year_obs_j = list_years_station_i[j];
            var hour_obs_j = list_hours_station_i[j];
            var traffic_level_1_obs_j = list_traffic_levels_1_station_i[j];
            var traffic_level_2_obs_j = list_traffic_levels_2_station_i[j];
            var traffic_count_1_obs_j = list_count_1_station_i[j];
            var traffic_count_2_obs_j = list_count_2_station_i[j];

            //year-month-day-hour --> our KEY
            var date_time_obs_j = (year_obs_j + "-" + month_obs_j + "-" + day_obs_j + ":" + hour_obs_j+":00");

            //[traffic_1, traffic_2] --> our VALUE
            var list_traffic_level_1_2_obs_j = [traffic_level_1_obs_j, traffic_level_2_obs_j, traffic_count_1_obs_j, traffic_count_2_obs_j];

            list_dicts_all_stations[i][date_time_obs_j] = list_traffic_level_1_2_obs_j;
        }       
        //output all the values for this station
        //console.log(list_dicts_all_stations[0]);
    }
    return(list_dicts_all_stations);
} 

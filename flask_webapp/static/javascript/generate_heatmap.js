//GLOBAL VARIABLES: list_predictions_all_stations, hosting all the future traffic levels and dates + hours for all stations
//                  list_dicts_predictions_all_stations {"date_time_hour" => "traffic_level"}
//                  list_names_stations_DE_split: a list containing all stations' names in german
//                  list_names_stations_IT_split: see above, but in Italian
//                  list_direzione_1_IT_split: a list containing all the direction 1's names in italian
//                  list_direzione_2_IT_split
//                  list_direzione_1_DE_split
//                  list_direzione_2_DE_split
//*******************************************************************
//*************FUNCTIONS' DECLARATIONS FOR HEATMAP GENERATION********
//*******************************************************************

    //Input: - text with HTML special chars: an input string having special HTML characters. Example: ' --> &#39
    //Output:- text with decoded HTML special chars: the input string with special HTML characters decoded.
    function decodeHTMLEntities(text) 
    {
        var entities = [
            ['amp', '&'],
            ['apos', '\''],
            ['#x27', '\''],
            ['#x2F', '/'],
            ['#39', '\''],
            ['#47', '/'],
            ['lt', '<'],
            ['gt', '>'],
            ['nbsp', ' '],
            ['quot', '"']
        ];

        for (var i = 0, max = entities.length; i < max; ++i) 
            text = text.replace(new RegExp('&'+entities[i][0]+';', 'g'), entities[i][1]);

        return text;
    }


    //Input: - div_name: the name of the div to which the options should be added
    //       - list_stations_names_IT: the list of all station names in italian
    //       - list_stations_names_DE: the list of all station names in German
    //Output: the names in Italian + "/" + German appended to a div having "div_name" as ID and having the index (integer) as value
    //        so as to generate a selector of stations
    function generate_stations_selector(div_name, selector_name, list_stations_names_IT, list_stations_names_DE)
    {
        var myDiv = document.getElementById(div_name);

        //Create and append select list to myDiv
        var selectList = document.createElement("select");
        selectList.setAttribute("id", selector_name);
        myDiv.appendChild(selectList);

        //Create and append the options
        for (var i = 0; i < list_stations_names_IT.length; i++) 
        {
            var option = document.createElement("option");
            var station_name_IT_DE = list_stations_names_IT[i] + "/" + list_stations_names_DE[i];
            option.setAttribute("value", i);
            option.setAttribute("class", "station_" + i);

            option.text = station_name_IT_DE;
            selectList.appendChild(option);
        }
    }

    //Input: -dict_predictions_cur_station: a dictionary containing all the single predictions for a station
    //        not grouped in any manner, so just like:
    //        "2019-11-18:0:00" => [<traffic_level_1>, <traffic_level_2>]
    //Output: - dict_predictions_grouped_station: a dictionary containing all the predictions for a single
    //          station grouped by date, so like:
    //          "2019-11-18" => [[traffic_level_1, traffic_level_2], [traffic_level_1, traffic_level_2], etc...] for all 24 hours
    function generate_dict_predictions_group_by_date(dict_predictions_cur_station)
    {
        var dict_predictions_grouped_by_date = {}

        //i is an index of the current hour[0-23]
        var i = 0;
        var current_date = null;

        Object.keys(dict_predictions_cur_station).forEach(function(date_hour) 
        {
            //the value is the list of traffic levels for a specific hour and date
            var list_traffic_levels = dict_predictions_cur_station[date_hour];

            //the key is the date and hour
            if( i % 24 == 0)
            {
                current_date = date_hour.substring(0, date_hour.indexOf(':'));
                //console.log(date);

                //let's create an empty list corresponding to the current date
                dict_predictions_grouped_by_date[current_date] = [];
            }

            //now let's assign all the traffic levels corresponding to the current date to the
            //corresponding position of the dictionary
            dict_predictions_grouped_by_date[current_date].push(dict_predictions_cur_station[date_hour]); 
          
           i++;
        });

        return(dict_predictions_grouped_by_date);      
    }

    //Input: day_index: the index of the day passed
    //Output: "Sunday" for day_index = 0
    //        "Monday" for day_index = 1
    //        "Tuesday" for day_index = 2
    //        ...
    //        "Saturday" for day_index = 6
    function get_day_name_given_day_index(day_index)
    {
        var day_name = "";

        switch(day_index)
        {
            case 0:
                day_name  = "Sunday";
                break;
            case 1:
                day_name = "Monday";
                break;
            case 2:
                day_name = "Tuesday";
                break;
            case 3:
                day_name = "Wednesday";
                break;
            case 4:
                day_name = "Thursday";
                break;
            case 5:
                day_name = "Friday";
                break;
            case 6:
                day_name = "Saturday";
                break;
        }
        return(day_name);
    }

    //Same as the method below, but with no coloring and no text adding (function used to add dates as table headers)
    function add_list_as_table_row_header(input_list, tbdy)
    {
         //let's create a single table row hosting the TABLE HEADERS
        var tr_headers = document.createElement('tr');
        for(var i = 0; i < input_list.length; i++ )
        {
            var td_header = document.createElement('td');
            td_header.classList.add('bold');
            var date_column = input_list[i];

            //the first entry is "Time", so we exclude it
            if(i != 0)
            {
                //we get the week day (like Monday, Tuesday, etc..) corresponding to the date passed as input
                var date_column_parsed = new Date(date_column);
                var week_day_index = date_column_parsed.getDay();

                //Monday, Tuesday, etc...
                var current_day_name = get_day_name_given_day_index(week_day_index);

                date_column = current_day_name + ", " + date_column;

                //console.log(date_column);
            }           

            td_header.appendChild(document.createTextNode(date_column));
            tr_headers.appendChild(td_header);
        }
       tbdy.appendChild(tr_headers);
    }

    //Input: - list_traffic_levels: an input list having six elements ready to be displayed on a table in the following
    //              format: ["00:00", <traffic_level_day_1>, <traffic_level_day_2>, <traffic_level_day_3>, <traffic_level_day_4>, <traffic_level_day_5>]
    //              where the traffic level is an integer from 1 to 4 for each one of the 5 days of traffic predictions.
    //       - list_traffic_numeric: an input list having six elements ready to be displayed on a table, in the following
    //              format: ["00:00", <traffic_numeric_1>]
    //       - tbdy: the body of the input table
    //Output: the input_list is appended as a new row of tbdy, where each cell is colored based on the traffic level contained therein
    function add_list_as_table_row_body(list_traffic_levels, list_traffic_numeric,tbdy)
    {

         //let's create a single table row hosting the TABLE HEADERS
        var tr_created = document.createElement('tr');
        for(var i = 0; i < list_traffic_levels.length; i++ )
        {
            var td_created = document.createElement('td');
            //for i = 0, the traffic_level is actually the date
            var traffic_level = list_traffic_levels[i];

            var traffic_level_number = Math.max(list_traffic_numeric[i], 0);

            //traffic level
            if(i != 0)
            {
                switch(traffic_level)
                {
                    case 1:
                        td_created.classList.add('color-normal-traffic');
                        break;
                    case 2:
                        td_created.classList.add('color-moderate-traffic');
                        break;
                    case 3:
                        td_created.classList.add('color-high-traffic');
                        break;
                     case 4:
                        td_created.classList.add('color-extreme-traffic');
                        break;
                }
            }
            //hour
            else if(i == 0)
            {
                traffic_level_number = traffic_level;
                td_created.classList.add('bold');
            }

            td_created.appendChild(document.createTextNode(traffic_level_number));
            tr_created.appendChild(td_created);
        }
       tbdy.appendChild(tr_created);
    }

    //Input: index_station_selected, the index [0-70] of the station currently selected
    //Output: the direction_1_id and direction_2_id having the names corresponding to the direction 1 and 2 of the station_name_selector,
    //        respectively, based on the station currently selected in index_station_selected
    function update_direction_selector(index_station_selected, direction_1_id, direction_2_id, list_direzione_1_IT_split, list_direzione_1_DE_split,
                                list_direzione_2_IT_split, list_direzione_2_DE_split)
    {
        var current_direction_1_IT = decodeHTMLEntities(list_direzione_1_IT_split[index_station_selected]);
        var current_direction_1_DE = decodeHTMLEntities(list_direzione_1_DE_split[index_station_selected]);
        var current_direction_2_IT = decodeHTMLEntities(list_direzione_2_IT_split[index_station_selected]);
        var current_direction_2_DE = decodeHTMLEntities(list_direzione_2_DE_split[index_station_selected]);
        

        //let's update direction 1 and direction 2 according to the station selected
        var direction_1_option = document.getElementById(direction_1_id);
        var direction_2_option = document.getElementById(direction_2_id);

        var new_str_direction_1 = "From: " + current_direction_1_IT + " / " + current_direction_1_DE + " to: " + 
                                    current_direction_2_IT + "/" + current_direction_2_DE;

        var new_str_direction_2 = "From: " + current_direction_2_IT + " / " + current_direction_2_DE + " to: " + 
                                    current_direction_1_IT + "/" + current_direction_1_DE;

        direction_1_option.text = new_str_direction_2;
        direction_2_option.text = new_str_direction_1;
    }


    //Input: index_station_selected: the index of the station currently selected[0-70]
    //       list_names_stations_IT_split: a list of station names in Italian 
    //       list_names_stations_DE_split: a list of station names in German
    //       selector_name: the name of the selector for station for which we would like to set the index_station_selected
    //Output: the station_selector passed has its name updated according to the index of the station selected
    function update_station_selector(station_selector, index_station_selected, list_names_stations_IT_split, list_names_stations_DE_split)
    {
        var current_name_station_IT = list_names_stations_IT_split[index_station_selected];
        var current_name_station_DE = list_names_stations_DE_split[index_station_selected];

        var text_element_selected = document.getElementsByClassName('station_' + index_station_selected)[0].value;

        document.getElementById(station_selector).selectedIndex = index_station_selected;
    }


    //Input: - list_dicts_predictions_all_stations, a list containing one dictionary per station
    //       - index_station_selected: the index of the station whose content we would like to show on a heatmap
    //       - direction_selector: the ID of the selector containing the direction picked
    //Output: a table created based on the level of traffic of the predictions
    function generate_heatmap_given_predictions(index_station_selected, list_dicts_predictions_all_stations, direction_selector)
    {
        //and let's get the current index of the direction that is selected
        var index_direction_selected = parseInt(document.getElementById(direction_selector).value);


        //let's get ALL the predictions corresponding to the station selected
        var dict_predictions_cur_station = list_dicts_predictions_all_stations[index_station_selected];

        //let's group the predictions of that station by date(as key)
        var dict_predictions_grouped_by_date = generate_dict_predictions_group_by_date(dict_predictions_cur_station);

        //let's now create a list of hours and append it to the dictionary of predictions grouped by date
        //so that the whole dictionary can be unrolled and displayed
        list_hours = []

        for(var i = 0; i < 24; i++)
        {
            list_hours.push(i + ":" + "00");
        }

        //let's remove the dummy_key and insert "Time" instead for displaying it as a table header
        var list_dates = Object.keys(dict_predictions_grouped_by_date);
        list_dates.unshift("Time");

        var body = document.getElementsByTagName('body')[0];

        //let's check if a table already exists.
        var table_heatmap = document.getElementById("table_heatmap");

        //if the heatmap already exists, then delete it
        if(table_heatmap != null)
        {
            table_heatmap.remove();
        }

        //generate new table
        var tbl = document.createElement('table');
        tbl.setAttribute("id", "table_heatmap");
        tbl.setAttribute('border', '2');
        var tbdy = document.createElement('tbody');

        //the list_dates are our table header
        add_list_as_table_row_header(list_dates, tbdy);

        //iterate over all the hours
        for(var i = 0; i < 24; i++) 
        {
            //TRAFFIC LEVELS: this list will host all the traffic levels for setting the color of the cells.
            var list_traffic_levels = [];

            //TRAFFIC NUMERIC: this list will host all the traffic level numerics to be displayed within the table's cells
            var list_traffic_numeric = []

            //first element of the list will be the hour
            list_traffic_levels.push(list_hours[i]);

            //first element, once again the hour
            list_traffic_numeric.push(list_hours[i]);


            //dict_predictions_grouped_by_date[key_date][i] = [TRAFFIC_1, TRAFFIC_2, COUNT_1, COUNT_2]
            for (var key_date in dict_predictions_grouped_by_date) 
            {
                //assign the traffic level
                traffic_level_hour_dir_selected = dict_predictions_grouped_by_date[key_date][i][index_direction_selected];

                //assign the traffic numeric
                traffic_numeric_hour_dir_selected = dict_predictions_grouped_by_date[key_date][i][2 + index_direction_selected];

                list_traffic_levels.push(traffic_level_hour_dir_selected);
                list_traffic_numeric.push(traffic_numeric_hour_dir_selected);
            }

            //the hour + all traffic levels now make up a single row
            add_list_as_table_row_body(list_traffic_levels, list_traffic_numeric, tbdy);
        }

       //finally, append the table body to the page
       tbl.appendChild(tbdy);
       body.appendChild(tbl);

    } 

    //Input: the name of the GET parameter whose value  we would like to get
    //Output: The value contained in the GET parameter passed in the URL
    function get_parameter_value_from_url(argument_name)
    {
        var queryDict = {};
        location.search.substr(1).split("&").forEach(function(item) {queryDict[item.split("=")[0]] = item.split("=")[1]});

        return(queryDict[argument_name]);
    }

    //Input: header_id, for example "station_name_title"
    //Output: The header (like an h1, h2) now has the text changed to the name of the station corresponding to the
    //        index of the station passed in Italian and German
    function set_title_content(title_id, list_names_stations_IT_split, list_names_stations_DE_split, index_station)
    {
         document.getElementById(title_id).innerHTML  = list_names_stations_IT_split[index_station] + "/" +
                                                        list_names_stations_DE_split[index_station] ;
    }



    //********************************************
    //*************EVENTS BEING INVOKED***********
    //********************************************


     //detect a direction being picked
    $('#direction_selector').change(function() 
    {
       var index_station_selected = document.getElementById("stations_options").value;
       generate_heatmap_given_predictions(index_station_selected, list_dicts_predictions_all_stations, "direction_selector");
    });

    //detect a station being picked
    $('#station_selector').change(function() 
    {
       var index_station_selected = document.getElementById("stations_options").value;

        generate_heatmap_given_predictions(index_station_selected, list_dicts_predictions_all_stations, "direction_selector");

        update_direction_selector(index_station_selected, "direction_1", "direction_2",
                            list_direzione_1_IT_split, list_direzione_1_DE_split,
                            list_direzione_2_IT_split, list_direzione_2_DE_split);

        document.getElementById("station_name_title").innerHTML  = list_names_stations_IT_split[index_station_selected] + "/"
                                                                 + list_names_stations_DE_split[index_station_selected] ;
    });


    //*******************************************
    //*************MAIN***************************
    //********************************************

    function main()
    {
        //convert the list of lists of predictions to a list of dictionaries
        list_dicts_predictions_all_stations = initialize_dicts_all_stations(list_predictions_all_stations);

        //remove the last element of each list, which is a null (empty) value
        list_names_stations_IT_split.splice(-1,1);
        list_names_stations_DE_split.splice(-1,1);
        list_direzione_1_IT_split.splice(-1,1);
        list_direzione_1_DE_split.splice(-1,1);
        list_direzione_2_IT_split.splice(-1,1);
        list_direzione_2_DE_split.splice(-1,1);

        //let's generate an options' button based on the names of the stations loaded
        generate_stations_selector("station_selector", "stations_options", list_names_stations_IT_split, list_names_stations_DE_split);

        //get the index of the station ID that was passed among the GET parameters
        var index_station_GET_param = get_parameter_value_from_url("station_id");

        //no station index passed as parameter, then just need to generate the heatmap for the
        //station selected in the button of the page (the first one)
        if(index_station_GET_param == null)
        {
            //get the index of the station that was selected in the options' button of the page
            var index_station_selected = document.getElementById("stations_options").value;

            update_direction_selector(index_station_selected, "direction_1", "direction_2",
                                     list_direzione_1_IT_split, list_direzione_1_DE_split,
                                   list_direzione_2_IT_split, list_direzione_2_DE_split);

            set_title_content("station_name_title",  list_names_stations_IT_split, list_names_stations_DE_split, index_station_selected);

            //let's update the heatmap                                                             
            generate_heatmap_given_predictions(index_station_selected,list_dicts_predictions_all_stations, "direction_selector");

        }
        //actually passed a station index as parameter, then use it to generate the heat map of the corresponding station
        else
        {
            update_direction_selector(index_station_GET_param, "direction_1", "direction_2",
                                     list_direzione_1_IT_split, list_direzione_1_DE_split,
                                   list_direzione_2_IT_split, list_direzione_2_DE_split);

            generate_heatmap_given_predictions(index_station_GET_param,list_dicts_predictions_all_stations, "direction_selector");

            update_station_selector("stations_options", index_station_GET_param, list_names_stations_IT_split, list_names_stations_DE_split);

            set_title_content("station_name_title",  list_names_stations_IT_split, list_names_stations_DE_split, index_station_GET_param);
           
        }
    }

    main();


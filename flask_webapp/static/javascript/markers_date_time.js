//GLOBAL VARIABLES: list_past_all_stations, hosting all the past traffic levels and dates + hours for all stations
//                  list_dicts_past_all_stations {"date_time_hour" => "traffic_level"}
//                  list_predictions_all_stations, hosting all the future traffic levels and dates + hours for all stations
//                  list_dicts_predictions_all_stations {"date_time_hour" => "traffic_level"}


//*******************************************************************
//*************FUNCTIONS' DECLARATIONS DATE, TIME AND MARKERS********
//*******************************************************************

  
    //Input: None
    //Output: Initialized datetime picker
    function initialize_date_time_picker()
    {

        //initialize the date-time picker so that we cannot pick the time there
          $(function() {
         $('#datetimepicker4').datetimepicker({
             pickTime: false
            });
         });

        //initialize the time picker to the current time
        var picker = $('#datetimepicker4').datetimepicker();          
        var slider = document.getElementById("myRange");
        var output = document.getElementById("displayValue");
        var hours = Math.floor(slider.value / 60);
        var minutes = slider.value % 60;
        var minuteOutput = $("#minutes")[0];
        var hourOutput = $("#hours")[0];
        var hours = Math.floor(slider.value / 60);
        var minutes = slider.value % 60;

        
        hourOutput.innerHTML = hours;
        minuteOutput.innerHTML = minutes;

         //let's initialize the date to today's date
        var today_date = new Date();
        var today_date_formatted = today_date.getFullYear() + "-" + (today_date.getMonth()+1) + "-" + today_date.getDate();
        document.getElementById("input_date_datepicker").value = today_date_formatted;
    }

    function initialize_time_slider()
    {
        var today_date = new Date();

        //the time slider has a range from 1 (00:00) to 1440(23:59)
        //it represents the number of seconds in a day
        //let's initialize the time slider to the current hour and minutes
        var hour_current = today_date.getHours();
        var minutes_current = today_date.getMinutes();

        var minuteOutput = $("#minutes")[0];
        var hourOutput = $("#hours")[0];

        //set the current hour
        hourOutput.innerHTML = hour_current;

        minutes_fixed = minutes_current;

        if (minutes_current.toString().length < 2)
        {
            minutes_fixed = "0" + minutes_current;
        }
       
        //set the current minutes
        minuteOutput.innerHTML = minutes_fixed;

        //set the initial value of the time slider to the current hour and minutes
        var slider = document.getElementById("myRange");
        slider.value = (hour_current * 60) + minutes_current;
    }

    //Input:- hours_picked: the hour picked in the corresponding slider
    //      - list_traffic_levels_all_stations: a list of traffic levels for all the different stations at the time picked
    //      - full_map_makers, which is a GLOBAL VARIABLE
    //Output: None, color the markers on the map based on the traffic level.
    function update_markers_traffic_level(hour_picked, list_traffic_levels_all_stations, fullmap_markers)
    {
        //console.log(list_traffic_levels_all_stations);
        //i is the index of a single station!
        for(var i = 0; i < list_traffic_levels_all_stations.length; i++)
        {
            //the prediction corresponding to the hour_picked for the current station
            traffic_level_station_i = list_traffic_levels_all_stations[i];
            //update the marker corresponding to this very station as well
            marker_station = fullmap_markers[i];

            var icon_url;

            switch(traffic_level_station_i) 
            {
                case  4:
                    icon_url = "./static/imgs/extreme_traffic_resized.png";
                    break;
                case 3:
                    icon_url = "./static/imgs/high_traffic_resized.png";
                    break;
                case 2:
                    icon_url = "./static/imgs/moderate_traffic_resized.png";
                    break;
                case 1: 
                    icon_url = "./static/imgs/normal_traffic_logo_resized.png";
                    break;
                     //a transparent icon
                default:
                    icon_url = "./static/imgs/transparent_logo_resized.png";
            }
            //let's now assign the new logo_url to the corresponding marker_station
            marker_station.setIcon(icon_url);
        }
    }


    //Input: date_time_formatted, example: 2019-12-10:8:00
    //Output: a list of the traffic levels of all the stations at the date, time and direction picked
    ///the traffic level can be: -1: if the traffic level is not present for station i at the date and time picked
    //                            1: normal traffic level
    //                            2: moderate traffic level
    //                            3: high traffic level
    //                            4: very high traffic level
    function get_traffic_level_all_stations(date_time_formatted, direction_picked, list_dicts_past_all_stations)
    {

        var list_traffic_levels_all_stations = []
        var direction_index = direction_picked.replace(/direction_/g,'');

        //iterate over all stations for which we have data
        for(var i = 0; i < list_dicts_past_all_stations.length; i++)
        {
            var dict_station_i = list_dicts_past_all_stations[i];

            //direction_1 --> index 0
            //direction_2 --> index 1
            var direction_index_fixed = parseInt(direction_index) - 1;

            //if this dictionary has no entry (i.e.,it is undefined) at the date and time provided
            // then, set the traffic level to -1 
            if ( dict_station_i[date_time_formatted] == undefined)
            {
                traffic_level_station_i = -1;
            }
            //the key is actually defined
            else
            {
                traffic_level_station_i = dict_station_i[date_time_formatted][direction_index_fixed];
            }
            //console.log(date_time_formatted);
            //console.log("Traffic level of station " + i + " is :" + traffic_level_station_i);
            list_traffic_levels_all_stations.push(traffic_level_station_i)
        }

        return(list_traffic_levels_all_stations);
    }


    //Input: - date_picked_parsed: the date picked by the user on the calendar
    //Output: - True if date_picked_parsed is in the past(i.e.: it is yesterday or even farther into the past)
    //        - False if date_picked_parsed is NOT in the past (i.e.: it is today, tomorrow or farther into the future)
    function is_input_date_past(date_picked_parsed)
    {
       var today_date = new Date();
       var today_date_formatted = today_date.getFullYear() + "-" + (today_date.getMonth()+1) + "-" + today_date.getDate();
       var today_date_parsed = new Date(today_date_formatted);

       return (date_picked_parsed < today_date_parsed)
    }


    //Input: date_picked_parsed(Date): the date for which markers should be updated
    //       hour_picked (integer): the current hour for which markers should be updated
    //       direction_picked ("direction_1") | ("direction_2"): the value of the current direction picked
    //Global vars input used: list_dicts_past_all_stations
    //Output: the markers updated with the date, hour and direction picked
    function update_markers_given_date_time_dir(date_picked_parsed, hour_picked, direction_picked)
    {
       var day_picked = date_picked_parsed.getDate();
       var month_picked = date_picked_parsed.getMonth() + 1; //because it starts from 0 [0 = January]
       var year_picked = date_picked_parsed.getFullYear();

       var date_time_picked = (year_picked + "-" + month_picked + "-" + day_picked + ":" + hour_picked+":00");
       //console.log(date_time_picked);

       //let's see if the date lies in the past or in the future
       var is_date_picked_past = is_input_date_past(date_picked_parsed);
      
       //if the date_time_picked is in the past
       if(is_date_picked_past == true)
       {
            //PAST DATA: let's use the date_time_formatted and the direction_picked to get a list of the traffic levels of all stations at the date_time picked.
            list_traffic_levels_all_stations = get_traffic_level_all_stations(date_time_picked, direction_picked,list_dicts_past_all_stations);
       }
       else
       {
            //PREDICTIONS, get the traffic levels for all stations at the date and time requested
            list_traffic_levels_all_stations = get_traffic_level_all_stations(date_time_picked, direction_picked,list_dicts_predictions_all_stations);

       }


       //and finally update the markers of all stations we have based on the new traffic levels!
       update_markers_traffic_level(hour_picked, list_traffic_levels_all_stations, fullmap_markers);
    }


     //********************************************
    //*************EVENTS BEING INVOKED***********
    //********************************************


     //detect a direction change
    $('#direction').change(function() 
    {
        //console.log(list_dicts_past_all_stations);

        //let's get the current date picked at the corresponding element
        var date_picked = document.getElementById("input_date_datepicker").value;
        var date_picked_parsed = new Date(date_picked);

        //let's get the current hour
        var hour_picked = document.getElementById("hours").innerText;

        //let's get the current direction
        var direction_picked = $('#direction').val();
       
        update_markers_given_date_time_dir(date_picked_parsed, hour_picked, direction_picked);
       
    });


  
    //detect time slider being held & value being changed
     document.getElementById("myRange").addEventListener('input', function(event) { 
        //let's get the properties we need for setting the new hours and minutes

        var slider = document.getElementById("myRange");
        var minuteOutput = $("#minutes")[0];
        var hourOutput = $("#hours")[0];
        var hours = Math.floor(slider.value / 60);
        var minutes = slider.value % 60;

        hourOutput.innerHTML = hours;

        if (minutes.toString().length < 2)
        {
            minutes = "0" + minutes;
        }
       
        minuteOutput.innerHTML = minutes;

        //also get the new date that was picked...
        var date_picked = document.getElementById("input_date_datepicker").value;
        var date_picked_parsed = new Date(date_picked);

        //let's get the current direction
        var direction_picked = $('#direction').val();

        //And then update the markers
        update_markers_given_date_time_dir(date_picked_parsed, hours, direction_picked);

   });

    //detect a date change - datetimepicker changed   
     $('#datetimepicker4').on('changeDate', function(e) {
        var date_picked = e.localDate.toString();
        var date_picked_parsed = new Date(date_picked);

        //console.log(date_picked_parsed);
        var hour_picked = document.getElementById("hours").innerText;

        var direction_picked = $('#direction').val();

        update_markers_given_date_time_dir(date_picked_parsed, hour_picked, direction_picked);
     });        




    //*******************************************
    //*************MAIN***************************
    //********************************************

    function main()
    {
        //initialize the date and time picker, using the current date
        initialize_date_time_picker();

        //let's initialize the time slider to the current time
        initialize_time_slider();

        //let's initialize a list of dictionaries for all the stations
        list_dicts_past_all_stations = initialize_dicts_all_stations(list_past_all_stations);

        list_dicts_predictions_all_stations = initialize_dicts_all_stations(list_predictions_all_stations);

        console.log(list_dicts_predictions_all_stations);

        //let's update the markers to the current date and time 50 ms after the page has loaded up
        $(document).ready(function() {
          var today_date = new Date();
          var hour_picked = today_date.getHours();
          var direction_picked = $('#direction').val();

          setTimeout(function() {  update_markers_given_date_time_dir(today_date, hour_picked, direction_picked); }, 50);
        });

    }

    main();


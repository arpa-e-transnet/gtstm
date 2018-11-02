from pymongo import MongoClient
conn = MongoClient('mongodb://venkat:d_v123@localhost:30000/')
db = conn['AirSage']
from bson.code import Code
map = Code("""
function(){
 var key = this["ID"];
 var day_of_the_week = this["LastUpdateTime"].getDay();
 var hour_of_the_day = this["LastUpdateTime"].getHours();
 var minute_window = ((this["LastUpdateTime"].getMinutes()%4)*15);
 var date = this["LastUpdateTime"].getDate();
                      var value = {
                                    speed: 1/this["Speed"],
                                    count: 1,
				    harmonic_speed : 0
                                   };

                      emit( {Date:date,Time:hour_of_the_day.toString()+':'+minute_window.toString(),Link:key}, value );
}
""")

reduce = Code("""
function(key, value){
var reducedObject = {
                                              speed: 0,
                                              count:0,
                                              harmonic_speed:0
                                            };

                        value.forEach( function(value) {
                                              reducedObject.speed += value.speed;
                                              reducedObject.count += value.count;
                                        }
                                      );
                        return reducedObject;
}
""")

finalize = Code("""function (key, reducedValue) {

                          if (reducedValue.count > 0)
                              reducedValue.harmonic_speed = reducedValue.count / reducedValue.speed;

                          return reducedValue;
                       }""")

op = db.game_data.map_reduce(map, reduce, {'reduce':'fifteen_game_speed_data'}, finalize=finalize)
#for entry in op:
print op

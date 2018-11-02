from pymongo import MongoClient
import datetime
conn = MongoClient('mongodb://venkat:d_v123@localhost:30000/')
db = conn['AirSage_Intermediate']
from bson.code import Code
map = Code("""
function(){
 var key = this["ID"];
 var date_link = this["LastUpdateTime"].getDate();
 var month_link = this["LastUpdateTime"].getMonth();
 var hour_of_the_day = this["LastUpdateTime"].getHours();
 var minute_window = ((this["LastUpdateTime"].getMinutes()%4)*15);
                      var value = {
                                    speed: 1/this["Speed"],
                                    count: 1,
				    harmonic_speed : 0
                                   };

                      emit( {ID:key,date:date_link,month:month_link,hour:hour_of_the_day,min_window:minute_window}, value );
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
feb_day_data = db.SpeedLink.find({"LastUpdateTime":{"$gt":datetime.datetime(2018,01,31,23,59,59),"$lt":datetime.datetime(2018,03,01,0,0,0)}})
op = db.SpeedLink.map_reduce(map, reduce, {'reduce':'HM_airsage_date_script'}, finalize=finalize)
#for entry in op:
#print op

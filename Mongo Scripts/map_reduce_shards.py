from pymongo import MongoClient
db = MongoClient()['AirSage']
from bson.code import Code
map = Code("""
function(){
 var key = this["@ID"];
                      var value = {
                                    linkid: this["@ID"],
                                    speed: 1/this["@Speed"],
                                    count: 1,
				    harmonic_speed : 0
                                   };

                      emit( key, value );
}
""")

reduce = Code("""
function(key, value){
var reducedObject = {
                                              linkid: key,
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

op = db.SpeedLink.map_reduce(map, reduce, {'reduce':'HM_airsage_speed'}, finalize=finalize)
#for entry in op:
#	print op

import pymongo
import csv
try:
                #conn = pymongo.MongoClient(host=['localhost:30000')
        conn = pymongo.MongoClient('mongodb://venkat:d_v123@localhost:30000/')
        print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e

db = conn['AirSage']


myquery = db.fifteen_formatted_game_speed_data.find() # I am getting everything !
output = csv.writer(open('Fifteen_Game_Dt_Speed.csv', 'wt')) # writng in this file
#output.writerow(myquery[0].keys())
for items in myquery:
    print items
    field_names = items.keys()
    id_idx = field_names.index("_id")
    field_names.remove("_id")
    output.writerow(field_names)
    break
for items in myquery:
    print items
    del items["_id"]
 # first 11 entries
    a = list(items.values()) # collections are importent as dictionary and I am making them as list
    #tt = list()
    #for chiz in a:
    #    if chiz is not None:
    #        tt.append(chiz.encode('ascii', 'ignore')) #encoding
    #    else:
    #        tt.append("none")
    output.writerow(a)

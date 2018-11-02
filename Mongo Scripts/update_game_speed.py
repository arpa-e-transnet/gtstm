import pymongo
try:
                #conn = pymongo.MongoClient(host=['localhost:30000')
	conn = pymongo.MongoClient('mongodb://venkat:d_v123@localhost:30000/')
        print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e

db = conn['AirSage']
collec = db.fifteen_game_speed_data
collection = db.fifteen_formatted_game_speed_data


for row in collec.find():
	Date = '2017-09-'+ str(row["_id"]["Date"])
	Time = row["_id"]["Time"]+":00"
	Link = row["_id"]["Link"]
	Speed = row["value"]["harmonic_speed"]
        collection.insert({"Date":Date,"Time":Time,"Link":Link,"Speed":Speed})	

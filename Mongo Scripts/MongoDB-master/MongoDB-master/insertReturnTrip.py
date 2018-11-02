import pymongo
from pymongo import MongoClient

# database and collection will be created at execution
def insertTrip(tripId, tripFile):
    client = MongoClient()
    db = client.Trips
    collection = db['trip']
    # insert document
    insertDoc = { 
        'tripId': tripId,
        'tripFile': tripFile}

    result = collection.insert_one(insertDoc)
    print result.inserted_id

def retrieveTrips():
    client = MongoClient()
    db = client.Trips
    collection = db['trip']
    cursor = collection.find({})
    for document in cursor:
        for row in document['tripFile'].split('\n'):
            # print each row
            print(row)
            # print each column
            for col in row.split(','):
                print(col)

# Test an insert for a trip, make sure mongoDB is running first
# Uncomment this section and specify a trip file to insert to database
'''
tripId = '1234'
tripFile = open('353627074744346.20170516.200038.csv', 'r').read()
insertTrip(tripId, tripFile)
'''
retrieveTrips()

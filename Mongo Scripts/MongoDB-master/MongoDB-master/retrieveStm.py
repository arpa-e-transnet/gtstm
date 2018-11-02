import pymongo
import getpass
from pymongo import MongoClient

def get_db(host, port, dbname, collname, user = '', passwd = ''):
    url = 'mongodb://'+user+':'+passwd+'@localhost:30000/'
    dbconn = pymongo.MongoClient(url)
    db = dbconn[dbname]
    coll = db[collname]
    return coll

def getStm(coll):
    cursor = coll.find({"LinkID":"137868_137872","TimeStamp":"08:30:00"})
    print(cursor.count())
    for document in cursor:
        print(document['LinkID'], document['TimeStamp'])

username = raw_input("Enter username to mongodb:")
pwd = getpass.getpass()
coll = get_db('localhost', '30000', 'StmLinkNetwork', 'WatsonPlots', username, pwd)
getStm(coll)

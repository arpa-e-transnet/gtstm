import pymongo
import pprint
from bson.code import Code
def return_db_data(db_name, clxn_name):
	try:
		clxn_data = pymongo.MongoClient()[db_name].clxn_name
	except:
		print 'connection error'
	return clxn_data
#def get_speed_fields(db_data):
#	pipeline= [{"$unwind": "$ID"},{"$group":{"_id":"$ID","count":{"$sum":1}}}]
#	pprint.pprint(list(db_data.aggregate(pipeline)))
def print_am():
	pipeline = [{'$group':{'_id':'$@ID','mean':{'$avg':'$@Speed'}}}]
        c = pymongo.MongoClient()['AirSage'].command('aggregate','sample',pipeline=pipeline)
        print c

def print_hm():
	pipeline = [{'$group':{'_id':'$@ID','count':{'$sum':1},'rec_sp':{'$sum':{'$divide':[1,'$@Speed']}}}}]
	c = pymongo.MongoClient()['AirSage'].command('aggregate','sample',pipeline=pipeline)
	print c['result']
        pymongo.MongoClient()['AirSage'].intermediate.insert(c['result'])
	nested_pipeline=[{'$project':{'_id':'$_id','harmonic_speed':{'$divide':['$count','$rec_sp']}}}]
	for i in range(20):
		print '#'*50
	d = pymongo.MongoClient()['AirSage'].command('aggregate','intermediate',pipeline = nested_pipeline)
	print d

	

def main():
	db_data = return_db_data('AirSage','sample')
	#for doc in db_data.find({}):
		#print doc
	#pipe = [{'$group': {'_id': '@ID'}]
	#db_data.aggregate(pipeline=pipe)
	#key = ['@ID']
	#initial = {'count':0,'sum':0}
	#condition = {}
	#reduce = 'function(doc, out){out.count++;out.sum+=doc["@Speed"]}'
	#grouped_data = db_data.group(key,condition,initial, reduce)
	#print grouped_data
	#for entry in grouped_data:
	#	print entry
	print_am()
	print_hm()
	
if __name__ == "__main__":
	main()
 

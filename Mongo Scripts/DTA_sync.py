import os
import datetime
import csv
import pandas as pd
import pymongo
from collections import defaultdict
os.chdir(os.getcwd())
files = [f for f in os.listdir('.') if os.path.isfile(f) and '.csv' in f]
total_headers = defaultdict(list)
print "enter user name"
user_name = str(raw_input())
print "enter password"
pswd = str(raw_input())
for f in files:
	with open(f) as contents:
		current_header_str = contents.readline()		
		current_header_arr = current_header_str.split(",")
		current_header_arr[-1] = current_header_arr[-1].split("\r\n")[0]
		for head in current_header_arr:
			total_headers[head].append(f)
	
try:
	conn = pymongo.MongoClient(host=['localhost:30000'])
	conn.admin.authenticate(user_name,pswd)
	print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e

	
db = conn['StmLinkNetwork']
DTA_df = pd.read_csv("Link_Grids_Nodes_ValidSpeed_stm_0227.csv")
collection = db.WatsonPlots
try:
	StmRow = collection.find()
	for row in StmRow:
		c = dict(row)
		#print c.keys()
		#print type(row),row
		break
except Exception,e:
	print str(e)		

'''print actual_headers
print "*"*200
print total_headers.keys()'''
entry_mapping = {}
'''loading attribute table data'''
attribute_dict = {}
db = conn['Stm_Attributes']
coll_attr = db.attr_stm_local
'''
with open("Export_Output.csv","rb") as infile:
	reader = csv.reader(infile)
	header = next(reader)
	for row_attr in reader:
		entry = dict(zip(header,row_attr))
		coll_attr.insert(entry)'''
#print "Number of records inserted", coll_attr.count()		
'''loading AirSage Data to Stm'''
#conn.admin.command('shardCollection','Stm_AirSage_DTA.AirSage_DTA_15_new',key={"stmLinkId":1})
def return_collection(coll_arg):
	db_cur_atr = conn['Stm_Attributes']
	coll_dict = {"opModeBinDistribution.csv":db_cur_atr["opModeBinDistribution.csv"],"ABM15_AM.csv":db_cur_atr["ABM15_AM.csv"],"ABM15_EA.csv":db_cur_atr["ABM15_EA.csv"],"ABM15_MD.csv":db_cur_atr["ABM15_MD.csv"],"ABM15_EV.csv":db_cur_atr["ABM15_EV.csv"],"ABM15_PM.csv":db_cur_atr["ABM15_PM.csv"]}
	return coll_dict[coll_arg]
db_Intermediate = conn['AirSage_Intermediate']
collection_airsage = db_Intermediate.SpeedLink
coll_dummy = db_Intermediate.dummy
print "*"*1000
print "beginning bulk insert"
try:
	AirSage_Row = collection_airsage.find(no_cursor_timeout=True)
	for idx,row in enumerate(AirSage_Row):
		#print "printing airsage row",idx,row
		entry_mapping = {}
		entry_mapping["stmSource"]="AirSage_weekday_15_min_avg"
		linkid_objectid = row["ID"]
		#print "filtering for",linkid_objectid
		timestamp = row["LastUpdateTime"]
		day_of_the_week = timestamp.weekday()
		hour = timestamp.hour
		min_window = (timestamp.minute%4)*15	
		attr_dict_object = coll_attr.find({"OBJECTID":str(linkid_objectid)})
		for doc in attr_dict_object:
		#	print "Attribute dictionary document",doc
			pass
		entry_mapping["stmLinkId"]=doc["A"]+"_"+doc["B"]
		print "time values", row["ID"], day_of_the_week, hour, min_window
		query =  db_Intermediate.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week,"_id.hour":hour,"_id.min_window":min_window})
		for x in query:
			#print x
			speed = int(x["value"]["harmonic_speed"])
			#print x["_id"]["day"]
			hour = str(int(x["_id"]["hour"])) if len(str(int(x["_id"]["hour"])))==2 else "0"+str(int(x["_id"]["hour"]))
			min_window = str(int(x["_id"]["min_window"])) if int(x["_id"]["min_window"])!=0 else "00"
			print DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_speed"]
			#print DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_ttime"]
			#print DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_dist"]
			#print "Before" * 30
			#print "*"*200,DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_speed"],type(DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_speed"])
			DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_speed"]=speed
			#DTA_entry[hour+min_window+"00_speed"]= speed
			#if DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"]][hour+min_window+"00_dist"]:
			#DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_ttime"]=float(DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_dist"])/float(speed)
			print DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_speed"]
                        #print DTA_df.loc[DTA_df["A_B"]==entry_mapping["stmLinkId"],hour+min_window+"00_ttime"]
                        #print "After" * 30

		#matched_values_link_id =db.HM_airsage_speed_script.find({"_id.ID":row["ID"]})
		#matched_values_link_day = db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week})
		#mathed_values_link_day_hour = db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week,"_id.hour":hour})
		#matched_values =  db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week,"_id.hour":hour,"_id.min_window":min_window})
		#hm_speed,hm_sp_l_d_h,hm_sp_l_d,hm_sp_l =None, None,None,None
		DTA_df.to_csv("AirSage_Speed_combined.csv",ignore_index = True)
		print "progress", idx/AirSage_Row.count()
except Exception as e:
	print(e)

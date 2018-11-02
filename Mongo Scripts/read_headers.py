import os
import datetime
import csv
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
	
actual_headers = ["stmId","stmSource","stmModelRunId","stmSampleSize","stmQuality","stmReliability","stmNetwork","stmLinkId","stmLinkName","stmLinkType","stmLinkDistance","stmLinkSpeedLimitArc","stmLinkSpeed","stmLinkSpeedImputed","stmLinkSpeedFreeFlow","stmLinkTime","stmLinkTimeFreeFlow","stmLinkDensity","stmLinkFlow","stmLinkCapacity","stmAbm15LinksLanes","stmAbm15LinksLanesAux","stmAbm15LinksAccess","stmAbm15LinksProhibitions","stmAbm15LinksWeaveFlag","stmTollData","stmAbm15LinksNodeA","stmAbm15LinksNodeB","stmAbm15LinksTaz","stmLinkSourceTypeDistribution","stmVehicleSourceTypeId","stmVehicleSourceTypeName","stmWatsonPlotId","stmWatsonPlotModelRunId","stmWatsonPlotVehicleSourceType","stmWatsonPlotVehicleSourceTypeName","stmOpModeBinDistributionId","stmOpModeBinDistributionModelRunId","stmOpModeBinDistributionVehicleSourceTypeId","stmOpModeBinVehicleSourceTypeName","stmOpModeBinDistributionPercents","stmIndicatorFileId","stmIndicatorFileModelRunId","stmIndicatorFileVehicleSourceTypeId","stmIndicatorFileVehicleSourceTypeName","stmIndicators","stmCostPerVehicleMile","stmEnergy","stmEmissions","stmCost","stmLinkCoding","stmTimeStamp","stmTimeDuration","stmComments","lastModified","stmAbm15LinksNodeA|stmAbm15LinksAccess|stmAbm15LinksAccess|stmAbm15LinksAreaType|stmAbm15LinksLanesAux|stmAbm15LinksNodeB|stmAbm15LinksCapacityAm|stmAbm15LinksCapacityMd|stmAbm15LinksCapacityNt|stmAbm15LinksCapacityPm|stmAbm15LinksCountyId|stmAbm15LinksCountyName|stmAbm15LinksCountDailyAverage|stmAbm15LinksDistance|stmAbm15LinksHovGpFactor|stmAbm15LinksFacilityType|stmAbm15LinksFacilityTypeHpms|stmAbm15LinksTollType|stmAbm15LinksFreewaySegment|stmAbm15LinksGpId|stmAbm15LinksHovMerge|stmAbm15LinkLinksId|stmAbm15LinksLanes|stmAbm15LinksLanesAm|stmAbm15LinksLanesEa|stmAbm15LinksLanesEv|stmAbm15LinksLanesMd|stmAbm15LinksLanesNt|stmAbm15LinksLanesPm|stmAbm15LinksLos|stmAbm15LinksSpeedBusMax|stmAbm15LinksCmsId|stmAbm15LinksMedianType|stmAbm15LinksSpeedBusMin|stmAbm15LinksName|stmAbm15LinksNumberHdt|stmAbm15LinksNumberMdt|stmAbm15LinksNumberCar|stmAbm15LinksProhibit|stmAbm15LinksProjectId|stmAbm15LinksSpeedLimitGdot|stmAbm15LinksShoulder|stmAbm15LinksSignalPreemtion|stmAbm15LinksSpeedFreeFlow|stmAbm15LinksSpeedLimitArc|stmAbm15LinksStrategic|stmAbm15LinksTaz|stmAbm15LinksTimeCongested|stmAbm15LinksTimeFreeFlow|stmAbm15LinksTmc|stmAbm15LinksTmcFlag|stmAbm15LinksTollAm|stmAbm15LinksTollId|stmAbm15LinksTollMd|stmAbm15LinksTollNt|stmAbm15LinksTollPm|stmAbm15LinksVehicleClassificationStation|stmAbm15LinksVhtTotalDaily|stmAbm15LinksVmtTotalDaily|stmAbm15LinksWeaveFlag|stmAbm15LinksAreaType|stmAbm15LinksLanesAux|stmAbm15LinksNodeB|stmAbm15LinksCapacityAm|stmAbm15LinksCapacityMd|stmAbm15LinksCapacityNt|stmAbm15LinksCapacityPm|stmAbm15LinksCountyId|stmAbm15LinksCountyName|stmAbm15LinksCountDailyAverage|stmAbm15LinksDistance|stmAbm15LinksHovGpFactor|stmAbm15LinksFacilityType|stmAbm15LinksFacilityTypeHpms|stmAbm15LinksTollType|stmAbm15LinksFreewaySegment|stmAbm15LinksGpId|stmAbm15LinksHovMerge|stmAbm15LinkLinksId|stmAbm15LinksLanes|stmAbm15LinksLanesAm|stmAbm15LinksLanesEa|stmAbm15LinksLanesEv|stmAbm15LinksLanesMd|stmAbm15LinksLanesNt|stmAbm15LinksLanesPm|stmAbm15LinksLos|stmAbm15LinksSpeedBusMax|stmAbm15LinksCmsId|stmAbm15LinksMedianType|stmAbm15LinksSpeedBusMin|stmAbm15LinksName|stmAbm15LinksNumberHdt|stmAbm15LinksNumberMdt|stmAbm15LinksNumberCar|stmAbm15LinksProhibit|stmAbm15LinksProjectId|stmAbm15LinksSpeedLimitGdot|stmAbm15LinksShoulder|stmAbm15LinksSignalPreemtion|stmAbm15LinksSpeedFreeFlow|stmAbm15LinksSpeedLimitArc|stmAbm15LinksStrategic|stmAbm15LinksTaz|stmAbm15LinksTimeCongested|stmAbm15LinksTimeFreeFlow|stmAbm15LinksTmc|stmAbm15LinksTmcFlag|stmAbm15LinksTollAm|stmAbm15LinksTollId|stmAbm15LinksTollMd|stmAbm15LinksTollNt|stmAbm15LinksTollPm|stmAbm15LinksVehicleClassificationStation|stmAbm15LinksVhtTotalDaily|stmAbm15LinksVmtTotalDaily|stmAbm15LinksWeaveFlag"]
#print actual_headers
try:
	conn = pymongo.MongoClient(host=['localhost:30000'])
	conn.admin.authenticate(user_name,pswd)
	print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e

db_attr = conn['Stm_Attributes']
for uniq_files in files:
	current_coll = db_attr[uniq_files]
	csv_reader_file = csv.reader(uniq_files)
	header_csv = next(csv_reader_file)
	for othr_rows in csv_reader_file:
		row_dict = dict(zip(header_csv,othr_rows))
		current_coll.insert(row_dict)
	
db = conn['StmLinkNetwork']
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
db = conn['AirSage_Intermediate']
db_final = conn["Stm_AirSage_DTA"]
#conn.admin.command('shardCollection','Stm_AirSage_DTA.AirSage_DTA_15_new',key={"stmLinkId":1})
coll_final = db_final.AirSage_DTA_15_new
def return_collection(coll_arg):
	db_cur_atr = conn['Stm_Attributes']
	coll_dict = {"opModeBinDistribution.csv":db_cur_atr["opModeBinDistribution.csv"],"ABM15_AM.csv":db_cur_atr["ABM15_AM.csv"],"ABM15_EA.csv":db_cur_atr["ABM15_EA.csv"],"ABM15_MD.csv":db_cur_atr["ABM15_MD.csv"],"ABM15_EV.csv":db_cur_atr["ABM15_EV.csv"],"ABM15_PM.csv":db_cur_atr["ABM15_PM.csv"]}
	return coll_dict[coll_arg]
collection_airsage = db.SpeedLink
print "*"*1000
print "beginning bulk insert"
try:
	AirSage_Row = collection_airsage.find(no_cursor_timeout=True)
	for idx,row in enumerate(AirSage_Row):
		print "printing airsage row",idx,row
		entry_mapping = {}
		entry_mapping["stmSource"]="AirSage_weekday_15_min_avg"
		linkid_objectid = row["ID"]
		entry_mapping["stmModelRunId"]=None
		entry_mapping["stmSampleSize"]=None
		entry_mapping["stmQuality"]=None
		entry_mapping["stmReliability"]= None
		entry_mapping["stmNetwork"]= "abm15_202k"
		print "filtering for",linkid_objectid
		timestamp = row["LastUpdateTime"]
		day_of_the_week = timestamp.weekday()
		hour = timestamp.hour
		min_window = (timestamp.minute%12)*5	
		attr_dict_object = coll_attr.find({"OBJECTID":str(linkid_objectid)})
		for doc in attr_dict_object:
			#print "Attribute dictionary document",doc
			pass
		entry_mapping["stmLinkId"]=doc["A"]+"_"+doc["B"]
		entry_mapping["stmLinkName"] = doc["NAME"]
		entry_mapping["stmLinkType"] = doc["FACTYPE"]
		entry_mapping["stmLinkDistance"]=doc["DISTANCE"]
		entry_mapping["stmLinkSpeedLimitArc"]=doc["SPEEDLIMIT"]
		print "time values", row["ID"], day_of_the_week, hour, min_window
		matched_values_link_id =db.HM_airsage_speed_script.find({"_id.ID":row["ID"]})
		matched_values_link_day = db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week})
		matched_values_link_day_hour = db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week,"_id.hour":hour})
		matched_values =  db.HM_airsage_speed_script.find({"_id.ID":row["ID"],"_id.day":day_of_the_week,"_id.hour":hour,"_id.min_window":min_window})
		hm_speed,hm_sp_l_d_h,hm_sp_l_d,hm_sp_l =None, None,None,None
		print "!"*200
		print list(matched_values)
		print list(matched_values_link_day_hour)
		print list(matched_values_link_day)
		print list(matched_values_link_id)
		entry_mapping[str(hour)+str(min_window)+"00"]=None
		for avg_sp_row in matched_values:
			#print "HM_row", avg_sp_row
			if not hm_speed and avg_sp_row:
				hm_speed = avg_sp_row["values"]["harmonic_speed"]
			
		for avg_l_d_h in matched_values_link_day_hour:
			#print "HM_row 1", avg_l_d_h
			if avg_l_d_h:
				entry_mapping[str(hour)+str(min_window)+"00"]=avg_l_d_h["values"]["harmonic_speed"]
			if not hm_sp_l_d_h and avg_l_d_h:
				hm_sp_l_d_h = avg_l_d_h["values"]["harmonic_speed"]
				#entry_mapping[str(hour)+str(min_window)+"00"]=avg_l_d_h["values"]["harmonic_speed"]
			
		for avg_l_d in matched_values_link_day:
			#print "HM_row 2", avg_l_d
			if not hm_sp_l_d and avg_l_d:
				hm_sp_l_d = avg_l_d["values"]["harmonic_speed"]
							
		for avg_l in matched_values_link_id:
			#print "HM_row 3", avg_l
			if not hm_sp_l and avg_l:
				hm_sp_l = avg_l["values"]["harmonic_speed"]
		hm_speed = hm_speed or hm_sp_l_d_h or hm_sp_l_d or hm_sp_l or 22
		entry_mapping["stmLinkSpeed"]= hm_speed
		print "hm_speed", hm_speed
		db_opMode = return_collection("opModeBinDistribution.csv").find({"opModeBinDistributionVehicleSourceType":"21","opModeBinDistributionAverageSpeed":str(int(row["Speed"]))})
		print "opmode values", "*"*200
		#print "$"*200, hm_speed
		for op_mod in db_opMode:
			print op_mod
		entry_mapping["stmLinkSpeedImputed"]=None
		entry_mapping["stmLinkSpeedFreeFlow"]=None
		entry_mapping["stmLinkTime"]=None
		entry_mapping["stmLinkTimeFreeFlow"]=None
		entry_mapping["stmLinkDensity"]=None
		entry_mapping["stmLinkFlow"]=None
		entry_mapping["stmLinkCapacity"]=None
		print entry_mapping["stmLinkId"]
		capacity_str = ""
		doc_AM, doc_EA,doc_MD, doc_EV,doc_PM = None, None,None,None,None
		for doc_AM in return_collection("ABM15_AM.csv").find({"A":doc["A"],"B":doc["B"]}):
			pass
                for doc_EA in return_collection("ABM15_EA.csv").find({"A":doc["A"],"B":doc["B"]}):
                        pass
                for doc_MD in return_collection("ABM15_MD.csv").find({"A":doc["A"],"B":doc["B"]}):
                        pass
                for doc_EV in return_collection("ABM15_EV.csv").find({"A":doc["A"],"B":doc["B"]}):
                        pass
                for doc_PM in return_collection("ABM15_PM.csv").find({"A":doc["A"],"B":doc["B"]}):
                        pass
		if not doc_AM and not doc_EA and not doc_MD and not doc_EV and not doc_PM:
			capacity_str = ""
		else:
			capacity_str += doc_AM["AMCAPACITY"]+"|"+doc_EA["EACAPACITY"]+"|"+doc_MD["MDCAPACITY"]+"|"+doc_EV["EVCAPACITY"]+"|"+doc_PM["PMCAPACITY"]
		print doc_AM, '*'*200, doc_EA, '*'*200, doc_MD, '*'*200, doc_EV,'*'*200, doc_PM
		entry_mapping["stmLinkCapacity"]=capacity_str
		entry_mapping["stmAbm15LinksLanes"]=doc["LANES"]
		entry_mapping["stmAbm15LinksLanesAux"]=doc["AUXLANE"]
		entry_mapping["stmAbm15LinksAccess"]=None
		entry_mapping["stmAbm15LinksProhibition"]=doc["PROHIBIT"]
		entry_mapping["stmAbm15LinksWeaveFlag"]=doc["WEAVEFLAG"]
		entry_mapping["stmTollData|ID|AM|EA|MD|EV|PM"]=doc["TOLLID"]+'|'+'None'
		entry_mapping["stmAbm15LinksNodeA"]=doc["A"]
		entry_mapping["stmAbm15LinksNodeB"]=doc["B"]
		entry_mapping["stmAbm15LinksTaz"]=None #doc["TAZ"]
		entry_mapping["stmLinkSourceTypeDistribution"]=None
		entry_mapping["stmVehicleSourceTypeId"]='Passenger cars'
		entry_mapping["stmVehicleSourceTypeName"]='Passenger cars'
		entry_mapping["stmWatsonPlotId"]='Passenger cars'+'|'+entry_mapping["stmLinkType"]+'|'+str(int(hm_speed)) 
		entry_mapping["stmWatsonPlotModelRunId"]= 1
		entry_mapping["stmWatsonPlotVehicleSourceType"]='Passenger cars'
		entry_mapping["stmWatsonPlotVehicleSourceTypeName"]= 'Passenger cars'
		entry_mapping["stmOpModeBinDistributionId"]=op_mod['opModeBinDistributionVehicleSourceType']+'|'+'00'+'|'+str(op_mod['opModeBinDistributionAverageSpeed'])
		entry_mapping["stmOpModeBinDistributionModelRunId"]=op_mod['opModeBinDistributionModelRunId']
		entry_mapping["stmOpModeBinDistributionVehicleSourceTypeId"]=op_mod['opModeBinDistributionVehicleSourceType']
		entry_mapping["stmOpModeBinVehicleSourceTypeName"]=op_mod['opModeBinDistributionVehicleSourceTypeName']
		entry_mapping["stmOpModeBinDistributionPercents|00|01|12|13|14|15|16|21|22|23|24|25|27|28|29|30|33|35|37|38|39|40"]=str(op_mod['opModeBinDistributionPercentOpModeBin00'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin01'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin12'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin13'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin14'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin15'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin16'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin21'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin22'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin23'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin24'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin25'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin27'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin28'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin29'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin30'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin33'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin35'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin37'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin38'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin39'])+'|'+str(op_mod['opModeBinDistributionPercentOpModeBin40'])
		entry_mapping["stmIndicatorFileId"]=None
		entry_mapping["stmIndicatorFileModelRunId"]=None
		entry_mapping["stmIndicatorFileVehicleSourceTypeId"]=None
		entry_mapping["stmIndicatorFileVehicleSourceTypeName"]=None
		entry_mapping["stmIndicators"]=None
		entry_mapping["stmCostPerVehicleMile"]=None
		entry_mapping["stmEnergy"]=None
		entry_mapping["stmEmissions"]=None
		entry_mapping["stmCost"]=None
		entry_mapping["stmLinkCoding"]=None
		entry_mapping["stmTimeStamp"]=datetime.datetime(2017,01,01,hour,min_window,0)
		entry_mapping["stmTimeDuration"]='15 minutes'
		entry_mapping["stmComments"]=None
		entry_mapping["lastModified"]=row["LastUpdateTime"] 
		print entry_mapping
		coll_final.insert(entry_mapping)

	AirSage_Row.close()
except Exception,e:
	print str(e)

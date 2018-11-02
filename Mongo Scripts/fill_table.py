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
	num_lines = sum(1 for line in open(uniq_files))
	print "file, number of lines", uniq_files, num_lines
	csv_reader_file = open(uniq_files)
	header_csv = next(csv_reader_file)
	print header_csv
	header_arr = header_csv.split(",")
	print header_arr
	header_arr[-1]=header_arr[-1].split("/r/n")[0]
	for idx,othr_rows in enumerate(csv_reader_file):
		print "Adding line" + str(idx) + " of "+ str(num_lines)
		print othr_rows
		other_rows = othr_rows.split(",")
		print other_rows
		other_rows[-1]=other_rows[-1].split("/r/n")[0]
		row_dict = dict(zip(header_arr,other_rows))
		print row_dict
		#if idx==20:
		#	break
		current_coll.insert(row_dict)
	
'''db = conn['StmLinkNetwork']
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
'''
'''print actual_headers
print "*"*200
print total_headers.keys()'''

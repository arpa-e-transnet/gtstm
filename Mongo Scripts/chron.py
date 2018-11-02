import os
import json
import xmljson
from lxml.etree import fromstring, tostring
import xml.etree.ElementTree as ET
import pymongo
import datetime
import schedule
import time


def job():
	path = os.getcwd()
	os.chdir(path)
	current_path_folders = os.listdir(path)
	print 'folders in current path', current_path_folders
	try:
		#conn = pymongo.MongoClient(host=['localhost:30000')
		conn = pymongo.MongoClient('mongodb://venkat:d_v123@localhost:30000/')
		print 'Connected successfully'
	except pymongo.errors.ConnectionFailure, e:
		print 'Could not connect, %s'%e

	db = conn['AirSage']
	collection = db.SpeedLink
	for file_or_folder in current_path_folders:
		if os.path.isdir(file_or_folder):
			print 'printing in loop current folders', file_or_folder
			data_files_path = os.path.join(path,file_or_folder)
			data_files_days = os.listdir(data_files_path)
			data = []
			#os.chdir(data_files_path)
			for sub_folder in data_files_days:
				xml_file_path = os.path.join(data_files_path,sub_folder)
				os.chdir(xml_file_path)
				xml_file_list = os.listdir(xml_file_path)
				for xml_file in xml_file_list:
					if db.existing_files.find({"file":xml_file}).count() == 0:
						db.existing_files.insert({"file":xml_file})
						print 'printing file name',xml_file
						if '.xml' in xml_file:
							xml_str = open(xml_file, 'r').read()
							xml_content = fromstring(xml_str)
							xml_to_json = xmljson.badgerfish.data(xml_content)
							try:
								list_of_records = xml_to_json['Events']['Event']
								for record in list_of_records:
									if isinstance(record,dict):
										keys_dict = record.keys()
										new_keys = []
										new_values = []
										for idx,key in enumerate(keys_dict):
											print key
											new_keys.append(key.split("@")[1])
											if key=="@LastUpdateTime":
												whole_date_str = record.values()[idx]
												date_split, time_stamp = whole_date_str.split(" ")
												dmy_split = date_split.split("/")
												hms_split = time_stamp.split(":")
												new_values.append(datetime.datetime(int(dmy_split[2]),int(dmy_split[1]),int(dmy_split[0]),int(hms_split[0]),int(hms_split[1]),int(hms_split[2])))
											else:
												new_values.append(record.values()[idx])
											print key, record.values()[idx]
										n_dict = dict(zip(new_keys, new_values))	 
										print n_dict
										collection.insert(n_dict)
							except Exception,e:
								print 'problem reading file', xml_file
								print str(e)
							print 'completed inserting', xml_file				
							print 'no of records', collection.count()
schedule.every(10).minutes.do(job)
while 1:
	schedule.run_pending()
	time.sleep(1)

import paramiko
import os
import json
#import xmljson
#from lxml.etree import fromstring, tostring
#import xml.etree.ElementTree as ET
import pymongo
import datetime
import csv
from collections import defaultdict
import pandas as pd
import json
#def get_records(records, start_time, timedelta):
path = "/mnt/data/load_to_mongo/2017"
print "Enter db pswd"
pswd = str(raw_input())

try:
        conn = pymongo.MongoClient(host=['localhost:30000'])
        conn.admin.authenticate("venkat",pswd)
        print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e
#print "Enter sys password"
#u_name = 'deepthi7'
#paswd = str(raw_input())
total_DataFrame = pd.DataFrame()
inter_db = conn["Navigator"]
coll = inter_db["Speed_Timestamp"]
header_names = ["Detect_ID","Device_ID","Timestamp","Volume","Dummy_1","Speed","Dummy_2","Dummy_3"]
current_path_folders = os.listdir(path)
print "current path", current_path_folders
for file_or_folder in current_path_folders:
        print 'printing in loop current folders', file_or_folder
        data_files_path = os.path.join(path,file_or_folder)
	if os.path.isdir(data_files_path):
        	data_files_days = os.listdir(data_files_path)
		for idx,files in enumerate(data_files_days):
			f_path = os.path.join(data_files_path,files)
			"""reader = csv.DictReader(f_path)

			for each in reader:
				print each
				row = {}
				for head_idx,field in enumerate(header):
					row[field] = each[head_idx]
				coll.insert(row)"""
			file_content = pd.read_csv(f_path,names = header_names)
			"""
			file_content = pd.read_csv(f_path,header = header_names, dtype={'Dummy':float,'Detect_ID':float,'Device_ID':float,'Timestamp':str, 'Volume':float,'Dummy_1':float,'Speed':float,'Dummy_2':float,'Dummy_3':float})
			"""
			print file_content
			file_content.drop_duplicates()
			data_json = json.loads(file_content.to_json(orient='records'))
			coll.insert(data_json,check_keys=False)
			#total_DataFrame = total_DataFrame.append(file_content)
			#total_DataFrame.drop_duplicates()
			#print total_DataFrame
			#records = json.loads(total_DataFrame.T.to_json()).values()
			#print records
			#coll.insert(records)
	"""
                        data = []
                        #os.chdir(data_files_path)
                        for sub_folder in data_files_days:
                                xml_file_path = os.path.join(data_files_path,sub_folder)
                                os.chdir(xml_file_path)
                                xml_file_list = os.listdir(xml_file_path)
				print xml_file_list
                                for xml_file in xml_file_list:
                                        if db.existing_files.find({"file":xml_file}).count() == 0:
                                                db.existing_files.insert({"file":xml_file})
                                                print 'printing file name',xml_file
                                                total_files += 1
                                                if '.xml' in xml_file:
                                                        xml_str = open(xml_file, 'r').read()
                                                        xml_content = fromstring(xml_str)
                              """

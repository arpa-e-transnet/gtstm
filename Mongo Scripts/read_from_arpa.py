import paramiko
import os
import json
import xmljson
from lxml.etree import fromstring, tostring
import xml.etree.ElementTree as ET
import pymongo
import datetime
from collections import defaultdict
#def get_records(records, start_time, timedelta):
print "Enter db pswd"
pswd = str(raw_input())
	
try:
        conn = pymongo.MongoClient(host=['localhost:30000'])
	conn.admin.authenticate("venkat",pswd)
        print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e
print "Enter sys password"
u_name = 'deepthi7'
paswd = str(raw_input())
myconn = paramiko.SSHClient()
inter_db = conn["AirSage_Intermediate"]
coll = inter_db["SpeedLink"]
myconn.set_missing_host_key_policy(paramiko.AutoAddPolicy())
myconn.connect("rg49-arpaweb2.ce.gatech.edu",22,u_name,paswd,timeout=20)
sftp = myconn.open_sftp()
(st_in,st_out,st_er) = myconn.exec_command('find /export/home1/www/commwar/httpsdocs/cwairsage/data/TT/2018/01/31')
c = st_out.read()
count_rec = 0
file_list = c.split("\n")
for idx,f_l in enumerate(file_list):
	#print f_l
	file_dict = defaultdict(list)
	if '.xml' in f_l:
		xml_str = sftp.open(f_l).read()
		#print xml_str
		xml_content = fromstring(xml_str)
                xml_to_json = xmljson.badgerfish.data(xml_content)
                try:
                	list_of_records = xml_to_json['Events']['Event']
                        for record in list_of_records:
				#print "record",record
				count_rec += 1
                        	if isinstance(record,dict):
                                	keys_dict = record.keys()
                                        new_keys = []
                                        new_values = []
                                        for idx,key in enumerate(keys_dict):
                                        	new_keys.append(key.split("@")[1])
                                               	if key=="@LastUpdateTime":
                                                        whole_date_str = record.values()[idx]
                                                        date_split, time_stamp = whole_date_str.split(" ")
                                                        dmy_split = map(int,date_split.split("/"))
                                                        hms_split = map(int,time_stamp.split(":"))
							#print "dates",dmy_split,hms_split
                                                        new_values.append(datetime.datetime(dmy_split[2],dmy_split[1],dmy_split[0],hms_split[0],hms_split[1],hms_split[2]))
							#print "datetime", new_values
						else:
							new_values.append(record.values()[idx])
					n_dict = dict(zip(new_keys, new_values))
					file_dict[n_dict["ID"]].append(n_dict)
						#print "final dict",n_dict
						#print "file dict", file_dict
                                        coll.insert(n_dict)
		except:
                	#print 'problem reading file', f_l
                        #print 'completed inserting', f_l
			print "no of records read", count_rec
                        print 'no of records inserted', coll.count()

	print "no of records read", count_rec
        print 'no of records inserted', coll.count()

import os
import json
import xmljson
from lxml.etree import fromstring, tostring
import xml.etree.ElementTree as ET
import pymongo
xml_str = open('tt_file_2017_07_01_09_32_511.xml', 'r').read()
xml_content = fromstring(xml_str)
xml_to_json = xmljson.badgerfish.data(xml_content)
try:
        conn = pymongo.MongoClient()
        print 'Connected successfully'
except pymongo.errors.ConnectionFailure, e:
        print 'Could not connect, %s'%e

db = conn['AirSage']
collection = db.sample
try:
	list_of_records = xml_to_json['Events']['Event']
        for record in list_of_records:
        	if isinstance(record,dict):
			collection.insert(record)
except:
	print 'problem reading file', xml_file


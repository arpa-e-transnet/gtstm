import csv
#import geopy.distance
import pandas as pd
import numpy as np
from collections import defaultdict
device_arr = []
detector_arr=[]
print("starting  script")
df_device = pd.read_csv("vds_device_20180212T1055.csv",index_col=False, header = 0, error_bad_lines = False,low_memory = False)
df_detector = pd.read_csv("vds_detector_20180212T10.csv",index_col=False, header=0, low_memory=False)


#df_device = pd.read_csv("vds_device_20180212T1055.csv",index_col=False, header = 0, error_bad_lines = False,dtype={'device_id':'int', 'version':'int', 'external_id':'int', 'name':'str', 'description':'str', 'agency':'str', 'make':'str','model':'str','vendor_id':'str' ,'active':'str','channel_uri':'str' ,'protocol_id':'int','deployed_by':'str'
# ,'deployed_date':'str' ,'modified_by':'str', 'modified_date':'str' ,'latitude':'float64' ,'longitude':'float64'
# ,'elevation':'int' ,'length':'int' ,'angle':'unicode' ,'orientation':'unicode' ,'mile_marker':'int' ,'road_type':'unicode'
# ,'primary_road':'unicode' ,'cross_road':'unicode' ,'direction':"unicode" ,'state':'unicode' ,'county':'unicode' ,'city':'unicode'
# ,'district':'unicode' ,'native_device_id':'int','vds_type':'int'})

print df_device.columns.values
print df_detector.columns.values

detector_device_mapping = {}
location_diction = {}
for idx,row in df_detector.iterrows():
	print idx
        current_detector = row["detector_id"]
        device_for_detector = row["device_id"]
        detector_device_mapping[row["detector_id"]] = row["device_id"]
        devices_columns = df_device.loc[df_device["device_id"]==device_for_detector]
        for idx_1,rw in devices_columns.iterrows():
                #print rw['device_id'],rw['latitude'], rw['longitude']
                if not np.isnan(rw['latitude']) and not np.isnan(rw['longitude']):
                        location_diction[rw['device_id']]=(rw['direction'],rw['latitude'], rw['longitude'])

print "device detector mapping complete"
filtered_devices = df_device[df_device["primary_road"]=="I-85"]["device_id"]
print filtered_devices
filtered_devices.to_csv('your.csv', index=False)
"""device_detector_mapping = defaultdict(list)
for k, v in detector_device_mapping.items():
	device_detector_mapping[v].append(k)

filtered_device_detector_mapping = defaultdict(list)
for k,v in device_detector_mapping.items():
	if k in filtered_devices:
		filtered_device_detector_mapping[k].append(v)

filtered_detector_values = filtered_device_detector_mapping.values()
filtered_detectors =  []
for vals in filtered_detector_values:
	filtered_detectors.extend(vals)

print "filtering complete"

with open('vds_device_20180212T1055.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
                header_device = row
		break
with open('vds_detector_20180212T10.csv', mode='r') as infile:
	reader = csv.reader(infile)
	for row in reader:
		header_detector = row
		break

with open('vds_device_20180212T1055.csv', mode='r') as infile:
        reader = csv.reader(infile)
	next(reader)
        for row in reader:
		if row["device_id"] in filtered_devices:
                	device_arr.append(dict(zip(header_device,row)))

print device_arr
with open('vds_detector_20180212T10.csv', mode='r') as infile:
        reader = csv.reader(infile)
	next(reader)
        for row in reader:
		corres_device = detector_device_mapping[row["detector_id"]]
		if corres_device in filtered_devices or row["detector_id"] in filtered_detectors:
                	detector_arr.append(dict(zip(header_detector,row)))
print detector_arr

mile_marker_direction = defaultdict(dict)
for entry in device_arr:
	mile_marker_direction[entry['direction']][entry['mile_marker']]= entry['device_id']

print mile_marker_direction.items()[:10]
device_mile_marker_dict = {}

with open('vds_device_20180212T1055.csv', mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)
        for row in reader:
		row_dict = dict(zip(header_device,row))
                device_mile_marker_dict[row_dict['device_id']] = row_dict['mile_marker']
"""	
"""df_device = pd.read_csv("vds_device_20180212T1055.csv",index_col=False, header = 0, error_bad_lines = False)
df_detector = pd.read_csv("vds_detector_20180212T10.csv",index_col=False, header=0)
detector_device_mapping = {}
location_diction = {}
for idx,row in df_detector.iterrows():
        current_detector = row["detector_id"]
        device_for_detector = row["device_id"]
	detector_device_mapping[row["detector_id"]] = row["device_id"]
        devices_columns = df_device.loc[df_device["device_id"]==device_for_detector]
        for idx_1,rw in devices_columns.iterrows():
                #print rw['device_id'],rw['latitude'], rw['longitude']
                if not np.isnan(rw['latitude']) and not np.isnan(rw['longitude']):
                        location_diction[rw['device_id']]=(rw['direction'],rw['latitude'], rw['longitude'])
"""

def get_sorted_mile_marker_device_ids(dictionary, current_device_mile_marker):
	all_mile_markers = dictionary.keys()
	further_mm = filter(lambda x: x>current_device_mile_marker, all_mile_markers)
	sorted_fmm = sorted(further_mm)
	return [dictionary[sfmm] for sfmm in sorted_fmm][:10]


## Train model on 2017 October, November, December data and predict for 2018 Jan data

"""with open('detector_device.csv','a') as f_csv:
	f_writer = csv.writer(f_csv)
        f_writer.writerow(['detector_id','device_id'])

curr_path = "/home/users/deepthi7/LSTM/2008/01/train"
files_in_dir = os.listdir(curr_path)
df_files = []
for f in files_in_dir:
	if ".csv" in f:
                        	#f_data = pd.read_csv(curr_path+'/'+f,header=['detect_id','timestamp','interval','vehicle count', 'not relevant', 'speed', 'not rel'])
        	f_data = pd.read_csv(curr_path+'/'+f,header=['detect_id','timestamp','interval','vehicle count', 'not relevant', 'speed', 'not rel'], dtypes={'detector_id':'int','timestamp':'str','interval':'int','vehicle count':'int','not relevant':'str','speed':'float','not rel':'str'}, parse_dates=['timestamp'])
		f_data = f_data.loc[df['detect_id']==filtered_detectors]
		for idx,cur_row in f_data.iterrows():
			current_detect = cur_row['detect_id']
			current_device = detector_device_mapping[current_detect]
			with open('detector_device.csv','a') as f_csv:
				f_writer = csv.writer(f_csv)
				f_writer.writerow([current_detect,current_device]) 			
			""""""location_attr = location_diction[current_device]
			mile_marker = device_mile_marker_dict[current_device]
			nearby_10_device_ids = get_sorted_mile_marker_device_ids(mile_marker_direction[location_attr[0]],mile_marker)
			current_lat_long = (location_attr[1], location_attr[2])
			detectors_for_nearby_devices = [detector_device_mapping[x] for x in nearby_10_device_ids]
			nearby_device_data = f_data.loc[f_data['detect_id']==detectors_for_nearby_devices]
			current_time_stamp = cur_row['timestamp']
			timestamp_range = [current_time_stamp+datetime.timedelta(seconds = 20), current_time_stamp-datetime.timedelta(seconds=20)]
			mask = (nearby_devices_data['timestamp']>=timestamp_range[0]) & (nearby_devices_data['timestamp']>=timestamp_range[1])
			nearby_timestamp_devices = nearby_devices_data.loc[mask]
			distance_neigh = []
			for i, near_row in nearby_timestamp_devices.iterrows():
				neigh_detect = near_row['detect_id']
				neigh_device = detector_device_mapping[neigh_detect]
				lat_long = (location_diction[neigh_device][1], location_diction[neigh_device[2]])	
				current_neigh_dist = vincenty(current_lat_long,lat_long).miles
				distance_neigh.append(current_neigh_dist)
			#distance_neigh = pd.Series(distance_neigh)
			nearby_timestamp_devices['dist']= pd.Series(distance_neigh,index=nearby_timestamp_devices.index)
			train_row = [0,cur_row['speed']]
			for i, near_row in nearby_timestamp_devices:
				train_row.append(near_row['dist'])
				train_row.append(near_row['speed'])
			minute_timestamp_range = [timestamp_range[0]+datetime.timedelta(minutes=15),timestamp_range[1]+datetime.timedelta(minutes=15)]
			target_mask =  (f_data['timestamp']>=minute_timestamp_range[0]) & (f_data['timestamp']>=minute_timestamp_range[1])
			target_dps = f_data.loc[target_mask]
			for j,ro in target_dps:
				train_y = [0,ro['speed']]
				break
			

		#df_files.append(f_data)
                        	
#dataset = pd.concat(df_files,ignore_index=True)

    



	
		
#print header_device
#print header_detector
#device_df = pd.DataFrame(device_arr)
#detector_df = pd.DataFrame(detector_arr)
#print device_df.columns.values
#print device_df['mile_marker']
#print detector_df"""


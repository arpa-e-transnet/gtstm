# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:54:13 2017

@author: Haobing
"""
#####################################################################
# <codecell> # upload module and path #############################

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPoint
from shapely.ops import nearest_points
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import math
import gdal
gdal.AllRegister()
gdal.UseExceptions()
import random
from pyproj import Proj, transform
dg_proj = Proj(init='epsg:4269')
ga_proj = Proj(init='epsg:2240')
## CHANGE PATH
path = "C:\\Users\hliu332\\Desktop\\transit_links_gis\\"
transit_path = "C:\\Users\hliu332\\Desktop\\trace_result\\"
f_m = 0.3048006096012192

#####################################################################
# <codecell> upload and build network #############################


import networkx as nx
## load a road network shapefile
global network, railstation_df,busstation_df,grtastation_df
network = gpd.read_file('C:\\Users\hliu332\\Desktop\\transit_links_gis\\atl_abm_new_conn1.shp')
network[['A','B']] = network[['A','B']].astype('str')
## build a network for shortest path algorithm
G = nx.DiGraph()
for i,row in network.iterrows():
    A = row['A']
    B = row['B']
    G.add_edge(A, B)
    G[A][B]['weight'] = row['Shape_Leng']
print 'network upload finished.'


#####################################################################
# <codecell> read rail, marta bus, and grta station information

## rail station
"""
railstation_df = pd.read_csv(path + 'rail_stop.csv')
railstation_df = railstation_df[['stop_id','stop_lat','stop_lon']]
railstation_df['stop_id'] = railstation_df['stop_id'].astype(str)

# marta bus station
busstation_df = pd.read_csv(path + 'bus_stop.csv')
busstation_df = busstation_df[['stop_id','nodeid']]
busstation_df['stop_id'] = busstation_df['stop_id'].astype(str)

# grta bus station
grtastation_df = pd.read_csv(path + 'grta_stop.csv')
grtastation_df = grtastation_df[['stop_id','nodeid']]
grtastation_df['stop_id'] = grtastation_df['stop_id'].astype(str)
"""

##################################################################
# <codecell> convert between ft geometry and longitude/latitude###
##################################################################

def transform_ft_to_digit(x,y):
    # input: x_o, y_o (longitude,latitude, US ft)
    # output: x, y (longitude, latitude)
    lon = transform(ga_proj,dg_proj, x*f_m,y*f_m)[0]
    lat = transform(ga_proj,dg_proj, x*f_m,y*f_m)[1]
    return lon, lat
    

def transform_digit_to_ft(x,y):
    # input: x_o, y_o (longitude,latitude, US ft)
    # output: x, y (longitude, latitude)
    lon = transform(dg_proj,ga_proj, x,y)[0]/f_m
    lat = transform(dg_proj,ga_proj, x,y)[1]/f_m
    return lon, lat

#####################################################################
# <codecell> drive module###########################################

#trip = pd.read_csv(path + 'haobing_drive.csv')
#trip['modeID'] = 1 ## NEED TO DELETE
def drive(trip):
    global network, railstation_df,busstation_df,grtastation_df
    ## merge these two, so we have geometric information appended to the driving route
    rt_list = pd.merge(network,trip,how='inner',on=['A','B'])
    ## for each link, change speed from mph to ft/second, since the length of link is in ft
    rt_list['speed_fts'] = rt_list['dist']/rt_list['time'] * 1.46667
    ## change from time in hour to tine in seconds (travel time spent in this link)
    for i in range(0,len(rt_list)):
        rt_list.loc[i,'time'] = rt_list.loc[i,'time'] * 3600
    ## make sure the link order is correct within the driving route
    rt_list = rt_list.sort_values(['sequence']).reset_index(drop=True)
    ## In summary, only the features listed below are needed as input
    ## geometry: geometry in ft for each link (node coodinator)
    ## speed: avg speed for the link in ft/s
    
    ## add avg speed ###
    rt_list = rt_list[['geometry','speed_fts','avgspeed']]
    ## algorithm ##
    v_list = []
    l_list = []
    deltx_list = []
    delty_list = []
    x_list = []
    y_list = []
    t_list = [0]
    d_list = [0]
    k = 0
    for i ,row in rt_list.iterrows():
        ctnd = len(row.geometry.coords[:])
        for cn in range(0,ctnd-1):
            v_list.append(row.speed_fts)
            x0 = row.geometry.coords[cn][0]
            y0 = row.geometry.coords[cn][1]
            x1 = row.geometry.coords[cn+1][0]
            y1 = row.geometry.coords[cn+1][1]
            x_list.append(x0)
            y_list.append(y0)
            deltx_list.append(x1-x0)
            delty_list.append(y1-y0)
            l = math.hypot(x1-x0, y1-y0)
            d_list.append(d_list[k]+l)
            l_list.append(math.hypot(x1-x0, y1-y0))
            t_list.append(t_list[k] + l/row.speed_fts)
            k += int(1)
    #print rt_list
            
    ## add avg speed ###
    rt_pt = gpd.GeoDataFrame(columns=('secondID','geometry','avgspeed'))
    rt_pt.loc[0] = [int(0),Point(rt_list.loc[0,'geometry'].coords[0]),rt_list.loc[0,'avgspeed']]
    step = 0
    l = len(v_list)
    m = 1
    
    for i in range(1, int(t_list[l])+1):
        for c in range(step,l):
            if (i > t_list[c]) and (i <= t_list[c+1]):
                t_left = i - t_list[c]
                l_left = t_left * v_list[c]
                x_new = x_list[c] + l_left*deltx_list[c]/l_list[c]
                y_new = y_list[c] + l_left*delty_list[c]/l_list[c]
                ## add avg speed ###
                rt_pt.loc[m] = [int(m),Point((x_new,y_new)), v_list[c]*0.3048]
                step = max(c-1,0)
                #print c,c+1
                m += 1
                break
        continue
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'drive'
    rt_pt['lat'] = 0.0
    rt_pt['lon'] = 0.0
    for i, row in rt_pt.iterrows():
        x = float(row['geometry'].x)
        y = float(row['geometry'].y)
        rt_pt.loc[i,'lon'],rt_pt.loc[i,'lat'] = transform_ft_to_digit(x,y)
    ## add avg speed ###
    rt_pt['time'] = rt_pt['secondID']
    return rt_pt[['modeID','mode','secondID','time','avgspeed','lat','lon']]
    
    #return rt_pt
    ### actually the rt_pt is the result (geometry per second)



#######################################################################
# <codecell> wait#################################################


def transit(trip):
    tmp_frames = []
    for i, row in trip.iterrows():
        #print type(row['route'])
        rr = str(row['route']).replace('/','---').replace('.0','')
        aa = row['A'].replace('/','---')
        bb = row['B'].replace('/','---')
        tmp_df = pd.read_csv(transit_path + rr+'__'+aa+'__'+bb+'.csv')
        tmp_frames.append(tmp_df)
    rt_pt = pd.DataFrame(pd.concat(tmp_frames))
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt = rt_pt.drop_duplicates(subset='time', keep="last").reset_index(drop=True)
    rt_pt['secondID'] = range(len(rt_pt))
    rt_pt = rt_pt.rename(columns={"speed (m/s)": "avgspeed"})
    return rt_pt[['modeID','mode','secondID','time','avgspeed','lat','lon']]
    
##################################################################
# <codecell> wait#################################################


def wait(trip):
    trip['time'] = trip['time']*3600
    trip = trip.round({'time': 0})
    rt_pt = gpd.GeoDataFrame(columns=('lat','lon','avgspeed'))
    t = 0
    lat = float(trip.loc[0,'B_loc'].split('_')[0])
    lon = float(trip.loc[0,'B_loc'].split('_')[1])
    for i, row in trip.iterrows():
        for j in range(int(row['time'])):
            rt_pt.loc[t] = [lat,lon,0]
            t+=1
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'wait'
    rt_pt['secondID'] = range(len(rt_pt))
    rt_pt['time'] = rt_pt['secondID']
    return rt_pt[['modeID','mode','secondID','time','avgspeed','lat','lon']]
    

##################################################################
# <codecell> walk#################################################


def walk(trip):
    trip['time'] = trip['time']*3600
    trip = trip.round({'time': 0})
    rt_pt = gpd.GeoDataFrame(columns=('secondID','lat','lon','avgspeed'))
    t = 0
    for i, row in trip.iterrows():
        for j in range(int(row['time'])):
            rt_pt.loc[t] = [t,'walk','walk',2]
            t+=1
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'walk'
    rt_pt['time'] = rt_pt['secondID']
    return rt_pt[['modeID','mode','secondID','time','avgspeed','lat','lon']]


##################################################################
# <codecell> implementation ######################################

#m_option = 'grta'
#comp_rt = pd.read_csv(path + 'haobing_grta_comp.csv')
# need to decide whether it is grta or marta involved route
#m_option = 'marta'
# uoload comprehensive route
comp_rt = pd.read_csv(path + 'routetest_new\\gtPnr_in.csv')
comp_rt = comp_rt[~(comp_rt['time']==0)].reset_index(drop=True)
#comp_rt['mode_new'] = comp_rt['mode']
#comp_rt.loc[comp_rt['mode']=='transit','mode_new'] = 'martabus'
#comp_rt.loc[comp_rt['route'].isin(['10909','10911','10912','10913',10909,10911,10912,10913]),'mode_new'] = 'martarail'
#if m_option == 'grta':
#    comp_rt.loc[comp_rt['mode']=='transit','mode_new'] = 'grta'
comp_rt['modeID'] = 1
for i in range(1,len(comp_rt)):
    if comp_rt.loc[i,'mode']==comp_rt.loc[i-1,'mode']:
        comp_rt.loc[i,'modeID']=comp_rt.loc[i-1,'modeID']
    else:
        comp_rt.loc[i,'modeID']=comp_rt.loc[i-1,'modeID']+1

## add avg speed ###
comp_rt['avgspeed'] = 0
comp_rt.loc[comp_rt['mode']=='walk','avgspeed'] = 2
comp_rt.loc[~comp_rt['mode'].isin(['wait','walk']),'avgspeed'] = \
(comp_rt[~comp_rt['mode'].isin(['wait','walk'])]['dist']/comp_rt[~comp_rt['mode'].isin(['wait','walk'])]['time']).values
## add avg speed ###

mode_uniq = comp_rt[['modeID','mode']].drop_duplicates(keep='first')
frames = []
for i, row in mode_uniq.iterrows():
    m = row['mode']
    trip = comp_rt[comp_rt['modeID']==row['modeID']].reset_index(drop=True)
    trip[['A','B']] = trip[['A','B']].astype('str')
    if m == 'drive':
        trip[['A','B']] = trip[['A','B']].astype('int').astype('str')
        frames.append(drive(trip))
    elif m == 'transit':
        frames.append(transit(trip))
    elif m == 'walk':
        frames.append(walk(trip))
    elif m == 'wait':
        frames.append(wait(trip))
   
trace_comp = pd.DataFrame(pd.concat(frames)).reset_index(drop=True)


for i, row in mode_uniq.iterrows():
    if row['mode'] == 'walk':
        trip = comp_rt[comp_rt['modeID']==row['modeID']].reset_index(drop=True)
        trip[['A','B']] = trip[['A','B']].astype('str')
        if not math.isnan(trip.loc[0,'A_loc']):
            st_lat, st_lon = [float(j) for j in trip.loc[0,'A_loc'].split('_')]
        else:
            st_lon, st_lat = trace_comp[trace_comp['modeID']==row['modeID']-1].iloc[[-1]][['lon','lat']].values.tolist()[0]
        if not math.isnan(trip.loc[0,'B_loc']):
            end_lat, end_lon = [float(j) for j in trip.loc[0,'B_loc'].split('_')]
        else:
            end_lon, end_lat = trace_comp[trace_comp['modeID']==row['modeID']+1].iloc[[0]][['lon','lat']].values.tolist()[0]
        cut = len(trace_comp[trace_comp['modeID']==row['modeID']])
        lat_list = np.linspace(st_lat,end_lat,cut)
        lon_list = np.linspace(st_lon,end_lon,cut)
        trace_comp.loc[trace_comp['modeID']==row['modeID'],'lon'] = lon_list
        trace_comp.loc[trace_comp['modeID']==row['modeID'],'lat'] = lat_list

trace_comp['sequence'] = range(0,len(trace_comp))
trace_comp.to_csv(path+'result_points.csv',index=False)


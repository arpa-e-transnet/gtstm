# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:54:13 2017

@author: Haobing
"""

import geopandas as gpd
import datetime
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPoint
from shapely.ops import nearest_points
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import math
#import gdal
#gdal.AllRegister()
#gdal.UseExceptions()
import random

def getTrace(driveFile):
    
    print("generating drive trace for:", driveFile)
    path = "/home/users/ywang936/transit_simulation-0710/Scripts/build_graph/traceBack/"
    f_m = 0.3048006096012192
    #load a road network shapefile
    network = gpd.read_file(path + 'route_2.shp')
    #load a driving route which composed of a list of links with (A,B) as linkID, and with speed in each link. Average speed for each link (given) - mph
    trip_csv = pd.read_csv(driveFile, header = 0)
    #merge these two, so we have geometric information appended to the driving route
    rt_list = pd.merge(network,trip_csv,how='inner',on=['A','B'])
    # for each link, change speed from mph to ft/second, since the length of link is in ft
    rt_list['speed_fts'] = rt_list['speed'] * 1.46667
    # change from time in hour to tine in seconds (travel time spent in this link)
    for i in range(0,len(rt_list)):
        rt_list.loc[i,'time'] = rt_list.loc[i,'duration'] * 3600
    # make sure the link order is correct within the driving route
    rt_list = rt_list.sort_values(['sequence']).reset_index(drop=True)

    # In summary, only the features listed below are needed as input
    # geometry: geometry in ft for each link (node coodinator)
    # speed: avg speed for the link in ft/s
    rt_list = rt_list[['geometry','speed_fts']]

    # method ############################################################################
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

    rt_pt = gpd.GeoDataFrame(columns=('secondID','geometry'))
    rt_pt.loc[0] = [int(0),Point(rt_list.loc[0,'geometry'].coords[0])]
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
                rt_pt.loc[m] = [int(m),Point((x_new,y_new))]
                step = max(c-1,0)
                #print c,c+1
                m += 1
                break
        continue
    return rt_pt
    ####################################################################################
    ### actually the rt_pt is the result (geometry per second)

def trace2LatLon(rt_pt, startTime, driveFile):
    todayDate = pd.to_datetime('today')
    todayString = todayDate.strftime('%Y-%m-%d')
    startTObj = datetime.datetime.combine(todayDate, datetime.datetime.strptime(startTime, '%H.%M.%S').time())
    startTStr = startTObj.strftime('%H:%M:%S')
    print(startTObj, startTStr)

    dg_proj = Proj(init='epsg:4269')
    ga_proj = Proj(init='epsg:2240')
    # feet to meter
    f_m = 0.3048006096012192
    # input: x_o, y_o (longitude,latitude, US ft)
    # output: x, y (longitude, latitude)
    rt_pt['longitude'] = 0.0
    rt_pt['latitude'] = 0.0
    curTimeObj = startTObj
    for i, row in rt_pt.iterrows():
        curTimeObj = curTimeObj + datetime.timedelta(seconds=1)
        curTimeStr = curTimeObj.strftime('%H:%M:%S')
        rt_pt.loc[i,'time'] = curTimeStr
        rt_pt.loc[i,'date'] = todayString
        rt_pt.loc[i,'longitude'] = transform(ga_proj,dg_proj, row.geometry.x*f_m,row.geometry.y*f_m)[0]
        rt_pt.loc[i,'latitude'] = transform(ga_proj,dg_proj, row.geometry.x*f_m,row.geometry.y*f_m)[1]
    cols = ['date','time','latitude','longitude']
    finalDf = rt_pt[cols]
    print(rt_pt)
    finalDf.to_csv(driveFile+'_trace.csv', index=False, header=False)
    return finalDf


if __name__=="__main__":
    traceFile = getTrace('drive_flask_1.csv')
    # startTime in the format of "H.MM.SS"
    trace2LatLon(traceFile, '8.03.00', 'drive_flask_1.csv')
    

# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:54:13 2017

@author: Haobing
"""
#####################################################################
# <codecell> # upload module and path #############################
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
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
#dg_proj = Proj(init='epsg:4269')
#ga_proj = Proj(init='epsg:2240')

f_m = 0.3048006096012192

#####################################################################
# <codecell> upload and build network #############################


import networkx as nx

##################################################################
# <codecell> convert between ft geometry and longitude/latitude###
##################################################################

def transform_ft_to_digit(x,y):
    # input: x_o, y_o (longitude,latitude, US ft)
    # output: x, y (longitude, latitude)
    dg_proj = Proj(init='epsg:4269')
    ga_proj = Proj(init='epsg:2240')
    lon = transform(ga_proj,dg_proj, x*f_m,y*f_m)[0]
    lat = transform(ga_proj,dg_proj, x*f_m,y*f_m)[1]
    return lon, lat
    

def transform_digit_to_ft(x,y):
    # input: x_o, y_o (longitude,latitude, US ft)
    # output: x, y (longitude, latitude)
    dg_proj = Proj(init='epsg:4269')
    ga_proj = Proj(init='epsg:2240')
    lon = transform(dg_proj,ga_proj, x,y)[0]/f_m
    lat = transform(dg_proj,ga_proj, x,y)[1]/f_m
    return lon, lat

#####################################################################
# <codecell> drive module###########################################

#trip = pd.read_csv(path + 'haobing_drive.csv')
#trip['modeID'] = 1 ## NEED TO DELETE
def drive(trip, network, railstation_df, busstation_df, grtastation_df):
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
    rt_list = rt_list[['geometry','speed_fts']]
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
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'drive'
    rt_pt['lat'] = 0.0
    rt_pt['lon'] = 0.0
    for i, row in rt_pt.iterrows():
        x = float(row['geometry'].x)
        y = float(row['geometry'].y)
        rt_pt.loc[i,'lon'],rt_pt.loc[i,'lat'] = transform_ft_to_digit(x,y)
    return rt_pt[['modeID','mode','secondID','lat','lon']]
    #return rt_pt
    ### actually the rt_pt is the result (geometry per second)


#####################################################################
# <codecell> marta rail module###########################################
####### metro : using straightline between stations ##########

#trip = pd.read_csv(path + 'haobing_rail.csv')
#trip['modeID'] = 1 ## NEED TO DELETE
def martarail(trip):
    trip['time'] = trip['time']*3600
    trip = trip.round({'time': 0})
    ## GET O/D rail station ID and coordinates
    trip['A1'] = [i[0] for i in trip['A'].str.split(pat='_')]
    trip['B1'] = [i[0] for i in trip['B'].str.split(pat='_')]
    trip = pd.merge(trip,railstation_df,how = 'left',left_on = 'A1', right_on = 'stop_id')
    trip = trip.rename(columns={"stop_lat": "A_lat", "stop_lon": "A_lon"})
    trip = pd.merge(trip,railstation_df,how = 'left',left_on = 'B1', right_on = 'stop_id')
    trip = trip.rename(columns={"stop_lat": "B_lat", "stop_lon": "B_lon"})
    ## generate output (time step and position coordinate)
    t = 0
    ## between each stations, generate points
    ### The dataframe "rt_pt" is the output
    rt_pt = gpd.GeoDataFrame(columns=('secondID','lat','lon'))
    for i, row in trip.iterrows():
        if t == 0:
            rt_pt.loc[0] = [0,row['A_lat'],row['A_lon']]
        cut = int(row['time'])
        delt_lat = (row['B_lat'] - row['A_lat'])/cut
        delt_lon = (row['B_lon'] - row['A_lon'])/cut
        tmp_lat = row['A_lat']
        tmp_lon = row['A_lon']
        for dt in range(1,cut+1):
            tmp_lat = tmp_lat + delt_lat
            tmp_lon = tmp_lon + delt_lon
            t+=1
            rt_pt.loc[t] = [t,tmp_lat,tmp_lon]
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'rail'
    return rt_pt[['modeID','mode','secondID','lat','lon']]



########################################################################
# <codecell> marta bus module###########################################

# read bus path
#trip = pd.read_csv(path + 'haobing_bus.csv')
#trip['modeID'] = 1 ## NEED TO DELETE
#trip = trip.round({'time': 0})
def martabus(trip):
    trip['time'] = trip['time']*3600
    trip['A1'] = [i.replace('_'+i.split('_')[-1],"") for i in trip['A']]
    trip['B1'] = [i.replace('_'+i.split('_')[-1],"") for i in trip['B']]
    trip = pd.merge(trip,busstation_df,how = 'left',left_on = 'A1', right_on = 'stop_id')
    trip = trip.rename(columns={"nodeid": "A_nodeid"})
    trip = pd.merge(trip,busstation_df,how = 'left',left_on = 'B1', right_on = 'stop_id')
    trip = trip.rename(columns={"nodeid": "B_nodeid"})
    trip = trip[['modeID','sequence','time','A_nodeid','B_nodeid']]
    
    
    rt_list = gpd.GeoDataFrame(columns=('geometry','speed_fts'))
    t = 0
    for i, row in trip.iterrows():
        start_pt = str(int(row['A_nodeid']))
        end_pt = str(int(row['B_nodeid']))
        if start_pt <> end_pt:
            nd_list = nx.algorithms.shortest_path(G, source=start_pt, target=end_pt, weight='weight')
            dist_sum = nx.algorithms.shortest_path_length(G, source=start_pt, target=end_pt, weight='weight')
            speed = dist_sum/row['time']
            #rdno_list = []
            for i in range(1,len(nd_list)):
                geo = network[(network['A'] == nd_list[i-1]) & \
                (network['B'] == nd_list[i])]['geometry'].values[0]
                rt_list.loc[t] = [geo,speed]
                t+=1
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
    #print rt_list
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
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'martabus'
    rt_pt['lat'] = 0.0
    rt_pt['lon'] = 0.0
    for i, row in rt_pt.iterrows():
        x = float(row['geometry'].x)
        y = float(row['geometry'].y)
        rt_pt.loc[i,'lon'],rt_pt.loc[i,'lat'] = transform_ft_to_digit(x,y)
    return rt_pt[['modeID','mode','secondID','lat','lon']]


########################################################################
# <codecell> grta module###########################################

# read bus path
#trip = pd.read_csv(path + 'haobing_grta.csv')
#trip['modeID'] = 1 ## NEED TO DELETE
#trip = trip.round({'time': 0})
def grta(trip):
    trip['time'] = trip['time']*3600
    trip['A1'] = [i.replace('_'+i.split('_')[-1],"") for i in trip['A']]
    trip['B1'] = [i.replace('_'+i.split('_')[-1],"") for i in trip['B']]
    #print [i.replace('_'+i.split('_')[-1],"") for i in trip['A']]
    #print [i.replace('_'+i.split('_')[-1],"") for i in trip['B']]
    trip = pd.merge(trip,grtastation_df,how = 'left',left_on = 'A1', right_on = 'stop_id')
    trip = trip.rename(columns={"nodeid": "A_nodeid"})
    trip = pd.merge(trip,grtastation_df,how = 'left',left_on = 'B1', right_on = 'stop_id')
    trip = trip.rename(columns={"nodeid": "B_nodeid"})
    trip = trip[['modeID','sequence','time','A_nodeid','B_nodeid']]
    
    
    rt_list = gpd.GeoDataFrame(columns=('geometry','speed_fts'))
    t = 0
    for i, row in trip.iterrows():
        start_pt = str(int(row['A_nodeid']))
        end_pt = str(int(row['B_nodeid']))
        #print start_pt, end_pt
        if start_pt <> end_pt:
            nd_list = nx.algorithms.shortest_path(G, source=start_pt, target=end_pt, weight='weight')
            dist_sum = nx.algorithms.shortest_path_length(G, source=start_pt, target=end_pt, weight='weight')
            speed = dist_sum/row['time']
            #rdno_list = []
            #print nd_list, dist_sum
            for i in range(1,len(nd_list)):
                geo = network[(network['A'] == nd_list[i-1]) & \
                (network['B'] == nd_list[i])]['geometry'].values[0]
                rt_list.loc[t] = [geo,speed]
                #print i, rt_list.loc[t]
                t+=1
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
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'grta'
    rt_pt['lat'] = 0.0
    rt_pt['lon'] = 0.0
    for i, row in rt_pt.iterrows():
        x = float(row['geometry'].x)
        y = float(row['geometry'].y)
        rt_pt.loc[i,'lon'],rt_pt.loc[i,'lat'] = transform_ft_to_digit(x,y)
    return rt_pt[['modeID','mode','secondID','lat','lon']]



##################################################################
# <codecell> wait#################################################


def wait(trip):
    trip['time'] = trip['time']*3600
    trip = trip.round({'time': 0})
    rt_pt = gpd.GeoDataFrame(columns=('secondID','lat','lon'))
    t = 0
    for i, row in trip.iterrows():
        for j in range(int(row['time'])):
            rt_pt.loc[t] = [t,'wait','wait']
            t+=1
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'wait'
    return rt_pt[['modeID','mode','secondID','lat','lon']]
    

##################################################################
# <codecell> walk#################################################


def walk(trip):
    trip['time'] = trip['time']*3600
    trip = trip.round({'time': 0})
    rt_pt = gpd.GeoDataFrame(columns=('secondID','lat','lon'))
    t = 0
    for i, row in trip.iterrows():
        for j in range(int(row['time'])):
            rt_pt.loc[t] = [t,'walk','walk']
            t+=1
    rt_pt['modeID'] = trip.loc[0,'modeID']
    rt_pt['mode'] = 'walk'
    return rt_pt[['modeID','mode','secondID','lat','lon']]




##################################################################
# <codecell> implementation ######################################


def getTraceTransit(comp_rt, m_option, network, railstation_df, busstation_df, grtastation_df, startTime):
    #m_option = 'grta'
    #comp_rt = pd.read_csv(path + 'haobing_grta_comp.csv')
    # need to decide whether it is grta or marta involved route
    m_option = 'marta'
    # uoload comprehensive route
    ##comp_rt = pd.read_csv(path + 'test_routes\\haobing_marta_only.csv')
    comp_rt = comp_rt[~(comp_rt['time']==0)].reset_index(drop=True)
    comp_rt['mode_new'] = comp_rt['mode']
    comp_rt.loc[comp_rt['mode']=='transit','mode_new'] = 'martabus'
    comp_rt.loc[comp_rt['route'].isin(['10909','10911','10912','10913',10909,10911,10912,10913]),'mode_new'] = 'martarail'
    if m_option == 'grta':
        comp_rt.loc[comp_rt['mode']=='transit','mode_new'] = 'grta'
    comp_rt['modeID'] = 1
    for i in range(1,len(comp_rt)):
        if comp_rt.loc[i,'mode_new']==comp_rt.loc[i-1,'mode_new']:
            comp_rt.loc[i,'modeID']=comp_rt.loc[i-1,'modeID']
        else:
            comp_rt.loc[i,'modeID']=comp_rt.loc[i-1,'modeID']+1
    mode_uniq = comp_rt[['modeID','mode_new']].drop_duplicates(keep='first')
    frames = []
    for i, row in mode_uniq.iterrows():
        m = row['mode_new']
        trip = comp_rt[comp_rt['modeID']==row['modeID']].reset_index(drop=True)
        trip[['A','B']] = trip[['A','B']].astype('str')
        if m == 'drive':
            trip[['A','B']] = trip[['A','B']].astype('int').astype('str')
            frames.append(drive(trip,network, railstation_df, busstation_df, grtastation_df))
        elif m == 'martabus':
            frames.append(martabus(trip))
        elif m == 'martarail':
            frames.append(martarail(trip))
        elif m == 'walk':
            frames.append(walk(trip))
        elif m == 'wait':
            frames.append(wait(trip))
        elif m == 'grta':
            frames.append(grta(trip))    
    trace_comp = pd.DataFrame(pd.concat(frames)).reset_index(drop=True)
    for i in reversed(trace_comp.index):
        if trace_comp.loc[i,'mode'] == 'wait':
            trace_comp.loc[i,'lon'] = trace_comp.loc[i+1,'lon']
            trace_comp.loc[i,'lat'] = trace_comp.loc[i+1,'lat']

    for i, row in mode_uniq.iterrows():
        if row['mode_new'] == 'walk':
            st_lon, st_lat = trace_comp[trace_comp['modeID']==row['modeID']-1].iloc[[-1]][['lon','lat']].values.tolist()[0]
            end_lon, end_lat = trace_comp[trace_comp['modeID']==row['modeID']+1].iloc[[0]][['lon','lat']].values.tolist()[0]
            cut = len(trace_comp[trace_comp['modeID']==row['modeID']])
            lat_list = np.linspace(st_lat,end_lat,cut)
            lon_list = np.linspace(st_lon,end_lon,cut)
            trace_comp.loc[trace_comp['modeID']==row['modeID'],'lon'] = lon_list
            trace_comp.loc[trace_comp['modeID']==row['modeID'],'lat'] = lat_list

    trace_comp['sequence'] = range(0,len(trace_comp))
    outputTrace(trace_comp, startTime)
    return trace_comp
    #trace_comp.to_csv('result_points.csv',index=False)

def outputTrace(trace_comp, startTime):
    todayDate = pd.to_datetime('today')
    todayString = todayDate.strftime('%Y-%m-%d')
    startTObj = datetime.datetime.combine(todayDate, datetime.datetime.strptime(startTime, '%H.%M.%S').time())
    startTStr = startTObj.strftime('%H:%M:%S')
    print(startTObj, startTStr)
    curTimeObj = startTObj
    for i, row in trace_comp.iterrows():
        curTimeObj = curTimeObj + datetime.timedelta(seconds=1)
        curTimeStr = curTimeObj.strftime('%H:%M:%S')
        trace_comp.loc[i,'time'] = curTimeStr
        trace_comp.loc[i,'date'] = todayString
    trace_comp = trace_comp.rename(columns={"lat": "latitude", "lon": "longitude"})
    cols = ['date','time','latitude','longitude']
    finalDf = trace_comp[cols]
    finalDf.to_csv('outputTrace.csv', index=False, header=False)


if __name__=="__main__":
    ## load a road network shapefile
    # global network, railstation_df,busstation_df,grtastation_df
    dirPath = "/home/users/ywang936/transit_simulation-0206"
    tracePath = "/home/users/ywang936/transit_simulation-0206/traceBack_all"
    #network = gpd.read_file(os.path.join(dirPath,'./data_node_link/abm15_links.shp'))
    network = gpd.read_file(os.path.join(dirPath,'./data_node_link/Link_Grids_Nodes.shp'))
    print(network)
    #network = gpd.read_file('D:\\TransportationEnvironment\\Dissertation\\GIS\\elev_data\\atl_abm_new_conn1.shp')
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
    railstation_df = pd.read_csv(os.path.join(tracePath, 'rail_stop.csv'))
    railstation_df = railstation_df[['stop_id','stop_lat','stop_lon']]
    railstation_df['stop_id'] = railstation_df['stop_id'].astype(str)

    # marta bus station
    busstation_df = pd.read_csv(os.path.join(tracePath, 'bus_stop.csv'))
    busstation_df = busstation_df[['stop_id','nodeid']]
    busstation_df['stop_id'] = busstation_df['stop_id'].astype(str)

    # grta bus station
    grtastation_df = pd.read_csv(os.path.join(tracePath, 'grta_stop.csv'))
    grtastation_df = grtastation_df[['stop_id','nodeid']]
    grtastation_df['stop_id'] = grtastation_df['stop_id'].astype(str)

    # set these
    comp_rt = pd.read_csv(os.path.join(tracePath, 'test_routes/mtOnly_in.csv'))
    m_option = 'marta'
    startTime = '8.03.00'
    getTraceTransit(comp_rt, m_option, network, railstation_df, busstation_df, grtastation_df, startTime)

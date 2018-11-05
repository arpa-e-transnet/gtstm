# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 11:31:04 2015

@author: fcastrillon3, deepthivenkat, ywang936
"""

dirPath = "/home/users/ywang936/transit_simulation-0206"
import os
os.chdir(dirPath)


from itertools import product
import trace_playback_func
from trace_playback_func import *
from comprehensive_trace_playback_cody_03012018 import *
from calceg_all import *
import MySQLdb
import json
#import pymongo
import networkx as nx
#import matplotlib.pyplot as plt

from collections import defaultdict
import sys
import csv
from geopy.distance import vincenty
from operator import itemgetter
from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd
import numpy as np
#import smopy
#import matplotlib as mpl
import shapely.geometry as geom
import geopandas as gpd
import datetime as dt
from datetime import datetime
from datetime import timedelta
from shapely.geometry import Point
import time
import math
from heapq import heappush, heappop
from itertools import count, islice

app = Flask(__name__)

#directory=r'C:\Users\Ann\Desktop\transit_simulation 07082017\transit_simulation\transit_simulation\Scripts\build_graph'
#directory=r'E:\mode comparison\transit_simulation\transit_simulation\Scripts\build_graph'
#os.chdir(directory)
#D = decimal.Decimal

# GRAPH=nx.Graph()
# GRAPH.add_edges_from(GRAPH_MARTA.edges(data=True)+GRAPH_GRTA.edges(data=True))
# GRAPH.add_nodes_from(GRAPH_GRTA.nodes(data=True)+GRAPH_MARTA.nodes(data=True))
'''
user=''
passwd=''
dbname='StmLinkNetwork'
collname='WatsonPlots'
'''
def get_db(dbname, collname, user = '', passwd = ''):
    url = 'mongodb://'+user+':'+passwd+'@rg49-mongodb1.ce.gatech.edu:30000/'
    dbconn = pymongo.MongoClient(url,connect=False)
    db = dbconn[dbname]
    coll = db[collname]
    return coll

def dbConnect(dbname):
    host = "somehost"
    user = "someuser"
    pwd = "somepwd"
    database = dbname
    db = MySQLdb.connect(host, user, pwd, database)
    cursor = db.cursor()
    return cursor


def sumList(lsts, totalWeight):
    return [float(sum(i))/float(totalWeight) for i in zip(*lsts)]

def k_shortest_paths(G, source, target, k, weight=None):
    print("using weight:", weight)
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except Exception, e:
        print("stm data is not available, in kshortest:", str(e))
        

def calcIsecPerc(main, others):
    isecs = [filter(lambda x: x in main, sublist) for sublist in others]
    try:
        #print([float(len(isecs[ind]))/float(len(main)) for ind in range(len(others))])
        selected_inds = [ind for ind in range(len(others)) if float(len(isecs[ind]))/float(len(main)) <= 0.9]
        print(selected_inds)
        return selected_inds
    except:
        # For python 3, cannot print first because printing first will exhaust the filter
        #print([float(len(list(isecs[ind])))/float(len(main)) for ind in range(len(others))])
        selected_inds = [ind for ind in range(len(others)) if float(len(list(isecs[ind])))/float(len(main)) <= 0.9]
        print(selected_inds)
        return selected_inds

def getAvgWatson(watsonList):
    weightSum = 0
    weightedWatsonList = []
    for watsonDict in watsonList:
        if watsonDict != None:
            weight = float(watsonDict['TravelTime'])
            weightSum += weight
            plot = watsonDict['WatsonPlot'].split('\n')
            watsonPlot = []
            for rowInd in range(len(plot)):
                # first row is speed
                if rowInd == 0 or len(plot[rowInd]) == 0:
                    continue
                else:
                    rowList = plot[rowInd].split(',')
                    try:
                        # First column is speed
                        floatList = [weight * float(item) for item in rowList[1:]]
                        #floatList = [float(item) for item in rowList[1:]]
                        watsonPlot.append(floatList)
                        # print(floatList)
                    except:
                        print(len(rowList), rowList)

            weightedWatsonList.append(watsonPlot)
            #print(weight, watsonPlot)
    '''
    for watson in weightedWatsonList:
        for row in watson:
            print(len(row), row)
    '''
    print("Number of weighted plots:", len(weightedWatsonList))
    print("Number of original plots:", len(watsonList))

    '''
    weightedWatsonList = []
    a = [[1,2,3,4],[2,3,4,5],[4,3,2,1],[5,4,3,2]]
    b = [[0,2,3,4],[0,3,4,5],[0,3,2,1],[5,4,3,0]]
    c = [[10,2,3,4],[10,3,4,5],[10,3,2,1],[5,4,3,10]]
    weightedWatsonList.append(a)
    weightedWatsonList.append(b)
    weightedWatsonList.append(c)
    '''

    avgWatson = [sumList(item,3) for item in zip(*weightedWatsonList)]
    return avgWatson

def getWatson(linkNodes):
    # TODO: change '_' to sep based on grta/marta
    watsonList = []
    prevNode = linkNodes[0]
    for node in linkNodes[1:]:
        linkID = prevNode + '_' + node
        watson = retWatson(collection, linkID, "07:30:00","passenger car")
        if watson != 0:
            watsonList.append(watson)
        prevNode = node
    return watsonList

def retWatson(coll, linkID, timeStamp, vehicleType):
    cursor = coll.find({"LinkID":linkID,"TimeStamp":timeStamp,"VehicleType":vehicleType})
    if cursor.count()> 0:
        for document in cursor:
            return document
    else:
        return 0

def printGraph():
    event_pos = {k:[int(i) for i in v] for k,v in EVENT_POS.items()}
    nx.draw(GRAPH,event_pos,node_size=15)
    plt.rcParams['figure.figsize'] = 20, 5    
    plt.show()
    
def clearAll():
    global STOP_EVENTS
    global EVENT_POS
    global GRAPH
    STOP_EVENTS = []
    EVENT_POS = []
    GRAPH = []  
    
    
    
def getShortestTravelTime_OptiArr(start_time,end_time, origin, destination, transitType='marta'):
    '''
    start_time=arrivalt
    origin=origin_stops
    destination=destination_stops
    transitType='marta'
    getShortestTravelTime_OptiDur(arrivalt,origin_stops,destination_stops,transitType)
    '''
    '''
    start_time=arrivalt_park
    origin=park_stops_data[2]
    destination=destination_stops
    transitType='marta'
    tup = getShortestTravelTime_OptiDur(arrivalt_park, park_stops_data[2],destination_stops,transitType)
    '''
    path = []
    '''
    foo = start_time.split('.') 
    startt = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
    
    foo = end_time.split('.') 
    endt = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
    '''          
    if transitType=='marta':
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop.json')))
    else:
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop_grta.json')))
        
    try:
        events_start = STOP_EVENTS[str(origin)]
        events_end  = STOP_EVENTS[str(destination)]
    except Exception as e:
        print('          ',transitType,' does not exist - first condition')
        return path 
     
    
    if transitType=='marta':
        pathGraph = GRAPH
    else:
        pathGraph = GRAPH_GRTA 
    
    # start searching from the earliest arrival time
    events_start.sort(key=lambda tup: tup[0],reverse=True)
    events_end.sort(key=lambda tup: tup[0],reverse=False)
    #events_start.sort(reverse=True)
    #events_end.sort(reverse=False) #reversed to find the shortest traveling time
    '''
    print('start looping')
    print(start_time)
    print(origin)
    print(destination)
    start_time='7.9.00'
    origin='WINDWARD PARK & RIDE_10266'
    destination='DORAVILLE STATION_384'
    '''
    j=1
    for t2,ttype in events_end:
        if t2<=end_time:
            events_start_new=[x[0] for x in events_start if x[0]<t2 and x[1]=='ride']
            for t1 in events_start_new:
                if t1>=start_time:
                    j=j+1
                    if j%200==0:
                    
                        print('====',t1,t2,'====')
                    #get start and end nodes
                    start_node = "s"+str(origin)+"t"+str(t1)            
                    end_node = "s"+str(destination)+"t"+str(t2)
                    '''
                    if t1 >= 7.81 and t1<= 7.82 and t2 > 8.55 and t2 <= 8.56:
                        print(start_node, end_node)
                    '''
                    #pathGraph_temp=pathGraph[u][v]['route']!!!
                    #start_node
                    '''
                    graph=pathGraph
                
                    for edgei in graph.out_edges(start_node):
                        if graph[edgei[0]][edgei[1]]['route']!='ride':
                            print(graph[edgei[0][edgei[1]]])
                            graph.remove_edge(edgei[0],edgei[1])
                    print(graph[edgei[0])
                    
                    for edgei in graph.in_edges(end_node):
                        if graph[edgei[0]][edgei[1]]['route']!='ride':
                            graph.remove_edge(edgei[0],edgei[1])  
                    '''
                    #print(start_node,end_node)
                    if nx.has_path(pathGraph,start_node,end_node):
                        arrival_time = t2
                        path = nx.dijkstra_path(pathGraph,start_node,end_node)
                        print('          ',transitType,' routing found')
                        return (arrival_time,path)
                
    print('          ',transitType,' does not exist - second condition')
    return path

def getShortestTravelTime_OptiDur(start_time, origin, destination, transitType='marta'):
    path = []
    foo = start_time.split('.') 
    startt = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
    
    if transitType=='marta':
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop.json')))
    else:
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop_grta.json')))
        
    try:
        events_start = STOP_EVENTS[str(origin)]
        events_end  = STOP_EVENTS[str(destination)]
    except Exception as e:
        print(str(e))
        return path 
     
    
    if transitType=='marta':
        pathGraph = GRAPH
    else:
        pathGraph = GRAPH_GRTA 
    
    events_start_list,events_end_list=pd.tools.util.cartesian_product([events_start,events_end])
    df_od_events=pd.DataFrame(dict(t1_list=events_start_list, t2_list=events_end_list))
    df_od_events=df_od_events[df_od_events['t1_list']<df_od_events['t2_list']]
    df_od_events['duration']=df_od_events['t2_list']-df_od_events['t1_list']
    df_od_events = df_od_events.sort(['duration','t2_list'], ascending=[1,1])
    for ind, row in df_od_events.iterrows():
        t1=row['t1_list']
        t2=row['t2_list']
        #get start and end nodes
        start_node = "s"+str(origin)+"t"+str(t1)            
        end_node = "s"+str(destination)+"t"+str(t2) 
        if nx.has_path(pathGraph,start_node,end_node):
            arrival_time = t2
            path = nx.dijkstra_path(pathGraph,start_node,end_node)
            return (arrival_time,path)
    return path


def getShortestTravelTime(start_time, origin, destination, transitType='marta'):
    # Obtains the shortest travel time based on origin station, destination station, and stop events at these two
    #print("Finding path...")
    #print(start_time, origin, destination)
    path = []
    #print("start time "+start_time)
    #print(type(start_time))
    print("start station "+str(origin))
    print("end station "+str(destination))
    foo = start_time.split('.') 
    #print(foo)
    startt = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
    #print("startt " + str(startt))

    if transitType=='marta':
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop.json')))
    else:
        STOP_EVENTS = json.load(open(os.path.join(dirPath,'events_at_stop_grta.json')))
    
    #get events at origin and destination
    try:
        events_start = STOP_EVENTS[str(origin)]
        events_end  = STOP_EVENTS[str(destination)]
    except Exception as e:
        print(str(e))
        return path

        
    events_start.sort()
    events_end.sort()
    #print("events start" + str(events_start))
    #print("events end" + str(events_end))

    #loop through event times and get nearest arrival time at destination
    arrival_time = 24 #hours in day    

    if transitType=='marta':
        pathGraph = GRAPH
    else:
        pathGraph = GRAPH_GRTA 
    
        
    
    for t1 in events_start:
        if t1 < startt or t1 > arrival_time:
            continue;
        for t2 in events_end:
            if t2 < t1 or t2 < startt or t2 >= arrival_time:
                continue;      
            
            #get start and end nodes
            start_node = "s"+str(origin)+"t"+str(t1)            
            end_node = "s"+str(destination)+"t"+str(t2) 
            #print("testing if shortest path from "+str(t1)+ "to"+str(t2)+"exists\n")
            #for node in GRAPH.nodes():
             #   if 'FIVE POINTS' in node or 'GARNETT STATION' in node:
              #      print(node)
            if nx.has_path(pathGraph,start_node,end_node):
                print("path exists")
                arrival_time = t2
                path = nx.dijkstra_path(pathGraph,start_node,end_node)
                print(arrival_time)

    if not path:
        print("no path exists")
    return (arrival_time,path)  

def getTimeInFormat(time_float):
    arrival_t = time_float
    hrs = int(math.floor(arrival_t/3600))
    mins = int(math.floor((arrival_t-3600*hrs)/60))
    secs = int(arrival_t-3600*hrs-60*mins)
    if len(str(mins)) == 1:
      mins = "0"+str(mins)
    if len(str(secs)) == 1:
      secs = "0"+str(secs)
    return str(hrs) +":"+ str(mins) +":"+ str(secs)  

def formatTime(original_time_str):
    l = original_time_str.split('.')
    return (float(l[0]) * 3600 + float(l[1]) * 60 + float(l[2]))/3600.0

def path_edge_attributes(path, transitType='marta'):
    # print 'path in original method', path
    if transitType == 'marta':
        transitGraph = GRAPH
    else:
        transitGraph = GRAPH_GRTA
    return [(transitGraph[u][v]['lat_or'],transitGraph[u][v]['lon_or'],transitGraph[u][v]['time_or'],transitGraph[u][v]['lat_dest'],transitGraph[u][v]['lon_dest'], transitGraph[u][v]['time_dest'], transitGraph[u][v]['distance']) for (u,v )in zip(path[0:],path[1:])]

def printDetailedPaths(path, outPaths, tripId, modeOption, transitType='marta'):
    # print 'path in original method', path
    if transitType == 'marta':
        transitGraph = GRAPH
    else:
        transitGraph = GRAPH_GRTA
    for (u,v) in zip(path[0:], path[1:]):
        outPaths.write(str(tripId)+','+modeOption+','+str(u)+','+str(v)+','+transitType+','+str(transitGraph[u][v]['time_dest'])+','+str(transitGraph[u][v]['distance'])+'\n')
        print('Graph element:', u,v)
        print('Origin:'+str(transitGraph[u][v]['lat_or'])+','+str(transitGraph[u][v]['lon_or'])+'|Destination:'+str(transitGraph[u][v]['lat_dest'])+','+str(transitGraph[u][v]['lon_dest']))
        print('Time at current destination:'+str(transitGraph[u][v]['time_dest']))
        print('Current distance travelled:'+str(transitGraph[u][v]['distance']))
    return outPaths
    #return [(transitGraph[u][v]['lat_or'],transitGraph[u][v]['lon_or'],transitGraph[u][v]['time_or'],transitGraph[u][v]['lat_dest'],transitGraph[u][v]['lon_dest'], transitGraph[u][v]['time_dest'], transitGraph[u][v]['distance']) for (u,v )in zip(path[0:],path[1:])]

def path_edge_attributes_dict(final_paths, transitType='marta'):
    if transitType == 'marta':
        transitGraph = GRAPH
    else:
        transitGraph = GRAPH_GRTA
    print('final paths', final_paths)
    paths = []
    p_list = []
    for path_no,tup in enumerate(final_paths):
        path_dict = {}
        path = tup[1]
        path_list_items = [(transitGraph[u][v]['lat_or'],transitGraph[u][v]['lon_or'],transitGraph[u][v]['time_or'],transitGraph[u][v]['lat_dest'],transitGraph[u][v]['lon_dest'], transitGraph[u][v]['time_dest'], transitGraph[u][v]['distance']) for (u,v )in zip(path[0:],path[1:])]
        print('Iteration tuple',tup)
        # print path_list_items
        # if len(path_list_items)>1:
        #     print 'origin time', path_list_items[0][2]
        #     o_t = path_list_items[0][2]
        #     print 'destination time', path_list_items[-1][5]
        #     d_t = path_list_items[-1][5]
        #     o_lat =path_list_items[0][0]
        #     o_lon =path_list_items[0][1]
        #     d_lat = path_list_items[-1][3]
        #     d_lon = path_list_items[-1][4]
        #     print path_list_items
        #     print len(path_list_items)
        #     time.sleep(10)
        #     p_list=[o_lat,o_lon,o_t,d_lat,d_lon,d_t]
        #     print 'p_list', p_list
        # else:
        #     print 'origin time', path_list_items[5]
        #     path_list_items[2] = path_list_items[2]
        #     print 'destination time', path_list_items[5]
        #     path_list_items[5] = path_list_items[5]
        #     print path_list_items
        #     print len(path_list_items)
        #     time.sleep(10)
        #     p_list = path_list_items
        #     print 'p_list',p_list

    
        # # for path_item in p_list:
        # path_dict = {}
        # path_dict["lat_or"] = p_list[0]
        # path_dict["lon_or"] = p_list[1]
        # path_dict["time_or"] = p_list[2]
        # path_dict["lat_dest"] = p_list[3]
        # path_dict["lon_dest"] = p_list[4]
        # path_dict["time_dest"] = p_list[5]
        # paths.append(path_dict)
        route_list = []
        for index,items in enumerate(path_list_items):
            route_dict = {}
            route_dict['sub_path_no'] = index 
            route_dict['start_lat'] = items[0]
            route_dict['start_long'] = items[1]
            route_dict['start_time'] = items[2]
            route_dict['dest_lat'] = items[3]
            route_dict['dest_long'] = items[4]
            route_dict['reach_time'] = items[5]
            route_dict['distance'] = items[6]
            route_list.append(route_dict)
        path_dict['path_'+str(path_no)] = route_list
        paths.append(path_dict)
    print(paths)
    #return jsonify({"paths":paths})

def drawGraph():
    global event_pos
    event_pos = {int(k):[int(i) for i in v] for k,v in event_pos.items()}
    nx.draw(GRAPH,event_pos,node_size=15)
    plt.rcParams['figure.figsize'] = 20, 5    
    plt.show()
           
def outputPath(edges):
    outfile = open(os.path.join(dirPath,"path.txt"),"w")
    outfile.write("lat,lon,time\n")
    for i,x in enumerate(edges):
        outfile.write(str(x[0])+","+str(x[1])+","+str(x[2])+"\n")
        if i == len(edges)-1:
          outfile.write(str(x[3])+","+str(x[4])+","+str(x[5])+"\n")
    outfile.close()

def add_xy(df,lat,lon,x,y,x_sq,y_sq,grid_size=25000.0):
    crs = {'init': 'epsg:4326', 'no_defs': True}
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    df = gpd.GeoDataFrame(df, crs=crs,geometry=geometry)
    df=df.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    df[x]=df['geometry'].apply(lambda x: x.coords[0][0])
    df[y]=df['geometry'].apply(lambda x: x.coords[0][1])
    df[x_sq]=round(df[x]/grid_size,0)
    df[y_sq]=round(df[y]/grid_size,0)
    return df 

def point_to_node(df_points,df_links,ifGrid=False,walk_speed=2.0,grid_size=25000.0,dist_thresh=5280.0):
    def find_grid(pt_x):
        return (round(pt_x/grid_size),0)
    def define_gridid(df_pts):
        df_pts['x_sq']=df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][0]))
        df_pts['y_sq']=df_pts['geometry'].apply(lambda x: find_grid(x.coords[0][1]))
        return df_pts
    
    def find_closestLink(point,lines):
        dists=lines.distance(point)
        return [dists.argmin(),dists.min()]
    def calculate_dist(x1,y1,x2,y2):
        return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    
    # INITIALIZATION
    if ifGrid:
        df_points=define_gridid(df_points)
    df_points['NodeID']=0
    df_points['Node_t']=0
    # CALCULATION
    for ind, row in df_points.iterrows():
        try:
            # find links in the grid
            df_links_i=df_links[df_links['minx_sq']<=row['x_sq']][df_links['maxx_sq']>=row['x_sq']][df_links['maxy_sq']>=row['y_sq']][df_links['miny_sq']<=row['y_sq']]
            # find the closest link and the distance
            LinkID_Dist=find_closestLink(row.geometry,gpd.GeoSeries(df_links_i.geometry))
            linki=df_links_i.loc[LinkID_Dist[0],:]
            # find the closest node on the link
            dist1=calculate_dist(df_points.loc[ind,'geometry'].coords[0][0],df_points.loc[ind,'geometry'].coords[0][1],linki['Ax'],linki['Ay'])
            dist2=calculate_dist(df_points.loc[ind,'geometry'].coords[0][0],df_points.loc[ind,'geometry'].coords[0][1],linki['Bx'],linki['By'])
            if (dist1>dist_thresh) and (dist2>dist_thresh):
                df_points.loc[ind,'NodeID']=-1
                df_points.loc[ind,'Node_t']=-1
            else:
                df_points.loc[ind,'NodeID'] = linki['A'] if dist1<dist2 else linki['B']
                df_points.loc[ind,'Node_t'] = dist1/walk_speed/5280.0 if dist1<dist2 else dist2/walk_speed/5280.0
        except Exception as e:
            print("in point to node:", str(e))
            df_points.loc[ind,'NodeID']=-1
            df_points.loc[ind,'Node_t']=0
    return df_points

def getDgWedgeId(df_points):
    global wedges
    global wedges_dt
    global DG
    global DG_reverse
    global DGs
    global DGs_reversed
    global DG_dt
    global DG_reverse_dt

    lat, lon = df_points.iloc[0]['ori_lat'], df_points.iloc[0]['ori_lon']
    lat2, lon2 = df_points.iloc[0]['dest_lat'], df_points.iloc[0]['dest_lon']
    geometry = [Point(xy) for xy in zip([lon], [lat])]
    geometry2 = [Point(xy) for xy in zip([lon2], [lat2])]
    crs = {'init': 'epsg:4326'}
    p = gpd.GeoDataFrame(pd.DataFrame({'lat':[lat],'lon':[lon]}), crs=crs, geometry=geometry)
    p2 = gpd.GeoDataFrame(pd.DataFrame({'lat':[lat2],'lon':[lon2]}), crs=crs, geometry=geometry2)
    print('origin:', p, 'dest:', p2)
    ori_loc = getPtLoc(p, wedges_dt, wedges)
    dest_loc = getPtLoc(p2, wedges_dt, wedges)
    print("origin loc: ", ori_loc)
    print("dest_loc: ", dest_loc)
    if ori_loc == 'downtown' and dest_loc == 'downtown':
        return (DG_dt, DG_reverse_dt)
    elif ori_loc == 'downtown' and dest_loc != 'downtown' and dest_loc != '':
        return (DGs[dest_loc], DGs_reversed[dest_loc])
    elif dest_loc == 'downtown' and ori_loc != 'downtown' and ori_loc != '':
        return (DGs[ori_loc], DGs_reversed[ori_loc])
    elif ori_loc == dest_loc:
        return (DGs[ori_loc], DGs_reversed[ori_loc])
    elif ori_loc != '' and dest_loc != '':
        return (DG, DG_reverse)
    else:
        return jsonify({"error":"ori or dest point not in network"})

def getPtLoc(p, wedges_dt, wedges):
    pt_loc = ''
    print("In getPtLoc, p is :", p)
    print('\n')
    print("in getPtLoc, wedges_dt is :", wedges_dt)
    try:
        pwd_dt = gpd.sjoin(p, wedges_dt, how='inner', op='intersects')
        print("point is in downtown")
        pt_loc = 'downtown'
        return pt_loc
    except Exception as e:
        print("in getPtLoc exception:", str(e))
        try:
            pwd = gpd.sjoin(p, wedges, how='inner', op='intersects')
            wid = pwd.iloc[0]['WID']
            print("point is in wedge:", wid)
            pt_loc = wid
            return pt_loc
        except:
            print("point is not in network")
            return pt_loc

@app.route("/getDriveFlask", methods=["GET"])
def getDriveFlask():
    global egall
    global railEnergyDict
    global df_link_grids
    global df_links_drive
    global df_grids
    global DG
    global DG_reverse
    global df_drive_trace
    #global trace_network
    
    strategy = ''
    #20180403
    if 'origin' in request.args and 'destination' in request.args:
        if 'startt' in request.args:
            strategy = 'start'
            try:
                origin_lat, origin_lon = request.args['origin'].split(',')
            except:
                print("wrong origin input:")
                return jsonify(request)
            try:
                destination_lat, destination_lon = request.args['destination'].split(',')
            except:
                print("wrong destination input:")
                return jsonify(request)
        
        elif 'endt' in request.args:
            strategy = 'end'
            try:
                origin_lat, origin_lon = request.args['destination'].split(',')
            except:
                print("wrong origin input:")
                return jsonify(request)
            try:
                destination_lat, destination_lon = request.args['origin'].split(',')
            except:
                print("wrong destination input:")
                return jsonify(request)
        else:
            print("wrong input: must provide either departure time or arrival time")
            return jsonify(request)
        
        if strategy == 'start':
            arrivalt = request.args['startt']
            #endt = request.args['endt']
        else:
            arrivalt = request.args['endt']
            #endt = request.args['startt']
        try:
            userId = request.args['userid']
        except:
            userId = 'unNamed'
            pass
        '''
        #####TESTING PURPOSE##########
        # far one
        origin_lat, origin_lon = '34.02208790', '-84.37511000'
        #origin_lat, origin_lon = '33.813331', '-84.373156'
        destination_lat, destination_lon = '33.778325','-84.399319'
        arrivalt = '8.03.00'
        endt = '10.00.00'
        userId = 'testing'
        ##############################
        '''
        foo = arrivalt.split('.')
        arrival_hours = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
        # 20180402
        #if arrival_hours >= 10 or arrival_hours <= 6.5:
         #   return jsonify({"drive":"route not found"})

        df_points = pd.DataFrame({'Trip':[1], 'ori_lat':[float(origin_lat)], 'ori_lon':[float(origin_lon)], 'dest_lat':[float(destination_lat)], 'dest_lon':[float(destination_lon)]})
    else:
        return jsonify({"error":"please provide latlon ori and dest"})
    

    # Wedge choose
    (DG_sel, DG_reverse_sel) = getDgWedgeId(df_points)

    if strategy == 'start':
        outPaths_drivei=returnDrivePaths_toDF_GlobalLoop(df_points,df_links_drive,DG_sel,1,arrivalt,strategy)
    else:
        outPaths_drivei=returnDrivePaths_toDF_GlobalLoop(df_points,df_links_drive,DG_reverse_sel,1,arrivalt,strategy)
    df_out=pd.DataFrame()
    df_out=df_out.append(outPaths_drivei,ignore_index=True)
    
    # second argument is starttime
    #df = splitOptions(df_out)
    df = calcTimeDuration_drive(df_out,arrival_hours)
    df_filled = fillEnergy(df,egall,railEnergyDict)
    optionUniques = df_filled['option_id'].unique().tolist()

    # create trip file directory if not exist
    now = datetime.now()
    dtstr = now.strftime("%Y%m")
    dateStr = now.strftime("%Y%m%d")
    tripPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr)
    outPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr,userId)
    if not os.path.exists(outPath):
        if not os.path.exists(tripPath):
            os.mkdir(tripPath)
        os.mkdir(outPath)

    traceFileDict = {}
    for optionId in optionUniques:
        driveFile = 'debug_files/drive_flask_'+str(optionId)+'.csv'
        df_filled[df_filled['option_id']==optionId].to_csv(driveFile, index=False)
        try:
            # 20180424
            traceFile = trace_playback_func.getTrace(driveFile, df_drive_trace)
            # 20180403

            if strategy == 'start':
                genFile = trace2LatLon(traceFile, arrivalt, driveFile)
            else:
                # 20180403: drivecolname is 'time'
                #startT = df_filled[df_filled['option_id']==optionId].iloc[0]['time'] - 0.001
                startT = df_filled[df_filled['option_id']==optionId].iloc[0]['time'] - 0.001
                
                startTObj = dt.datetime.combine(dt.date.today(), dt.time()) + dt.timedelta(hours=startT)
                startTStr = startTObj.strftime('%H.%M.%S')
                genFile = trace2LatLon(traceFile, startTStr, driveFile)
                
            firstRow = genFile.iloc[0]
            startTime = firstRow[1]
            startTime = startTime.replace(':', '')
            trName = os.path.join(outPath, userId+'.'+dateStr+'.'+startTime+'.'+str(optionId)+'.csv')
            traceFileDict[int(optionId)] = trName
            print("output trace File to:", trName)
            genFile.to_csv(trName, index=False, header=False)
        except Exception, e:
            print(str(e))
    print("traceFileDict:", traceFileDict)
    return calcDfEg(df_filled, userId, traceFileDict=traceFileDict)

@app.route("/getMartaFlask", methods=["GET"])
def getMartaFlask():
    global egall
    global railEnergyDict
    global dict_marta
    global dict_marta_rail
    global df_links
    global df_links_trace
    global railstation_df
    global busstation_df
    global grtastation_df
    
    #20180403
    if 'origin' in request.args and 'destination' in request.args:
        try:
            origin_lat, origin_lon = request.args['origin'].split(',')
        except:
            print("wrong origin input:")
            return jsonify(request)
        try:
            destination_lat, destination_lon = request.args['destination'].split(',')
        except:
            print("wrong destination input:")
            return jsonify(request)
        
        try:
            userId = request.args['userid']
        except:
            userId = 'unNamed'
            pass
        '''
        #####TESTING PURPOSE##########
        #transfer pair
        origin_lat, origin_lon = '34.024370', '-84.360619'
        destination_lat, destination_lon = '33.781863', '-84.386071'
        
        #transfer bus pair 
        #origin_lat, origin_lon = '33.6639', '-84.46723'
        #destination_lat, destination_lon = '33.69361959','-84.4899292'
        
        arrivalt = '7.00.00'
        endt = '10.00.00'
        ##############################                
        '''
        arrival_hours = end_hours = 0
        if 'startt' in request.args:
            strategy = 1
            arrivalt = request.args['startt']
            foo = arrivalt.split('.')
            stmTime = getCur15(arrivalt)
            arrival_hours = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
        if 'endt' in request.args:
            strategy = 3
            endt = request.args['endt']
            foo2 = endt.split('.')
            stmTime = getCur15(endt)
            end_hours = float(foo2[0]) + float(foo2[1]) / 60 + float(foo2[2])/3600
        ####033018####
        # buffer is the max time allowed to arrive earlier than the set destination time
        bufferTime = 1/3.0
        if arrival_hours == 0 and end_hours == 0:
            return jsonify({"error": "must provide departure or arrival time"})
        else:
            if arrival_hours == 0:
                arrival_hours = end_hours - 3.0
            elif end_hours == 0:
                end_hours = arrival_hours + 3.0
        print("arrival hours:", arrival_hours, "end hours:", end_hours, "strategy:", strategy)
        dict_settings={}
        # 211
        dict_settings={'cutoffs':np.arange(15,121,15),'commuteTime':[arrival_hours,end_hours],'wait_thresh':[0,2],'buffer': bufferTime, 'walk_speed':2.0,'cutoff_max':2*60.0,'grid_size':25000.0,'ntp_dist_thresh':5280.0}
        dict_settings['network']={'marta-pnr':dict_marta_rail,'marta-only':dict_marta}
        #dict_settings['cutoffs_label']=['063000', '064500', '070000','071500','073000','074500','080000','081500','083000','084500','090000','091500','093000','094500']
        #strategy: 1, minimum total time excluding wait1; 2, earliest arrival, leave when commute starts 
        dict_settings['strategy']={'marta-only':strategy,'grta-only':strategy,'marta-pnr':strategy,'grta-pnr':strategy}
        dict_settings['walk_thresh']={'marta-only':0.5,'marta-pnr':0.5,'marta-rail':0.5,'grta-only':1,'grta-pnr':1}
        dict_settings['num_options']={'drive':3,'marta-only':1,'marta-pnr':1,'marta-rail':1,'grta-only':1,'grta-pnr':1}        
        
        walk_speed, grid_size,ntp_dist_thresh=dict_settings['walk_speed'],dict_settings['grid_size'],dict_settings['ntp_dist_thresh']
        df_points = pd.DataFrame({'trip_id':[1], 'ori_lat':[float(origin_lat)], 'ori_lon':[float(origin_lon)], 'dest_lat':[float(destination_lat)], 'dest_lon':[float(destination_lon)]})
        df_points=add_xy(df_points,'ori_lat','ori_lon','x','y','x_sq','y_sq')    
        df_points=point_to_node(df_points,df_links,False, walk_speed,grid_size,ntp_dist_thresh).rename(columns={'NodeID':'o_node','Node_t':'o_t','x':'ox','y':'oy','x_sq':'ox_sq','y_sq':'oy_sq'})
        df_points=add_xy(df_points,'dest_lat','dest_lon','x','y','x_sq','y_sq')
        df_points=point_to_node(df_points,df_links,False, walk_speed,grid_size,ntp_dist_thresh).rename(columns={'NodeID':'d_node','Node_t':'d_t','x':'dx','y':'dy','x_sq':'dx_sq','y_sq':'dy_sq'})
    else:
        return jsonify({"error":"please provide latlon ori and dest"})
    
    
    print("current stmTime is:", stmTime)
    row = df_points.iloc[0]
    walk_dist=np.sqrt((row['ox']-row['dx'])**2+(row['oy']-row['dy'])**2)/5280.0
    resultsPath_martaOnly,runningLogi=Transit(df_points,'marta-only',walk_dist,dict_settings) 

    df_ods = pd.DataFrame()
    dict_transitRoutes={}
    # Park-and-Ride
    df_ods_i,runningLogi,dict_transitRoutes['marta-pnr']=pnr_Transit(row,'marta-pnr',dict_settings)
    if len(df_ods_i)>0:
        df_ods_i['option']='marta-pnr'
        df_ods=df_ods.append(df_ods_i)
    if len(df_ods) > 0:
        _df_ods,dict_driveRoutes,errMessage=get_drive_routes(str(row['o_node']),row['o_t'],df_ods,dict_settings,stmTime)
        if len(_df_ods) > 0:
            resultsPath_martaPnr,runningLogi=output_pnr(row,_df_ods,dict_settings,dict_driveRoutes,dict_transitRoutes,['marta-pnr'])
        else:
            resultsPath_martaPnr = pd.DataFrame()
    else:
        resultsPath_martaPnr = pd.DataFrame()
    
    
    mtOnly = False
    mtPnr = False

    traceFileDict = {}

    # create trip file directory if not exist
    now = datetime.now()
    dtstr = now.strftime("%Y%m")
    dateStr = now.strftime("%Y%m%d")
    tripPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr)
    outPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr,userId)
    if not os.path.exists(outPath):
        if not os.path.exists(tripPath):
            os.mkdir(tripPath)
        os.mkdir(outPath)


    destination_lat_lon = destination_lat+'_'+destination_lon
    origin_lat_lon = origin_lat+'_'+origin_lon
    # originally len > 0, but encountered some issue
    if len(resultsPath_martaOnly) > 1:
        #todo: change output station/route split by _
        mtOnly = True
        mtOnly_in = resultsPath_martaOnly.reset_index(drop=True)
        mtOnly_in = fillLoc(mtOnly_in, origin_lat_lon, destination_lat_lon, option='marta')
        mtOnly_in.to_csv('debug_files/mtOnly_in.csv', index=False)
        mtOnly_in = pd.read_csv('debug_files/mtOnly_in.csv')
        traceFile_direct = getTraceTransit(mtOnly_in, df_links_trace)

        firstRow = traceFile_direct.iloc[0]
        startTime = firstRow[1]
        startTime = startTime.replace(':', '')
        trName = os.path.join(outPath, userId+'.'+dateStr+'.'+startTime+'.mtOnly'+'.csv')
        traceFileDict['mtOnly'] = trName
        print("output trace File to:", trName)
        traceFile_direct.to_csv(trName, index=False, header=False)

    if len(resultsPath_martaPnr) > 1:
        #todo: change output station/route split by _
        mtPnr = True
        mtPnr_in = resultsPath_martaPnr.reset_index(drop=True)
        mtPnr_in.to_csv('debug_files/mtPnrBefore_in.csv', index=False)
        mtPnr_in = fillLoc(mtPnr_in, origin_lat_lon, destination_lat_lon, option='marta')
        mtPnr_in.to_csv('debug_files/mtPnr_in.csv', index=False)
        mtPnr_in = pd.read_csv('debug_files/mtPnr_in.csv')
        traceFile_pnr = getTraceTransit(mtPnr_in, df_links_trace)

        firstRow = traceFile_pnr.iloc[0]
        startTime = firstRow[1]
        startTime = startTime.replace(':', '')
        trName = os.path.join(outPath, userId+'.'+dateStr+'.'+startTime+'.mtPnr'+'.csv')
        traceFileDict['mtPnr'] = trName
        print("output trace File to:", trName)
        traceFile_pnr.to_csv(trName, index=False, header=False)

    #print(traceFile)

    #TODO:
    #1. Rebuild transit graphs with larger range of time
    #2. Output energy and final json strings
    
    df_out=pd.DataFrame()
    df_out=df_out.append(resultsPath_martaOnly,ignore_index=True)
    df_out=df_out.append(resultsPath_martaPnr,ignore_index=True)
    df_out.to_csv('debug_files/martaPaths.csv', index=False)

    

    if len(df_out) != 0:
        # second argument is starttime
        #df = splitOptions(df_out)
        df = calcTimeDuration(df_out,arrival_hours)
        results_wName=names_adder(df,['data_node_link/marta/stops.txt','data_node_link/grta/stops.txt'],['data_node_link/marta/routes.txt','data_node_link/grta/routes.txt'])
        df_filled = fillEnergy(results_wName,egall,railEnergyDict)
        df_filled.to_csv('debug_files/df_filled_marta.csv', index=False)
        
        return calcDfEg(df_filled, userId, mtTraceDict = traceFileDict)
    else:
        return jsonify({"marta":"route not found"})
        
@app.route("/getGrtaFlask", methods=["GET"])
def getGrtaFlask():
    global egall
    global railEnergyDict
    global dict_grta
    global df_links
    global df_links_trace
    global railstation_df
    global busstation_df
    global grtastation_df

    if 'origin' in request.args and 'destination' in request.args:
        try:
            origin_lat, origin_lon = request.args['origin'].split(',')
        except:
            print("wrong origin input:")
            return jsonify(request)
        try:
            destination_lat, destination_lon = request.args['destination'].split(',')
        except:
            print("wrong destination input:")
            return jsonify(request)
        try:
            userId = request.args['userid']
        except:
            userId = 'unNamed'
            pass
        '''
        #####TESTING PURPOSE##########
        # No transfer for grta
        #origin_lat, origin_lon = '33.781078', '-84.386157'
        origin_lat, origin_lon = '33.783078', '-84.386157'
        destination_lat, destination_lon = '33.751986', '-84.392470'
        arrivalt = '7.03.00'
        endt = '9.00.00'
        ##############################                
        '''
        arrival_hours = end_hours = 0
        if 'startt' in request.args:
            strategy = 1
            arrivalt = request.args['startt']
            foo = arrivalt.split('.')
            stmTime = getCur15(arrivalt)
            arrival_hours = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
        if 'endt' in request.args:
            strategy = 3
            endt = request.args['endt']
            foo2 = endt.split('.')
            stmTime = getCur15(endt)
            end_hours = float(foo2[0]) + float(foo2[1]) / 60 + float(foo2[2])/3600

        # buffer is the max time allowed to arrive earlier than the set destination time
        bufferTime = 1/3.0
        if arrival_hours == 0 and end_hours == 0:
            return jsonify({"error": "must provide departure or arrival time"})
        else:
            if arrival_hours == 0:
                arrival_hours = end_hours - 3.0
            elif end_hours == 0:
                end_hours = arrival_hours + 3.0
        print("arrival hours:", arrival_hours, "end hours:", end_hours, "strategy:", strategy)
        dict_settings={}

        # 211
        ####033018####
        # buffer is the max time allowed to arrive earlier than the set destination time
        dict_settings={'cutoffs':np.arange(15,121,15),'commuteTime':[arrival_hours,end_hours],'wait_thresh':[0,2],'buffer': bufferTime, 'walk_speed':2.0,'cutoff_max':2*60.0,'grid_size':25000.0,'ntp_dist_thresh':5280.0}
        dict_settings['network']={'grta-pnr':dict_grta,'grta-only':dict_grta}
        #dict_settings['cutoffs_label']=['063000', '064500', '070000','071500','073000','074500','080000','081500','083000','084500','090000','091500','093000','094500']
        #strategy: 1, minimum total time excluding wait1; 2, earliest arrival, leave when commute starts 
        dict_settings['strategy']={'marta-only':3,'grta-only':3,'marta-pnr':3,'grta-pnr':3}
        dict_settings['walk_thresh']={'marta-only':0.5,'marta-pnr':0.5,'marta-rail':0.5,'grta-only':1,'grta-pnr':1}
        dict_settings['num_options']={'drive':3,'marta-only':1,'marta-pnr':1,'marta-rail':1,'grta-only':1,'grta-pnr':1}        
        
        walk_speed, grid_size,ntp_dist_thresh=dict_settings['walk_speed'],dict_settings['grid_size'],dict_settings['ntp_dist_thresh']
        df_points = pd.DataFrame({'trip_id':[1], 'ori_lat':[float(origin_lat)], 'ori_lon':[float(origin_lon)], 'dest_lat':[float(destination_lat)], 'dest_lon':[float(destination_lon)]})
        df_points=add_xy(df_points,'ori_lat','ori_lon','x','y','x_sq','y_sq')    
        df_points=point_to_node(df_points,df_links,False, walk_speed,grid_size,ntp_dist_thresh).rename(columns={'NodeID':'o_node','Node_t':'o_t','x':'ox','y':'oy','x_sq':'ox_sq','y_sq':'oy_sq'})
        df_points=add_xy(df_points,'dest_lat','dest_lon','x','y','x_sq','y_sq')
        df_points=point_to_node(df_points,df_links,False, walk_speed,grid_size,ntp_dist_thresh).rename(columns={'NodeID':'d_node','Node_t':'d_t','x':'dx','y':'dy','x_sq':'dx_sq','y_sq':'dy_sq'})
        
        
    else:
        return jsonify({"error":"please provide latlon ori and dest"})

    
    print("current stmTime is:", stmTime)
    row = df_points.iloc[0]
    walk_dist=np.sqrt((row['ox']-row['dx'])**2+(row['oy']-row['dy'])**2)/5280.0
    resultsPath_grtaOnly,runningLogi=Transit(df_points,'grta-only',walk_dist,dict_settings)        
        
    df_ods = pd.DataFrame()
    dict_transitRoutes={}
    # Park-and-Ride
    df_ods_i,runningLogi,dict_transitRoutes['grta-pnr']=pnr_Transit(row,'grta-pnr',dict_settings)
    if len(df_ods_i)>0:
        df_ods_i['option']='grta-pnr'
        df_ods=df_ods.append(df_ods_i)

    if len(df_ods) > 0:
        _df_ods,dict_driveRoutes,errMessage=get_drive_routes(str(row['o_node']),row['o_t'],df_ods,dict_settings,stmTime)
        if len(_df_ods) > 0:
            resultsPath_grtaPnr,runningLogi=output_pnr(row,_df_ods,dict_settings,dict_driveRoutes,dict_transitRoutes,['grta-pnr'])
        else:
            resultsPath_grtaPnr = pd.DataFrame()
            print('cannot arrive to parking lots within required time')
        
    else:
        resultsPath_grtaPnr = pd.DataFrame()
        print('destination does not have pnr stop')
    
    
    #TRACE GENERATION
    gtOnly = False
    gtPnr = False

    traceFileDict = {}

    # create trip file directory if not exist
    now = datetime.now()
    dtstr = now.strftime("%Y%m")
    dateStr = now.strftime("%Y%m%d")
    tripPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr)
    outPath = os.path.join('/mnt/data/www/commwar/httpdocs/data/trips/',dtstr,userId)
    if not os.path.exists(outPath):
        if not os.path.exists(tripPath):
            os.mkdir(tripPath)
        os.mkdir(outPath)
        
    destination_lat_lon = destination_lat+'_'+destination_lon
    origin_lat_lon = origin_lat+'_'+origin_lon

    if len(resultsPath_grtaOnly) > 0:
        #todo: change output station/route split by _
        gtOnly = True
        gtOnly_in = resultsPath_grtaOnly.reset_index(drop=True)
        gtOnly_in = fillLoc(gtOnly_in, origin_lat_lon, destination_lat_lon, option='grta')
        gtOnly_in.to_csv('debug_files/gtOnly_in.csv', index=False)
        gtOnly_in = pd.read_csv('debug_files/gtOnly_in.csv')
        traceFile_direct = getTraceTransit(gtOnly_in, df_links_trace)

        firstRow = traceFile_direct.iloc[0]
        startTime = firstRow[1]
        startTime = startTime.replace(':', '')
        trName = os.path.join(outPath, userId+'.'+dateStr+'.'+startTime+'.gtOnly'+'.csv')
        traceFileDict['gtOnly'] = trName
        print("output trace File to:", trName)
        traceFile_direct.to_csv(trName, index=False, header=False)

    if len(resultsPath_grtaPnr) > 0:
        #todo: change output station/route split by _
        gtPnr = True
        gtPnr_in = resultsPath_grtaPnr.reset_index(drop=True)
        gtPnr_in = fillLoc(gtPnr_in, origin_lat_lon, destination_lat_lon, option='grta')
        gtPnr_in.to_csv('debug_files/gtPnr_in.csv', index=False)
        gtPnr_in = pd.read_csv('debug_files/gtPnr_in.csv')
        traceFile_pnr = getTraceTransit(gtPnr_in, df_links_trace)

        firstRow = traceFile_pnr.iloc[0]
        startTime = firstRow[1]
        startTime = startTime.replace(':', '')
        trName = os.path.join(outPath, userId+'.'+dateStr+'.'+startTime+'.gtPnr'+'.csv')
        traceFileDict['gtPnr'] = trName
        print("output trace File to:", trName)
        traceFile_pnr.to_csv(trName, index=False, header=False)
            
    df_out=pd.DataFrame()
    df_out=df_out.append(resultsPath_grtaOnly,ignore_index=True)
    df_out=df_out.append(resultsPath_grtaPnr,ignore_index=True)
    df_out.to_csv('debug_files/grtaPaths.csv', index=False)

    if len(df_out) != 0:
        # second argument is starttime
        #df = splitOptions(df_out)
        df = calcTimeDuration(df_out,arrival_hours)
        df_filled = fillEnergy(df,egall,railEnergyDict)
        
        #df_out_in = df_out.iloc[1:-1]
        #traceFile = getTraceTransit(df_out_in, 'grta', df_links_drive, railstation_df, busstation_df, grtastation_df, arrivalt,traceG)
        #print(traceFile)
        df_filled.to_csv('debug_files/df_filled_grta.csv', index=False)
        return calcDfEg(df_filled, userId, gtTraceDict=traceFileDict)
    else:
        return jsonify({"grta":"route not found"})


def getNearbyStopsGrtaFlask(df_points, nearness):
    global df_stations_grta
    transitType = 'grta'
    sep = '|'

    if nearness=='origin':
        tail = 'ori'
        geometry = [Point(xy) for xy in zip(df_points.lon_ori, df_points.lat_ori)]
    else:
        tail = 'des'
        geometry = [Point(xy) for xy in zip(df_points.lon_des, df_points.lat_des)]

    crs = {'init': 'epsg:4326', 'no_defs': True}
    df_points = gpd.GeoDataFrame(df_points, crs=crs,geometry=geometry)
    df_points=df_points.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    df_points['y']=df_points['geometry'].apply(lambda p: p.y)
    df_points['x']=df_points['geometry'].apply(lambda p: p.x)
    x1=df_points.loc[0,'x']
    y1=df_points.loc[0,'y']

    ##### Option 1: walk to the closest stop(s)
    candStops = []
    candDists = []
    if transitType=='grta':
        walk_dist_thresh=2
    else:
        walk_dist_thresh=0.5

    df_distances=pd.DataFrame()
    for ind, row in df_stations_grta.iterrows():
        distancei = vincenty((df_points.loc[0,'lat_'+tail], df_points.loc[0,'lon_'+tail]),(row['stop_lat'],row['stop_lon'])).miles
        df_distances=df_distances.append(pd.DataFrame({'distance':[distancei],'stopName':[row['stop_name']],'stopID':[row['stop_id']],'x':[row['x']],'y':[row['y']]}))
    
    df_distances=df_distances.sort_values(['distance'],ascending=[1])
    for ind, row in df_distances.iterrows():
        #print(row['distance'], walk_dist_thresh)
        if row['distance']<=walk_dist_thresh:
            
            candStops.append(row['stopName']+sep+str(row['stopID']))
            #candDists.append(df_distances.iloc[1]['distance'])    
            x2=row['x']
            y2=row['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([row['distance'],manhattan_dist])

    ##### park and ride
    ## 2.2.3 Find the closest node for each origin point
    df_points=point_to_node(df_points,df_links,df_grids)
    df_points=df_points.rename(columns={'Node_Lat':'lat_'+tail,'Node_Lon':'lon_'+tail,'Link_Dist':'link_dist_'+tail,'LinkID':'linkid_'+tail,'Node_Dist':'node_dist_'+tail,'NodeID':'nodeid_'+tail})

    for ind, df_points_i in df_points.iterrows():
        if ind>=0:
            paths=nx.single_source_dijkstra_path(DG,str(df_points_i['nodeid_'+tail]), weight='weight')
            path_len=nx.single_source_dijkstra_path_length(DG,str(df_points_i['nodeid_'+tail]), weight='weight')

    df_park=df_stations_grta[df_stations_grta['has_park']==True]   
    stops_node_id = np.array(df_stations_grta.loc[df_stations_grta['has_park']==True]['nodeid'])


    pkrData = []
    # the sorting length part
    df_path_len=pd.DataFrame.from_dict(path_len,orient='index')
    df_path_len.columns=['len']
    df_path_len=df_path_len.sort_values(['len'],ascending=[1])
    df_path_len['node_id']=df_path_len.index.astype(str)                   
         
    if nearness=='origin':
        i=0
        for ind, row in df_path_len.iterrows():
            nodeKey=row['node_id']
            if int(nodeKey) in stops_node_id:
                #(nodeKey)
                if(i==4):
                    break
                selectedStops = df_park.loc[df_stations_grta['nodeid']==int(nodeKey)]
                for ind, row in selectedStops.iterrows():
                    #print(row['stop_name'], row['stop_id'])
                    pkrData.append((path_len[nodeKey],row['stop_name']+sep+str(row['stop_id']),paths[nodeKey]))
                    print(row['stop_name'])
                i=i+1

    
    return candStops, candDists, pkrData, df_points

def getNearbyStopsMartaFlask(df_points, nearness):
    global df_all_transit
    transitType = 'marta'
    sep = '_'
        
    if nearness=='origin':
        tail = 'ori'
        geometry = [Point(xy) for xy in zip(df_points.lon_ori, df_points.lat_ori)]
    else:
        tail = 'des'
        geometry = [Point(xy) for xy in zip(df_points.lon_des, df_points.lat_des)]

    crs = {'init': 'epsg:4326', 'no_defs': True}
    df_points = gpd.GeoDataFrame(df_points, crs=crs,geometry=geometry)
    df_points=df_points.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    df_points['y']=df_points['geometry'].apply(lambda p: p.y)
    df_points['x']=df_points['geometry'].apply(lambda p: p.x)
    x1=df_points.loc[0,'x']
    y1=df_points.loc[0,'y']
                              
    ##### Option 1: walk to the closest stop(s)
    candStops = []
    candDists = []
    if transitType=='grta':
        walk_dist_thresh=2
    else:
        walk_dist_thresh=0.5

    df_distances=pd.DataFrame()
    for ind, row in df_all_transit.iterrows():
        distancei = vincenty((df_points.loc[0,'lat_'+tail], df_points.loc[0,'lon_'+tail]),(row['stop_lat'],row['stop_lon'])).miles
        df_distances=df_distances.append(pd.DataFrame({'distance':[distancei],'stopName':[row['stop_name']],'stopID':[row['stop_id']],'x':[row['x']],'y':[row['y']]}))
    
    df_distances=df_distances.sort_values(['distance'],ascending=[1])
    for ind, row in df_distances.iterrows():
        #print(row['distance'], walk_dist_thresh)
        if row['distance']<=walk_dist_thresh:
            
            candStops.append(row['stopName']+sep+str(row['stopID']))
            #candDists.append(df_distances.iloc[1]['distance'])    
            x2=row['x']
            y2=row['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([row['distance'],manhattan_dist])

    ##### park and ride
    ## 2.2.3 Find the closest node for each origin point
    df_points=point_to_node(df_points,df_links,df_grids)
    df_points=df_points.rename(columns={'Node_Lat':'lat_'+tail,'Node_Lon':'lon_'+tail,'Link_Dist':'link_dist_'+tail,'LinkID':'linkid_'+tail,'Node_Dist':'node_dist_'+tail,'NodeID':'nodeid_'+tail})

    for ind, df_points_i in df_points.iterrows():
        if ind>=0:
            paths=nx.single_source_dijkstra_path(DG,str(df_points_i['nodeid_'+tail]), weight='weight')
            path_len=nx.single_source_dijkstra_path_length(DG,str(df_points_i['nodeid_'+tail]), weight='weight')

    df_park=df_all_transit[df_all_transit['has_park']==True]   
    stops_node_id = np.array(df_all_transit.loc[df_all_transit['has_park']==True]['nodeid'])


    pkrData = []
    # the sorting length part
    df_path_len=pd.DataFrame.from_dict(path_len,orient='index')
    df_path_len.columns=['len']
    df_path_len=df_path_len.sort_values(['len'],ascending=[1])
    df_path_len['node_id']=df_path_len.index.astype(str)                   
         
    if nearness=='origin':
        i=0
        for ind, row in df_path_len.iterrows():
            nodeKey=row['node_id']
            if int(nodeKey) in stops_node_id:
                #(nodeKey)
                # change to 10 stops
                #if(i==10):
                if(i==4):
                    break
                selectedStops = df_park.loc[df_all_transit['nodeid']==int(nodeKey)]
                for ind, row in selectedStops.iterrows():
                    #print(row['stop_name'], row['stop_id'])
                    pkrData.append((path_len[nodeKey],row['stop_name']+sep+str(row['stop_id']),paths[nodeKey]))
                    print("pkr stops:", row['stop_name'])
                i=i+1
    print("candidate stops:", candStops)
    return candStops, candDists, pkrData, df_points

def errMessages(row,option, condition):
    if condition==1:
        numRoutes,err_message=0,'no ['+option+'] stops could be found from destination that are within walking distance'
        print('Trip' , row['trip_id'], err_message)
    if condition==2:   
        numRoutes,err_message=0,'no ['+option+'] stops could be found from origin that are within walking distance & time'
        print('Trip' , row['trip_id'], err_message)
    if condition==3:
        numRoutes,err_message=0,'no ['+option+'] transit routes could be found between identified transit stops'
        print('Trip' , row['trip_id'], err_message)
    if condition==4:
        numRoutes,err_message=0,'no ['+option+'] transit routes could be found between parking and destination'
        print('Trip' , row['trip_id'], err_message)  
    if condition==5:
        numRoutes,err_message=0,'Destination is reachable via transit from parking, however, the ['+option+'] parking  is not reachable'
        print('Trip' , row['trip_id'], err_message)  
    if condition==6:
        numRoutes,err_message=0,'No transit available, thus ['+option+'] was not launched to find driving routes'
        print('Trip' , row['trip_id'], err_message) 
    return numRoutes,err_message

def getNearbyStops(x,y,dfs,walk_thresh):
    def calculate_dist(x1,y1,x2,y2):
        return (math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))/5280.0
    dfs['man_dist']=dfs.apply(lambda row: calculate_dist(x,y,row['x'],row['y']), axis=1)
    return dfs[dfs['man_dist']<=walk_thresh]


def transit_finder(df, option,dict_settings):
    try:
        commuteTime,walk_speed,cutoff_max=dict_settings['commuteTime'],dict_settings['walk_speed'],dict_settings['cutoff_max']
        DG_transit,transit_links, transfer_links=dict_settings['network'][option]['DG'],dict_settings['network'][option]['links'],dict_settings['network'][option]['t_links']
        num_options,strategy=dict_settings['num_options'][option],dict_settings['strategy'][option]
        t_start=commuteTime[0]
        err_message=''
        df_path=pd.DataFrame()      
        l,p=nx.multi_source_dijkstra(DG_transit, df['o'].tolist(), cutoff=cutoff_max)    
        df_transitDuration=pd.DataFrame()
        df_transitDuration['d'] =l.keys()
        #df_transitDuration['d'],df_transitDuration['transit_duration'] =l.keys(),l.values()
        #df_transitDuration['transit_duration']=df_transitDuration['transit_duration']/60.0
        dfp=pd.DataFrame()
        dfp['d'],dfp['p']=p.keys(),p.values()
        df_transitDuration=df_transitDuration.merge(dfp,how='left',on='d')
        df_transitDuration['o']=df_transitDuration['p'].apply(lambda p: p[0])
        df=df[df['d'].isin(p.keys())].merge(df_transitDuration,on=['o','d'],how='left')
        df=df[~df['p'].isnull()]                
        
        if len(df)==0: # if o-d transit route could be found
            err_message='no ['+option+'] route could be found'
            return df_path, err_message, 0
        else:
            num_AvaiOptions=len(df)
            num_options_found=0
            
            #033018
            #if choose to use arrival time as strategy, filter the dataframe and use strategy1 
            if strategy==3:
                strategy = 1
                if len(df[df['arrival']>=commuteTime[1]-dict_settings['buffer']]) > 0:
                    df=df[df['arrival']>=commuteTime[1]-dict_settings['buffer']]

            if strategy==1:
                df=df.sort_values(['total_duration','arrival','walk_duration'],ascending=[1,1,1])# minimum time spent
            elif strategy==2: # ann 03282018
                df=df.sort_values(['arrival','total_duration','walk_duration'],ascending=[1,1,1])# earliest arrival
            
            #else: # ann 03282018
             #   df['gap']=commuteTime[1]-df['arrival']
              #  df=df.sort(['gap','total_duration','walk_duration'],ascending=[1,1,1])# closest to expected arrival time

            df_path=pd.DataFrame()
            
            if len(transfer_links)==0:
                err_message_extra=', no transfer considered'
                for oid in np.arange(0,len(df),1):
                    row_out=df.iloc[oid]
                    o,d=row_out['o'],row_out['d']
                    num_options_found+=1
                    
                    for ind in np.arange(0,len(p[d])-1,1):
                        node_a=p[d][ind]
                        node_b=p[d][ind+1]
                        row=transit_links[transit_links['stop_key1']==node_a][transit_links['stop_key2']==node_b].iloc[0]
                        if ind==0:
                            # walk link
                            #df_pathi=pd.DataFrame({'A':['origin'],'B':[o],'option':[option],'mode':['walk'],'timeStamp':[t_start+row_out['o_duration']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})
                            # 20180401
                            df_pathi=pd.DataFrame({'A':['origin'],'B':[o],'option':[option],'mode':['walk'],'timeStamp':[row['thrs1']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})

                            df_path=df_path.append(df_pathi)
                            # wait link
                            #df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[row['thrs1']],'time':[row['thrs1']-df_pathi.iloc[-1]['timeStamp']],'dist':[0],'route':['wait1'],'sequence':[2],'option_id':[num_options_found]})
                            # 20180401
                            df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[row['thrs1']],'time':[0],'dist':[0],'route':['wait1'],'sequence':[2],'option_id':[num_options_found]})
                            df_path=df_path.append(df_pathi)
                            seq=3
                        # ride links
                        df_pathi=pd.DataFrame({'A':[node_a],'B':[node_b],'option':[option],'mode':['transit'],'timeStamp':[row['thrs2']],'time':[row['duration']],'dist':[row['dist']],'route':[row['route_id1']],'sequence':[seq],'option_id':[num_options_found]})
                        df_path=df_path.append(df_pathi)
                        seq+=1
                    
                    df_pathi=pd.DataFrame({'A':[d],'B':['destination'],'option':[option],'mode':['walk'],'timeStamp':[df_path.iloc[-1]['timeStamp']+row_out['d_duration']],'time':[row_out['d_duration']],'dist':[row_out['d_dist']],'route':['walk2'],'sequence':[seq],'option_id':[num_options_found]})        
                    df_path=df_path.append(df_pathi)
                    
                    if num_options_found==num_options:
                        break
                err_message= str(len(df))+' ['+option+'] routes found, '+str(num_options_found)+' reported'+err_message_extra
                
            else:
                err_message_extra=', transfer filtered'
                for oid in np.arange(0,len(df),1):
                    row_out=df.iloc[oid]
                    o,d=row_out['o'],row_out['d']
                    first_transitLink=transit_links[transit_links['stop_key1']==p[d][0]][transit_links['stop_key2']==p[d][1]][transit_links['type']!='transfer']
                    last_transitLink=transit_links[transit_links['stop_key1']==p[d][-2]][transit_links['stop_key2']==p[d][-1]][transit_links['type']!='transfer']
                    if (len(first_transitLink)==0) or (len(last_transitLink)==0): 
                        err_message_extra=' , if walk farther than threshold, ['+option+'] would save more time'
                        num_AvaiOptions-=1
                        continue
                    else:
                        num_options_found+=1
                        first_transitLink=first_transitLink.iloc[0]
                        
                        # walk link
                        #df_pathi=pd.DataFrame({'A':['origin'],'B':[o],'option':[option],'mode':['walk'],'timeStamp':[t_start+row_out['o_duration']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})
                        # 20180401
                        df_pathi=pd.DataFrame({'A':['origin'],'B':[o],'option':[option],'mode':['walk'],'timeStamp':[first_transitLink['thrs1']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})
                        df_path=df_path.append(df_pathi)
                        
                        # wait link
                        seq=df_path.iloc[-1]['sequence']+1
                        #df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[first_transitLink['thrs1']],'time':[first_transitLink['thrs1']-df_pathi.iloc[-1]['timeStamp']],'dist':[0],'route':['wait1'],'sequence':[seq],'option_id':[num_options_found]})
                        # 20180401
                        df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[first_transitLink['thrs1']],'time':[0],'dist':[0],'route':['wait1'],'sequence':[seq],'option_id':[num_options_found]})
                        df_path=df_path.append(df_pathi)
                        
                        # ride links
                        seq+=1
                        for ind in np.arange(0,len(p[d])-1,1):
                            node_a=p[d][ind]
                            node_b=p[d][ind+1]
                            rows=transit_links[transit_links['stop_key1']==node_a][transit_links['stop_key2']==node_b]
                            if ind==0:
                                row=first_transitLink
                            elif len(rows['type'].unique().tolist())==2:
                                '''
                                if transfer & ride links can both be found from graph
                                    determine the link type based on "route_id of prev path link" ?="route_id of cur ride link"
                                '''
                                row=rows[rows['type']=='ride'].iloc[0] if rows[rows['type']=='ride'].iloc[0]['route_id1']==df_pathi.iloc[-1]['route'] else rows[rows['type']=='transfer'].iloc[0]   
                            else:
                                row=rows.iloc[0]
                                
                            if row['type']=='transfer':
                                '''
                                1 transfer link converted to 3 links {walk, wait, ride}
                                '''
                                _t=transfer_links[transfer_links['stop_key1']==node_a][transfer_links['stop_key3']==node_b].iloc[0,:]
                                df_pathi=pd.DataFrame({'A':[node_a,_t['stop_key2'],_t['stop_key2']],'B':[_t['stop_key2'],_t['stop_key2'],_t['stop_key3']],
                                'option':[option,option,option],'mode':['walk','wait','transit'],
                                'timeStamp':[_t['thrs1'],_t['thrs2'],_t['thrs3']],'time':[_t['t_walk'],_t['t_wait'],_t['thrs3']-_t['thrs2']],
                                'dist':[_t['man_dist'],0,_t['dist']],'route':['walk','wait',_t['route_id2']],'sequence':[seq,seq+1,seq+2],'option_id':[num_options_found,num_options_found,num_options_found]})
                            else:
                                df_pathi=pd.DataFrame({'A':[node_a],'B':[node_b],'option':[option],'mode':['transit'],'timeStamp':[row['thrs2']],'time':[row['duration']],'dist':[row['dist']],'route':[row['route_id1']],'sequence':[seq],'option_id':[num_options_found]})
                            df_path=df_path.append(df_pathi)
                            seq+=1
                        
                        df_pathi=pd.DataFrame({'A':[d],'B':['destination'],'option':[option],'mode':['walk'],'timeStamp':[df_path.iloc[-1]['timeStamp']+row_out['d_duration']],'time':[row_out['d_duration']],'dist':[row_out['d_dist']],'route':['walk2'],'sequence':[seq],'option_id':[num_options_found]})        
                        df_path=df_path.append(df_pathi)
                        
                        if num_options_found==num_options:
                            break
                if num_options_found==0:
                    err_message_extra=' , if walk farther than threshold, ['+option+'] could be possible'
                err_message= str(len(df))+' ['+option+'] routes found, '+str(num_options_found)+' reported'+err_message_extra
                              
        return df_path, err_message, num_AvaiOptions
    except Exception as e:
        print('comes the error', str(e))
        return pd.DataFrame(),err_message, 0

def getNearbyStops_old(df_points, nearness, df, transitType='marta'):
    if transitType == 'marta':
        sep = '_'
    else:
        sep = '|'
    
    if nearness=='origin':
        tail = 'ori'
        geometry = [Point(xy) for xy in zip(df_points.lon_ori, df_points.lat_ori)]
    else:
        tail = 'des'
        geometry = [Point(xy) for xy in zip(df_points.lon_des, df_points.lat_des)]

    crs = {'init': 'epsg:4326', 'no_defs': True}
    df_points = gpd.GeoDataFrame(df_points, crs=crs,geometry=geometry)
    df_points=df_points.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    df_points['y']=df_points['geometry'].apply(lambda p: p.y)
    df_points['x']=df_points['geometry'].apply(lambda p: p.x)
    x1=df_points.loc[0,'x']
    y1=df_points.loc[0,'y']
                              
    ##### Option 1: walk to the closest stop(s)
    candStops = []
    candDists = []
    if transitType=='grta':
        walk_dist_thresh=2
    else:
        walk_dist_thresh=0.25

    df_distances=pd.DataFrame()
    for ind, row in df.iterrows():
        distancei = vincenty((df_points.loc[0,'lat_'+tail], df_points.loc[0,'lon_'+tail]),(row['stop_lat'],row['stop_lon'])).miles
        df_distances=df_distances.append(pd.DataFrame({'distance':[distancei],'stopName':[row['stop_name']],'stopID':[row['stop_id']],'x':[row['x']],'y':[row['y']]}))
    
    df_distances=df_distances.sort_values(['distance'],ascending=[1])
    for ind, row in df_distances.iterrows():
        #print(row['distance'], walk_dist_thresh)
        if row['distance']<=walk_dist_thresh:
            
            candStops.append(row['stopName']+sep+str(row['stopID']))
            #candDists.append(df_distances.iloc[1]['distance'])    
            x2=row['x']
            y2=row['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([row['distance'],manhattan_dist])
        '''
        df_rails=df[df['type']=='rail']
        df_bus=df[df['type']=='bus']
        if df_distances.iloc[0]['distance']<=walk_dist_thresh:
            candStops.append(df_distances.iloc[0]['stopName']+sep+str(df_distances.iloc[0]['stopID']))
            #candDists.append(df_distances.iloc[0]['distance'])    
            x2=df_distances.iloc[0]['x']
            y2=df_distances.iloc[0]['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([df_distances.iloc[0]['distance'],manhattan_dist]) 
        if df_distances.iloc[1]['distance']<=walk_dist_thresh:
            candStops.append(df_distances.iloc[1]['stopName']+sep+str(df_distances.iloc[1]['stopID']))
            #candDists.append(df_distances.iloc[1]['distance'])    
            x2=df_distances.iloc[1]['x']
            y2=df_distances.iloc[1]['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([df_distances.iloc[0]['distance'],manhattan_dist]) 
        # 'rail' --> marta rail/grta
        df_distances=pd.DataFrame()
        for ind, row in df_rails.iterrows():
            distancei = vincenty((df_points.loc[0,'lat_'+tail], df_points.loc[0,'lon_'+tail]),(row['stop_lat'],row['stop_lon'])).miles
            df_distances=df_distances.append(pd.DataFrame({'distance':[distancei],'stopName':[row['stop_name']],'stopID':[row['stop_id']],'x':[row['x']],'y':[row['y']]}))
        df_distances=df_distances.sort_values(['distance'],ascending=[1])
        
        if df_distances.iloc[0]['distance']<=walk_dist_thresh:
            candStops.append(df_distances.iloc[0]['stopName']+sep+str(df_distances.iloc[0]['stopID']))
            x2=df_distances.iloc[0]['x']
            y2=df_distances.iloc[0]['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([df_distances.iloc[0]['distance'],manhattan_dist])            
        if df_distances.iloc[1]['distance']<=walk_dist_thresh:
            candStops.append(df_distances.iloc[1]['stopName']+sep+str(df_distances.iloc[1]['stopID']))
            #candDists.append(df_distances.iloc[1]['distance'])
            x2=df_distances.iloc[1]['x']
            y2=df_distances.iloc[1]['y']
            manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280 
            candDists.append([df_distances.iloc[0]['distance'],manhattan_dist]) 
        '''
    
    ##### park and ride
    ## 2.2.3 Find the closest node for each origin point
    df_points=point_to_node(df_points,df_links,df_grids)
    df_points=df_points.rename(columns={'Node_Lat':'lat_'+tail,'Node_Lon':'lon_'+tail,'Link_Dist':'link_dist_'+tail,'LinkID':'linkid_'+tail,'Node_Dist':'node_dist_'+tail,'NodeID':'nodeid_'+tail})

    for ind, df_points_i in df_points.iterrows():
        if ind>=0:
            paths=nx.single_source_dijkstra_path(DG,str(df_points_i['nodeid_'+tail]), weight='weight')
            path_len=nx.single_source_dijkstra_path_length(DG,str(df_points_i['nodeid_'+tail]), weight='weight')

    df_park=df[df['has_park']==True]   
    stops_node_id = np.array(df.loc[df['has_park']==True]['nodeid'])


    pkrData = []
    # the sorting length part
    df_path_len=pd.DataFrame.from_dict(path_len,orient='index')
    df_path_len.columns=['len']
    df_path_len=df_path_len.sort_values(['len'],ascending=[1])
    df_path_len['node_id']=df_path_len.index.astype(str)                   
         
    if nearness=='origin':
        
        i=0
        '''
        try: #if origin node has park-n-ride stations
            print()
            rows=df_park[df_park['nodeid']==int(df_points.loc[0,'nodeid_'+tail])]
            for ind,row in rows.iterrows():
                print(row)
                if(i==4):
                    break
                pkrData.append((0,row['stop_name']+sep+str(row['stop_id']),[row['nodeid'],row['nodeid']]))
                print(0)
                i=i+1
        except:
            pass
        '''
        for ind, row in df_path_len.iterrows():
            nodeKey=row['node_id']
            if int(nodeKey) in stops_node_id:
                #(nodeKey)
                if(i==4):
                    break
                selectedStops = df_park.loc[df['nodeid']==int(nodeKey)]
                for ind, row in selectedStops.iterrows():
                    #print(row['stop_name'], row['stop_id'])
                    pkrData.append((path_len[nodeKey],row['stop_name']+sep+str(row['stop_id']),paths[nodeKey]))
                    print(row['stop_name'])
                i=i+1
    '''
    stops_parking_id = np.array(df.loc[df['has_park']==True]['stop_id'])
    stops_node_id = np.array(df.loc[df['has_park']==True]['nodeid'])
    stops_parking_names = np.array(df.loc[df['has_park']==True]['stop_name'])

    pkrStops = []
    pkrStopIds = []
    pkrDists = []
    pkrPaths = []
    pkrStopNames = []
    for nodeKey in paths:
        if int(nodeKey) in stops_node_id:
            ind = list(stops_node_id).index(int(nodeKey))
            pkrStopNames.append(stops_parking_names[ind])
            pkrStopIds.append(stops_parking_id[ind])
            pkrStops.append(nodeKey)
            pkrDists.append(path_len[nodeKey]) # obtained from DG (graph's weight) --> unit: minutes of travel time
            pkrPaths.append(paths[nodeKey])
    minPkrIndSorted = np.argsort(pkrDists)
    first2Ind = minPkrIndSorted[:2]
    
    pkrData = []
    for ind in first2Ind:
        pkrData.append((pkrDists[ind], pkrStopNames[ind]+sep+str(pkrStopIds[ind]), pkrStopNames[ind]+sep+str(pkrStopIds[ind]), pkrPaths[ind]))
    '''
    return candStops, candDists, pkrData, df_points

def getNearbyStopsBefore(df_points, nearness, df_rails, pkrDF, transitType='marta'):
    if transitType == 'marta':
        sep = '_'
    else:
        sep = '|'
    # gets nearby stops, park-and-ride stops, and the paths from origin location to these stations. Paths are nodes, and when concatenated they form the link_ids
    
    if nearness=='origin':
        tail = 'ori'
        geometry = [Point(xy) for xy in zip(df_points.lon_ori, df_points.lat_ori)]
    else:
        tail = 'des'
        geometry = [Point(xy) for xy in zip(df_points.lon_des, df_points.lat_des)]
    crs = {'init': 'epsg:4326', 'no_defs': True}
    
    df_points = gpd.GeoDataFrame(df_points, crs=crs,geometry=geometry)
    
    ## 2.2.2 Convert the origins points to the same coordinate system as Link file
    #print("current df:")
    #print(df_points)
    df_points=df_points.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    

    ## 2.2.3 Find the closest node for each origin point
    df_points=point_to_node(df_points,df_links,df_grids)
    
    df_points=df_points.rename(columns={'Node_Lat':'lat_'+tail,'Node_Lon':'lon_'+tail,'Link_Dist':'link_dist_'+tail,'LinkID':'linkid_'+tail,'Node_Dist':'node_dist_'+tail,'NodeID':'nodeid_'+tail})
    ## 2.3 Find the closest 2 rail stations !!Driving Distance!! (station ID, distance to the station, routes) for each origin's node
    df_points['rail1_'+tail]=0 # closest rail station's node ID to the origin node
    df_points['rail2_'+tail]=0 # 2nd closest rail station's node ID to the origin node
    df_points['dist1_'+tail]=0 # distance to the closest rail station to the origin node
    df_points['dist2_'+tail]=0 # distance to the 2nd closest rail station to theorigin node
    df_points['routes1_'+tail]='' # route from the closest rail station's node to the origin node
    df_points['routes2_'+tail]='' # route from the 2nd closest rail station's node to the origin node

    #extract the unique nodes
    #df_points= pd.concat([df_links.A,df_links.B]).unique()
    #df_points = pd.DataFrame(df_points)
    #df_points.columns=['nodeid_'+tail]
    start_time = time.time()
    #errorRows = []

    for ind, df_points_i in df_points.iterrows():
        if ind>=0:
            paths=nx.single_source_dijkstra_path(DG,str(df_points_i['nodeid_'+tail]), weight='weight')
            path_len=nx.single_source_dijkstra_path_length(DG,str(df_points_i['nodeid_'+tail]), weight='weight')
            df_path_len=pd.DataFrame.from_dict(path_len, orient='index').reset_index()
            df_path_len.columns=['nodeid','drive_dist']
            df_path_len_rail=df_path_len[df_path_len['nodeid'].isin([str(x) for x in df_rails.nodeid.unique().tolist()])].reset_index()
            ## for the closest 
            try:
                min1_nodeid=df_path_len_rail.loc[np.argsort(df_path_len_rail.loc[:,'drive_dist'])[0],'nodeid']
                df_points.loc[ind,'rail1_'+tail]=min1_nodeid 
                df_points.loc[ind,'dist1_'+tail]=path_len[min1_nodeid]
                df_points.loc[ind,'routes1_'+tail]='->'.join(paths[min1_nodeid])
            except:
                errorRows.append(ind)
            ## for the 2nd closest
            try:
                min2_nodeid=df_path_len_rail.loc[np.argsort(df_path_len_rail.loc[:,'drive_dist'])[1],'nodeid']       
                df_points.loc[ind,'rail2_'+tail]=min2_nodeid 
                df_points.loc[ind,'dist2_'+tail]=path_len[min2_nodeid]
                df_points.loc[ind,'routes2_'+tail]='->'.join(paths[min2_nodeid]) 
            except:
                errorRows.append(ind)
            
    #elapsed_time = time.time() - start_time 
    #print(elapsed_time)
    candStops = []
    candDists = []
    
    df = pkrDF
    # ToDo: process stops into nearest node, then replace "stops_parking_id" below with these nearest nodes
    stops_parking_id = np.array(df.loc[df['has_park']==True]['stop_id'])
    stops_node_id = np.array(df.loc[df['has_park']==True]['nodeid'])
    stops_parking_names = np.array(df.loc[df['has_park']==True]['stop_name'])

    pkrStops = []
    pkrStopIds = []
    pkrDists = []
    pkrPaths = []
    pkrStopNames = []
    #print("Park and ride Stations:", stops_parking_id)
    for nodeKey in paths:
        if int(nodeKey) in stops_node_id:
            ind = list(stops_node_id).index(int(nodeKey))
            pkrStopNames.append(stops_parking_names[ind])
            pkrStopIds.append(stops_parking_id[ind])
            pkrStops.append(nodeKey)
            pkrDists.append(path_len[nodeKey])
            pkrPaths.append(paths[nodeKey])
    minPkrIndSorted = np.argsort(pkrDists)
    minPkrInd = minPkrIndSorted[0]
    minPkrDist = pkrDists[minPkrInd]
    minPkrStop = pkrStops[minPkrInd]
    minPkrName = pkrStopNames[minPkrInd]

    # Note the distance here is not correct
    '''
    print("Nearest park and ride stop:", minPkrStop)
    print("Nearest park and ride name:", minPkrName)
    print("Path to nearest:", pkrPaths[minPkrInd])
    print("Dist to nearest:", pkrDists[minPkrInd])
    '''
    stopDict = pd.Series(df_rails.stop_name.values,index=df_rails.nodeid).to_dict()
    idDict = pd.Series(df_rails.stop_id.values,index=df_rails.nodeid).to_dict()
    #print("park and ride closest:", stopDict[int(minPkrStop)])
    #for ind, row in df_points.iterrows():
    row = df_points.iloc[0]
    #if row['dist1_'+tail] <= 0.01: # distance threshold !
    #TODO: set distance threshold
    #TODO: add walk time calculation
    
           
    if row['dist1_'+tail] <= 1:
        stopName = str(stopDict[int(row['rail1_'+tail])])
        stopID = str(idDict[int(row['rail1_'+tail])])
        #candStops.append(stopDict[int(row['rail1_'+tail])])
        candStops.append(stopName+sep+stopID)
        candDists.append(row['dist1_'+tail])
    #if row['dist2_'+tail] <= 0.01:
    if row['dist2_'+tail] <= 1:
        stopName = str(stopDict[int(row['rail2_'+tail])])
        stopID = str(idDict[int(row['rail2_'+tail])])
        #candStops.append(stopDict[int(row['rail2_'+tail])])
        candStops.append(stopName+sep+stopID)
        candDists.append(row['dist2_'+tail])
    #print(df_points)

    pkrData = []
    first2Ind = minPkrIndSorted[:2]
    for ind in first2Ind:
        # TODO: add distance threshold for destination
        
        #print("Path to station:", pkrPaths[ind])
        #pkrData.append((pkrDists[ind], pkrStops[ind], pkrStopNames[ind], pkrPaths[ind]))
        pkrData.append((pkrDists[ind], pkrStopNames[ind]+sep+str(pkrStopIds[ind]), pkrStopNames[ind]+sep+str(pkrStopIds[ind]), pkrPaths[ind]))
    #print(row['routes1_'+tail], row['routes2_'+tail])
    
    '''
    # cody
    watsonRoute1 = getWatson(row['routes1_'+tail].split('->'))
    avgWatson = getAvgWatson(watsonRoute1)
    watsonRoute2 = getWatson(row['routes2_'+tail].split('->'))
    avgWatson2 = getAvgWatson(watsonRoute2)
    watsonPkr = getWatson(pkrPaths[minPkrInd])
    avgWatsonPkr = getAvgWatson(watsonPkr)
    # nearby_parking.append((stops_parking_time[index],stops_parking_id[index],stops_parking_names[index], stops_parking_lattitude[index], stops_parking_longitude[index]))
    '''
    return candStops, candDists, pkrData, df_points

def getNearByStopsFromLocation(location_tuple, nearness):
    nearby = []
    distance_between = []
    with open(os.path.join(dirPath,'links.txt')) as csvfile_marta:
        linksreader = csv.reader(csvfile_marta)
        # with open('links_grta.txt', 'rb') as csvfile_grta:
        #     linksreader_grta = csv.reader(csvfile_grta)
        #     linksreader = list(linksreader_grta) + list(linksreader_marta)
        if nearness=="origin":
            for row in linksreader:
                current_stop_location = (row[4],row[6])
                distance = vincenty(location_tuple, current_stop_location).miles
                if distance<0.01:
                    nearby.append(row[0].split("s")[1].split("t")[0])
                    distance_between.append(distance)
        elif nearness == "dest":
            for row in linksreader:
                current_stop_location = (row[5],row[7])
                distance = vincenty(location_tuple, current_stop_location).miles
                if distance<0.1:
                    nearby.append(row[0].split("s")[1].split("t")[0])
                    distance_between.append(distance)
        return [(dist,stp) for (dist,stp) in sorted(zip(distance_between,nearby))]

def getNearbyParkandRide(origin_tuple):
    df = pd.read_csv(os.path.join(dirPath,"../../Data/stops_parking.txt"))
    stops_parking_id =np.array(df.loc[df['has_park']==True]['stop_id'])
    stops_parking_names = np.array(df.loc[df['has_park']==True]['stop_name'])
    stops_parking_lattitude = np.array(df.loc[df['has_park']==True]['stop_lat'])
    stops_parking_longitude = np.array(df.loc[df['has_park']==True]['stop_lon'])
    stops_parking_location = zip(stops_parking_lattitude,stops_parking_longitude)
    stops_parking_time = []
    df = pd.read_csv(os.path.join(dirPath,"../../Data/stops_parking.txt"))
    nearby_parking = []

    for index,stop_parking_name in enumerate(stops_parking_names):
        print(stop_parking_name)
        distance = vincenty(origin_tuple,stops_parking_location[index]).miles
        time_to_travel = (float(distance)/35)*60
        print(distance,time_to_travel)
        stops_parking_time.append(time_to_travel)
    for index,stop_parking_name in enumerate(stops_parking_names):
        if (stops_parking_time[index]*35)/60 < 10: #check if distance is less than 10 miles
            nearby_parking.append((stops_parking_time[index],stops_parking_id[index],stops_parking_names[index], stops_parking_lattitude[index], stops_parking_longitude[index]))
    return nearby_parking


def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            length += G.edge[u][v].get(weight, 1)
    return length    
    
def find_grid(df_grids,pt_x,pt_y):
    #print([(ind, it) for ind,it in df_grids[df_grids.maxx>=pt_x][df_grids.minx<=pt_x][df_grids.maxy>=pt_y][df_grids.miny<=pt_y].iloc[0,:].iteritems()])
    #print('here')
    return df_grids[df_grids.maxx>=pt_x][df_grids.minx<=pt_x][df_grids.maxy>=pt_y][df_grids.miny<=pt_y].iloc[0,:]['GridID']

def define_gridid(df_grids,df_pts):
    df_pts['GridID']=df_pts['geometry'].apply(lambda x: find_grid(df_grids,x.coords[0][0],x.coords[0][1]))
    #print(df_pts)
    return df_pts
    
def find_closestLink(point,lines):
    dists=lines.distance(point)
    return [dists.argmin(),dists.min()]
    
def find_closestNodes(point,stations):
    dists=stations.distance(point)
    return [np.argsort(dists)[0],dists[np.argsort(dists)[0]],np.argsort(dists)[1],dists[np.argsort(dists)[1]]]

def calculate_dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    
def point_to_node_drive(df_points,df_links,df_grids):
    # INITIALIZATION
    df_points=define_gridid(df_grids,df_points)
    df_points['Link_Dist']=0
    df_points['LinkID']=0
    df_points['Node_Dist']=0
    df_points['NodeID']=0
    df_points['Node_Lat']=0
    df_points['Node_Lon']=0
    # CALCULATION
    for ind, df_points_i in df_points.iterrows():
        # find links in the grid
        df_links_i=df_links[df_links['GridID']==df_points_i['GridID']]
        # find the closest link and the distance
        LinkID_Dist=find_closestLink(df_points_i.geometry,gpd.GeoSeries(df_links_i.geometry))
        # LinkID_Dist is the magnitude of distance
        df_points.loc[ind,'Link_Dist']=LinkID_Dist[1]
        linki=df_links_i.loc[LinkID_Dist[0]]
        df_points.loc[ind,'LinkID']=linki['LinkID']
        # find the closest node on the link
        dist1=calculate_dist(df_points.loc[ind,'geometry'].coords[0][0],df_points.loc[ind,'geometry'].coords[0][1],linki['A_lon'],linki['A_lat'])
        dist2=calculate_dist(df_points.loc[ind,'geometry'].coords[0][0],df_points.loc[ind,'geometry'].coords[0][1],linki['B_lon'],linki['B_lat'])
        if (dist1<dist2):
            df_points.loc[ind,'Node_Dist']=dist1
            df_points.loc[ind,'NodeID']=linki['A']
            df_points.loc[ind,'Node_Lat']=linki['A_lat']
            df_points.loc[ind,'Node_Lon']=linki['A_lon']
        else:
            df_points.loc[ind,'Node_Dist']=dist2
            df_points.loc[ind,'NodeID']=linki['B']
            df_points.loc[ind,'Node_Lat']=linki['B_lat']
            df_points.loc[ind,'Node_Lon']=linki['B_lon']
    return df_points

'''
df_final_paths = returnPaths_toDF_GlobalLoop(origin_stops_list_marta,ori_dists_marta,destination_stops_list_marta,des_dists_marta,arrivalt,endt,pkrStopsOri_marta,df_points_ori_marta,'marta')
origin_stops_list=origin_stops_list_marta
ori_dists=ori_dists_marta
destination_stops_list=destination_stops_list_marta
des_dists=des_dists_marta
transitType='marta'
pkrStopsOri=pkrStopsOri_marta
pkrStops=pkrStops_marta


origin_stops_list=origin_stops_list_grta
ori_dists=ori_dists_grta
destination_stops_list=destination_stops_list_grta
des_dists=des_dists_grta
transitType='grta'
pkrStopsOri=pkrStopsOri_grta
pkrStops=pkrStops_grta
'''
def returnPaths_toDF_GlobalLoop(origin_stops_list,ori_dists,destination_stops_list,des_dists,arrivalt,endt,pkrStopsOri,df_points,transitType):
    
    startInfo = df_points.iloc[0]
    
    hours,minutes,seconds = arrivalt.split('.')
    start_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0
    hours,minutes,seconds = endt.split('.')
    end_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0   
                   
    try:
        del df_paths
    except:
        print('################## starting transit ##################')
    
    startTime = time.time()
    print("getting transit. Current time"+str(startTime))
    # craete od pairs
    df_od_pairs=pd.DataFrame()
     
    walk_speed=2
    '''
    origin_stops_list=['DUNWOODY STATION_40910']
    destination_stops_list=['DUNWOODY PL @ ROBERTS DR']
    origin_stops_list=['BRIARCLIFF RD@3055_61017']
    origin_stops_list=['BRIARCLIFF RD @ BRUCE RD_60134']
    destination_stops_list=['DUNWOODY PL @ ROBERTS DR_27002']
    destination_stops_list=['DUNWOODY PL @ CEDAR RUN_27012']
    ori_dists=[[0,0]]
    des_dists=[[0,0]]
    '''
    for i in range(len(origin_stops_list)):
        origin = origin_stops_list[i]
        walk1_dist=ori_dists[i][1]
        walk1_time_stamp=walk1_dist/walk_speed/1.0+start_time_hh
        for j in range(len(destination_stops_list)):
            destination = destination_stops_list[j]
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            try:
                if transitType=='marta':
                    events_start = STOP_EVENTS[str(origin)]
                    events_end  = STOP_EVENTS[str(destination)]
                    events_start_filtered=[x for x in events_start if x[1]=='ride']
                    events_end_filtered=[x for x in events_end if x[1]=='ride']
                else:
                    events_start = STOP_EVENTS_GRTA[str(origin)]
                    events_end  = STOP_EVENTS_GRTA[str(destination)]
                    events_start_filtered=events_start
                    events_end_filtered=events_end
                
                
                #if origin=='BRIARCLIFF RD @ BRUCE RD_60134' and destination=='DUNWOODY PL @ CEDAR RUN_27012':
                   #print(events_start_filtered)
                   #print(events_end_filtered)
                k=1
                
                #TODO: add ttype in data_prep_grta
                for t2,ttype in events_end_filtered:
                    if t2<=end_time_hh-walk2_time:
                        events_start_new=[x[0] for x in events_start_filtered if x[0]<t2 and x[0]>=walk1_time_stamp]
                        for t1 in events_start_new:
                            #if t1>=walk1_time_stamp:
                            k=k+1
                            if k%200==0:
                                print('====',t1,t2,'====')
                            start_node = "s"+str(origin)+"t"+str(t1)
                            end_node = "s"+str(destination)+"t"+str(t2)
                            df_od_pairsi=pd.DataFrame({'start_node':[start_node],'end_node':[end_node],'walk1_dist':[walk1_dist],'walk2_dist':[walk2_dist],'t1':[t1],'t2':[t2]})
                            df_od_pairs=df_od_pairs.append(df_od_pairsi,ignore_index=True)
            except:
                print('cannot build od dataframe')
    
    option_id=0
    df_paths=pd.DataFrame()
    try:
        df_od_pairs=df_od_pairs.sort_values(['t2','t1'],ascending=[1,0])
    except:
        pass
    '''
    transitGraph['sDUNWOODY STATION_40910t7.533333333333333']
    transitGraph['sDUNWOODY STATION_53t7.5520062938012416']
    transitGraph['sDUNWOODY STATION_53t7.683333333333334']
    '''
    print('################## Finding Transit Routes ##################')
    numPaths = 0
    for ind,row in df_od_pairs.iterrows():
        #print(ind)
        origin_stops=row['start_node']
        destination_stops=row['end_node']
        
        if transitType == 'marta':
            transitGraph = GRAPH
        else: 
            transitGraph = GRAPH_GRTA
        tup=[]        
        try:
            #tup = getShortestTravelTime_OptiArr(walk1_time_stamp,end_time_hh-walk2_time,origin_stops,destination_stops,transitType)
            if nx.has_path(transitGraph,origin_stops,destination_stops):
                tup = nx.dijkstra_path(transitGraph,origin_stops,destination_stops)
                #tup = nx.shortest_path(transitGraph,origin_stops,destination_stops)

                if tup != []:
                    option_id=option_id+1
                    link_seq=0
                    # output walking part
                    link_seq=link_seq+1 # first link walk1
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':['origin'],'B':[origin_stops],'option':['TransitOnly'],'mode':['walk1'],'time':[walk1_time_stamp],'dist':[walk1_dist],'route':['walking1'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    
                    link_seq=link_seq+1
                    df_pathsi2=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[origin_stops],'B':[origin_stops],'option':['TransitOnly'],'mode':['wait1'],'time':[float(tup[0].split('t')[-1])],'dist':[0],'route':['waiting1'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi2,ignore_index=True) 
                    
                    # output transit part
                    for (u,v) in zip(tup[0:], tup[1:]):
                        link_seq=link_seq+1
                        df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['TransitOnly'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths=df_paths.append(df_pathsi,ignore_index=True)
                        
                    # output walking part
                    link_seq=link_seq+1 # last link walk2
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['TransitOnly'],'mode':['walk2'],'time':[walk2_time+df_paths.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    
                    print('          ',transitType,' routing found')
                    numPaths += 1
                    if numPaths >= 2:
                        break
        except Exception as e:
            print(str(e))
            tup = []
            
    
    elapsed_time = time.time() - startTime 
    print("Transit finder used time:", elapsed_time)
    
    startTime = time.time()
    print("getting park and Ride. Current time"+str(startTime))
    
    
    #pkrStopsOri_backup=pkrStopsOri
    #pkrStopsOri=[pkrStopsOri[1]]
    #pkrStopsOri=pkrStopsOri_backup
    df_od_pairs_park=pd.DataFrame()
    print('################## starting park and ride ##################')
    for j in range(len(destination_stops_list)):
        destination_stops = destination_stops_list[j]
        for park_stops_data in pkrStopsOri:
            
            arrivalt_park=park_stops_data[0]/60.0+start_time_hh
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            origin=park_stops_data[1]
            destination = destination_stops_list[j]
            
            try:
                if transitType=='marta':
                    events_start = STOP_EVENTS[str(origin)]
                    events_end  = STOP_EVENTS[str(destination)]
                    events_start_filtered=[x for x in events_start if x[1]=='ride']
                    events_end_filtered=[x for x in events_end if x[1]=='ride']
                else:
                    events_start = STOP_EVENTS_GRTA[str(origin)]
                    events_end  = STOP_EVENTS_GRTA[str(destination)]
                    events_start_filtered=events_start
                    events_end_filtered=events_end
                #print(origin,events_start_filtered)
                k=1
                for t2,ttype in events_end_filtered:
                    if t2<=end_time_hh-walk2_time:
                        events_start_new=[x[0] for x in events_start_filtered if x[0]<t2 and x[0]>=arrivalt_park]
                        for t1 in events_start_new:
                            k=k+1
                            if k%200==0:
                                print('====',t1,t2,'====')
                            start_node = "s"+str(origin)+"t"+str(t1)
                            
                            end_node = "s"+str(destination)+"t"+str(t2)
                            df_od_pairs_parki=pd.DataFrame({'start_node':[start_node],'end_node':[end_node],'walk2_dist':[walk2_dist],'t1':[t1],'t2':[t2]})
                            df_od_pairs_park=df_od_pairs_park.append(df_od_pairs_parki,ignore_index=True)
            except Exception as e:
                print(str(e))
                print('cannot build od dataframe')
            
    df_paths_park=pd.DataFrame()
    try:
        df_od_pairs_park=df_od_pairs_park.sort_values(['t2','t1'],ascending=[1,0])
    except:
        # no odpairs
        pass
    print('################## Finding Transit Routes ##################')
    numPaths = 0
    for ind,row in df_od_pairs_park.iterrows():
        origin_stops=row['start_node']
        destination_stops=row['end_node']
        
        if transitType == 'marta':
            transitGraph = GRAPH
        else: 
            transitGraph = GRAPH_GRTA
        tup=[]        
        try:
            if nx.has_path(transitGraph,origin_stops,destination_stops):
                tup = nx.dijkstra_path(transitGraph,origin_stops,destination_stops)
                #tup = nx.shortest_path(transitGraph,origin_stops,destination_stops)

                if tup != []:
                    option_id=option_id+1
                    link_seq=0
                    # output driving part
                    lastNode = park_stops_data[2][0]
                    foo = arrivalt.split('.')            
                    curTime = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
                    for node in park_stops_data[2][1:]:
                        link_seq=link_seq+1
                        curNode = node
                        linkTime = DG.edge[lastNode][curNode].get('weight', 1)
                        # linkTime is in minutes
                        curTime += linkTime / 60
                        dist = DG.edge[lastNode][curNode].get('dist', 1)
                        df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[lastNode],'B':[curNode],'option':['ParkNRide'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':['driving'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                        lastNode = curNode
                    # output waiting 
                    link_seq=link_seq+1
                    df_paths_parki2=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[origin_stops],'B':[origin_stops],'option':['ParkNRide'],'mode':['wait1'],'time':[float(tup[0].split('t')[-1])],'dist':[0],'route':['waiting1'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths_park=df_paths_park.append(df_paths_parki2,ignore_index=True) 
                    
                    # output transit part
                    for (u,v) in zip(tup[0:], tup[1:]):
                        link_seq=link_seq+1
                        df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['ParkNRide'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                        
                    # output walking part
                    link_seq=link_seq+1 # last link walk2
                    df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['ParkNRide'],'mode':['walk2'],'time':[walk2_time+df_paths_park.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                    
                    print('          ',transitType,' Park-N-Ride found')
                    numPaths += 1
                    if numPaths >= 2:
                        break
        except Exception as e:
            print(str(e))
            tup = []
            
    elapsed_time = time.time() - startTime
    print("Pkr finder used time:", elapsed_time)
    return df_paths, df_paths_park

def returnTransitPaths_toDF_GlobalLoop(origin_stops_list,ori_dists,destination_stops_list,des_dists,arrivalt,endt,df_points,transitType):
    global STOP_EVENTS
    global STOP_EVENTS_GRTA
    global GRAPH
    global GRAPH_GRTA

    startInfo = df_points.iloc[0]   
    hours,minutes,seconds = arrivalt.split('.')
    start_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0
    hours,minutes,seconds = endt.split('.')
    end_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0               
    walk_speed=2
    
    try:
        del df_paths
    except:
        print('################## starting transit ##################')
    
    startTime = time.time()
    print("getting transit. Current time"+str(startTime))
    
    # craete od pairs
    df_od_pairs=pd.DataFrame() 
    for i in range(len(origin_stops_list)):
        origin = origin_stops_list[i]
        walk1_dist=ori_dists[i][1]
        walk1_time_stamp=walk1_dist/walk_speed/1.0+start_time_hh
        for j in range(len(destination_stops_list)):
            destination = destination_stops_list[j]
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            try:
                if transitType=='marta':
                    events_start = STOP_EVENTS[str(origin)]
                    events_end  = STOP_EVENTS[str(destination)]
                    events_start_filtered=[x for x in events_start if x[1]=='ride']
                    events_end_filtered=[x for x in events_end if x[1]=='ride']
                else:
                    events_start = STOP_EVENTS_GRTA[str(origin)]
                    events_end  = STOP_EVENTS_GRTA[str(destination)]
                    events_start_filtered=events_start
                    events_end_filtered=events_end
                
                k=1
                for t2,ttype in events_end_filtered:
                    if t2<=end_time_hh-walk2_time:
                        events_start_new=[x[0] for x in events_start_filtered if x[0]<t2 and x[0]>=walk1_time_stamp]
                        for t1 in events_start_new:
                            '''
                            k=k+1
                            if k%200==0:
                                print('====',t1,t2,'====')
                            '''
                            start_node = "s"+str(origin)+"t"+str(t1)
                            end_node = "s"+str(destination)+"t"+str(t2)
                            df_od_pairsi=pd.DataFrame({'start_node':[start_node],'end_node':[end_node],'walk1_dist':[walk1_dist],'walk2_dist':[walk2_dist],'t1':[t1],'t2':[t2]})
                            df_od_pairs=df_od_pairs.append(df_od_pairsi,ignore_index=True)
            except:
                print('cannot build od dataframe')
    
    option_id=0
    
    try:
        #df_od_pairs=df_od_pairs.sort_values(['t2','t1'],ascending=[1,0])
        df_od_pairs['walk1_time_stamp']=df_od_pairs['t1']-df_od_pairs['walk1_dist']/walk_speed/1.0#ann 12292017
        df_od_pairs['walk2_time_stamp']=df_od_pairs['walk2_dist']/walk_speed/1.0+df_od_pairs['t2']#ann 12292017
        df_od_pairs['duration']=df_od_pairs['walk2_time_stamp']-df_od_pairs['walk1_time_stamp'] #ann 01012018
        df_od_pairs=df_od_pairs.sort_values(['duration','walk2_time_stamp','walk1_time_stamp'],ascending=[1,1,0])#ann 01012018

    except:
        pass
    df_paths=pd.DataFrame()
    print('################## Finding Transit Routes ##################')
    k=1
    numPaths = 0
    for ind,row in df_od_pairs.iterrows():
        k=k+1

        origin_stops=row['start_node']
        destination_stops=row['end_node']
        walk1_dist=row['walk1_dist']
        walk2_dist=row['walk2_dist']
        walk1_time_stamp=walk1_dist/walk_speed/1.0+start_time_hh
        walk2_time=walk2_dist/walk_speed/1.0
        
        #if k%200==0:
            #print('====',origin_stops,destination_stops,'====')
        if transitType == 'marta':
            transitGraph = GRAPH
        else: 
            transitGraph = GRAPH_GRTA
        tup=[]        
        try:
            if nx.has_path(transitGraph,origin_stops,destination_stops):
                tup = nx.dijkstra_path(transitGraph,origin_stops,destination_stops)
                if tup != []:
                    # make sure last node is not transfer
                    if transitGraph[tup[-2]][tup[-1]]['route']!='waiting':
                        option_id=option_id+1
                        link_seq=0
                        # output walking part
                        link_seq=link_seq+1 # first link walk1
                        df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':['origin'],'B':[origin_stops],'option':['TransitOnly'],'mode':['walk1'],'time':[walk1_time_stamp],'dist':[walk1_dist],'route':['walking1'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    
                        link_seq=link_seq+1
                        df_pathsi2=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[origin_stops],'B':[origin_stops],'option':['TransitOnly'],'mode':['wait1'],'time':[float(tup[0].split('t')[-1])],'dist':[0],'route':['waiting1'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths=df_paths.append(df_pathsi2,ignore_index=True) 
                    
                        # output transit part
                        for (u,v) in zip(tup[0:], tup[1:]):
                            link_seq=link_seq+1
                            df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['TransitOnly'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                            df_paths=df_paths.append(df_pathsi,ignore_index=True)
                        
                        # output walking part
                        link_seq=link_seq+1 # last link walk2
                        df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['TransitOnly'],'mode':['walk2'],'time':[walk2_time+df_paths.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    
                        print('          ',transitType,' routing found')
                        numPaths += 1
                        if numPaths >= 2:
                            break
        except Exception as e:
            print(str(e))
            tup = []
    elapsed_time = time.time() - startTime 
    print("Transit finder used time:", elapsed_time)
    return df_paths
'''
df_park_final_paths_grta=returnPNRPaths_toDF_GlobalLoop(destination_stops_list_grta,des_dists_grta,arrivalt,endt,pkrStopsOri_grta,df_points_ori_grta,'grta')

destination_stops_list=destination_stops_list_grta
des_dists=des_dists_grta
transitType='grta'
pkrStopsOri=pkrStopsOri_grta
pkrStops=pkrStops_grta                
'''
def returnPNRPaths_toDF_GlobalLoop(destination_stops_list,des_dists,arrivalt,endt,pkrStopsOri,df_points,transitType):
    global STOP_EVENTS
    global STOP_EVENTS_GRTA
    global GRAPH
    global GRAPH_GRTA

    stmTime = getCur15(arrivalt)
    startInfo = df_points.iloc[0]
    hours,minutes,seconds = arrivalt.split('.')
    start_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0
    hours,minutes,seconds = endt.split('.')
    end_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0   
                   
    try:
        del df_paths
    except:
        print('################## starting park and ride  ##################')
    
    startTime = time.time()
    # craete od pairs
    walk_speed=2
 
    df_od_pairs_park=pd.DataFrame()
    for j in range(len(destination_stops_list)):
        destination_stops = destination_stops_list[j]
        for park_stops_data in pkrStopsOri:
            
            arrivalt_park=park_stops_data[0]/60.0+start_time_hh
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            origin=park_stops_data[1]
            destination = destination_stops_list[j]
            
            try:
                if transitType=='marta':
                    events_start = STOP_EVENTS[str(origin)]
                    events_end  = STOP_EVENTS[str(destination)]
                    events_start_filtered=[x for x in events_start if x[1]=='ride']
                    events_end_filtered=[x for x in events_end if x[1]=='ride']
                else:
                    events_start = STOP_EVENTS_GRTA[str(origin)]
                    events_end  = STOP_EVENTS_GRTA[str(destination)]
                    events_start_filtered=events_start
                    events_end_filtered=events_end
                #print(origin,events_start_filtered)
                k=1
                for t2,ttype in events_end_filtered:
                    if t2<=end_time_hh-walk2_time:
                        events_start_new=[x[0] for x in events_start_filtered if x[0]<t2 and x[0]>=arrivalt_park]
                        for t1 in events_start_new:
                            k=k+1
                            if k%200==0:
                                print('====',t1,t2,'====')
                            start_node = "s"+str(origin)+"t"+str(t1)
                            end_node = "s"+str(destination)+"t"+str(t2)
                            df_od_pairs_parki=pd.DataFrame({'start_node':[start_node],'drive_path':[park_stops_data[2]],'end_node':[end_node],'walk2_dist':[walk2_dist],'t1':[t1],'t2':[t2]})
                            df_od_pairs_park=df_od_pairs_park.append(df_od_pairs_parki,ignore_index=True)
            except Exception as e:
                print(str(e))
                print('cannot build od dataframe')
            
    df_paths_park=pd.DataFrame()

    try:
        # this is different from transitonly
        #df_od_pairs_park=df_od_pairs_park.sort_values(['t2','t1'],ascending=[1,0])
        df_od_pairs_park['walk1_time_stamp']=df_od_pairs_park['t1']#ann 12292017
        df_od_pairs_park['walk2_time_stamp']=df_od_pairs_park['walk2_dist']/walk_speed/1.0+df_od_pairs_park['t2']#ann 12292017
        df_od_pairs_park=df_od_pairs_park.sort_values(['walk2_time_stamp','walk1_time_stamp'],ascending=[1,1])#ann 12292017

    except:
        pass
    option_id=0
    
    
    print('################## Finding Transit Routes ##################')
    for ind,row in df_od_pairs_park.iterrows():
        origin_stops=row['start_node']
        destination_stops=row['end_node']
        
        walk2_dist=row['walk2_dist']
        walk2_time=walk2_dist/walk_speed/1.0
        
        if transitType == 'marta':
            transitGraph = GRAPH
        else: 
            transitGraph = GRAPH_GRTA
        tup=[]        
        try:
            if nx.has_path(transitGraph,origin_stops,destination_stops):
                #print(origin_stops,destination_stops)
                tup = nx.dijkstra_path(transitGraph,origin_stops,destination_stops)
                if tup != []:
                    if transitGraph[tup[-2]][tup[-1]]['route']!='waiting':
                        option_id=option_id+1
                        link_seq=0
                    
                        # output driving part
                        lastNode=row['drive_path'][0]
                        #lastNode = park_stops_data[2][0]
                        foo = arrivalt.split('.')            
                        curTime = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
                        #for node in park_stops_data[2][1:]:
                        for node in row['drive_path'][1:]:
                            link_seq=link_seq+1
                            curNode = node
                            linkTime = DG.edge[lastNode][curNode].get(stmTime, 1)
                            # linkTime is in minutes
                            curTime += linkTime / 60.0
                            dist = DG.edge[lastNode][curNode].get('dist', 1)
                            df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[lastNode],'B':[curNode],'option':['ParkNRide'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':['driving'],'sequence':[link_seq],'option_id':[option_id]})
                            df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                            lastNode = curNode
                    
                        # output waiting 
                        link_seq=link_seq+1
                        df_paths_parki2=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[origin_stops],'B':[origin_stops],'option':['ParkNRide'],'mode':['wait1'],'time':[float(tup[0].split('t')[-1])],'dist':[0],'route':['waiting1'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths_park=df_paths_park.append(df_paths_parki2,ignore_index=True) 
                    
                        # output transit part
                        for (u,v) in zip(tup[0:], tup[1:]):
                            link_seq=link_seq+1
                            df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['ParkNRide'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                            df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                        
                        # output walking part
                        link_seq=link_seq+1 # last link walk2
                        df_paths_parki=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['ParkNRide'],'mode':['walk2'],'time':[walk2_time+df_paths_park.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                        df_paths_park=df_paths_park.append(df_paths_parki,ignore_index=True)
                    
                        print('          ',transitType,' Park-N-Ride found')
                        break
        except Exception as e:
            print(str(e))
            tup = []
            
    elapsed_time = time.time() - startTime
    print("Pkr finder used time:", elapsed_time)
    return df_paths_park

def returnPaths_toDF(origin_stops_list,ori_dists,destination_stops_list,des_dists,arrivalt,endt,pkrStopsOri,df_points,transitType):
    '''
    origin_stops_list=origin_stops_list_marta
    destination_stops_list=destination_stops_list_marta
    ori_dists=ori_dists_marta
    des_dists=des_dists_marta
    '''
    #outPaths = open('outPaths_'+transitType+'.csv', 'w')
    paths = []
    startInfo = df_points.iloc[0]
    
    hours,minutes,seconds = arrivalt.split('.')
    start_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0
    hours,minutes,seconds = endt.split('.')
    end_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0   
                   
    try:
        del df_paths
    except:
        print('################## starting transit ##################')
    
    startTime = time.time()
    print("getting transit. Current time"+str(startTime))
    
    option_id=0
    df_paths=pd.DataFrame()
    
    '''
    df_trips=pd.DataFrame()
    df_tripsi=pd.DataFrame({'origin':origin_stops,'destination':destination_stops,'transitType':transitType,'path_id':df_trips.shape[0]})
    df_trips=df_trips.append(df_tripsi,ignore_index=True)
    '''
    walk_speed=2
    for i in range(len(origin_stops_list)):
        origin_stops = origin_stops_list[i]
        walk1_dist=ori_dists[i][1]
        walk1_time_stamp=walk1_dist/walk_speed/1.0+start_time_hh
        
        for j in range(len(destination_stops_list)):
            destination_stops = destination_stops_list[j]
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            print(origin_stops,destination_stops)
            try:
                tup = getShortestTravelTime_OptiArr(walk1_time_stamp,end_time_hh-walk2_time,origin_stops,destination_stops,transitType)
            except Exception as e:
                print(str(e))
                tup = []
            
            if tup != [] and tup not in paths:
                paths.append(tup)
                
                option_id=option_id+1
                link_seq=0
                '''                
                # ouput driving part
                driveNodes = startInfo['routes'+pathInd+'_ori'].split('->')
                lastNode = driveNodes[0]
                
                for node in driveNodes[1:]:
                    link_seq=link_seq+1
                    curNode = node
                    ttime = DG.edge[lastNode][curNode].get('weight', 1)
                    dist = DG.edge[lastNode][curNode].get('dist', 1)
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[lastNode],'B':[curNode],'option':['TransitOnly'],'mode':[transitType],'time':[ttime],'dist':[dist],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    lastNode = curNode
                '''
                # output walking part
                link_seq=link_seq+1 # first link walk1
                df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':['origin'],'B':[origin_stops],'option':['TransitOnly'],'mode':['walk1'],'time':[walk1_time_stamp],'dist':[walk1_dist],'route':['walking1'],'sequence':[link_seq],'option_id':[option_id]})
                df_paths=df_paths.append(df_pathsi,ignore_index=True)
                # add waiting time at initial stop
                
                link_seq=link_seq+1
                df_pathsi2=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[origin_stops],'B':[origin_stops],'option':['TransitOnly'],'mode':['wait1'],'time':[float(tup[1][0].split('t')[-1])],'dist':[0],'route':['waiting1'],'sequence':[link_seq],'option_id':[option_id]})
                df_paths=df_paths.append(df_pathsi2,ignore_index=True) 
                # output transit part
                if transitType == 'marta':
                    transitGraph = GRAPH
                else:
                    transitGraph = GRAPH_GRTA
                    
                for (u,v) in zip(tup[1][0:], tup[1][1:]):
                    link_seq=link_seq+1
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['TransitOnly'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    '''
                    df_pathsi=pd.DataFrame({'A':[str(u)],'B':[str(v)],'option':['TransitOnly'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'sequence':[link_seq],'path_id':[path_id],'trip_id':[startInfo['Trip']]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    '''
                # output walking part
                link_seq=link_seq+1 # last link walk2
                df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['TransitOnly'],'mode':['walk2'],'time':[walk2_time+df_paths.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                df_paths=df_paths.append(df_pathsi,ignore_index=True)
    
    elapsed_time = time.time() - startTime 
    print("Transit finder used time:", elapsed_time)
    
    park_paths = []
    
   
    startTime = time.time()
    print("getting park and Ride. Current time"+str(startTime))
    
    print('################## starting park and ride ##################')
    for j in range(len(destination_stops_list)):
        destination_stops = destination_stops_list[j]
    #for destination_stops in destination_stops_list:
        for park_stops_data in pkrStopsOri:
            
            arrivalt_park=park_stops_data[0]/1.0+start_time_hh
            
            #time_to_arrive = str(int(park_stops_data[0]))
            #hours,minutes,seconds = arrivalt.split('.')
            #hours = int(hours)
            #minutes = int(minutes)+int(time_to_arrive)
            #while minutes > 60:
             #   minutes -= 60
              #  hours +=1
            #arrivalt_park = '.'.join([str(hours),str(minutes),str(seconds)])
            
            walk2_dist=des_dists[j][1]
            walk2_time=walk2_dist/walk_speed/1.0
            
            # TODO: print separate links in paths
            #outPaths.write(','+park_stops_data[2]+','+destination_stops+','+pkrStartPath+','+str(arrivalt_park)+'\n')
            #print('Time to arrive at pkr station', arrivalt_park)
            try:
                #print("Getting shortest path for park and ride data...")
                #print('entering pnr!')
                tup = getShortestTravelTime_OptiArr(arrivalt_park, end_time_hh-walk2_time,park_stops_data[2],destination_stops,transitType)
            except Exception as e:
                print(str(e))
                tup = []
                #outPaths.write(tripInd+',optionPkr,No Paths\n')
                
            if tup != [] and tup not in park_paths:
                park_paths.append(tup)
                option_id=option_id+1
                link_seq=0
                lastNode = park_stops_data[3][0]
                foo = arrivalt.split('.')            
                curTime = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600
                for node in park_stops_data[3][1:]:
                    link_seq=link_seq+1
                    curNode = node
                    linkTime = DG.edge[lastNode][curNode].get('weight', 1)
                    # linkTime is in minutes
                    curTime += linkTime / 60
                    dist = DG.edge[lastNode][curNode].get('dist', 1)
                    #df_pathsi=pd.DataFrame({'A':[lastNode],'B':[curNode],'option':['ParkNRide'],'mode':['drive'],'time':[curTime],'dist':[dist],'sequence':[link_seq],'option_id':[option_id],'trip_id':[startInfo['Trip']]})
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[lastNode],'B':[curNode],'option':['ParkNRide'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':['driving'],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    lastNode = curNode
                    
                if transitType == 'marta':
                    transitGraph = GRAPH
                else:
                    transitGraph = GRAPH_GRTA
                for (u,v) in zip(tup[1][0:], tup[1][1:]):
                    link_seq=link_seq+1
                    #df_pathsi=pd.DataFrame({'A':[str(u)],'B':[str(v)],'option':['ParkNRide'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'sequence':[link_seq],'path_id':[path_id],'trip_id':[startInfo['Trip']]})
                    df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[str(u)],'B':[str(v)],'option':['ParkNRide'],'mode':[transitType],'time':[transitGraph[u][v]['time_dest']],'dist':[transitGraph[u][v]['distance']],'route':[transitGraph[u][v]['route']],'sequence':[link_seq],'option_id':[option_id]})
                    df_paths=df_paths.append(df_pathsi,ignore_index=True)
                    
                link_seq=link_seq+1 # last link walk2
                df_pathsi=pd.DataFrame({'trip_id':[startInfo['Trip']],'A':[destination_stops],'B':['destination' ],'option':['ParkNRide'],'mode':['walk2'],'time':[walk2_time+df_paths.iloc[-1]['time']],'dist':[walk2_dist],'route':['walking2'],'sequence':[link_seq],'option_id':[option_id]})
                df_paths=df_paths.append(df_pathsi,ignore_index=True)
                
    elapsed_time = time.time() - startTime
    print("Pkr finder used time:", elapsed_time)
    
    return df_paths



def returnPaths(origin_stops_list,destination_stops_list,arrivalt,pkrStopsOri,df_points,transitType):
    outPaths = open(os.path.join(dirPath,'outPaths_'+transitType+'.csv'), 'w')
    print("in returnPaths")
    print(origin_stops_list, destination_stops_list)
    final_paths = []
    paths = []
    pathInd = '1'
    startInfo = df_points.iloc[0]
    tripInd = str(startInfo['Trip'])
    
    print(startInfo)
    print('------------------')
    #outPaths.write('TripId,'+str(startInfo['Trip'])+'\n')
    if len(origin_stops_list) == 0 or len(destination_stops_list) == 0:
        outPaths.write(tripInd+',optionTransit,No Transit Path\n')
    for i in range(len(origin_stops_list)):
        origin_stops = origin_stops_list[i]
        
        print("Trip ID:", startInfo['Trip'])
        print("Start Station:", startInfo['rail'+pathInd+'_ori']) 
        print("Path to start station,", startInfo['routes'+pathInd+'_ori']) 
        print("Distance to start station,", startInfo['dist'+pathInd+'_ori'])
        
        for j in range(len(destination_stops_list)):
            destination_stops = destination_stops_list[j]
            
            try:
                print("Getting shortest path for direct transit:")
                tup = getShortestTravelTime(arrivalt,origin_stops,destination_stops,transitType)
            except Exception as e:
                print(str(e))
                #outPaths.write(tripInd+',optionTransit,No Paths\n')
                tup = []
            if tup != [] and tup not in paths:
                paths.append(tup)
                tripOption = 'optionTransit'
                driveNodes = startInfo['routes'+pathInd+'_ori'].split('->')
                lastNode = driveNodes[0]
                curTime = arrivalt
                for node in driveNodes[1:]:
                    curNode = node
                    ttime = DG.edge[lastNode][curNode].get('weight', 1)
                    dist = DG.edge[lastNode][curNode].get('dist', 1)
                    print(curTime, ttime)
                    #curTime += ttime
                    outPaths.write(tripInd+',optionTransit,'+str(lastNode)+','+str(curNode)+','+'drive,'+str(ttime)+','+str(dist)+'\n')
                    lastNode = curNode
                outPaths = printDetailedPaths(tup[1],outPaths,startInfo['Trip'],tripOption,transitType)

            #else:
             #   outPaths.write(tripInd+',optionTransit,No Paths\n')
        pathInd = str(int(pathInd) + 1)

    # Only takes the shortest time path
    print("All possible paths:", paths)
    #shortest_time_path = sorted(paths, key=itemgetter(0))[-1]
    if len(paths) > 0:
        shortest_time_path = sorted(paths, key=itemgetter(0))[0]
        final_paths.append(shortest_time_path)

    park_paths = []

    for destination_stops in destination_stops_list:
        for park_stops_data in pkrStopsOri:
            pkrStartPath = '->'.join(park_stops_data[3])
            time_to_arrive = str(int(park_stops_data[0]))
            ''' cody
            print("Routes to pkr station:", park_stops_data[3])
            '''
            # arrivalt_park = formatTime('00.'+time_to_arrive+'.00') + formatTime(arrivalt)
            hours,minutes,seconds = arrivalt.split('.')
            hours = int(hours)
            minutes = int(minutes)+int(time_to_arrive)
            while minutes > 60:
                minutes -= 60
                hours +=1
            arrivalt_park = '.'.join([str(hours),str(minutes),str(seconds)])

            # TODO: print separate links in paths
            #outPaths.write(','+park_stops_data[2]+','+destination_stops+','+pkrStartPath+','+str(arrivalt_park)+'\n')
            print('Time to arrive at pkr station', arrivalt_park)
            try:
                print("Getting shortest path for park and ride data...")
                tup = getShortestTravelTime(arrivalt_park, park_stops_data[2],destination_stops,transitType)
            except Exception as e:
                print(str(e))
                tup = []
                #outPaths.write(tripInd+',optionPkr,No Paths\n')
            if tup != [] and tup not in park_paths:
                lastNode = park_stops_data[3][0]
                curTime = arrivalt
                for node in park_stops_data[3][1:]:
                    curNode = node
                    ttime = DG.edge[lastNode][curNode].get('weight', 1)
                    dist = DG.edge[lastNode][curNode].get('dist', 1)
                    #print(curTime, ttime)
                    #curTime += ttime
                    outPaths.write(tripInd+',optionPkr,'+str(lastNode)+','+str(curNode)+','+'drive,'+str(ttime)+','+str(dist)+'\n')
                    lastNode = curNode
                tripOption = 'optionPkr'
                outPaths = printDetailedPaths(tup[1],outPaths,startInfo['Trip'],tripOption,transitType)
                park_paths.append(tup)
            #else:
             #   outPaths.write(tripInd+',optionPkr,No Paths\n')

    if len(park_paths) > 0:
        #shortest_parked_path = sorted(park_paths, key=itemgetter(0))[-1]
        shortest_parked_path = sorted(park_paths, key=itemgetter(0))[0]
        final_paths.append(shortest_parked_path)
    
    outPaths.close()
    return final_paths
    
    
def add_xy_drive(df,lat,lon):
    crs = {'init': 'epsg:4326', 'no_defs': True}
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    df = gpd.GeoDataFrame(df, crs=crs,geometry=geometry)
    df=df.to_crs(epsg=2240) ### NAD83 / Georgia West (ftUS):  EPSG:2240
    return df 
    
def returnDrivePaths_toDF_GlobalLoop(df_points,df_links,DG,tripId,arrivalt,strategy='start',optionId=1):
    print("number of edges in graph is:", DG.number_of_edges())
    stmTime = getCur15(arrivalt)
    print("current stmTime is:", stmTime)
    hours,minutes,seconds = arrivalt.split('.')
    start_time_hh=int(hours)+int(minutes)/60.0+int(seconds)/3600.0
    curTime=start_time_hh
    
    df_paths_drive=pd.DataFrame()
    '''
    # changed network
    #df_points_node=point_to_node_drive(add_xy_drive(df_points,'lat_ori','lon_ori'),df_links,df_grids)
    ####################
    
    df_points_node=df_points_node.loc[:,['NodeID','Trip','lat_des','lon_des']]
    df_points_node=df_points_node.rename(columns={'NodeID':'NodeID_ori'})
    # changed network
    #df_points_node=point_to_node_drive(add_xy_drive(df_points_node,'lat_des','lon_des'),df_links,df_grids)
    ####################
    
    df_points_node=df_points_node.loc[:,['NodeID','Trip','NodeID_ori']]
    df_points_node=df_points_node.rename(columns={'NodeID':'NodeID_des'})
    '''

    df_points=add_xy(df_points,'ori_lat','ori_lon','x','y','x_sq','y_sq')    
    df_points=point_to_node(df_points,df_links).rename(columns={'NodeID':'o_node','Node_t':'o_t','x':'ox','y':'oy','x_sq':'ox_sq','y_sq':'oy_sq'})
    df_points=add_xy(df_points,'dest_lat','dest_lon','x','y','x_sq','y_sq')
    df_points=point_to_node(df_points,df_links).rename(columns={'NodeID':'d_node','Node_t':'d_t','x':'dx','y':'dy','x_sq':'dx_sq','y_sq':'dy_sq'})
    df_points_node = df_points

    print("df_points_node:", df_points_node)
    if df_points_node.loc[0,'d_node']==df_points_node.loc[0,'o_node']:
        print('no need to drive')

        df_points=add_xy_drive(df_points,'lat_ori','lon_ori')
        df_points['y']=df_points['geometry'].apply(lambda p: p.y)
        df_points['x']=df_points['geometry'].apply(lambda p: p.x)
        x1=df_points.loc[0,'x']
        y1=df_points.loc[0,'y']

        df_points=add_xy_drive(df_points,'lat_des','lon_des')
        df_points['y']=df_points['geometry'].apply(lambda p: p.y)
        df_points['x']=df_points['geometry'].apply(lambda p: p.x)
        x2=df_points.loc[0,'x']
        y2=df_points.loc[0,'y']

        manhattan_dist=(abs(x1-x2)+abs(y1-y2))/5280
        walk_speed=2.0
        walk_time=manhattan_dist/walk_speed
        #TODO: change walk to drive, if encounter bug in energy processing
        df_paths_drive=pd.DataFrame({'trip_id':[tripId],'A':['origin'],'B':[df_points_node.loc[0,'d_node']],'option':['drive-only'],'mode':['walk'],'time':[curTime+walk_time],'dist':[manhattan_dist],'route':[''],'sequence':[1],'option_id':[optionId]})
    else:
        arriveIn15 = False
        try:
            source = str(int(df_points_node.loc[0,'o_node']))
            pathDict = nx.single_source_dijkstra_path(DG, source, cutoff=15, weight=stmTime)
            #print(pathDict.keys())
            sink = str(int(df_points_node.loc[0,'d_node']))
            if sink in pathDict:
                print("can arrive dest within 15 minutes")
                arriveIn15 = True

            else:
                arriveIn15 = False
                ### CHANGED ###
                print("cannot arrive dest within 15 minutes, finding k shortest...")
                drive_paths = k_shortest_paths(DG,str(int(df_points_node.loc[0,'o_node'])), str(int(df_points_node.loc[0,'d_node'])), 30, stmTime)
                select_nodes = list(set.union(*map(set,drive_paths)))
                subGraph = nx.DiGraph(DG.subgraph(select_nodes))
                pathSubDict = nx.single_source_dijkstra_path(subGraph, source, cutoff=15, weight=stmTime)
                curCutoff = 15
                while not sink in pathSubDict:                   
                    unreached_nodes = list(set(select_nodes)-set(pathSubDict.keys()))
                    print("Updating edges, current curoff is: ", curCutoff, "; Number of unreached nodes:", len(unreached_nodes))
                    edges_toUpdate = subGraph.edges(unreached_nodes)
                    newStmTime = getNewStmTime(stmTime, curCutoff, strategy)
                    for edge in edges_toUpdate:
                        try:
                            old_weight = subGraph[edge[0]][edge[1]][stmTime]
                            
                            
                            #TODO: replace new weight with weight retrieved in real-time based on stm data
                            '''
                            link_row = df_link_grids[(df_link_grids['A']==int(edge[0])) & (df_link_grids['B']==int(edge[1]))]
                            new_weight = link_row.iloc[0]['ttime_'+str(curCutoff)]
                            '''
                            
                            new_weight = subGraph[edge[0]][edge[1]][newStmTime]
                            
                            if old_weight != new_weight:
                                print("Update changed weights from:", old_weight, " to ", new_weight, "newStmTime is:", newStmTime)
                                subGraph[edge[0]][edge[1]][stmTime] = new_weight
                        except Exception as e:
                            print(str(e))
                            pass
                    curCutoff += 15
                    pathSubDict = nx.single_source_dijkstra_path(subGraph, source, cutoff=curCutoff, weight=stmTime)
                    
            if arriveIn15:
                print("using original graph")
                graph2Use = DG
            else:
                print("using subgraph")
                graph2Use = subGraph
                
            drive_paths = k_shortest_paths(graph2Use,str(int(df_points_node.loc[0,'o_node'])), str(int(df_points_node.loc[0,'d_node'])), 30, stmTime)
            
            main_path = drive_paths[0]
            other_paths = [pa for pa in drive_paths[1:]]
            # For longer paths, might need to reduce difference threshold to retain more options
            pathInds = calcIsecPerc(main_path, other_paths)
            selected_paths = [main_path]+[other_paths[ind] for ind in pathInds[:2]]
            
                        
            for drive_nodes in selected_paths:
                node_a=drive_nodes[0]
                curTime=start_time_hh
                # 20180402
                if strategy == 'start':
                    link_seq=0
                else:
                    link_seq=len(drive_nodes)

                for node_b in drive_nodes[1:]:
                    linkTime = graph2Use.adj[node_a][node_b].get(stmTime, 1) # in minutes
                    dist = graph2Use.adj[node_a][node_b].get('dist', 1)

                    if strategy == 'start':
                        curTime += linkTime / 60
                        link_seq=link_seq+1
                    else:
                        curTime -= linkTime / 60
                        link_seq=link_seq-1
                    # 20180402
                    if strategy == 'start':
                        df_paths_drivei=pd.DataFrame({'trip_id':[tripId],'A':[node_a],'B':[node_b],'option':['drive-only'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':[''],'sequence':[link_seq],'option_id':[optionId]})
                    else:
                        df_paths_drivei=pd.DataFrame({'trip_id':[tripId],'A':[node_b],'B':[node_a],'option':['drive-only'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':[''],'sequence':[link_seq],'option_id':[optionId]})
                    # can change to below, add 'driving' to route
                    #df_paths_drivei=pd.DataFrame({'trip_id':[tripId],'A':[node_a],'B':[node_b],'option':['Drive'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':['driving'],'sequence':[link_seq],'option_id':[optionId]})
                    df_paths_drive=df_paths_drive.append(df_paths_drivei,ignore_index=True)

                    node_a=node_b
                    # in driving df, dist column is in miles. In marta it is km
                    # In post-processed df (by calctimeenergy function,
                    # dist_mile column is mile for all modes
                optionId += 1
       
            
        except Exception as e:
            print(str(e))
            print('no drive routes could be found')
           
        # return 20 shortest paths, get 3 different by 20%
        
        
        '''    
        drive_nodes=nx.dijkstra_path(DG,str(int(df_points_node.loc[0,'o_node'])), str(int(df_points_node.loc[0,'d_node'])), weight='weight')
        #drive_nodes=nx.dijkstra_path(DG,str(993520), str(993520),weight='weight')
        node_a=drive_nodes[0]
        
        link_seq=0
        for node_b in drive_nodes[1:]:
            linkTime = DG.edge[node_a][node_b].get('weight', 1) # in minutes
            dist = DG.edge[node_a][node_b].get('dist', 1)
            curTime += linkTime / 60
            link_seq=link_seq+1
            df_paths_drivei=pd.DataFrame({'trip_id':[tripId],'A':[node_a],'B':[node_b],'option':['Drive'],'mode':['drive'],'time':[curTime],'dist':[dist],'route':[''],'sequence':[link_seq],'option_id':[optionId]})
            df_paths_drive=df_paths_drive.append(df_paths_drivei,ignore_index=True)
            
            node_a=node_b
            
        '''
    df_paths_drive.to_csv('debug_files/test1.csv', index=False)
    df_paths_drive.iloc[::-1].to_csv('debug_files/test2.csv', index=False)
    if strategy == 'start':
        return df_paths_drive
    else:
        return df_paths_drive.iloc[::-1]

def makeDG(stmTime):
    DG=nx.DiGraph()
    # prepare network with congestion levels
    df_link_grids=pd.read_csv(os.path.join(dirPath,'data_node_link/stmFilled/Link_Grids_Nodes_ValidSpeed_stm_0920.csv'),header=0)

    speed_col=stmTime+'_speed'
    df_link_grids.loc[~df_link_grids[speed_col].isnull(),'ttime']=df_link_grids.loc[~df_link_grids[speed_col].isnull(),'DISTANCE']/df_link_grids.loc[~df_link_grids[speed_col].isnull(),speed_col]*60.0
    df_link_grids.loc[df_link_grids[speed_col].isnull(),'ttime']=df_link_grids.loc[df_link_grids[speed_col].isnull(),'DISTANCE']/df_link_grids.loc[df_link_grids[speed_col].isnull(),'SPEEDLIMIT']*60.0

    for ind, row2 in df_link_grids.iterrows():
        DG.add_weighted_edges_from([(str(row2['A']),str(row2['B']),float(row2['ttime']))],dist=row2['DISTANCE'])

    print('graph created for stm time:',stmTime)
    return DG

def output_pnr(row,df_ods,dict_settings,dict_driveRoutes,dict_transitRoutes,options):
    resultsPath=pd.DataFrame()
    runningLog=pd.DataFrame()
    t1=time.time()
    err_message_extra,err_message_extra2='',''
    commuteTime,walk_speed=dict_settings['commuteTime'],dict_settings['walk_speed']
    for option in list(set(['marta-pnr','grta-pnr']) & set(options)):
        df_odsi=df_ods[df_ods['option']==option]
        num_options=dict_settings['num_options'][option]
        
        if len(df_odsi)==0:
            numRoutes,err_message=errMessages(row,option,6)
            runningLog=runningLog.append(pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]}))
        else:
            transit_links, transfer_links=dict_settings['network'][option]['links'],dict_settings['network'][option]['t_links']
            num_AvaiOptions=len(df_odsi)
            num_options_found=0
            df_path=pd.DataFrame()

            # 20180401 #
            
            df_odsi['wait1'] = df_odsi['o_start']-commuteTime[0]-df_odsi['driving_duration']-df_odsi['o_duration']
            df_odsi['total_duration_1'] = df_odsi['total_duration'] - df_odsi['wait1']

            df_odsi=df_odsi.sort_values('arrival', ascending=0)
            #df_odsi.to_csv('debug_files/test1.csv', index=False)

            print(dict_settings['buffer'], len(df_odsi[df_odsi['arrival']>=commuteTime[1]-dict_settings['buffer']]))
            if len(df_odsi[df_odsi['arrival']>=commuteTime[1]-dict_settings['buffer']]) > 0:
                df_odsi=df_odsi[df_odsi['arrival']>=commuteTime[1]-dict_settings['buffer']]


            # need to add else condition

            df=df_odsi.sort_values(['total_duration_1', 'arrival'], ascending=[1,0])
            #df.to_csv('debug_files/test2.csv', index=False)
            ###########
            

            #df=df_odsi.sort_values(['total_duration','arrival'],ascending=[1,1])
            p=dict_transitRoutes[option]
            if len(p[df.iloc[0]['d']])==1:
                err_message= str(len(df))+' ['+option+'] routes found, 0 reported because driving makes more sense (drive to the transit parking lot, and directly walk)'
                df_path['trip_id']=row['trip_id']                        
                resultsPath=resultsPath.append(df_path,ignore_index=True)
                runningLog=runningLog.append(pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[num_AvaiOptions],'runTime':[time.time()-t1]}),ignore_index=True)
                print('Trip' , row['trip_id'], err_message)
            else:
                if len(transfer_links)>0:
                    err_message_extra=', transfer filtered'
                    for oid in np.arange(0,len(df),1):
                        
                        row_out=df.iloc[oid]
                        o,d=row_out['o'],row_out['d']
                        first_transitLink=transit_links[transit_links['stop_key1']==p[d][0]][transit_links['stop_key2']==p[d][1]][transit_links['type']!='transfer']
                        last_transitLink=transit_links[transit_links['stop_key1']==p[d][-2]][transit_links['stop_key2']==p[d][-1]][transit_links['type']!='transfer']
                        if (len(first_transitLink)==0) or (len(last_transitLink)==0): 
                            err_message_extra2=' , if walk farther than threshold, ['+option+'] would save more time'
                            num_AvaiOptions-=1
                        else:
                            num_options_found+=1
                            first_transitLink=first_transitLink.iloc[0]
                            
                            # walk1 link
                            df_path_drive,o_node,err_message=output_driving_routes(row_out['park_node'],dict_driveRoutes)
                            df_pathi=pd.DataFrame({'A':['origin'],'B':[o_node],'option':[option],'mode':['walk'],'timeStamp':[commuteTime[0]+row_out['o_duration']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})
                            df_path=df_path.append(df_pathi,ignore_index=True)
                            # drive links
                            if len(df_path_drive)==0:
                                print('Trip' , row['trip_id'], err_message)
                            else:
                                df_path_drive['option'],df_path_drive['option_id']=option,num_options_found
                                df_path=df_path.append(df_path_drive.sort_values('sequence',ascending=1),ignore_index=True)
                                
                            # 20180401
                            df_path['timeStamp']=df_path['timeStamp']+row_out['wait1']
                            # 20180401
                            # wait link
                            seq=df_path.iloc[-1]['sequence']+1
                            df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[first_transitLink['thrs1']],'time':[0],'dist':[0],'route':['wait1'],'sequence':[seq],'option_id':[num_options_found]})
                            df_path=df_path.append(df_pathi,ignore_index=True)
                            seq += 1

                            # ride links
                            for ind in np.arange(0,len(p[d])-1,1):
                                node_a=p[d][ind]
                                node_b=p[d][ind+1]
                                rows=transit_links[transit_links['stop_key1']==node_a][transit_links['stop_key2']==node_b]
                                if ind==0:
                                    _row=first_transitLink
                                elif len(rows['type'].unique().tolist())==2:
                                    '''
                                    if transfer & ride links can both be found from graph
                                        determine the link type based on "route_id of prev path link" ?="route_id of cur ride link"
                                    '''
                                    _row=rows[rows['type']=='ride'].iloc[0] if rows[rows['type']=='ride'].iloc[0]['route_id1']==df_pathi.iloc[-1]['route'] else rows[rows['type']=='transfer'].iloc[0]   
                                else:
                                    _row=rows.iloc[0]
                                
                                if _row['type']=='transfer':
                                    '''
                                    1 transfer link converted to 3 links {walk, wait, ride}
                                    '''
                                    _t=transfer_links[transfer_links['stop_key1']==node_a][transfer_links['stop_key3']==node_b].iloc[0,:]
                                    df_pathi=pd.DataFrame({'A':[node_a,_t['stop_key2'],_t['stop_key2']],'B':[_t['stop_key2'],_t['stop_key2'],_t['stop_key3']],
                                    'option':[option,option,option],'mode':['walk','wait','transit'],
                                    'timeStamp':[_t['thrs1'],_t['thrs2'],_t['thrs3']],'time':[_t['t_walk'],_t['t_wait'],_t['thrs3']-_t['thrs2']],
                                    'dist':[_t['man_dist'],0,_t['dist']],'route':['walk','wait',_t['route_id2']],'sequence':[seq,seq+1,seq+2],'option_id':[num_options_found,num_options_found,num_options_found]})
                                else:
                                    df_pathi=pd.DataFrame({'A':[node_a],'B':[node_b],'option':[option],'mode':['transit'],'timeStamp':[_row['thrs2']],'time':[_row['duration']],'dist':[_row['dist']],'route':[_row['route_id1']],'sequence':[seq],'option_id':[num_options_found]})
                                df_path=df_path.append(df_pathi,ignore_index=True)
                                seq+=1
                            # walk2 link
                            df_pathi=pd.DataFrame({'A':[d],'B':['destination'],'option':[option],'mode':['walk'],'timeStamp':[df_path.iloc[-1]['timeStamp']+row_out['d_duration']],'time':[row_out['d_duration']],'dist':[row_out['d_dist']],'route':['walk2'],'sequence':[seq],'option_id':[num_options_found]})        
                            df_path=df_path.append(df_pathi,ignore_index=True)
                            
                            if num_options_found==num_options:
                                break
                    if num_options_found==0:
                        err_message_extra=' , if walk farther than threshold, ['+option+'] could be possible'
                    err_message= str(len(df))+' ['+option+'] routes found, '+str(num_options_found)+' reported'+err_message_extra+err_message_extra2
                    df_path['trip_id']=row['trip_id']                        
                    resultsPath=resultsPath.append(df_path,ignore_index=True)
                    runningLog=runningLog.append(pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[num_AvaiOptions],'runTime':[time.time()-t1]}),ignore_index=True)
                    print('Trip' , row['trip_id'], err_message)
                else:# transfer links<=0
                    err_message_extra=', no transfer considered'
                    for oid in np.arange(0,len(df),1):
                        row_out=df.iloc[oid]
                        o,d=row_out['o'],row_out['d']
                        num_options_found+=1
                        for ind in np.arange(0,len(p[d])-1,1):
                            node_a=p[d][ind]
                            node_b=p[d][ind+1]
                            _row=transit_links[transit_links['stop_key1']==node_a][transit_links['stop_key2']==node_b].iloc[0]
                            if ind==0:
                                # walk1 link
                                df_path_drive,o_node,err_message=output_driving_routes(row_out['park_node'],dict_driveRoutes)
                                df_pathi=pd.DataFrame({'A':['origin'],'B':[o_node],'option':[option],'mode':['walk'],'timeStamp':[commuteTime[0]+row_out['o_duration']],'time':[row_out['o_duration']],'dist':[row_out['o_duration']*walk_speed],'route':['walk1'],'sequence':[1],'option_id':[num_options_found]})
                                df_path=df_path.append(df_pathi,ignore_index=True)
                                # drive links
                                if len(df_path_drive)==0:
                                    print('Trip' , row['trip_id'], err_message)
                                else:
                                    df_path_drive['option'],df_path_drive['option_id']=option,num_options_found
                                    df_path=df_path.append(df_path_drive.sort_values('sequence',ascending=1),ignore_index=True)
                                # wait link
                                seq=df_path.iloc[-1]['sequence']+1
                                df_pathi=pd.DataFrame({'A':[o],'B':[o],'option':[option],'mode':['wait'],'timeStamp':[_row['thrs1']],'time':[_row['thrs1']-df_path.iloc[-1]['timeStamp']],'dist':[0],'route':['wait1'],'sequence':[seq],'option_id':[num_options_found]})
                                df_path=df_path.append(df_pathi,ignore_index=True)
                                seq+=1
                                
                            # ride links
                            df_pathi=pd.DataFrame({'A':[node_a],'B':[node_b],'option':[option],'mode':['transit'],'timeStamp':[_row['thrs2']],'time':[_row['duration']],'dist':[_row['dist']],'route':[_row['route_id1']],'sequence':[seq],'option_id':[num_options_found]})
                            df_path=df_path.append(df_pathi,ignore_index=True)
                            seq+=1
                        # walk2 link
                        df_pathi=pd.DataFrame({'A':[d],'B':['destination'],'option':[option],'mode':['walk'],'timeStamp':[df_path.iloc[-1]['timeStamp']+row_out['d_duration']],'time':[row_out['d_duration']],'dist':[row_out['d_dist']],'route':['walk2'],'sequence':[seq],'option_id':[num_options_found]})        
                        df_path=df_path.append(df_pathi,ignore_index=True)
                        if num_options_found==num_options:
                            break
                    err_message= str(len(df))+' ['+option+'] routes found, '+str(num_options_found)+' reported'+err_message_extra
                    df_path['trip_id']=row['trip_id']
                    resultsPath=resultsPath.append(df_path,ignore_index=True)
                    runningLog=runningLog.append(pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[num_AvaiOptions],'runTime':[time.time()-t1]}),ignore_index=True)
                    print('Trip' , row['trip_id'], err_message)
    return resultsPath,runningLog

def output_driving_routes(di,dict_driveRoutes):
    # output results
    o,p,tlabel,DGo,timeStamp=dict_driveRoutes['o'],dict_driveRoutes['p'],dict_driveRoutes['cutoffs'],dict_driveRoutes['DGo'],dict_driveRoutes['drive_start']
    df_pathi=pd.DataFrame()
    if di in p.keys():
        if len(p[di])==1:
            err_message='no need to drive, can directly ride transit'
            return df_pathi, o, err_message
    seq=1
    if di in p.keys():
        for i in np.arange(0,len(p[di])-1,1):
            node_a=p[di][i]
            node_b=p[di][i+1]
            duration=DGo[node_a][node_b][tlabel]/60.0
            route=''#DGo[node_a][node_b]['name']
            seq+=1
            dist=DGo[node_a][node_b]['dist']
            dfi=pd.DataFrame({'A':[node_a],'B':[node_b],'mode':['drive'],'time':[duration],'dist':[dist],'route':[route],'sequence':[seq]})
            df_pathi=dfi.append(df_pathi)
    df_pathi=df_pathi.sort_values('sequence',ascending=1)
    df_pathi['timeStamp']=df_pathi['time'].cumsum()+timeStamp
    return df_pathi,o, '1 driving route was found'

def get_drive_routes(o,ot,df_ods,dict_settings,stmTime):
    global DG
    
    commuteTime,t_start=dict_settings['commuteTime'],dict_settings['commuteTime'][0]
    stop_nodes=df_ods['park_node'].unique().tolist()
    
    l,p={},{}
    
    try:
        l,p=nx.single_source_dijkstra(DG,o, weight=stmTime)
        
        df_toStops = pd.DataFrame()
        df_toStops['park_node'] = l.keys()
        df_toStops['driving_duration'] = l.values()
        df_toStops['driving_duration']= df_toStops['driving_duration']/60.0
        df_toStops=df_toStops[df_toStops['park_node'].isin(stop_nodes)].sort_values('driving_duration',ascending=True)
        
        
        os=df_ods.merge(df_toStops,on='park_node',how='left')
        os['o_duration']=ot
        os=os[os['o_start']>=commuteTime[0]+os['driving_duration']+os['o_duration']]
        
        if len(os) > 0:
            dict_driveRoutes={'p':p,'l':l,'o':o,'cutoffs':stmTime,'cutoffs_label':stmTime,'DGo':DG,'drive_start':ot+t_start}
            return os,dict_driveRoutes,"no error"
        else:
            return pd.DataFrame(), {}, ', cannot arrive to parking lots within required time'
    except Exception as e:
        print("in get_drive_routes:", str(e))
        return pd.DataFrame(), {}, ', cannot arrive to parking lots within required time'

def pnr_Transit(row,option,dict_settings):
    t1=time.time()
    transit_nodes=dict_settings['network'][option]['nodes']
    walk_thresh=dict_settings['walk_thresh'][option]
    walk_speed=dict_settings['walk_speed']
    commuteTime,cutoff_max=dict_settings['commuteTime'],dict_settings['cutoff_max']
    DG_transit=dict_settings['network'][option]['DG']
    dfstops=transit_nodes.drop_duplicates('stop_id',keep='first').loc[:,['stop_id','x','y']]
    ds_transit=getNearbyStops(row['dx'],row['dy'],dfstops,walk_thresh)
    # to remove
    #print("ds_transit:", ds_transit)
    df_ods_out=pd.DataFrame()
    p={}
    if len(ds_transit)==0:
        numRoutes,err_message=errMessages(row,option,1)
        runningLogi=pd.DataFrame({'trip_id':[row['trip_id']],'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]})
    else:
        ds=ds_transit.merge(transit_nodes[['thrs','stop_id','stop_key']],how='left',on='stop_id').rename(columns={'thrs':'d_end','stop_key':'d','man_dist':'d_dist'})
        ds['d_duration']=ds['d_dist']/walk_speed
        ds['arrival']=ds['d_duration']+ds['d_end']
        ds=ds[ds['arrival']<= commuteTime[1]]
        if len(ds)==0:
            numRoutes,err_message=errMessages(row,option,1)
            runningLogi=pd.DataFrame({'trip_id':[row['trip_id']],'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]})
        else:
            os=transit_nodes[transit_nodes['has_park']==True].astype(str)
            os=os[['stop_key','thrs','nodeid','stop_id']].rename(columns={'stop_id':'stop_ido','stop_key':'o','nodeid':'park_node','thrs':'o_start'})
            os['o_start']=os['o_start'].astype(float)        
            df_ods=pd.DataFrame(list(product(os['o'].tolist(),ds['d'].tolist())), columns=['o', 'd'])
            df_ods=df_ods.merge(os,on='o',how='left').merge(ds[['d_dist','d','arrival','d_duration','d_end','stop_id']],how='left',on='d')
            df_ods=df_ods[df_ods['stop_id']!=df_ods['stop_ido']]
            df_ods['total_duration']=df_ods['d_end']+df_ods['d_duration']-commuteTime[0]

            df_ods=df_ods[df_ods['d_end']>df_ods['o_start']] 
            
            try:
                l,p=nx.multi_source_dijkstra(DG_transit,df_ods['o'].unique().tolist(), cutoff=cutoff_max)
                df_transitDuration=pd.DataFrame()
                df_transitDuration['d'],df_transitDuration['transit_duration'] =l.keys(),l.values()
                df_transitDuration['transit_duration']=df_transitDuration['transit_duration']/60.0
                dfp=pd.DataFrame()
                dfp['d'],dfp['p']=p.keys(),p.values()
                df_transitDuration=df_transitDuration.merge(dfp,how='left',on='d')
                df_transitDuration['o']=df_transitDuration['p'].apply(lambda p: p[0])
                df_ods=df_ods[df_ods['d'].isin(p.keys())].merge(df_transitDuration,on=['o','d'],how='left')
                df_ods=df_ods[~df_ods['p'].isnull()]
                df_ods_out=df_ods.sort_values(['total_duration','arrival'],ascending=[1,1])# earliest arrival = minimum time spent
                numRoutes=len(df_ods_out)
                err_message=str(numRoutes)+' ['+option+'] transit pairs found between park & destinations, not look at driving part yet'
                runningLogi=pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]})
                print('Trip',row['trip_id'],err_message)
            except Exception as e:
                print("in pnr_transit, error:")
                print(str(e))
                numRoutes,err_message=errMessages(row,option,4)
                runningLogi=pd.DataFrame({'trip_id':[row['trip_id']],'option':[option],'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]})
    return df_ods_out,runningLogi,p

def Transit(row,option,walk_dist,dict_settings):
    t1=time.time()
    walk_thresh=dict_settings['walk_thresh'][option]
    walk_speed=dict_settings['walk_speed']
    commuteTime=dict_settings['commuteTime']
    
    transit_nodes=dict_settings['network'][option]['nodes']
    dfstops=transit_nodes.drop_duplicates('stop_id',keep='first').loc[:,['stop_id','x','y']]
    
    resultsPathi,runningLogi=pd.DataFrame(),pd.DataFrame()
    err_message,err_message_extra='',''
    
    ds_transit=getNearbyStops(row['dx'],row['dy'],dfstops,walk_thresh)
    
    if len(ds_transit)==0:
        numRoutes,err_message=errMessages(row,option,1)
    else:
        ds=ds_transit.merge(transit_nodes[['thrs','stop_id','stop_key']],how='left',on='stop_id').rename(columns={'thrs':'d_end','stop_key':'d','man_dist':'d_dist'})
        ds['d_duration']=ds['d_dist']/walk_speed
        ds['arrival']=ds['d_duration']+ds['d_end']
        
        ds=ds[ds['arrival']<= commuteTime[1]]
        if len(ds)==0:
            numRoutes,err_message=errMessages(row,option,1)
        else:
            os_transit=getNearbyStops(row['ox'],row['oy'],dfstops,walk_thresh)
            if len(os_transit)==0:
                numRoutes,err_message=errMessages(row,option,2)
            else:   
                os=os_transit.merge(transit_nodes[['thrs','stop_id','stop_key']],how='left',on='stop_id').rename(columns={'thrs':'o_start','stop_key':'o','man_dist':'o_dist'})
                os['o_duration']=os['o_dist']/walk_speed
                os=os[os['o_duration']+commuteTime[0]<=os['o_start']]
                if len(os)==0:
                    numRoutes,err_message=errMessages(row,option,2)
                else:
                    jointStops=list(set(os['stop_id']) & set(ds_transit['stop_id']))
                    if len(jointStops)>0:
                        err_message_extra='. Note: Walk distance is '+str(round(walk_dist,0))+' mile, '+str(jointStops[0])+' is within walking distance from origin and destinations'
                    
                    df_ods=pd.DataFrame(list(product(os['o'].tolist(),ds['d'].tolist())), columns=['o', 'd'])
                    df_ods=df_ods.merge(os[['o_dist','o','o_duration','o_start']],on='o',how='left').merge(ds[['d_dist','arrival','d','d_duration','d_end']],how='left',on='d')
                    df_ods=df_ods[df_ods['d_end']>df_ods['o_start']][['o','arrival','o_dist','o_duration','o_start','d','d_end','d_dist','d_duration']]
                    if len(df_ods)==0:
                        numRoutes,err_message=errMessages(row,option,3)
                    else:
                        df_ods['walk_duration']=df_ods['o_duration']+df_ods['d_duration']
                        df_ods['transit_duration']=df_ods['d_end']-df_ods['o_start']
                        df_ods['total_duration']=df_ods['walk_duration']+df_ods['transit_duration']
                        resultsPathi,err_message,numRoutes=transit_finder(df_ods, option,dict_settings)
                        resultsPathi['trip_id']=row['trip_id']
                        err_message+=err_message_extra
                        print('Trip' , row['trip_id'], err_message)
    runningLogi=pd.DataFrame({'trip_id':[row['trip_id']],'option':[option], 'state':[err_message],'numRoutes':[numRoutes],'runTime':[time.time()-t1]})
    return resultsPathi,runningLogi


def calcDfEg(df, userId, traceFileDict=None, mtTraceDict=None, gtTraceDict=None):
    print("in calcDfEg:", traceFileDict, mtTraceDict, gtTraceDict)
    ### calculate energy per link per traveler
    # assumptions for ridership
    marta_rail_occupancy=.8
    marta_bus_ridership=10.0
    grta_ridership=40.0
    auto_ridership=1.0
    walkwait_energy=0.0
    # assign scaled energy values
    #railLabels = ['RED', 'GOLD', 'BLUE', 'GREEN']
    railLabels = [10909, 10911, 10912, 10913]
    jsonResults = defaultdict(dict)
    for ind, row in df.iterrows():
        eg=row['energy']
        if row['mode']=='drive':
            df.set_value(ind,'energy_scaled',eg/auto_ridership)
            df.set_value(ind,'cost_drive',0.54*row['dist_mile'])
        elif row['option'] in ['grta-only', 'grta-pnr']:
            if row['mode'] == 'transit':
                df.set_value(ind,'energy_scaled',eg/grta_ridership)
                df.set_value(ind,'cost_drive',0) 
            elif row['mode'] in ['walk','wait']:
                df.set_value(ind,'energy_scaled',walkwait_energy)
        elif row['option'] in ['marta-only', 'marta-pnr']:
            df.set_value(ind,'cost_drive',0)
            if row['route'] in railLabels:
                df.set_value(ind,'energy_scaled',eg/marta_rail_occupancy)
            elif row['mode'] in ['walk','wait']:
                df.set_value(ind,'energy_scaled',walkwait_energy)
            else:
                df.set_value(ind,'energy_scaled',eg/marta_bus_ridership)
        else:
            print("shouldn't be in calcDfEg here.")
            df.set_value(ind,'cost_drive',0)
            df.set_value(ind,'energy_scaled',walkwait_energy)
    options = ['grta-only', 'grta-pnr', 'marta-only', 'marta-pnr', 'drive-only']

    # connect to database
    cursorDev = dbConnect("commuteWarrior")

    for option in options:
        #TODO: change drive df TripOption to option
        modeDf = df[df['option']==option]
        #print('----------------'+option+'-----------------')
        for alter in modeDf['option_id'].unique():
            if option in ['grta-only', 'grta-pnr']:
                jsonResults[option+str(int(alter))]['cost'] = 2.88
            elif option in ['marta-only', 'marta-pnr']:
                jsonResults[option+str(int(alter))]['cost'] = 2.19
            else:
                jsonResults[option+str(int(alter))]['cost'] = 0
            optionDf = modeDf[modeDf['option_id'] == alter]
            totalE = round(optionDf['energy_scaled'].sum(),2)
            totalDist = round(optionDf['dist_mile'].sum(),2)
            driveCost = round(optionDf['cost_drive'].sum(),2)
            # when optionDf['route']='waiting1', can remove the duration for that line, assuming later departure
            duration = round(optionDf['duration'].sum()*60.0,2)

            nonWaitDf = optionDf[optionDf['route']!='wait1']
            nonWaitDuration = round(nonWaitDf['duration'].sum()*60.0,2)
            
            # kwH to gallon
            totalE = totalE / 36.6
            jsonResults[option+str(int(alter))]['energy'] = totalE
            jsonResults[option+str(int(alter))]['cost'] += driveCost
            jsonResults[option+str(int(alter))]['distance'] = totalDist
            jsonResults[option+str(int(alter))]['duration'] = duration

            if mtTraceDict != None:
                if 'mtOnly' in mtTraceDict and option=='marta-only':
                    traceId = insertTrace(mtTraceDict['mtOnly'], totalE, totalDist, driveCost, duration, userId)
                    jsonResults['marta-only'+str(int(alter))]['traceFile'] = traceId
                if 'mtPnr' in mtTraceDict and option=='marta-pnr':
                    traceId = insertTrace(mtTraceDict['mtPnr'], totalE, totalDist, driveCost, duration, userId)
                    jsonResults['marta-pnr'+str(int(alter))]['traceFile'] = traceId
            elif gtTraceDict != None:
                if 'gtOnly' in gtTraceDict and option=='grta-only':
                    traceId = insertTrace(gtTraceDict['gtOnly'], totalE, totalDist, driveCost, duration, userId)
                    jsonResults['grta-only'+str(int(alter))]['traceFile'] = traceId
                if 'gtPnr' in gtTraceDict and option=='grta-pnr':
                    traceId = insertTrace(gtTraceDict['gtPnr'], totalE, totalDist, driveCost, duration, userId)
                    jsonResults['grta-pnr'+str(int(alter))]['traceFile'] = traceId

            elif traceFileDict != None and int(alter) in traceFileDict:
                traceFile = traceFileDict[int(alter)]
                fName = os.path.basename(traceFile)
                tripDF = pd.read_csv(traceFile, header=None)
                lastRow = tripDF.iloc[-1]
                endDate = lastRow[0]
                endTime = lastRow[1]
                endLat = lastRow[2]
                endLon = lastRow[3]
                
                firstRow = tripDF.iloc[0]
                startDate = firstRow[0]
                startTime = firstRow[1]
                startLat = firstRow[2]
                startLon = firstRow[3]

                if traceFile != None:
                    dataComp = 0
                    tripPart = 1
                    lastTripIdDev = ''
                    lastQueryDev = "SELECT tripId FROM trip where deviceID='%s' order by tripId DESC limit 1" % userId
                    statusDev = cursorDev.execute(lastQueryDev)
                    dayWeek = datetime.today().weekday()
                    if statusDev == 1:
                        resultDev = cursorDev.fetchone()
                        lastTripIdDev = resultDev[0]
                        print("last trip ID Dev:", lastTripIdDev)
                        queryDev = "INSERT INTO trip(tripFileName, tripPart, tripStartDate, tripStartTime, tripEndDate, tripEndTime, tripDistance, tripDuration,  tripOriginLongitude, tripOriginLatitude, tripDestinationLongitude, tripDestinationLatitude, deviceId, tripDayWeek, tripShowStatus, lastTripId, dataComplete, tripCost, tripEnergy) values('%s', '1', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '0', '%s', '%s', '%s', '%s')" % (fName, startDate, startTime, endDate, endTime, totalDist, duration, startLon, startLat, endLon, endLat, userId, dayWeek, lastTripIdDev, dataComp, driveCost, totalE)
                        print(queryDev)
                        cursorDev.execute(queryDev)
                        traceId = cursorDev.lastrowid
                        jsonResults[option+str(int(alter))]['traceFile'] = traceId
            print(option+str(int(alter)), "total Energy:", totalE, "total Distance:", totalDist, "duration with wait:", duration, "duration w/o wait:", nonWaitDuration, "driveCost:", driveCost, "totalCost:", jsonResults[option+str(int(alter))]['cost'])
    return jsonify(jsonResults)

def insertTrace(traceFName, totalE, totalDist, driveCost, duration, userId):
    # connect to database
    cursorDev = dbConnect("commuteWarrior")

    fName = os.path.basename(traceFName)
    tripDF = pd.read_csv(traceFName, header=None)
    lastRow = tripDF.iloc[-1]
    endDate = lastRow[0]
    endTime = lastRow[1]
    endLat = lastRow[2]
    endLon = lastRow[3]
    
    firstRow = tripDF.iloc[0]
    startDate = firstRow[0]
    startTime = firstRow[1]
    startLat = firstRow[2]
    startLon = firstRow[3]

    if traceFName != None:
        dataComp = 0
        tripPart = 1
        lastTripIdDev = ''
        lastQueryDev = "SELECT tripId FROM trip where deviceID='%s' order by tripId DESC limit 1" % userId
        statusDev = cursorDev.execute(lastQueryDev)
        dayWeek = datetime.today().weekday()
        if statusDev == 1:
            resultDev = cursorDev.fetchone()
            lastTripIdDev = resultDev[0]
            print("last trip ID Dev:", lastTripIdDev)
            queryDev = "INSERT INTO trip(tripFileName, tripPart, tripStartDate, tripStartTime, tripEndDate, tripEndTime, tripDistance, tripDuration,  tripOriginLongitude, tripOriginLatitude, tripDestinationLongitude, tripDestinationLatitude, deviceId, tripDayWeek, tripShowStatus, lastTripId, dataComplete, tripCost, tripEnergy) values('%s', '1', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '0', '%s', '%s', '%s', '%s')" % (fName, startDate, startTime, endDate, endTime, totalDist, duration, startLon, startLat, endLon, endLat, userId, dayWeek, lastTripIdDev, dataComp, driveCost, totalE)
            print(queryDev)
            cursorDev.execute(queryDev)
            traceId = cursorDev.lastrowid
    return traceId

def buildTransitNetwork(flink,fnode,ftlink):
    df_links=pd.read_csv(flink)
    df_links['thrs1'],df_links['thrs2']=df_links['thrs1'].astype(float),df_links['thrs2'].astype(float)
    df_nodes=pd.read_csv(fnode)
    df_nodes['thrs'],df_nodes['nodeid']=df_nodes['thrs'].astype(float),df_nodes['nodeid'].astype(str)
    if ftlink!='':
        df_links_transfer=pd.read_csv(ftlink)
    else:
        df_links_transfer=pd.DataFrame()
    DG=nx.DiGraph()
    # weight is in minutes
    for ind, row in df_links.iterrows():
        DG.add_weighted_edges_from([(row['stop_key1'],row['stop_key2'],row['duration']*60.0)], ttype=row['type'])
    return DG,df_nodes,df_links,df_links_transfer


def getNewStmTime(stmTime, curCutoff, strategy='start'):
    stmObj = datetime.strptime(stmTime,'%H%M%S')
    #20180402
    if strategy == 'start':
        stmObjNew = stmObj + timedelta(minutes=curCutoff)
    else:
        stmObjNew = stmObj - timedelta(minutes=curCutoff)
    dt100 = datetime.strptime('100000','%H%M%S')
    dt630 = datetime.strptime('063000','%H%M%S')
    if stmObjNew > dt100 or stmObjNew < dt630:
        ############
        stmTimeNew = '100000'
    else:
        stmTimeNew = datetime.strftime(stmObjNew, '%H%M%S')
    return stmTimeNew

def getCur15(startt):
    [hr, min, sec] = startt.split('.')
    minute = round(float(min) / 15) * 15

    if int(hr) < 10:
        hr = '0'+str(hr)
    if minute == 0:
        stmTime = str(hr) + '0000'
    elif minute == 60:
        hr = int(hr) + 1
        if hr < 10:
            hr = '0'+str(hr)
        stmTime = str(hr) + '0000'
    else:
        stmTime = str(hr) + str(int(minute)) + '00'
    return stmTime


def round15(startt):
    [hr, min, sec] = startt.split('.')
    minute = round(float(min) / 15) * 15

    if int(hr) < 10:
        hr = '0'+str(hr)
    if minute == 0:
        stmTime = str(hr) + '0000'
    elif minute == 60:
        hr = int(hr) + 1
        if hr < 10:
            hr = '0'+str(hr)
        stmTime = str(hr) + '0000'
    else:
        stmTime = str(hr) + str(int(minute)) + '00'

    dtobj = datetime.strptime(stmTime, '%H%M%S')
    dt630 = datetime.strptime('063000','%H%M%S')
    dt100 = datetime.strptime('100000','%H%M%S')

    dtobjm_15 = dtobj - timedelta(minutes=15)
    dtobjm_30 = dtobj - timedelta(minutes=30)
    dtobjm15 = dtobj + timedelta(minutes=15)
    dtobjm30 = dtobj + timedelta(minutes=30)
    dtobjm45 = dtobj + timedelta(minutes=45)
    dtobjm60 = dtobj + timedelta(minutes=60)

    dtobjm75 = dtobj + timedelta(minutes=75)
    dtobjm90 = dtobj + timedelta(minutes=90)
    dtobjm105 = dtobj + timedelta(minutes=105)
    dtobjm120 = dtobj + timedelta(minutes=120)

    arrivaltobj_15 = datetime.strptime(startt, '%H.%M.%S') - timedelta(minutes=15)
    arrivaltobj_30 = datetime.strptime(startt, '%H.%M.%S') - timedelta(minutes=30)
    arrivaltobj15 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=15)
    arrivaltobj30 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=30)
    arrivaltobj45 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=45)
    arrivaltobj60 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=60)
    arrivaltobj75 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=75)
    arrivaltobj90 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=90)
    arrivaltobj105 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=105)
    arrivaltobj120 = datetime.strptime(startt, '%H.%M.%S') + timedelta(minutes=120)

    arrivalt_15 = datetime.strftime(arrivaltobj_15, '%H.%M.%S')
    arrivalt_30 = datetime.strftime(arrivaltobj_30, '%H.%M.%S')
    arrivalt15 = datetime.strftime(arrivaltobj15, '%H.%M.%S')
    arrivalt30 = datetime.strftime(arrivaltobj30, '%H.%M.%S')
    arrivalt45 = datetime.strftime(arrivaltobj45, '%H.%M.%S')
    arrivalt60 = datetime.strftime(arrivaltobj60, '%H.%M.%S')
    arrivalt75 = datetime.strftime(arrivaltobj75, '%H.%M.%S')
    arrivalt90 = datetime.strftime(arrivaltobj90, '%H.%M.%S')
    arrivalt105 = datetime.strftime(arrivaltobj105, '%H.%M.%S')
    arrivalt120 = datetime.strftime(arrivaltobj120, '%H.%M.%S')

    #dtobjects = [dtobj, dtobjm_15, dtobjm_30, dtobjm15, dtobjm30, dtobjm45, dtobjm60]
    #for dtobject in dtobjects:

    if dtobj < dt630:
        stmTime = '063000'
    elif dtobj > dt100:
        stmTime = '100000'
    else:
        pass
    if dtobjm_15 < dt630:
        stmTime_15 = '063000'
    elif dtobjm_15 > dt100:
        stmTime_15 = '100000'
    else:
        stmTime_15 = datetime.strftime(dtobjm_15, '%H%M%S')
    if dtobjm_30 < dt630:
        stmTime_30 = '063000'
    elif dtobjm_30 > dt100:
        stmTime_30 = '100000'
    else:
        stmTime_30 = datetime.strftime(dtobjm_30, '%H%M%S')

    if dtobjm15 > dt100:
        stmTime15 = '063000'
    elif dtobjm15 < dt630:
        stmTime15 = '100000'
    else:
        stmTime15 = datetime.strftime(dtobjm15, '%H%M%S')
    if dtobjm30 > dt100:
        stmTime30 = '063000'
    elif dtobjm30 < dt630:
        stmTime30 = '100000'
    else:
        stmTime30 = datetime.strftime(dtobjm30, '%H%M%S')

    if dtobjm45 > dt100:
        stmTime45 = '063000'
    elif dtobjm45 < dt630:
        stmTime45 = '100000'
    else:
        stmTime45 = datetime.strftime(dtobjm45, '%H%M%S')

    if dtobjm60 > dt100:
        stmTime60 = '063000'
    elif dtobjm60 < dt630:
        stmTime60 = '100000'
    else:
        stmTime60 = datetime.strftime(dtobjm60, '%H%M%S')

    if dtobjm75 > dt100:
        stmTime75 = '063000'
    elif dtobjm75 < dt630:
        stmTime75 = '100000'
    else:
        stmTime75 = datetime.strftime(dtobjm75, '%H%M%S')

    if dtobjm90 > dt100:
        stmTime90 = '063000'
    elif dtobjm90 < dt630:
        stmTime90 = '100000'
    else:
        stmTime90 = datetime.strftime(dtobjm90, '%H%M%S')

    if dtobjm105 > dt100:
        stmTime105 = '063000'
    elif dtobjm105 < dt630:
        stmTime105 = '100000'
    else:
        stmTime105 = datetime.strftime(dtobjm105, '%H%M%S')

    if dtobjm120 > dt100:
        stmTime120 = '063000'
    elif dtobjm120 < dt630:
        stmTime120 = '100000'
    else:
        stmTime120 = datetime.strftime(dtobjm120, '%H%M%S')
    return stmTime, stmTime15, stmTime30, stmTime45, stmTime60, stmTime75, stmTime90,stmTime105, stmTime120, stmTime_15, stmTime_30, arrivalt15, arrivalt30, arrivalt45, arrivalt60, arrivalt75, arrivalt90, arrivalt105, arrivalt120, arrivalt_15, arrivalt_30


def fillLoc(tripFile, origin_lat_lon, destination_lat_lon, option='marta'):
    global marta_loc_dict
    global grta_loc_dict

    
    for ind, row in tripFile.iterrows():
        if row['mode'] in ['walk', 'wait']:
            stopKeyA = row['A']
            stopKeyB = row['B']
            if option == 'marta':
                if stopKeyA in marta_loc_dict:
                    latA, lonA = marta_loc_dict[stopKeyA]
                    aLoc = str(latA)+'_'+str(lonA)
                    tripFile.set_value(ind, 'A_loc', aLoc)
                if stopKeyB in marta_loc_dict:
                    latB, lonB = marta_loc_dict[stopKeyB]
                    bLoc = str(latB)+'_'+str(lonB)
                    tripFile.set_value(ind, 'B_loc', bLoc)

            else:
                if stopKeyA in grta_loc_dict:
                    latA, lonA = grta_loc_dict[stopKeyA]
                    aLoc = str(latA)+'_'+str(lonA)
                    tripFile.set_value(ind, 'A_loc', aLoc)
                if stopKeyB in grta_loc_dict:
                    latB, lonB = grta_loc_dict[stopKeyB]
                    bLoc = str(latB)+'_'+str(lonB)
                    tripFile.set_value(ind, 'B_loc', bLoc)
    ############################
    #20180403: transit colname is 'timeStamp'
    startT = tripFile['timeStamp'].iloc[0]
    startTObj = dt.datetime.combine(dt.date.today(), dt.time()) + dt.timedelta(hours=startT)
    startTStr = startTObj.strftime('%H%M%S')
    origin_hhmmss = origin_lat_lon+'_'+startTStr
    '''
    startH, startM, startS = arrivalt.split('.')
    if len(startH) == 1:
        startH = '0'+startH
    origin_hhmmss = origin_lat_lon+'_'+startH+startM+startS
    '''
    tripFile['A_loc'].iloc[0] = origin_hhmmss

    endT = tripFile['timeStamp'].iloc[-1]
    endTObj = dt.datetime.combine(dt.date.today(), dt.time()) + dt.timedelta(hours=endT)
    endTStr = endTObj.strftime('%H%M%S')
    dest_hhmmss = destination_lat_lon+'_'+endTStr
    tripFile['B_loc'].iloc[-1] = dest_hhmmss
    print('A_loc:', origin_hhmmss, 'B_loc:', dest_hhmmss)
    ################################3


    return tripFile

    
#================BEGINS MAIN==================#
def main(__name__):
    args_startt = '8.03.00'
    args_endt = '10.00.00'
    stmTime, stmTime15, stmTime30, stmTime45, stmTime60, stmTime75, stmTime90,stmTime105, stmTime120, stmTime_15, stmTime_30, arrivalt15, arrivalt30, arrivalt45, arrivalt60, arrivalt75, arrivalt90, arrivalt105, arrivalt120, arrivalt_15, arrivalt_30=round15(args_startt)

    stmTimeOptions = [stmTime15, stmTime30, stmTime45,stmTime60,stmTime75, stmTime90,stmTime105, stmTime120, stmTime_15, stmTime_30]
    arrivaltOptions = [arrivalt15, arrivalt30, arrivalt45,arrivalt60, arrivalt75, arrivalt90, arrivalt105, arrivalt120, arrivalt_15, arrivalt_30]
    print(stmTime, stmTime15, stmTime30, stmTime45, stmTime60, stmTime75, stmTime90,stmTime105, stmTime120, stmTime_15, stmTime_30, arrivalt15, arrivalt30, arrivalt45,arrivalt60,  arrivalt75, arrivalt90, arrivalt105, arrivalt120,arrivalt_15, arrivalt_30)

    beginTime = time.time()
    
    # Building dictionaries for filling in walk wait locations
    global grta_loc_dict
    global marta_loc_dict

    grta_loc_dict = {}
    marta_loc_dict = {}
    locDirPath = '/home/users/ywang936/transit_simulation-0206/data_node_link/graph_file'
    mtLocDf = pd.read_csv(os.path.join(locDirPath, 'nodes_marta.csv'))
    gtLocDf = pd.read_csv(os.path.join(locDirPath, 'nodes_grta.csv'))

    for ind,row in mtLocDf.iterrows():
        stopKey = row['stop_key']
        lat, lon = row['stop_lat'], row['stop_lon']
        marta_loc_dict[stopKey] = (lat, lon)
    for ind,row in gtLocDf.iterrows():
        stopKey = row['stop_key']
        lat, lon = row['stop_lat'], row['stop_lon']
        grta_loc_dict[stopKey] = (lat, lon)
    ########################################################



    print("building marta network. Current time"+str(beginTime)) 
    # ==============================MARTA====================================== #
    dir_graph='data_node_link/graph_file/'

    global dict_marta
    dict_marta={}
    dict_marta['DG'],dict_marta['nodes'],dict_marta['links'],dict_marta['t_links']=buildTransitNetwork(dir_graph+'links_marta.csv',dir_graph+'nodes_marta.csv',dir_graph+'transfer_pair_marta.csv')

    global dict_marta_rail
    dict_marta_rail={}
    dict_marta_rail['DG'],dict_marta_rail['nodes'],dict_marta_rail['links'],dict_marta_rail['t_links']=buildTransitNetwork(dir_graph+'links_marta-rail.csv',dir_graph+'nodes_marta-rail.csv',dir_graph+'transfer_pair_marta-rail.csv')

    
    print("building grta network.") 
    # ==============================GRTA======================================= #
    dir_graph='data_node_link/graph_file/'

    global dict_grta
    dict_grta={}
    dict_grta['DG'],dict_grta['nodes'],dict_grta['links'],dict_grta['t_links']=buildTransitNetwork(dir_graph+'links_grta.csv',dir_graph+'nodes_grta.csv','')

    global railEnergyDict
    railEnergyDict = makeEgDict(os.path.join(dirPath, 'energy_rate/energy_per_passenger_with_occ_100.csv'))
    
    '''
    global STOP_EVENTS_GRTA
    STOP_EVENTS_GRTA = json.load(open(os.path.join(dirPath, 'events_at_stop_grta.json')))

    global GRAPH_GRTA
    GRAPH_GRTA = nx.read_edgelist(os.path.join(dirPath,'links_grta.txt'), nodetype=str, delimiter=",", \
                                  data=(('weight',float),('type',str),('lat_or',float),('lat_dest',float),('lon_or',float),
                                        ('lon_dest',float),('time_or',float),('time_dest',float),('distance',float),('route',str),),create_using=nx.DiGraph() )
    '''
    
    print("building roadway network.") 
    # ==============================ABM15======================================= #
    
    global df_links
    #df_links = df_links_drive
    #changed network 2/28/2018
    df_links = gpd.read_file(os.path.join(dirPath,'./data_node_link/abm15_links.shp')) # link file from ARC (note that this is pre-processed by adding the Grid ID to each link)
    ##########################

    global df_drive_trace
    df_drive_trace = gpd.read_file(os.path.join(dirPath,'./data_node_link/abm15_links.shp')) # link file from ARC (note that this is pre-processed by adding the Grid ID to each link)

    global df_links_drive
    # changed network 3/01/2018
    df_links_drive = df_links
    #df_links_drive = gpd.read_file(os.path.join(dirPath,'./data_node_link/Link_Grids_Nodes.shp')) # link file from ARC (note that this is pre-processed by adding the Grid ID to each link)
    #df_links_drive = df_links_drive[(df_links_drive['FACTYPE']>0) & (df_links_drive[df_links_drive['FACTYPE']<50])]
    #print(df_links_drive.columns)
    #fcids=df_links_drive[['FACTYPE','A_B']].drop(['FACTYPE','A_B'],keep='first')


    #print(df_links.columns.values.tolist())
    #df_links = df_links[(df_links['FACTYPE']>0) & (df_links[df_links['FACTYPE']<50])]

    
    global df_links_trace
    ##changed network 3/01/2018
    #df_links_trace = gpd.read_file(os.path.join(dirPath,'./data_node_link/Link_Grids_Nodes.shp')) # link file from ARC (note that this is pre-processed by adding the Grid ID to each link)
    df_links_trace = df_links
    #df_links_trace = df_links_trace[(df_links_trace['FACTYPE']>0) & (df_links_trace[df_links_trace['FACTYPE']<50])]
    #print(df_links_trace.columns)

    # DISTANCE field is in mile, Shape_len is in Feet
    df_links_trace[['A','B']] = df_links_trace[['A','B']].astype('str')
    '''
    global traceG
    traceG = nx.DiGraph()
    for i,row in df_links_trace.iterrows():
        A = row['A']
        B = row['B']
        #if row['FACTYPE'] > 0 and row['FACTYPE'] < 50:
        #   print("shouldn't be here")
        traceG.add_edge(A, B)
        traceG[A][B]['weight'] = row['DISTANCE']*5280
    print 'network upload finished.'
    '''
    
    # LinkID is the index of the link in file
    df_links['LinkID']=df_links.index
    df_links_drive['LinkID']=df_links_drive.index


    global df_grids
    df_grids=pd.read_csv(os.path.join(dirPath, 'data_node_link/Grids.csv')) # grid file, has the boundary coordinates of each grid

    # 0.2 Build direction graphs
    # note: Link_Grids.csv is a ".csv" format version of the link shape file
    
    # prepare network with congestion levels

    global df_link_grids
    df_link_grids=pd.read_csv(os.path.join(dirPath, 'data_node_link/stmFilled/Link_Grids_Nodes_ValidSpeed_stm_0920.csv'),header=0)
    
 


    #df_link_grids = df_link_grids[(df_link_grids['FACTYPE']>0) & (df_link_grids[df_link_grids['FACTYPE']<50])]
    
    # update ttime to remove cases where speed calculated is too big
    # dist is in feet
    # speed is in mph
    # ttime is in hour

    df_link_grids['new']=-1
    
    for ind, row in df_link_grids.iterrows():
        for col in df_link_grids.columns.values.tolist():
            if col[-5:]=='ttime':
                dist=row[col[:-5]+'dist']
                ttime=row[col[:-5]+'ttime']
                speed=row[col[:-5]+'speed']
                ttime_new=dist/5280.0/speed
                if ttime_new>ttime:
                    df_link_grids.set_value(ind,col[:-5]+'ttime',ttime_new)
                    df_link_grids.set_value(ind,'new',1)
        
    wedgeDir = 'wedge_files'
    print("building individual wedges...")
    link_wedge=pd.read_csv(os.path.join(wedgeDir, 'abm15_203k_wid_link.csv'))
    df_link_grids_merged=df_link_grids.merge(link_wedge[['A_B','WID']],on=['A_B'],how='left')
    #TODO: instead of filling ttime with stmTime, stmTime15, stmTIme30, fill it when user requests through php Flask
    # currently, weights are filled in based on when flask started, suppose 8:03 was the cmd line argument when starting flask,
    # stmTime15 will be 8:15 and stmTime30 will be 8:30
    #TODO: Eventually, need to get back weights from Mongo based on real time
    # 'DISTANCE' col unit: miles
    # ttime_col unit: mph
    # 'ttime' unit: minutes

    
    global DGs
    DGs={}
    global DGs_reversed
    DGs_reversed={}

    print("finished reading wedge files")
    def dg_looper(dflx):
        #change here add all time
        all_ttimes = ['063000_ttime', '073000_ttime', '083000_ttime', '093000_ttime', '064500_ttime', '074500_ttime', '084500_ttime', '094500_ttime', '070000_ttime', '080000_ttime', '090000_ttime', '100000_ttime', '071500_ttime', '081500_ttime', '091500_ttime']
        DGx = nx.DiGraph()
        DG_reversex = nx.DiGraph()
        for col in all_ttimes:
            dflx.loc[dflx[col].isnull(),col]=dflx.loc[dflx[col].isnull(),'DISTANCE']/dflx.loc[dflx[col].isnull(),'SPEEDLIMIT']
            # weight is in minutes
        for ind, row2 in dflx.iterrows():
            for col in all_ttimes:
                edgeLabel = col.split('_')[0]
                #if row2['FACTYPE'] > 0 and row2['FACTYPE'] < 50:
                #   print("shoudn't be here")
                DGx.add_weighted_edges_from([(str(row2['A']),str(row2['B']),float(row2[col])*60.0)],weight=edgeLabel, dist=row2['DISTANCE'])
                DG_reversex.add_weighted_edges_from([(str(row2['B']),str(row2['A']),float(row2[col])*60.0)],weight=edgeLabel, dist=row2['DISTANCE'])
        return DGx, DG_reversex

    # downtown
    global DG_dt
    global DG_reverse_dt

    link_downtown = pd.read_csv(os.path.join(wedgeDir, 'link_downtown.csv'))
    df_links_downtown = df_link_grids[df_link_grids['A_B'].isin(link_downtown['A_B'].tolist())]
    DG_dt,DG_reverse_dt = dg_looper(df_links_downtown)

    global DG
    #DG=nx.DiGraph()
    
    global DG_reverse
    #DG_reverse=nx.DiGraph()

    
    # wedged_network
    for wid in df_link_grids_merged['WID'].unique().tolist():
        print("creating graph for wedge:", wid)
        dfx=df_link_grids_merged[df_link_grids_merged['WID']==wid]
        # remove 'wid' column and merge with downtown wedge
        dfx=dfx.drop(['WID'], axis=1)
        dfx_merged=pd.concat([dfx,df_links_downtown]).drop_duplicates().reset_index(drop=True)
        print("wedge file length:", len(dfx_merged))
        DGs[wid],DGs_reversed[wid]=dg_looper(dfx_merged)
    print('All graphs created')

    # the big network
    print("creating big network...")
    DG,DG_reverse=dg_looper(df_link_grids)
    

    global wedges
    wedges = gpd.read_file(os.path.join(wedgeDir, 'wedge_prj.shp'))
    global wedges_dt
    wedges_dt = gpd.read_file(os.path.join(wedgeDir, 'downtown_prj.shp'))
    print("downtown shape:")
    print(wedges_dt)
    print("==================")

    #df_points = pd.DataFrame({'Trip':[1], 'lat_ori':[float(origin_lat)], 'lon_ori':[float(origin_lon)], 'lat_des':[float(destination_lat)], 'lon_des':[float(destination_lon)]})
    #print(df_points)
    
    arrivalt = args_startt
    
    foo = arrivalt.split('.') 
    
    print('Preparing dataframes for trace generation')
    tracePath = "/home/users/ywang936/transit_simulation-0206/traceBack_all"
    global railstation_df
    global busstation_df
    global grtastation_df
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


    #arrivalt_hours = float(foo[0]) + float(foo[1]) / 60 + float(foo[2])/3600

    '''
    node_marta= pd.read_csv(os.path.join(dirPath,'data_node_link/marta/bus_stops_node.csv'))
    node_marta['key']=node_marta['stop_name']+'_'+node_marta['stop_id'].astype(str)

    global df_all_transit
    df_all_transit=node_marta[node_marta['key'].isin(STOP_EVENTS.keys())]
    

    node_grta = pd.read_csv(os.path.join(dirPath,'data_node_link/grta/bus_stops_node.csv'))
    node_grta['key']=node_grta['stop_name']+'|'+node_grta['stop_id'].astype(str)

    global df_stations_grta
    df_stations_grta=node_grta[node_grta['key'].isin(STOP_EVENTS_GRTA.keys())]
    '''
    '''
    outPaths_marta = pd.DataFrame()
    outPaths_grta = pd.DataFrame()
    outPaths_marta_pnr = pd.DataFrame()
    outPaths_grta_pnr = pd.DataFrame()
    outPaths_drive=pd.DataFrame()
    
    
    ## 2.1 Create input dataframe
    df_out=pd.DataFrame()
    print('---- Processing Trip ------------------')
    #df_points = pd.DataFrame({'Trip':[tripId], 'lat_ori':[origin_lat], 'lon_ori':[origin_lon], 'lat_des':[destination_lat], 'lon_des':[destination_lon]})

    app.run(host='0.0.0.0', port=5005, threaded=True, debug=True)
    

    print('---- Finding GRTA Routes ------------------')  
    origin_stops_list_grta, ori_dists_grta, pkrStopsOri_grta, df_points_ori_grta= getNearbyStops(df_points, 'origin', df_stations_grta, 'grta')
    destination_stops_list_grta, des_dists_grta, pkrStops_grta, df_points_dest_grta = getNearbyStops(df_points, 'destination', df_stations_grta, 'grta')
    
    df_final_paths_grta= returnTransitPaths_toDF_GlobalLoop(origin_stops_list_grta,ori_dists_grta,destination_stops_list_grta,des_dists_grta,arrivalt,endt,df_points_ori_grta,'grta')
    df_park_final_paths_grta=returnPNRPaths_toDF_GlobalLoop(destination_stops_list_grta,des_dists_grta,arrivalt,endt,pkrStopsOri_grta,df_points_ori_grta,'grta')
                
    df_out=df_out.append(df_final_paths_grta,ignore_index=True)
    df_out=df_out.append(df_park_final_paths_grta,ignore_index=True)
            
    print('---- Finding MARTA Routes ------------------')
    origin_stops_list_marta, ori_dists_marta, pkrStopsOri_marta, df_points_ori_marta= getNearbyStops(df_points, 'origin', df_all_transit)
    destination_stops_list_marta, des_dists_marta, pkrStops_marta, df_points_dest_marta = getNearbyStops(df_points, 'destination', df_all_transit)
    df_final_paths_marta= returnTransitPaths_toDF_GlobalLoop(origin_stops_list_marta,ori_dists_marta,destination_stops_list_marta,des_dists_marta,arrivalt,endt,df_points_ori_marta,'marta')
    df_park_final_paths_marta=returnPNRPaths_toDF_GlobalLoop(destination_stops_list_marta,des_dists_marta,arrivalt,endt,pkrStopsOri_marta,df_points_ori_marta,'marta')
        
    df_out=df_out.append(df_final_paths_marta,ignore_index=True)
    df_out=df_out.append(df_park_final_paths_marta,ignore_index=True)

    print('---- Finding Driving Routes ---------------')
    # parameter "1" is tripId
    outPaths_drivei=returnDrivePaths_toDF_GlobalLoop(df_link_grids,df_points,df_links,df_grids,DG,1,arrivalt)
    df_out=df_out.append(outPaths_drivei,ignore_index=True)
    for i in range(len(stmTimeOptions)):
        DG = makeDG(stmTimeOptions[i])
        outPaths_driveM=returnDrivePaths_toDF_GlobalLoop(df_link_grids,df_points,df_links,df_grids,DG,1,arrivaltOptions[i],i+2)
        df_out=df_out.append(outPaths_driveM,ignore_index=True)
    # 15 minutes before
    
    #outPaths_drive=outPaths_drive.append(outPaths_drivei,ignore_index=True)
    df_out.to_csv(os.path.join(dirPath, args.userid+'_comalts.csv'),index=False)
    '''
    #=====================ENERGY COST CALCULATION=========================#
    #  energy consump for MARTA RAIL
    global railEnergyDict
    railEnergyDict = makeEgDict(os.path.join(dirPath, 'energy_rate/energy_per_passenger_with_occ_100.csv'))

    # energy consump for other modes
    df_marta_bus_rate=PrepareMartaRate(os.path.join(dirPath, 'energy_rate/marta_bus_energyrate.csv'))
    eg_df=pd.DataFrame({'speed_mph':df_marta_bus_rate['avg speed mph'],'ww_kwhpermile':df_marta_bus_rate['ww_kwhpermile']})
    eg_df['veh_type']='marta'
    df_grta_rate=PrepareGRTA(os.path.join(dirPath, 'energy_rate/energy_rate_auto_grta.csv'))
    dfi=df_grta_rate.loc[:,['speed_mph','ww_kwhpermile']]
    dfi['veh_type']='grta'
    eg_df=eg_df.append(dfi)
    df_drive_rate=PrepareDrive(os.path.join(dirPath,'energy_rate/energy_rate_auto_grta.csv'))
    dfi=df_drive_rate.loc[:,['speed_mph','ww_kwhpermile']]
    dfi['veh_type']='drive'
    eg_df=eg_df.append(dfi)
    eg_df.to_csv(os.path.join(dirPath,'energy_rate/combined_rate.csv'),index=False)
    global egall
    egall = createEgDictAll(eg_df)
    
    '''
    # 20180424 remove, use df_links / df_links_trace(if 'A' 'B' are str type)
    print("preparing network for trace generation...")
    global trace_network
    path = "/home/users/ywang936/transit_simulation-0206/traceBack/"
    trace_network = gpd.read_file(path + 'route_2.shp')
    '''
    print("finished setting up")
    #=======================RUN FLASK=====================================#
    #app.run(host='0.0.0.0', port=5005, threaded=True, debug=True, use_reloader=False)
    '''
    f = args.userid+'_comalts'
    df = pd.read_csv(os.path.join(dirPath,f+'.csv'))
    df=splitOptions(df)
    df.to_csv(os.path.join(dirPath,f+'_withID.csv'),index=False)
    # second argument is starttime
    df = calcTimeDuration(df,arrivalt_hours)
    df_filled = fillEnergy(df, egall,railEnergyDict)
    df_filled.to_csv(os.path.join(dirPath,f+'_egAll.csv'), index=False)
    '''

    '''
    #################
    ### calculate energy per link per traveler
    # assumptions for ridership
    marta_rail_occupancy=.8
    marta_bus_ridership=10.0
    grta_ridership=40.0
    auto_ridership=1.0
    walkwait_energy=0.0
    # assign scaled energy values
    railLabels = ['RED', 'GOLD', 'BLUE', 'GREEN']
    filenames = [args.userid+'_comalts']


    for filen in filenames:
        df=pd.read_csv(os.path.join(dirPath,filen+'_egAll.csv'))
        for ind, row in df.iterrows():
            eg=row['energy']
            if row['mode']=='grta':
                df.set_value(ind,'energy_scaled',eg/grta_ridership)
                df.set_value(ind,'cost_drive',0)
            elif row['mode']=='drive':
                df.set_value(ind,'energy_scaled',eg/auto_ridership)
                df.set_value(ind,'cost_drive',0.54*row['dist_mile'])
            elif row['mode']=='marta':
                df.set_value(ind,'cost_drive',0)
                if row['route'] in railLabels:
                    df.set_value(ind,'energy_scaled',eg/marta_rail_occupancy)
                elif row['route'] in ['walking','waiting']:
                    df.set_value(ind,'energy_scaled',walkwait_energy)
                else:
                    df.set_value(ind,'energy_scaled',eg/marta_bus_ridership)
            else:
                df.set_value(ind,'cost_drive',0)
                df.set_value(ind,'energy_scaled',walkwait_energy)
        df.to_csv(os.path.join(dirPath,filen+'_egall_scaled.csv'),index=False)

        options = ['grta_only', 'grta_pnr', 'marta_only', 'marta_pnr', 'drive_only']
        jsonResults = defaultdict(dict)

        for option in options:
            modeDf = df[df['TripOption']==option]
            print('----------------'+option+'-----------------')
            for alter in modeDf['option_id'].unique():
                if option in ['grta_only', 'grta_pnr']:
                    jsonResults[option+str(alter)]['cost'] = 2.88
                elif option in ['marta_only', 'marta_pnr']:
                    jsonResults[option+str(alter)]['cost'] = 2.19
                else:
                    jsonResults[option+str(alter)]['cost'] = 0
                optionDf = modeDf[modeDf['option_id'] == alter]
                totalE = round(optionDf['energy_scaled'].sum(),2)
                totalDist = round(optionDf['dist_mile'].sum(),2)
                driveCost = round(optionDf['cost_drive'].sum(),2)
                duration = round(optionDf['duration'].sum()*60.0,2)
                jsonResults[option+str(alter)]['energy'] = totalE
                jsonResults[option+str(alter)]['cost'] += driveCost
                jsonResults[option+str(alter)]['distance'] = totalDist
                jsonResults[option+str(alter)]['duration'] = duration
                print(option+str(alter), totalE, totalDist, duration, driveCost, jsonResults[option+str(alter)]['cost'])

        sys.exit(json.dumps(jsonResults))




        outAlternatives = '/mnt/data/www/commwar/httpdocs/geopandas-scripts/commuteOptions/'+args.userid+'_comalts.json'
        with open(outAlternatives,'w') as fp:
            json.dump(jsonResults, fp)
    if os.path.isfile(outAlternatives):
        print("successfully generated commute alternatives for user:", args.userid)
        sys.exit(json.dumps(jsonResults))
    else:
        print("did not generate commute alternative for user:", args.userid)
        sys.exit(0)
    '''
main(__name__)
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=True, use_reloader=False)

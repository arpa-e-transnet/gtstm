from xml.etree import ElementTree
import pandas as pd
import argparse
from datetime import datetime, time, timedelta
import os

def parseXML(xmlf, abmDict, rdTypeDict):
    outFile = xmlf+'_stm'
    tree = ElementTree.parse(xmlf)
    root = tree.getroot()
    ind = 0
    fPartCount = 0
    df = pd.DataFrame(columns=['LinkID', 'LinkType', 'LinkDist', 'TimeStamp', 'Speed', 'TravelTime', 'VehicleType', 'WatsonPlot'])

    for link in root.findall('Link'):
        linkIDRaw = link.get('ID')
        linkParts = linkIDRaw.split('n')
        linkAB = linkParts[1]+'_'+linkParts[2]

        time = link.find('Time')
        timeStamp = time.get('Time')
        midNight = datetime.strptime('00:00:00', '%H:%M:%S')
        morning = datetime.strptime('06:00:00', '%H:%M:%S')
        timeDelta = (datetime.strptime(timeStamp, '%H:%M:%S') - midNight)
        actualTime = morning + timeDelta
        actualTimeStr = datetime.strftime(actualTime, '%H:%M:%S')

        speed = time.get('Speed')
        travelTime = time.get('TravelTime')
        rdType = rdTypeDict[abmDict[linkAB]['FACTYPE']]
        rdLength = abmDict[linkAB]['Shape_Leng']

        watsonFCar = 'passenger car ' + rdType + ' AvgSpeed '+ speed + 'mph_watsonplot_data.csv_mat' 
        watsonFTruck = 'combination long-haul truck ' + rdType + ' AvgSpeed '+ speed + 'mph_watsonplot_data.csv_mat' 
        watsonFBus = 'transit bus ' + rdType + ' AvgSpeed '+ speed + 'mph_watsonplot_data.csv_mat' 
        watsonFiles = [watsonFCar, watsonFTruck, watsonFBus]        
        watsonPaths = [os.path.join('/mnt/data/load_to_mongo/watson/ConvertedAll', wtf) for wtf in watsonFiles]
        
        watsonContents = [open(wtf,'r').read() for wtf in watsonPaths]
        #print(linkAB, rdType, actualTimeStr, speed, travelTime, 'passenger car', watsonContents[0])
        df.loc[ind] = [linkAB, rdType, rdLength, actualTimeStr, speed, travelTime, 'passenger car', watsonContents[0]]
        ind += 1

        #print(linkAB, rdType, actualTimeStr, speed, travelTime, 'heavy truck', watsonContents[1])
        df.loc[ind] = [linkAB, rdType, rdLength, actualTimeStr, speed, travelTime, 'heavy truck', watsonContents[1]]
        ind += 1

        #print(linkAB, rdType, actualTimeStr, speed, travelTime, 'transit bus', watsonContents[2])
        df.loc[ind] = [linkAB, rdType, rdLength, actualTimeStr, speed, travelTime, 'transit bus', watsonContents[2]]
        ind += 1
        
        if ind % 1000 == 0:
            fPartCount += 1
            print("Processed ", fPartCount, "parts")
            df.to_json('/home/users/ywang936/stmLoad2/'+outFile+str(fPartCount)+'.json', orient='records')
            df.to_csv('/home/users/ywang936/stmLoad2/'+outFile+str(fPartCount)+'.csv', index=False, header=False)
            df = pd.DataFrame(columns=['LinkID', 'LinkType', 'LinkDist', 'TimeStamp', 'Speed', 'TravelTime', 'VehicleType', 'WatsonPlot'])
            ind = 0

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("xmlFile", type=str, help="xml file")
    
    args = parser.parse_args()
    xmlF = args.xmlFile
    
    abmDF = pd.read_csv('/mnt/data/load_to_mongo/ABM_15_Links.csv')
    abmDict = abmDF.set_index('A_B').to_dict(orient='index')
    print(len(abmDict.keys()))

    rdTypeDict = {}
    for abmtype in [1,3,4,5,6,7,8,9,10,11]:
        rdTypeDict[abmtype] = 'urban highway'
    for abmtype in [2,12,13,14,15,16,17,18,19,20,50,51,52,53,54]:
        rdTypeDict[abmtype] = 'urban local'
    print(rdTypeDict)

    parseXML(xmlF, abmDict, rdTypeDict)
    
    


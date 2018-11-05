import pandas as pd
from collections import defaultdict
import datetime as dt
from datetime import date, datetime, timedelta
import time
from time import mktime



def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

if __name__=="__main__":
    aspeed = pd.read_csv('Fifteen_Game_Dt_Speed.csv')

    aspeed_906 = aspeed[aspeed['Date'] == '2017-09-6.0']
    print(aspeed_906)

    abmNet = pd.read_csv('Link_Grids_Nodes_ValidSpeed_stm.csv')
    print(abmNet)


    time1 = dt.time(14,0,0)
    time2 = dt.time(19,15,0)
    gameTimes = []
    # 9/13
    for result in perdelta(dt.datetime.combine(dt.date(1, 1, 1), time1), dt.datetime.combine(date(1, 1, 1), time2), timedelta(minutes=15)):
        print(result.time().strftime('%H:%M:%S'))
        gameTimes.append(result.time())
    print(gameTimes)
    speedCols = set()
    
    speedDict = defaultdict(dict)
    for ind, row in aspeed_906.iterrows():
        linkId = row['Link']
        timeStamp = row['Time']
        speed = row['Speed']
        speedDict[linkId][timeStamp] = speed

    for ind, row in abmNet.iterrows():
        objectId = row['OBJECTID']
        aSpeedsObj = speedDict[objectId]
        for timeStp in aSpeedsObj:
            timeObj = time.strptime(timeStp, '%H:%M:%S')
            timeObj = datetime.fromtimestamp(mktime(timeObj))
            #print(timeStp, timeObj)
            if timeObj.time() in gameTimes:
                print(objectId, timeObj.time().strftime('%H:%M:%S'), aSpeedsObj[timeStp])
                abmNet.set_value(ind, timeObj.time().strftime('%H%M%S')+'_speed',aSpeedsObj[timeStp])
                speedCols.add(timeObj.time().strftime('%H%M%S')+'_speed')
    speedColsFinal = list(speedCols)
    for col in speedColsFinal:
        abmNet[col] = abmNet[col].fillna(abmNet['073000_speed'])
    
    abmNet.to_csv('linkSpeed_gameday_906_final.csv', index=False)


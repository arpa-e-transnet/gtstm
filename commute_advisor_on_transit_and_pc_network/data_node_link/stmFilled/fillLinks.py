import pandas as pd
#from datetime import datetime as dt
df = pd.read_csv('Link_Grids_Nodes_ValidSpeed_stm_0920.csv')
initialTStr = '101500'
initObj = pd.to_datetime(initialTStr, format='%H%M%S', errors='ignore')

newObjs = [initObj + pd.Timedelta(15*x, 'm') for x in range(0, 81)]
newStrs = [item.strftime('%H%M%S') for item in newObjs]
speedCols = [item+'_speed' for item in newStrs]
distCols = [item+'_dist' for item in newStrs]
timeCols = [item+'_ttime' for item in newStrs]

for ind, row in df.iterrows():
    for col in speedCols:
        df.set_value(ind, col, row['100000_speed'])
    for col2 in distCols:
        df.set_value(ind, col2, row['100000_dist'])
    for col3 in timeCols:
        df.set_value(ind, col3, row['100000_ttime'])

df.to_csv('Link_Grids_Nodes_ValidSpeed_stm_0227.csv', index=False)



#newColsSpeed = ['101500_speed', '103000_speed', '104500_speed', '110000_speed', '111500_speed', '113000_speed', '114500_speed', '120000_speed', '121500_speed', '123000_speed', '124500_speed', '130000_speed', '131500_speed', '133000_speed', '134500_speed', '140000_speed', '141500_speed', '143000_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed', '121500_speed']

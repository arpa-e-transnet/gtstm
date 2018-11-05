#A	B	dist	mode	option	option_id	route	sequence	time	trip_id	speed	dist_mile	duration	rail_eg
import pandas as pd
import json
from collections import defaultdict

def makeEgDict(railEgFile):
    energyDict = {}
    railEg = pd.read_csv(railEgFile)
    for ind, row in railEg.iterrows():
        ori = row['Station_start']
        dest = row['Station_end']
        energyDict[ori, dest] = row['energy_consumption_kwh_per_passenger']
        energyDict[dest, ori] = row['energy_consumption_kwh_per_passenger']
    return energyDict

def createEgDictAll(eg_df):
	egall = defaultdict(dict)
	for ind, row in eg_df.iterrows():
		mode = row['veh_type']
		speed = row['speed_mph']
		egrate = row['ww_kwhpermile']
		egall[mode][speed] = egrate
	return egall
	
def names_adder(df,fs,fr):
    # f_s: stops fie; f_r: routes file
    fss,frs=pd.DataFrame(),pd.DataFrame()
    for fi in fs:
        fss=fss.append(pd.read_csv(fi))
    for fi in fr:
        frs=frs.append(pd.read_csv(fi))
    frs['route_id'],fss['stop_id']=frs['route_id'].astype(str),fss['stop_id'].astype(str)
    df['A'],df['B']=df['A'].astype(str),df['B'].astype(str)
    
    #df['route_id']=df['route'].apply(lambda x: x[:-2] if len(x)>2 and x[-2:]=='.0' else x)
    #df=df.merge(frs[['route_id','route_short_name']],how='left',on='route_id').rename(columns={'route_short_name':'route_name'})
    
    df['stop_id']=df['A'].apply(lambda x: str(x).split('_')[0] if len(str(x).split('_'))>1 else '')
    df=df.merge(fss[['stop_id','stop_name']],how='left',on='stop_id').rename(columns={'stop_name':'A_name'})
    df['stop_id']=df['B'].apply(lambda x: str(x).split('_')[0] if len(str(x).split('_'))>1 else '')
    df=df.merge(fss[['stop_id','stop_name']],how='left',on='stop_id').rename(columns={'stop_name':'B_name'})
    df=df.drop(['stop_id'], axis=1)
    
    return df

def fillEnergy(df, egall, railEnergyDict):
    railRoutes = [10909, 10911, 10912, 10913]
    for ind, row in df.iterrows():
        mode = row['mode']
        speed = int(round(row['speed']))
        route = row['route']
        dist = row['dist_mile']
        option = row['option']
        # default vehicle type is driving (car)
        vehType = 'drive'
        
        if mode=='transit' or mode=='drive':
            if option in ['marta-only', 'marta-pnr']:
                vehType = 'marta'
            elif option in ['grta-only', 'grta-pnr']:
                vehType = 'grta'
                
            if speed!=0:
                if route in railRoutes:
                    A = row['A_name']
                    B = row['B_name']
                    if A.startswith('MIDTOWN'):
                        A = 'MIDTOWN STATION'
                    if B.startswith('MIDTOWN'):
                        B = 'MIDTOWN STATION'
                    eg = railEnergyDict[A,B]
                    #print("here:", A,B, eg)
                else:
                    if speed > 80:
                        rate = egall[vehType][80]
                    else:
                        rate = egall[vehType][speed]
                    eg = rate * dist
                
            else:
                eg=0
        else:
            eg=0
        #print(mode, option, vehType, route, speed, eg)
        df.set_value(ind, 'energy', eg)
    return df

def calcTimeDuration_drive(tripPath, startTime):
    #can get rid of this function
    #energy = 0
    prevTime = startTime
    
    for ind, row in tripPath.iterrows():
        if row['sequence'] == 1:
            if row['option'] == 'drive-only':
                if row['option_id'] == 2:
                    #prevTime = startTime + 0.25
                    prevTime = startTime
                elif row['option_id'] == 3:
                    #prevTime = startTime + 0.5
                    prevTime = startTime
                elif row['option_id'] == 4:
                    prevTime = startTime - 0.25
                elif row['option_id'] == 5:
                    prevTime = startTime - 0.5
                else:
                    prevTime = startTime
                #20180402
                prevTime = row['time'] - 0.01
                ######
            else:
                prevTime = startTime
        curTime = row['time']      
        duration = curTime - prevTime
        mile = row['dist']

        if duration > 0:
            speed = mile / duration
        else:
            #20180403
            speed = 0
        tripPath.set_value(ind, 'speed', speed)
        tripPath.set_value(ind, 'dist_mile', mile)
        tripPath.set_value(ind, 'duration', duration)
        prevTime = curTime
    return tripPath

def calcTimeDuration(tripPath, startTime):
    #can get rid of this function
    #energy = 0
    #prevTime = startTime
    #walkWaitLabels = ['walking', 'waiting']
    
    for ind, row in tripPath.iterrows():
        #if row['sequence'] == 1:
            #prevTime = startTime
        #curTime = row['timeStamp']      
        duration = row['time']
        mile = row['dist']
         
        #Ask Ann if needed           
        #if ((row['mode']=='marta') and (row['route'] not in walkWaitLabels)):
            #mile = 0.621371*row['dist']

        if duration > 0:
            speed = mile / duration
        else:
            # 20180403 change 0 to a tiny number
            speed = 0
        tripPath.set_value(ind, 'speed', speed)
        tripPath.set_value(ind, 'dist_mile', mile)
        tripPath.set_value(ind, 'duration', duration)
        #prevTime = curTime
    return tripPath

def calcRailEg2(tripPath, railEnergyDict, startTime):
    #can get rid of this function
    #energy = 0
    prevTime = startTime
    walkWaitLabels = ['walking', 'waiting']
    railLabels = ['RED', 'GOLD', 'BLUE', 'GREEN']
    
    for ind, row in tripPath.iterrows():
        if row['sequence'] == 1:
            prevTime = startTime
        curTime = row['time']      
        duration = curTime - prevTime
        mile = row['dist']
        if row['route'] in railLabels:
            A = row['A'].split('_')[0][1:]
            B = row['B'].split('_')[0][1:]
            if A.startswith('MIDTOWN'):
                A = 'MIDTOWN STATION'
            if B.startswith('MIDTOWN'):
                B = 'MIDTOWN STATION'
            curEg = railEnergyDict[A,B]
            tripPath.set_value(ind, 'rail_eg', curEg)
            #energy += curEg
            
        if ((row['mode']=='marta') and (row['route'] not in walkWaitLabels)):
            mile = 0.621371*row['dist']

        if duration > 0:
            speed = mile / duration
        else:
            speed = 0
        tripPath.set_value(ind, 'speed', speed)
        tripPath.set_value(ind, 'dist_mile', mile)
        tripPath.set_value(ind, 'duration', duration)
        prevTime = curTime
    return tripPath

def splitOptions(df):
    mode=[]
    ind_start=0
    
    # can use option col insted
    for ind, row in df.iterrows():
        if ((row['sequence']==1) and (ind!=0)) or (ind==df.shape[0]-1):
            # after a mode finishes
            mode=list(set(mode))
            #print(mode)
            if ('drive' in mode) and ('grta' in mode):
                TripOption='grta_pnr'
            elif ('drive' in mode) and ('marta' in mode):
                TripOption='marta_pnr'
            elif 'marta' in mode:
                TripOption='marta_only'
            elif 'grta' in mode:
                TripOption='grta_only'
            elif 'drive' in mode:
                TripOption='drive_only'
            else:
                TripOption='bug'
                print('----- bug coming ------')
                print(mode)
                print([ind_start,ind])
                print(df.loc[ind_start:ind,:])
                
            df.loc[ind_start:ind,'TripOption']=TripOption
        
            ind_start=ind
            mode=[row['mode']]
        else:
            mode.append(row['mode'])
            
    return df

def PrepareMartaRate(filen):
    df_raw_marta=pd.read_csv(filen)
    # update emission rate for marta bus by adding upstream for MARTA
    df_raw_marta['ww_kwhpermile']=df_raw_marta.apply(lambda x: AddWPRate(x['energy_kwhpermile'],x['fuel type']),axis=1)
    df_marta_diesel=df_raw_marta[(df_raw_marta['road type']=='local') & (df_raw_marta['fuel type']=='diesel')]
    df_marta_cng=df_raw_marta[(df_raw_marta['road type']=='local') & (df_raw_marta['fuel type']=='cng')]
    df_marta_diesel=df_marta_diesel.rename(columns={'energy_kwhpermile':'energy_kwhpermile_diesel','ww_kwhpermile':'ww_kwhpermile_diesel'})
    df_marta_cng=df_marta_cng.rename(columns={'energy_kwhpermile':'energy_kwhpermile_cng','ww_kwhpermile':'ww_kwhpermile_cng'})
    # calculate the fleet average emission rate by applying number of buses
    diesel_percent=	145/565.0
    cng_percent=420/565.0
    df=df_marta_diesel.merge(df_marta_cng.loc[:,['avg speed mph','energy_kwhpermile_cng','ww_kwhpermile_cng']],how='left',on='avg speed mph')
    df['ww_kwhpermile']=diesel_percent*df['ww_kwhpermile_diesel']+cng_percent*df['ww_kwhpermile_cng']
    return df

def PrepareDrive(filen):
    df_raw=pd.read_csv(filen)
    df=df_raw[df_raw['veh_type']=='drive']
    df['ww_kwhpermile']=df.apply(lambda x:AddWPRate(x['energyrate_kwhpermile'],'gasoline'),axis=1)
    return df

def PrepareGRTA(filen):
    df_raw=pd.read_csv(filen)
    df=df_raw[df_raw['veh_type']=='grta']
    df['ww_kwhpermile']=df.apply(lambda x:AddWPRate(x['energyrate_kwhpermile'],'diesel'),axis=1)
    return df

def AddWPRate(pw,mode):
    # GREET MODEL upstream emission rate
    wp_bus_diesel=0.204
    wp_bus_cng=0.163
    wp_gas_auto=0.281
    if mode=='diesel':
        ww=pw+wp_bus_diesel*pw
    elif mode=='gasoline':
        ww=pw+pw*wp_gas_auto
    elif mode=='cng':
        ww=pw+pw*wp_bus_cng
    return ww

if __name__=='__main__':
    #  energy consump for MARTA RAIL
    railEnergyDict = makeEgDict('energy_rate/energy_per_passenger_with_occ_100.csv')
    
    # energy consump for other modes
    df_marta_bus_rate=PrepareMartaRate('energy_rate/marta_bus_energyrate.csv')
    df=pd.DataFrame({'speed_mph':df_marta_bus_rate['avg speed mph'],'ww_kwhpermile':df_marta_bus_rate['ww_kwhpermile']})
    df['veh_type']='marta'
    df_grta_rate=PrepareGRTA('energy_rate/energy_rate_auto_grta.csv')
    dfi=df_grta_rate.loc[:,['speed_mph','ww_kwhpermile']]
    dfi['veh_type']='grta'
    df=df.append(dfi)
    df_drive_rate=PrepareDrive('energy_rate/energy_rate_auto_grta.csv')
    dfi=df_drive_rate.loc[:,['speed_mph','ww_kwhpermile']]
    dfi['veh_type']='drive'
    df=df.append(dfi)
    df.to_csv('energy_rate/combined_rate.csv',index=False)
    
    eg_df=df
    #eg_df = pd.read_csv('energy_rate/energy_rate_auto_grta_marta.csv')

    ###### creating energy dictioary for all transit modes #####
    egall = createEgDictAll(eg_df)
    
    # generating mode ID
    fs = ['0820out']
    for f in fs:
        df=pd.read_csv(f+'.csv')
        #### optional depending on shortest path output ####
        df=splitOptions(df)
        df.to_csv(f+'_withID.csv',index=False)
        df = calcTimeDuration(df,7.0)
        df_filled = fillEnergy(df, egall,railEnergyDict)
        df_filled.to_csv(f+'_egAll.csv', index=False)
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
    filenames = ['0820out']
    for filen in filenames:
        df=pd.read_csv(filen+'_egAll.csv')
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
        df.to_csv(filen+'_egall_scaled.csv',index=False)
        
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

        with open('commuteAlts.json', 'w') as fp:
            json.dump(jsonResults, fp)

        
        
        

import pandas as pd

df_speeds=pd.read_csv('linkSpeed_gameday_913.csv')

print('101568',df_speeds[df_speeds['OBJECTID'] == 101568])
print('101568',df_speeds[df_speeds['OBJECTID'] == 101568]['190000_speed'])
print('91001',df_speeds[df_speeds['OBJECTID'] == 91001])
print('91001',df_speeds[df_speeds['OBJECTID'] == 91001]['153000_speed'])
print('90869',df_speeds[df_speeds['OBJECTID'] == 90869])
print('90869',df_speeds[df_speeds['OBJECTID'] == 90869]['190000_speed'])

'''
df_speeds_large = df_speeds[df_speeds['180000_ttime'] >= 1]
print(df_speeds_large)
'''

print(df_speeds['174500_speed'].isnull().sum())
print(df_speeds['164500_speed'].isnull().sum())
print(df_speeds['154500_speed'].isnull().sum())
print(df_speeds['144500_speed'].isnull().sum())
print(df_speeds['184500_speed'].isnull().sum())
print(df_speeds['171500_speed'].isnull().sum())
print(df_speeds['173000_speed'].isnull().sum())
print(df_speeds['181500_speed'].isnull().sum())
print(len(df_speeds))
print(df_speeds[df_speeds['A'] == 259])

df_speeds2=pd.read_csv('Link_Grids_Nodes_ValidSpeed_stm_0920.csv')
print(df_speeds2[df_speeds2['A'] == 259])

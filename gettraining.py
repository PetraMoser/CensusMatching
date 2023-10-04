#! /usr/bin/env python3


################################################
#
#
# Initial filter, for each individual:
#
#  1) Matching state
#  2) Within 3 years of birth
#  3) 0.2 Jaro-Winkler Distance
#
# #
################################################ 


import os, csv, sqlite3, re
import sys
import multiprocessing, random
import pandas as pd
import numpy as np
print(np.__version__) 
import time
import jellyfish #Jaro-Winkler and Soundex Distances
from filelock import FileLock
from multiprocessing import Lock
from functools import partial
from contextlib import contextmanager
from scipy.stats import norm
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

#------------------------------------------------------------------------------ 


############ Customable parameters ##################
#### Init variables
pool_size = 1
censusyear = 1870 
birthyear_bandwidth = 3
myseed = 201
random_size = 1000

############ Directories ##################
root_dir = ''
dataLog_dir  = root_dir + 'data/log/'
dataProc_dir = root_dir + 'data/processed/'
root_lockPath = root_dir + 'data/locks/'
dataRecord_dir = root_dir + 'data/raw/'
DataCensus_dir = '/homes/data/census-ipums/v2019/mx/'+ str(censusyear)+ '/csv/'

#------------------------------------------------------------------------------ 
##### Get list of individuals to be matched ######

df_record_init=pd.read_csv(dataRecord_dir +'record_sample.csv',encoding='ISO-8859-1')
df_record_init = df_record_init[['ID','Gender','FirstName','MiddleName','LastName','BirthYear','BirthYear_Census','BirthState']]

## Drop duplicates
df_record = df_record_init.drop_duplicates().reset_index()  
 
## drop the females
df_record=df_record[~(df_record['Gender']=='female')]

## Drop observations with missing for birthstate or birthyear
df_record = df_record[df_record['BirthState'].notnull()]
df_record = df_record[df_record['BirthYear'].notnull()]
df_record = df_record[df_record['BirthYear']!=0]
df_record = df_record.reset_index(drop =True)

## Subset to those born before or on census year - birthyear_bandwidth
df_record = df_record[df_record['BirthYear']<=(censusyear-birthyear_bandwidth)]


## Make sure all the years are integral
df_record['BirthYear_Census'] = df_record['BirthYear_Census'].astype('int')


### Now I want to grab 1000 random id's from the unique ids
ID_values=df_record['ID'].unique()
random.seed(myseed)
random_PIDs=random.sample(ID_values.tolist(),random_size)

#------------------------------------------------------------------------------ 
############ Functions ##################
state_dict=states_dict = {'1':'Alabama',
			'2':'Alaska',
			'4':'Arizona',
			'5':'Arkansas',
			'6':'California',
			'8':'Colorado',
			'9':'Connecticut',
			'10':'Delaware',
			'11':'District of Columbia',
			'12':'Florida',
			'13':'Georgia',
			'15':'Hawaii',
			'16':'Idaho',
			'17':'Illinois',
			'18':'Indiana',
			'19':'Iowa',
			'20':'Kansas',
			'21':'Kentucky',
			'22':'Louisiana',
			'23':'Maine',
			'24':'Maryland',
			'25':'Massachusetts',
			'26':'Michigan',
			'27':'Minnesota',
			'28':'Mississippi',
			'29':'Missouri',
			'30':'Montana',
			'31':'Nebraska',
			'32':'Nevada',
			'33':'New Hampshire',
			'34':'New Jersey',
			'35':'New Mexico',
			'36':'New York',
			'37':'North Carolina',
			'38':'North Dakota',
			'39':'Ohio',
			'40':'Oklahoma',
			'41':'Oregon',
			'42':'Pennsylvania',
			'44':'Rhode Island',
			'45':'South Carolina',
			'46':'South Dakota',
			'47':'Tennessee',
			'48':'Texas',
			'49':'Utah',
			'50':'Vermont',
			'51':'Virginia',
			'53':'Washington',
			'54':'West Virginia',
			'55':'Wisconsin',
			'56':'Wyoming'}

inv_map = {v: k for k, v in state_dict.items()}
def state_to_name(x):
	dict=inv_map[x]
	name='mx'+str(dict)+'001.csv'
	return name


def open_census_data(state,year,dir):
	filename=state_to_name(state)
	df=pd.read_csv(dir+filename,encoding='latin1')
	
	# These states also have Indian Territories (two csv files associated with them)
	if state=='New Mexico':
		df2=pd.read_csv(DataCensus_dir+'mx3511.csv',encoding='latin1')
		df=df.append(df2)
	if state=='Idaho':
		df2=pd.read_csv(DataCensus_dir+'mx16101.csv',encoding='latin1')
		df=df.append(df2)
	if state=='Oklahoma':
		df2=pd.read_csv(DataCensus_dir+'mx40101.csv',encoding='latin1')
		df=df.append(df2)
	if state=='South Dakota':
		df2=pd.read_csv(DataCensus_dir+'mx46101.csv',encoding='latin1')
		df=df.append(df2)
	if state=='Utah':
		df2=pd.read_csv(DataCensus_dir+'mx49101.csv',encoding='latin1')
		df=df.append(df2)
	if state=='Wyoming':
		df2=pd.read_csv(DataCensus_dir+'mx5611.csv',encoding='latin1')
		df=df.append(df2)
	
	# Calculate the year from age and subset for the year we need
	df['birthyr']=censusyear-df['age']
	year_min=year-birthyear_bandwidth
	year_max=year+birthyear_bandwidth
	df=df[df['birthyr']<=year_max]
	df=df[df['birthyr']>=year_min]

	# Fix the first names: drop the null ones, make everything in lower case.
	df=df[~(df['namefrst'].isnull())]
	df=df[~(df['namelast'].isnull())]
	df=df[~(df['namefrst']=='!')]
	df=df[~(df['namelast']=='!')]
	df=df[~(df['namefrst']=='*')] 
	df=df[~(df['namelast']=='*')]
	df=df[~(df['namefrst']=='---')] 
	df=df[~(df['namelast']=='---')]


	df['namefrst']=df['namefrst'].apply(lambda x: x.lower())
	df['namelast']=df['namelast'].apply(lambda x: x.lower())
	fname1 = df['namefrst'].str.partition(' ')[[0,2]].rename(columns={0:'namefrst',2:'namemiddle'})
	df=df.drop(['namefrst'],axis=1)
	df = pd.concat([df, fname1], axis=1)
	#print(df[['namefrst','namemiddle']].head(20))


	## fix first names and last name too by dropping special characters and spaces in the beginning
	## The special characters affect JW, spaces affect the initial match
	search=['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m','1','2','3','4','5','6','7','8','9','\x85']
	#print((df[(~df['namefrst'].str[0].isin(search))]))
	while df[(~df['namefrst'].str[0].isin(search))&(df['namefrst'].str.len()>0)].shape[0]>0:
		df['namefrst']=df['namefrst'].apply(lambda x: x if len(x)==0 else (x if x[0] in search else x[1:]))
		#print((df[(~df['namefrst'].str[0].isin(search))&(df['namefrst'].str.len()>0)]))

	#print((df[(~df['namelast'].str[0].isin(search))]))
	while df[(~df['namelast'].str[0].isin(search))&(df['namelast'].str.len()>0)].shape[0]>0:
		df['namelast']=df['namelast'].apply(lambda x: x if len(x)==0 else (x if x[0] in search else x[1:]))
		#print((df[(~df['namelast'].str[0].isin(search))&(df['namelast'].str.len()>0)]))
	
	df=df[~(df['namefrst']=='')]
	df=df[~(df['namelast']=='')]


	## Fix initial
	#print(np.sort(df['namemiddle'].unique()))
	#print((df[(~df['namemiddle'].str[0].isin(search))&(df['namemiddle'].str.len()>0)]))
	while df[(~df['namemiddle'].str[0].isin(search))&(df['namemiddle'].str.len()>0)].shape[0]>0:
		df['namemiddle']=df['namemiddle'].apply(lambda x: x if len(x)==0 else (x if x[0] in search else x[1:]))
		#print((df[(~df['namemiddle'].str[0].isin(search))&(df['namemiddle'].str.len()>0)]))
	df['namemiddle']=df['namemiddle'].apply(lambda x: np.nan if len(x) == 0 else x[0])	

	df=df[['bpl','namefrst','namelast','namemiddle','histid','birthyr']]
	return df


	
def get_training(random_PID, census_dir, df_record, path_to_data_filelock, path_to_data_file):
	
	errFunc = False
	df_result = None

	try:
		print('Entering unique_id'+str(random_PID))
		df_toMatch = df_record[(df_record['ID_new']==random_PID)].copy()
		del df_toMatch['index']
		#print(df_toMatch)
		birthStatelist = df_toMatch['BirthState'].tolist()
		birthState=birthStatelist[0]
		birthYr = df_toMatch['BirthYear_Census']
		

		##Birth Years
		birthYr = int(birthYr)
		df_census_orig=open_census_data(birthState,birthYr,census_dir)

		#Save index values in column
		df_census_orig['indexValues'] = df_census_orig.index
		

		
		for ix, rowData in df_toMatch.iterrows():
			
			#print(rowData)
			#Make a copy of the census dataframe so we can edit it.
			df_census = df_census_orig.copy()
			#print(f"df census before JW: {df_census.head()}")

			
			#Create Jaro Winkler Distances based on last names and drop all pairs with a distance of more than 0.2
			jw_lastName_dict = {}
			for i, row in df_census.iterrows():
				jw_lastName_dict[i] = 1 - jellyfish.jaro_winkler(row['namelast'],rowData['LastName'])
			df_census['jw_LastName'] = df_census['indexValues'].map(jw_lastName_dict)
			df_census = df_census[df_census['jw_LastName']<=0.2]
			#print(f"df census after JW last name: {df_census.head()}")
            #print('df census after JW last name')
			

			#Create Jaro Winkler Distances based on first names and drop all pairs with a distance of more than 0.2
			jw_firstName_dict = {}
			for i, row in df_census.iterrows():
				jw_firstName_dict[i] = 1 - jellyfish.jaro_winkler(row['namefrst'],rowData['FirstName'])
			df_census['jw_FirstName'] = df_census['indexValues'].map(jw_firstName_dict)
			initial_MoS=rowData['FirstName'][0]
			df_census['initial_fname']=(df_census['namefrst'].str.len()==1)*1
			df_census = df_census[(df_census['jw_FirstName']<=0.2)|((df_census['initial_fname']==1)&(df_census['namefrst']==initial_MoS))]
			#print(f"df census after JW first name: {df_census.head()}")
            
			df_census=df_census.drop(['initial_fname'],axis=1)
		
			#Only proceed with the matching process if there are still possible matches to consider
			if not df_census.empty:
				df_export=df_census[['bpl','namelast','namefrst','namemiddle','histid','birthyr']].copy()
				
	
				df_export['record_lastName'] = rowData['LastName']
				df_export['record_firstName'] = rowData['FirstName']
				df_export['record_middleName'] = rowData['MiddleName']
				df_export['record_birthYear'] = rowData['BirthYear']
				df_export['record_BirthYear_Census'] = rowData['BirthYear_Census']
				df_export['record_birthplc'] = rowData['BirthState']
				df_export['record_gender'] = rowData['Gender']
				df_export['record_original_ID'] = rowData['ID']
				df_export['FailJW'] = 0

				#print('print df_export')
				#print(df_export.head())

				if not (df_export is None):
					#print('create df_result')
					if df_result is None:
						#print('df_result is None')
						df_result = df_export
					else:
						df_result = df_result.append(df_export, sort=False)
				#print('now df_result',df_result.head())
			else:
				data={'bpl':['Fail'],'namelast':['Fail'],'namefrst':['Fail'],'namemiddle':['Fail'],'histid':['Fail'],'birthyr':['Fail']}
				df_export = pd.DataFrame(data, columns=['bpl','namelast','namefrst','namemiddle','histid','birthyr'])
				df_export['record_lastName'] = rowData['LastName']
				df_export['record_firstName'] = rowData['FirstName']
				df_export['record_middleName'] = rowData['MiddleName']
				df_export['record_birthYear'] = rowData['BirthYear']
				df_export['record_BirthYear_Census'] = rowData['BirthYear_Census']
				df_export['record_birthplc'] = rowData['BirthState']
				df_export['record_gender'] = rowData['Gender']
				df_export['record_original_ID'] = rowData['ID']
				df_export['FailJW'] = 1

				print('print df_export census empty')
				#print(df_export)

				if not (df_export is None):
					
					if df_result is None:
						df_result = df_export
					else:
						df_result = df_result.append(df_export, sort=False)
				#print('now df_result',df_result.head())
		
		lock = FileLock(path_to_data_filelock)
		with lock:
			dataFile = open(path_to_data_file,'a')
			dataWriter = csv.writer(dataFile)
			if not df_result is None:
				for ix,row in df_result.iterrows():
					dataWriter.writerow(row)
			dataFile.close()


		
	except Exception as e:
		msg=str(e)
		#error_code=e.sqlite_errorcode
		#error_name=e.sqlite_name
		print(msg)
		print('ERROR') 
		
	
	print('Exiting unique_id'+str(random_PID))

# Allow pool to accept keyword arguments
@contextmanager
def poolcontext(*args, **kwargs):
	pool = multiprocessing.Pool(*args, **kwargs)
	yield pool
	pool.terminate()


#### File names
data_file_name =  'record_links_' + str(censusyear) + 'census.csv'
## Create data file
if data_file_name not in os.listdir(dataProc_dir):
	
	## Create header for dataset
	dataFile = open(dataProc_dir + data_file_name,'w')
	dataWriter = csv.writer(dataFile)

	## Column names
	cols = ['bpl','namelast','namefrst','namemiddle','histid','birthyr']
	add_cols = ['record_firstName','record_lastName','record_middleName','record_birthYear',
             'record_BirthYear_Census','record_birthplc','gender_mos','record_original_ID','FailJW']
	cols = cols + add_cols

	## Add header
	dataWriter.writerow(cols)
	
	## Close file
	dataFile.close()

path_to_data_filelock = root_lockPath + 'filelock_' + 'data_census_training' + str(censusyear) + '.txt.lock'
path_to_data_file = dataProc_dir + data_file_name


if __name__ == "__main__":
	print('Ready to run function.')
	## Try using a pool of workers
	with poolcontext(processes=pool_size) as pool:
		pool.map(partial(get_training, census_dir=DataCensus_dir, df_record=df_record,path_to_data_filelock=path_to_data_filelock,path_to_data_file=path_to_data_file),random_PIDs)


for random_PID in random_PIDs:
	print(random_PID)
	get_training(str(random_PID), DataCensus_dir, df_record, path_to_data_filelock, path_to_data_file)



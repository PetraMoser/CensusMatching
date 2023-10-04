#! /usr/bin/env python3

##### NOW GO WITH THE MATCHING

#Import packages
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
import re
import math
import pickle
import random
import jellyfish
import statsmodels.api as sm
from scipy.optimize import minimize


#------------------------------------------------------------------------------ 
############ Customable parameters ##################
#Specify gamma parameter (how much weight you place on PPV relative to TPR)
#eg. Gamma=1 means we place equal weight on PPV and TPR
# PPV = Positive Predictive Value, TPR = True Positive Rate
gamma = 1
seed=10

training_path = '/homes/data/cens1930.work/moser/shared/training_MOScensus/Final_1880_files/'
train=pd.read_csv(training_path+'data_regressors/training1880james3.csv',encoding='latin-1')
### Define the revant variables: the id, the match, and the birthyear_mos
birth_mos='BirthYear_Census'
match_variable='match'
id_census='histid'
mos_id='record_original_ID'

#------------------------------------------------------------------------------ 
def split_sample(df,training_size,seed):
	df.loc[:,mos_id]=df[mos_id].astype('str')
	random.seed(seed)
	id=df[mos_id].unique().tolist()
	train_id=random.sample(id,round(len(id)*training_size))
	train=df[df[mos_id].isin(train_id)].copy()
	oos=df[~df[mos_id].isin(train_id)].copy()
	return train,oos

#train,oos=split_sample(df,0.5,seed)

def training(train_orig,size_b,type_grid):
	## Make sure the data are in the right format and drop fails
	total_size=train_orig[mos_id].nunique()
	print(train_orig.shape)
	print(train_orig[(train_orig[id_census]=="Fail")|(train_orig['hits']=="0")|(train_orig['hits']==0)])

	train = train_orig[~((train_orig[id_census]=="Fail")|(train_orig['hits']=="0")|(train_orig['hits']==0))].copy()
	regressors = ['exact','fdist','ldist','ydist1','ydist2','ydist3','fsoundex','lsoundex','hits', 
		'hits2','exact.mult','fstart','lstart','fend','lend','mimatch']
	train['exact.all.mult']=train['exact.all.mult'].astype('float64')

	for x in regressors:
		train[x]=train[x].astype('float64')
	#print(train[train['exact.all.mult']!=0])

	train=train[train['exact.all.mult']==0]

	print(train.shape)

	#print(train.dtypes)

	#Building the Probit Model

	#Independent Variables
	##James ends up deciding to use the following variables from Table 3
	##I.E. James uses these variables:
	#print(train.dtypes)
	X = train[regressors].copy()
	X = sm.add_constant(X)
	#Dependent Variable
	Y = train[match_variable]

	#Fit the model
	probit_model = sm.Probit(Y.astype(float), X.astype(float)).fit(method='newton',maxiter=5000)

	print(probit_model.summary())

	pd.set_option("display.max_columns", 101)

	train["predictions"] = probit_model.predict(sm.add_constant(train[regressors]))
	temp = pd.DataFrame(train[mos_id].value_counts().index).rename(columns={0:mos_id})
	for index, row in temp.iterrows():
		a = train[row[mos_id]==train[mos_id]].copy()
		imax = a["predictions"].idxmax(axis=1)
		temp.loc[index,match_variable] = a.loc[imax,match_variable]
		max_prediction = a.loc[imax,"predictions"]
		temp.loc[index,id_census]=a.loc[imax,id_census]
		if a.shape[0]>1:
			a.drop(index=imax,inplace=True)
			second_max_prediction = a.loc[a["predictions"].idxmax(axis=1),"predictions"]
			temp.loc[index,"ratio_max_to_second"] = max_prediction/second_max_prediction
		else:
			temp.loc[index,"ratio_max_to_second"] = 2
		temp.loc[index,"max_prediction"] = max_prediction

   
	#### here we have to count the number of false negatives coming from 
	#### observations that are correct matches but do not have the highest score
	#### Step 1: do a dictionary, indicate in the main frame what is the pid that has been selected by the probit
	temp2=temp.set_index(mos_id)
	temp_dict=temp2[id_census]
	dictionary=temp_dict.T.to_dict()
	train["pid_highest"] = train[mos_id].map(dictionary)
	#print(train_hyper)

	# At this point, I can count the false negatives coming from the probit scores
	not_highest=train[train[id_census]!=train['pid_highest']]
	fn_highest_score=sum(not_highest[match_variable]==1)
	print('False negatives')
	print(fn_highest_score)

	## Calculate the grid and the maximizers
	## for an explanation of the code,
	## see bottom of this page.
	if type_grid=="grid_search":
		b1=np.linspace(0.001,0.4,size_b)
		b2=np.linspace(1.001,2,size_b)
	elif type_grid=="random_search":
		b1=np.random.uniform(0.001,0.4,size_b)
		b2=np.random.uniform(1.001,2,size_b)
	m=temp['max_prediction'].values
	m=np.vstack(m)
	B1=np.tile(b1,(m.shape[0],1))
	r=temp['ratio_max_to_second'].values
	r=np.vstack(r)
	B2=np.tile(b2,(m.shape[0],1))
	trueB1=(m>B1)*1
	trueB2=(r>B2)*1
	t1=trueB1.T
	t2=trueB2.T
	predicted=np.resize(t1,(b1.shape[0],1,m.shape[0]))*t2
	cr=temp[match_variable].values
	#print(predicted)
	#print(cr)
	tp_temp=((predicted==cr)&(cr==1))*1
	tp=np.sum(tp_temp,2)
	#print(cr, tp)
	#note: for the false negatives, I need to sum the fn coming from the first step (probit scores)
	# and those coming from the second: not selected because below (b1 and b2)
	fn_temp=((predicted!=cr)&(cr==1))*1
	fn=np.sum(fn_temp,2)+fn_highest_score
	fp_temp=((predicted==1)&(predicted!=cr))
	fp=np.sum(fp_temp,2)
	tpr=np.zeros((b1.shape[0],b2.shape[0]))
	tpr=(tp/(tp+fn))
	tpr[np.isnan(tpr)] = 0
	ppv=np.zeros((b1.shape[0],b2.shape[0]))
	ppv=tp/(tp+fp)
	ppv[np.isnan(ppv)] = 0
	U=tpr+gamma*ppv 
	maxs=np.where(U == np.max(U))
	max_U=U[maxs[0],maxs[1]]
	b1_max=b1[maxs[0]]
	b2_max=b2[maxs[1]]
	b1_list=b1_max.tolist()
	b2_list=b2_max.tolist()
	x0 = np.array([sum(b1_list)/len(b1_list),sum(b2_list)/len(b2_list)])
	def dist_func(x0):
    		return sum(((np.full(len(b1_list),x0[0])-b1_list)**2+(np.full(len(b1_list),x0[1])-b2_list)**2)**(1/2))
	res = minimize(dist_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
	print(res.x)
	b1_median,b2_median=res.x
	TPR_m=tpr[maxs[0],maxs[1]]
	PPV_m=ppv[maxs[0],maxs[1]]
	print(b1_median,b2_median,max_U.mean(), TPR_m.mean(), PPV_m.mean()) 

	### get the 90% PPV 
	#######
	ppv2=ppv.copy()
	ppv2[ppv2<0.90]=0
	U2=tpr+gamma*ppv2 
	maxs2=np.where(U2 == np.max(U2))
	max_U2=U2[maxs2[0],maxs2[1]]
	b1_max2=b1[maxs2[0]]
	b2_max2=b2[maxs2[1]]
	TPR_m2=tpr[maxs2[0],maxs2[1]]
	PPV_m2=ppv2[maxs2[0],maxs2[1]]
	b1_list2=b1_max2.tolist()
	b2_list2=b2_max2.tolist()
	x0 = np.array([sum(b1_list2)/len(b1_list2),sum(b2_list2)/len(b2_list2)])
	def dist_func(x0):
    		return sum(((np.full(len(b1_list2),x0[0])-b1_list2)**2+(np.full(len(b1_list2),x0[1])-b2_list2)**2)**(1/2))
	res2 = minimize(dist_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
	print(res2.x)
	b1_median2,b2_median2=res2.x

	print(b1_median2,b2_median2,max_U2[0], TPR_m2[0], PPV_m2[0]) 

	return probit_model, b1_median,b2_median,max_U[0], TPR_m[0], PPV_m[0], total_size, train

model,b1,b2,maxU,TPR,PPV,size, df_train=training(train,1000,"grid_search")
#### Save the outcomes of the probit
a=model.summary()
summ=open(training_path+'models/summary_probit_1880_james.txt','w')
summ.write(str(a))
summ.close

coeff=model.params
coeff.to_csv(training_path+'models/probit_coeff_1880_james.csv')

hyper=open(training_path+'models/hyper_1880_james.csv','w')
hyper.write('b1,b2')
hyper.write('\n')
hyper.write(str(b1)+','+str(b2))
hyper.close()

## save model as pickle file
model.save(training_path+"models/probit_1880_james.pickle")

print(df_train.head())
print(df_train.columns)


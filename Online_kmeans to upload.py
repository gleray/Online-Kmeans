########### DISCLAIMER #############
## The read_format function has to be implemented as 
##it depends on the nature of the data file storage type
###################################
import csv
import itertools
import math
import os
import random
import scipy.spatial.distance
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fastcluster import linkage
from cdtw import pydtw
from functools import partial
from itertools import chain
from multiprocessing import Pool
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.stats import mode
from workalendar.europe import Denmark
from datetime import datetime

sys.setrecursionlimit(10000)


def init_centers(data,K=2):
	'''
	Randomly initialize the  K centers

	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	'''
	return([random.randint(0,len(data)-1) for k in range(K)])

def euclidean_dist(x,y):
	'''
	Calculate the euclidean distance

	x: A 2D N*M numpy array (data)
	y: A vector M (a centroid)
	'''
	dist=np.sqrt(np.nansum((np.array(x,dtype=np.float)-np.array(y,dtype=np.float))**2,axis=1))

	return(dist)

def dCor(x,y,beta=None):
	'''
	Calculate the Correlation distance

	x: A 2D N*M numpy array (data)
	y: A vector M (a centroid)
	beta: if none perform the first type dcor otherwise
	'''
	cor=[np.corrcoef(i,y)[0,1] for i in x]
	if beta is None:
		#dcor=np.sqrt((2*[a_i-b_i for a_i,b_i in zip([1]*len(cor),cor)]))
		dcor=2*(np.array([1]*len(cor))-np.array(cor))
	else :
		dcor=(np.array([1]*len(cor))-np.array(cor))/(np.array([1]*len(cor))+np.array(cor))**beta
	return(dcor)

def dissCORT(x,y,q=1):
	'''
	Calculate the dissimilarity distance using the DTW and first and first order temporal correlation
	x: A 2D N*M numpy array (data)
	y: A vector M (a centroid)
	q: tunning factor between DTW and first order temporal correlation 0<q<5
	'''

	p=len(y)
	x1Index=np.ix_(range(len(x)),range(1,p))
	x2Index=np.ix_(range(len(x)),range(p-1))
	x=np.array(x)
	y=np.array(y)

	distance = [pydtw.dtw(i,y,pydtw.Settings(dist = 'euclid', 
                             	step  = 'dp1', 
                                window = 'palival_mod', param = 0.1,
                                 compute_path = True)).get_dist()  for i in x]
	
	
	corrTempOrder=np.nansum((x[x1Index]-x[x2Index])*(y[1:]-y[:(p-1)]),axis=1) / (np.sqrt( np.nansum((x[x1Index]-x[x2Index])**2,axis=1) )*np.sqrt(np.nansum((y[1:]-y[:(p-1)])**2)))

	if np.isnan(corrTempOrder).any():
		for i in np.where(np.isnan(corrTempOrder))[0]:
			corrTempOrder[i]=0

	return((2/( 1+ np.exp(q*corrTempOrder)))*distance)
	

def assign_points(data,centroids,K):
	'''
	Calculate the distance of each point to the centroids and affect them to the closest 

	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	'''
	for k in range(K):
		if k==0:
			dist=dissCORT(data,centroids[k])
		else:
			dist=np.vstack((dist,dissCORT(data,centroids[k])))

	return([np.argmin(dist,axis=0),dist]) # for each consumer in data affect it to the closest cluster.

def compute_means(data,assigned,K):
	'''
	Calculate the mean for each cluster to generate the centroids

	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	assigned: list of index specifying to which cluster each individual is assigned to
	'''
	return(np.array([np.nanmean(np.array([data[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0) for k in range(K)]))
	

def objective_function(data,centroids,assigned,K):
	'''
	Calculate the sum of the distance of the points to their centroids and return the sum of it

	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	assigned: list of index specifying to which cluster each individual is assigned to
	'''
	
	 sumcort=0
	 for k in range(K):
	 	subdata=[data[i] for i in np.where(assigned==k)[0]]
	 	sumcort+=np.nansum(dissCORT(subdata,centroids[k]) )
	 return(sumcort)

def clustering(data,centroids,K=2,eps=1.e-4):
	'''
	main K-means clustering function (used for the consensus clustering)
	
	data: the dataset N*M as a numpy array
	centroids: initial centroids as a 2D numpy array N*M*K
	K: the number of cluster to generate
	eps: the threshold used to break the loop

	'''	
	WCSS=[]
	# iterate up to 300 loop; stop if the Winthin sum of square is not reducing anymore (<1.e-4)
	for loop in xrange(500):
		print('loop:'+str(loop))
		#assign points to a cluster (center defined by centroid)
		assigned,dist=assign_points(data,centroids,K)
		#recalculate the centroid
		centroids=compute_means(data,assigned,K)
		#check convergence
		WCSS.append(objective_function(data,centroids,assigned,K))
		if len(WCSS)>2:
			if (WCSS[len(WCSS)-2]-WCSS[-1])<eps and (WCSS[len(WCSS)-2]-WCSS[-1])>=0:
				break
		print(WCSS[-1])
	return([WCSS,assigned,centroids,dist])


def probability_distance(poolResults,N,I,K):
	'''
	Create a probability distance matrix from all the partition in poolResults

	PoolResults: list of the results from the different instances
	N: number of individuals
	I: number of instances
	K: list of the number of cluster in each instances
	'''

	prob=np.zeros(shape=(N,N))
	for i in range(I): 
		print(i)
		for k in range(K):
			for x,y in itertools.product(np.where(poolResults[1+(i*4)]==k)[0],np.where(poolResults[1+(i*4)]==k)[0]):
				prob[x,y]+=1
	print prob
	prob=scipy.spatial.distance.squareform(1-prob/I)

	return(prob)


def Kmpp(data,K,outlierT=2.5,npoints=2):
	'''
	K means++ seeding; find the most extreme point of the data to start as centers (unless they are isolated)
	
	data: the dataset N*M as a numpy array
	K: the number of cluster to generate
	outlierT
	npoints
	'''
	# first point randomly seeded
	centroids=[random.randint(0,len(data)-1)]
	outliers=[]
	k=1
	while k<(K+1): # get the K farthest points as cluster centres. the +1 is to make sure the last one is not an outlier.
		print(k)
		if k==1:
			dist=dissCORT(data,data[centroids[-1]])	
			centroids.append(np.argmax(dist))
			k+=1
		else:
			disttemp=dissCORT(data,data[centroids[-1]])	
			# if a center is too far from the closest point, it is classified as outlier and cannot be seeded
			if sorted(disttemp)[npoints]>outlierT:
				# classified as outlier
				outliers.append(centroids[-1])
				# set the distance to 0 so that it does not get taken in next rounds
				dist[-1][outliers[-1]]=0
				# delete it from centroids' list
				del centroids[-1]
				dist=np.vstack((dist,disttemp))
				# get one step back
				k-=1
			else:
				dist=np.vstack((dist,disttemp))

			centroids.append(np.argmax(np.min(dist,axis=0)))
			k+=1
			
	del centroids[-1]		
	return(centroids,outliers)


def new_clsuter(distTemp,fc,N):
	'''
	check if a new cluster center
	'''
	proba=distTemp/fc
	probboulean=proba>np.random.random(len(distTemp))
	return(np.where(probboulean)[0])

def online_clustering(nameOutput,assigned,ids,datelist,bda=0.85,fc=850.,outliers=5,lbound=0.13):
	'''
	nameOutput: the name and path of the HDF5 to store the results
	assigned: the partition from the ensemble clustering
	ids: list of the ids of the customer to include in the analysis
	datelist: list of the date to run the online clustering
	bda: the lambda use in the exponential forgetting
	fc: the facility cost
	outliers: the number of points used to create a new cluster
	lbound: the smallest distance between 2 clusters centers

	'''
	# set the initial parameter K0, omega0, Gamma0, D(Omega0,Gamma0)
	K=max(assigned)+1

	data=read_format(datelist[0],ids=ids) #read_format function that reads and normalize the daily loads as (ids,time of the days) from data file(s)
	N=len(data)
	
	centroids=compute_means(data,assigned,K)
	dist=assign_points(data,centroids,K)[1]

	# store the results in a HDF5 file
	output=pd.HDFStore('./'+nameOutput+'.h5')
	output[str(0)+'/centroids']=pd.DataFrame(centroids)
	output[str(0)+'/distances']=pd.DataFrame(dist)
	output[str(0)+'/assigned']=pd.Series(assigned)
	output.close()
	# below online part
	for date in range(1,len(datelist)):
		
		print 'date '+ str(date)
		
		# load the new day data
		data=read_format(datelist[date],ids=ids) #read_format function that reads and normalize the daily loads as (ids,time of the days) from data file(s)
	N=len(data)
		
		# calculate the distance between the previous step centroids (Gamma t-1) and the new data (Omega t)
		for k in range(K):
			if k==0:

				distT=dissCORT(data,centroids[k])
			else:
				distT=np.vstack((distT,dissCORT(data,centroids[k])))

		# calculate the distance with the exponential forgetting
		distTemp=(distT+dist*bda)/(1.+bda)

		# adaptive part of the algorithm
		if date>1:
			# Monte_carlos simulations run 10000 times the calculation of the probability of each point to be above the threshold (random)
			Nprob=[]
			for i in range(10000):
				Nprob.append(new_clsuter(np.min(distTemp,axis=0),fc,N))
			
			dist2=dist
			# Check if enough customers are above the threshold
			while mode([len(i) for i in Nprob])[0]>outliers:
				# if yes create a new cluster center
				reshist=np.histogram([item for sublist in Nprob for item in sublist],bins=range(N))[0]
				centroids=np.vstack((centroids,data[np.argmax(reshist)]))

				K=K+1
				print K
				# recalculate the distance matrix with the new cluster center
				for k in range(K):
					if k==0:
						distT=dissCORT(data,centroids[k])
					else:
						distT=np.vstack((distT,dissCORT(data,centroids[k])))

			
				distTemp=np.vstack(((distT[:-1]+dist2*bda)/(1.+bda),distT[-1]))
				dist2=distTemp
				Nprob=[]
				for i in range(10000):
					Nprob.append(new_clsuter(np.min(distTemp,axis=0),fc,N))
		
		dist=distTemp	

		assigned=np.argmin(dist,axis=0)
		

		# update the distance matrix with the updated assignments
		for k in range(K):
			if k==0:
		 		distC=dissCORT(centroids,centroids[k])
			else:
		 		distC=np.vstack((distC,dissCORT(centroids,centroids[k])))

		# check if 2 clusters centers are too close
		np.fill_diagonal(distC,np.max(distC)+10)
		tooClose=sorted(np.unique(distC[distC<lbound]))
		for l in tooClose:
			coordinates=zip(*np.where(distC==l))[0]
			np.put(assigned,np.where(assigned==coordinates[1])[0],coordinates[0])
			np.put(assigned,np.where(assigned>coordinates[1])[0],[assigned[i]-1 for i in np.where(assigned>coordinates[1])[0]])

		# if yes merge them and update the assigned and the distance matrix
		count=np.array([len(np.where(assigned==k)[0])for k in range(K)])
		if np.any(count==0):
			whereCount0=np.where(count==0)[0]
			for l in whereCount0[::-1]:
				centroids=np.delete(centroids,(l),axis=0)
				K=K-1
				np.put(assigned,np.where(assigned>l)[0],[assigned[i]-1 for i in np.where(assigned>l)[0]])
				dist=np.delete(dist,(l),axis=0)
		
		
		
		print 'K = '+str(K)
		# calculate the centroids
		centroids=np.array(compute_means(data,assigned,K))
		
		# store the results
		output=pd.HDFStore('./'+nameOutput+'.h5')
		output[str(date)+'/centroids']=pd.DataFrame(centroids)
		output[str(date)+'/distances']=pd.DataFrame(dist)
		output[str(date)+'/assigned']=pd.Series(assigned)
		output.close()	



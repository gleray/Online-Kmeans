import random
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.spatial.distance
from fastdtw import fastdtw
import math
from scipy.cluster.hierarchy import dendrogram, fcluster
from itertools import chain
import csv
import sys
sys.setrecursionlimit(10000)
from fastcluster import linkage
import copy
from scipy.stats import mode 
import os

# import the HDFS (as structure in sqlquery.py) file and clean it
def read_format(filename, ids=None):
	'''
	read the data for a starting date keep only the one with min level of consumption and certain variance during the day.
	then at the following date, can take the ids from the first date and return their values, if some ids are missing fill with none.
	'''
	store=pd.HDFStore(filename) 
	data=[]

	if ids is None:
		ids=store['/ID']#str(filename)+
		toRemove=[]
		for id in ids: 
			if (np.sum(store['/refPower'][np.where(ids==id)[0]]*store['/data/'+str(id)])) and (np.var(store['/data/'+str(id)])!=0) and (all(store['/data/'+str(id)])):
				data.append(np.array(store['/data/'+str(id)]))#str(filename)+
			else:
				toRemove.append(id)


		for id in sorted([i for i in range(len(ids)) if ids[i] in toRemove],reverse=True):
			del ids[id]
		return(data,ids)

	else:
		for id in ids:
			if id in list(store['/ID']):#str(filename)+
				data.append(np.array(store['/data/'+str(id)]))#str(filename)+
			else:
				data.append([None]*24)
		return(data)
	store.close()
# randomly initialize the centroids
def init_centers(data,K):
	'''
	randomly initialize the  K centers
	'''
	return([random.randint(0,len(data)-1) for k in range(K)])

# euclidean distance
def euclidean_dist(x,y,weight=False):
	'''
	calculate the euclidean distance with or without weight between 1 matrix (or vector) and a vector (cannot be a matrix)
	'''
	if weight:
		#print y
		weight=exponential_forgetting(len(y)/24)
		dist=np.nansum(weight*(np.array(x,dtype=np.float)-np.array(y,dtype=np.float))**2,axis=1)
		#print dist
	else:
		dist=np.nansum((np.array(x,dtype=np.float)-np.array(y,dtype=np.float))**2,axis=1)

	return(dist)

# Correlation distance
def dCor(x,y,beta=None):
	'''
	The Correlation distance between 1 matrix (or vector) and a vector (cannot be a matrix)
	'''
	cor=[np.corrcoef(i,y)[0,1] for i in x]
	if beta is None:
		#dcor=np.sqrt((2*[a_i-b_i for a_i,b_i in zip([1]*len(cor),cor)]))
		dcor=2*(np.array([1]*len(cor))-np.array(cor))
	else :
		dcor=(np.array([1]*len(cor))-np.array(cor))/(np.array([1]*len(cor))+np.array(cor))**beta
	return(dcor)

# dissimilarity distance using the DTW
def dissCORT(x,y,q=2):
	'''
	dissimilarity distance using the DTW between 1 matrix (or vector) and a vector (cannot be a matrix)
	'''

	p=len(y)
	x1Index=np.ix_(range(len(x)),range(1,p))
	x2Index=np.ix_(range(len(x)),range(p-1))
	x=np.array(x)
	y=np.array(y)
	distance=np.array([fastdtw(i,y)[0] for i in x])

	
	# if vector is not constant
	corrTempOrder=np.nansum((x[x1Index]-x[x2Index])*(y[1:]-y[:(p-1)]),axis=1) / (np.sqrt( np.nansum((x[x1Index]-x[x2Index])**2,axis=1) )*np.sqrt(np.nansum((y[1:]-y[:(p-1)])**2)))
	#print distance
	#print corrTempOrder
	if np.isnan(corrTempOrder).any():
		for i in np.where(np.isnan(corrTempOrder))[0]:
			corrTempOrder[i]=0
	#print corrTempOrder
	return((2/( 1+ np.exp(q*corrTempOrder)))*distance)
	

# Calculate the distance of each point to the centroids and affect them to the closest (----DISTANCE----)
def assign_points(data,centroids,K):
	'''
	assign the points (time series) to the closest centroid
	'''
	for k in range(K):
		if k==0:
			#dist=euclidean_dist(data,centroids[k])
			#dist=dCor(data,centroids[k])
			dist=dissCORT(data,centroids[k])
		else:
			#dist=np.vstack((dist,euclidean_dist(data,centroids[k])))
			#dist=np.vstack((dist,dCor(data,centroids[k])))
			dist=np.vstack((dist,dissCORT(data,centroids[k])))

	return([np.argmin(dist,axis=0),dist]) # for each consumer in data affect it to the closest cluster.

# calculate the mean for each cluster to generate the centroids
def compute_means(dat,assigned,K):
	'''
	calculate the average for each cluster
	'''
	return(np.array([np.nanmean(np.array([dat[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0) for k in range(K)]))
	#return(np.array([mode(np.array([dat[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0,nan_policy='omit')[0].flatten() for k in range(K)]))

# calculate the sum of the distance of the points to their centroids and return the sum of it (----DISTANCE----) 
def objective_function(dat,centroids,assigned,K):
	'''
	caclculate the Within Cluster Sum of Square at each iteration
	'''
	#return(np.nansum([np.nansum([(np.array(dat[i],dtype=np.float)-np.array(centroids[k],dtype=np.float))**2 for i in np.where(assigned==k)[0]]) for k in range(K)]))
	############### Cor ###############
	# sumdcor=0
	# for k in range(K):
	#  	cor=[np.corrcoef(data[i],centroids[k])[0,1] for i in np.where(assigned==k)[0]]
	#  	sumdcor+=np.nansum(2*(np.array([1]*len(cor))-np.array(cor)))
	# return(sumdcor)
	################# dissCort ############333
	sumcort=0
	for k in range(K):
		subdata=[data[i] for i in np.where(assigned==k)[0]]
		sumcort+=np.nansum(dissCORT(subdata,centroids[k]) )
	return(sumcort)

# main clustering function
def clustering(data,centroids,K=2,eps=1.e-4):
	'''
	overall K-means function looping over up to 300 iterations

	'''	
	WCSS=[]
	# iterate up to 300 loop; stop if the Winthin sum of square is not reducing anymore (<1.e-4)
	for loop in xrange(300):
		#print(loop)
		#assign points to a cluster (center defined by centroid)
		assigned,dist=assign_points(data,centroids,K)
		#recalculate the centroid
		#print assigned
		centroids=compute_means(data,assigned,K)
		#check convergence
		#print centroids
		WCSS.append(objective_function(data,centroids,assigned,K))
		if len(WCSS)>2:
			if (WCSS[len(WCSS)-2]-WCSS[-1])<eps and (WCSS[len(WCSS)-2]-WCSS[-1])>=0:
				break
		#print(WCSS[-1])
	return([WCSS,assigned,centroids,dist])

# First K-means pick the centers using Kmpp and use adaptive to check if K the number of centroids is optimal and rerun it.
def clustering_init(K):
	'''
	First K-means pick the centers using Kmpp and use adaptive to check if K the number of centroids is optimal and rerun it.
	'''
	centres,outliers=Kmpp(data,K,outlierT=25)
	centroidsInit=[data[i] for i in centres]
	#print K
	WCSS,assigned,centroids,dist=clustering(data,centroidsInit,K,eps=3)
	#index,listdist=adaptive(assigned,dist)
	#centroids=centroids.append(data[np.argmax(listdist)])
	output=pd.HDFStore('/home/gleray/Work/2017-04-HOFOR/instances/'+str(K)+'.h5')
	output['/centroids']=pd.DataFrame(centroids)
	output['/assigned']=pd.Series(assigned)
	output['/dist']=pd.DataFrame(dist)
	output['/WCSS']=pd.Series(WCSS)
	output.close()
	#return([WCSS,assigned,centroids,dist])

def probability_distance(poolResults,N,Iterations,K):

	# create a probability distance matrix
	prob=np.zeros(shape=(N,N))
	for i in range(Iterations): 
		print(i)
		for k in range(K):
			for x,y in itertools.product(np.where(poolResults[1+(i*4)]==k)[0],np.where(poolResults[1+(i*4)]==k)[0]):
				prob[x,y]+=1
	print prob
	prob=scipy.spatial.distance.squareform(1-prob/Iterations)

	return(prob)

# finds the K most spread points in the cloud (also detect outliers) (----DISTANCE----) 
def Kmpp(data,K,outlierT=2.5,npoints=2):
	'''
	K means++ seeding; find the most extreme point of the data to start as centers (unless they are isolated)
	'''
	# first point randomly seeded
	centroids=[random.randint(0,len(data)-1)]
	outliers=[]
	k=1
	while k<(K+1): # get the K farthest points as cluster centres. the +1 is to make sure the last one is not an outlier.
		#print(k)
		if k==1:
			#dist=euclidean_dist(data,data[centroids[-1]])
			#dist=dCor(data,data[centroids[-1]])
			dist=dissCORT(data,data[centroids[-1]])	
			centroids.append(np.argmax(dist))
			k+=1
		else:
			#disttemp=euclidean_dist(data,data[centroids[-1]])
			#disttemp=dCor(data,data[centroids[-1]])
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

def plot_meanCF(data,clusters,date,ylim=[0.0,1.0]):
	#plot the results of the clustering for the 1st of january                                                 
	x=np.arange(0,np.shape(data)[1],1)
	maxClusters=max(clusters)+1
	print maxClusters
	if divmod(maxClusters,4)[1]!=0:              
		line=int(math.ceil(max(clusters)/4)+1) 
	else:
		line=int(math.ceil(int((maxClusters)/4)))
	f, axs = plt.subplots(line,4,figsize=(4*5,5*line), sharex='col', sharey='row')
	f.subplots_adjust(hspace = .2, wspace=.05)
	axs = axs.ravel()  
	for i in range(maxClusters):                                                                        
		#axs[i].axis('off')                                    
		print(i)                                                    
		Y = np.array([data[int(j)] for j in np.where(clusters==i)[0]],dtype=float)
		#Y = [mydatanorm[int(j)] for j in list(list(clusters)[16])]
		    
		axs[i].set_xlabel('Time (hour)')                  
		axs[i].set_ylabel('Power consumption (% max)')
		axs[i].set_ylim(ylim)                                                    
		axs[i].set_title('Cluster: '+str(i)+'; Size: '+str(len(Y)))                          
		# plot                            
		if len(Y)>1:
			for per in zip([5,10,25],[95,90,75]):
				axs[i].fill_between(x,np.nanpercentile(Y,per[0],axis=0),np.nanpercentile(Y,per[1],axis=0), color="#3F5D7D",alpha=0.3)                                                            
			axs[i].plot(x,np.nanmean(Y,axis=0),lw=1, color="white")
		else:
			axs[i].plot(x,np.nanmean(Y,axis=0),lw=1, color="#3F5D7D")
		#plt.show()  
	f.suptitle('date = '+date)          
	f.savefig('./'+date+'-K'+ str(maxClusters) +'.png',bbox_inches='tight')
	plt.clf()   
	plt.close()

def sankey_export(assigned,filename):
	'''
	Export the results of the online clustering into a CSV in order to produce a sankey flow chart.
	'''
	nbClust=np.append([0],np.cumsum([max(i)+1 for i in assigned]))

	count=[]
	for j in range(len(assigned)-1):
		count.append({})                                                                                   
		for i in itertools.product(range(nbClust[j],nbClust[j+1]),range(nbClust[j+1],nbClust[j+2])):
			count[j][i]=0 

		for c1,c2 in zip(assigned[j]+nbClust[j],assigned[j+1]+nbClust[j+1]): 
			count[j][(c1,c2)]+=1


	countToWrite=np.column_stack((list(chain(*count)),list(chain(*[count[i].values() for i in range(len(assigned)-1)]))))


	with open(str(filename)+'.csv', 'wb') as output_file:
	    dict_writer = csv.writer(output_file, delimiter=';', quoting=csv.QUOTE_NONE)
	    dict_writer.writerows(np.array([row for row in countToWrite if row[2]!=0]))


def datacheck(data,centroids,assigned):
	for i in range(len(data)):
		if len(data[i])<24:
			data[i]=centroids[assigned[i]]
		if (data[i]==np.array([None]*24)).all():
			data[i]=centroids[assigned[i]]
	return(data)

def new_clsuter(*args):
	distTemp=args[0]
	fc=args[1]
	N=args[2]
	proba=np.min(distTemp,axis=0)/fc
	probboulean=proba>np.random.random(N)
	return(np.where(probboulean)[0])

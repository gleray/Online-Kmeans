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

# import the HDFS (as structure in sqlquery.py) file and clean it
def read_format(filename, ids=None):
	'''
	read the data for a starting date keep only the one with min level of consumption and certain variance during the day.
	then at the following date, can take the ids from the first date and return their values, if some ids are missing fill with none.
	'''
	store=pd.HDFStore(filename) 
	data=[]

	if ids is None:
		ids=store['ID']
		toRemove=[]
		for id in ids: 
			if (np.sum(store['refPower'][np.where(ids==id)[0]]*store['data/'+str(id)])) and (np.var(store['data/'+str(id)])!=0) and (all(store['data/'+str(id)])):
				data.append(np.array(store['data/'+str(id)]))
			else:
				toRemove.append(id)


		for id in sorted([i for i in range(len(ids)) if ids[i] in toRemove],reverse=True):
			del ids[id]
		return(data,ids)

	else:
		for id in ids:
			if id in list(store['ID']):
				data.append(np.array(store['data/'+str(id)]))
			else:
				data.append([None]*24)
		return(data)

# randomly initialize the centroids
def init_centers(data,K):
	'''
	randomly initialize the  K centers
	'''
	return([[random.randint(0,len(data)-1)] for k in range(K)])

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

	corrTempOrder=np.nansum((x[x1Index]-x[x2Index])*(y[1:]-y[:(p-1)]),axis=1) / (np.sqrt( np.nansum((x[x1Index]-x[x2Index])**2,axis=1) )*np.sqrt(np.nansum((y[1:]-y[:(p-1)])**2)))
	distance=np.array([fastdtw(i,y)[0] for i in x])
	#print distance
	#print corrTempOrder
	return((2/( 1+ np.exp(q*corrTempOrder)))*distance)

# Not used now
def exponential_forgetting(window):
	'''
	generate a vector of weight calculated based on the exponential forgeting (need to adjust the slope with a coeff)
	'''
	return(list(itertools.chain.from_iterable([[0.99**(window-i)]*24 for i in range(window)])))

# Calculate the distance of each point to the centroids and affect them to the closest (----DISTANCE----)
def assign_points(data,centroids,K):
	'''
	assign the points (time series) to the closest centroid
	'''
	for k in range(K):
		if k==0:
			dist=euclidean_dist(data,centroids[k])
			#dist=dCor(data,centroids[k])
			#dist=dissCORT(data,centroids[k])
		else:
			dist=np.vstack((dist,euclidean_dist(data,centroids[k])))
			#dist=np.vstack((dist,dCor(data,centroids[k])))
			#dist=np.vstack((dist,dissCORT(data,centroids[k])))

	return([np.argmin(dist,axis=0),dist]) # for each consumer in data affect it to the closest cluster.

# calculate the mean for each cluster to generate the centroids
def compute_means(dat,assigned,K):
	'''
	calculate the average for each cluster
	'''
	return(np.array([np.nanmean(np.array([dat[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0) for k in range(K)]))

# calculate the sum of the distance of the points to their centroids and return the sum of it (----DISTANCE----) 
def objective_function(dat,centroids,assigned,K):
	'''
	caclculate the Within Cluster Sum of Square at each iteration
	'''
	return(np.nansum([np.nansum([(np.array(dat[i],dtype=np.float)-np.array(centroids[k],dtype=np.float))**2 for i in np.where(assigned==k)[0]]) for k in range(K)]))
	############### Cor ###############
	# sumdcor=0
	# for k in range(K):
	#  	cor=[np.corrcoef(data[i],centroids[k])[0,1] for i in np.where(assigned==k)[0]]
	#  	sumdcor+=np.nansum(2*(np.array([1]*len(cor))-np.array(cor)))
	# return(sumdcor)
	################# dissCort ############333
	# sumcort=0
	# for k in range(K):
	# 	subdata=[data[i] for i in np.where(assigned==k)[0]]
	# 	sumcort+=np.nansum(dissCORT(subdata,centroids[k]) )
	# return(sumcort)

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
		centroids=compute_means(data,assigned,K)
		#check convergence
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
	centres,outliers=Kmpp(data,K)
	centroidsInit=[data[i] for i in centres]
	
	WCSS,assigned,centroids,dist=clustering(data,centroidsInit,K)
	#index,listdist=adaptive(assigned,dist)
	#centroids=centroids.append(data[np.argmax(listdist)])
	return([WCSS,assigned,centroids,dist])

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
			dist=euclidean_dist(data,data[centroids[-1]])
			#dist=dCor(data,data[centroids[-1]])
			#dist=dissCORT(data,data[centroids[-1]])	
			centroids.append(np.argmax(dist))
			k+=1
		else:
			disttemp=euclidean_dist(data,data[centroids[-1]])
			#disttemp=dCor(data,data[centroids[-1]])
			#disttemp=dissCORT(data,data[centroids[-1]])	
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

def adaptive(data,assigned,dist,theta=0.001):
	'''
	adaptive split clusters that are too large into 2
	'''
	# create list of distance from each points centroids
	listdist=np.array([item for sublist in [dist[i,np.where(assigned==i)[0]] for i in range(max(assigned)+1)] for item in sublist])

	# generate the density 
	x=np.linspace(0,math.ceil(max(listdist)),1000)
	gaussiandist=stats.gaussian_kde(listdist)(x)
	thetax=[x[np.min(np.where(gaussiandist<0.001))]]

	#distcentroids=assign_points(centroids,centroids,max(assigned)+1)[1]
	#listdistcentroids=np.array([item for sublist in distcentroids for item in sublist])
	#xc=np.linspace(0,math.ceil(max(listdistcentroids)),1000)
	#gaussiandistcentroids=stats.gaussian_kde(listdistcentroids)(xc)

	# split until the variation of the intercept between theta=0.001 and the density is lower than 1 (arbitrary)
	#print(thetax)
	deltaThetax=2
	while deltaThetax>1:
		
		#plt.plot(x,gaussiandistcentroids)
		#plt.axhline(y=0.001,xmin=0,xmax=max(listdist),c="blue",linewidth=0.5,zorder=0)

		clusterSplit=np.unique(assigned[np.where(listdist>=thetax[-1])[0]])

		for toSplit in clusterSplit:
			#print(toSplit)
			dataSplit=np.array([data[i] for i in np.where(assigned==toSplit)[0]])
			centreSplit=np.array([dataSplit[i] for i in init_centers(dataSplit,K=2)])
			assignedSplit,centroidsSplit,distSlipt=clustering(dataSplit,centreSplit)[1:]
			assignedSplit[np.where(assignedSplit==1)[0]]=max(assigned)+1
			assignedSplit[np.where(assignedSplit==0)[0]]=toSplit
			index=np.where(assigned==toSplit)[0]
			
			for i in range(len(index)):
				#print(i)
				assigned[index[i]]=assignedSplit[i]


		centroids=compute_means(data,assigned,max(assigned)+1)
		dist=assign_points(data,centroids,max(assigned)+1)[1]

		listdist=np.array([item for sublist in [dist[i,np.where(assigned==i)[0]] for i in range(max(assigned)+1)] for item in sublist])

		x=np.linspace(0,math.ceil(max(listdist)),1000)
		gaussiandist=stats.gaussian_kde(listdist)(x)
		thetax=np.append(thetax,x[np.min(np.where(gaussiandist<0.001))])
		#print(thetax)
		deltaThetax=thetax[(len(thetax)-2)]-thetax[-1]
		#distcentroids=assign_points(centroids,centroids,max(assigned)+1)[1]
		#listdistcentroids=np.array([item for sublist in distcentroids for item in sublist])
		#xc=np.linspace(0,math.ceil(max(listdistcentroids)),1000)
		#gaussiandistcentroids=stats.gaussian_kde(listdistcentroids)(xc)
		#plot_meanCF(data,assigned,datelist[4])
	return([dist,assigned,centroids])

def plot_meanCF(data,clusters,date):
	#plot the results of the clustering for the 1st of january                                                 
	x=np.arange(0,np.shape(data)[1],1)
	maxClusters=max(clusters)+1
	if divmod(maxClusters,5)[1]!=0:              
		line=int(math.ceil(max(clusters)/5)+1) 
	else:
		line=int((maxClusters)/5)
	f, axs = plt.subplots(line,5,figsize=(5*line,22), sharex='col', sharey='row')
	f.subplots_adjust(hspace = .2, wspace=.05)
	axs = axs.ravel()  
	for i in range(maxClusters):                                                                        
		#axs[i].axis('off')                                    
		print(i)                                                    
		Y = np.array([data[int(j)] for j in np.where(clusters==i)[0]],dtype=float)
		#Y = [mydatanorm[int(j)] for j in list(list(clusters)[16])]
		    
		#axs[i].set_xlabel('Time (hour)')                  
		#axs[i].set_ylabel('Power consumption (% max)')                                                    
		axs[i].set_title('Cluster: '+str(i)+'; Size: '+str(len(Y)))                          
		# plot                            
		if len(Y)!=0:                                                            
			axs[i].fill_between(x,np.nanpercentile(Y,2.5,axis=0),np.nanpercentile(Y,97.5,axis=0), color="#3F5D7D")
			axs[i].plot(x,np.nanmean(Y,axis=0),lw=1, color="white")
		#plt.show()             
	f.savefig('./graphs/meanCF-online-'+date+'K'+ str(maxClusters) +'.pdf',bbox_inches='tight')
	plt.clf()   

def sankey_export(assigned,filename):
	'''
	Export the results of the online clustering into a CSV in order to produce a sankey flow chart.
	'''
	nbClust=np.append([0],np.cumsum([max(i)+1 for i in assigned]))

	count=[]
	for j in range(len(assigned)-1):
		count.append({})                                                                                   
		for i in itertools.product(range(nbClust[j],nbClust[j+1]),range(nbClust[j+1]+1),nbClust[j+2]):
			count[j][i]=0                                                                                           
		for c1,c2 in zip(assigned[j]+nbClust[j],assigned[j+1]+nbClust[j+1]): 
			count[j][(c1,c2)]+=1


	countToWrite=np.column_stack((list(chain(*count)),list(chain(*[count[i].values() for i in range(5)]))))


	with open(str(filename)+'.csv', 'wb') as output_file:
	    dict_writer = csv.writer(output_file, delimiter=';', quoting=csv.QUOTE_NONE)
	    dict_writer.writerows(np.array([row for row in countToWrite if row[2]!=0]))


#Main codes: group function above.

def main_pseudo_online(dateStart,nDays=7,window=5,shift=24,K=10,Ninit=15):

	datelist = pd.date_range(pd.to_datetime(dateStart,format='%Y-%m-%d'), periods=nDays)
	datelist = datelist.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

	for start in datelist[:(nDays-window+1)]:
		print start
		if start==datelist[0]:
			iterable=['./data/days/'+str(i)+'.h5' for i in datelist[:window]]
			for i in iterable:
				
				if i==iterable[0]:
					data,ids=read_format(i)
				else:
					data=np.hstack((data,read_format(i,ids=ids)))
			res=[clustering(K=K)]
			res[-1].append(data[:,((window-1)*24):])
		else:
			data=data[:,shift:]
			print int(np.where([i==start for i in datelist])[0])+window-1
			data=np.hstack((data,read_format('./data/days/'+str(datelist[(int(np.where([i==start for i in datelist])[0])+window-1)])+'.h5',ids=ids)))
			res.append(clustering(K=K))
			res[-1].append(data[:,((window-1)*24):])

	return(res)


def main_online(dateStart,lengthInit=1,nDays=5,K=20,bda=0.8,Ninit=30):
	'''
	dateStart		: The date at which the clustering starts
	lengthInit=1	: The length (in days) of the initiation set
	nDays=5			: The number of days to do the online clustering
	K=20			: The number of clusters for the initial solution (before adaptive is called)
	bda=0.8			: The coefficients for the exponential forgetting
	Ninit=30		: The number of run for each days of the initiation
	'''

	datelist = pd.date_range(pd.to_datetime(dateStart,format='%Y-%m-%d'), periods=nDays+lengthInit)
	datelist = datelist.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

	iterable=['/home/gleray/Work/2016-10-Radius-project/data/days/'+str(i)+'.h5' for i in datelist]

	# Initialize phase, uses the data from the lengthInit to create the clusters using Kmpp
	# Start Ninit different initialization for each date in Init
	for t in range(lengthInit):
		if t==0:
			
			global data
			data,ids=read_format(iterable[t])
			outliers=Kmpp(data,K=50,outlierT=3)[1]
			for i in outliers:
				del data[i]
				del ids[i]
			pool=Pool(processes=6)
			Iterations=[K]*Ninit
			res=pool.map(clustering_init,Iterations)
			#WCSS,assigned,centroids,dist=res[np.argmin([min(res[i][0]) for i in range(30)])]
			#plot_meanCF(data,assigned,datelist[t])

		else:
			data=read_format(iterable[t],ids=ids)
			res=np.append(res,pool.map(clustering_init,Iterations))
	
	if lengthInit==1:
		res=np.array(res).flatten()

	probaDist=probability_distance(res,N=len(ids),Iterations=Ninit*lengthInit,K=K)
	

	hac=linkage(probaDist,'ward')

	assigned=fcluster(hac,K,criterion='maxclust')-1

	centroids=compute_means(data,assigned,max(assigned)+1)
	dist=assign_points(data,centroids,max(assigned)+1)[1]
	dist,assigned,centroids=adaptive(data,assigned,dist)

	Knd=max(assigned)+1
	for t in range(lengthInit,lengthInit+nDays):		
			# for each iteration read the data and calculate the distance from centers then add the 
		data=read_format(iterable[t],ids=ids)
		
		if t==lengthInit:
			for k in range(Knd):
				if k==0:
					distTemp=euclidean_dist(data,centroids[k])
				else:
					distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[k])))
			
			dist=np.stack((dist,bda*dist+distTemp))
			assigned=np.vstack((assigned,np.argmin(dist[-1],axis=0)))
			centroids=np.stack((centroids,compute_means(data,assigned[-1],Knd)))
		else:
			for k in range(Knd):
				if k==0:
					distTemp=euclidean_dist(data,centroids[-1][k])
				else:
					distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[-1][k])))

			lastDist=bda*dist[-1]+distTemp
			dist=np.append(dist,lastDist[None,:,:],axis=0)

			assigned=np.vstack((assigned,np.argmin(dist[-1],axis=0)))

			lastCentroids=np.array(compute_means(data,assigned[-1],Knd))
			centroids=np.append(centroids,lastCentroids[None,:,:],axis=0)

		plot_meanCF(data,assigned[-1],datelist[t])

	return(dist,assigned,centroids)




###################### playground ##################
dist,assigned,centroids=main_online(
	dateStart='2015-01-12',
	nDays=12,
	Ninit=10
	)

dateStart='2015-01-12'
nDays=12
bda=0.8
K=15
## import data
datelist = pd.date_range(pd.to_datetime(dateStart,format='%Y-%m-%d'), periods=nDays)
datelist = datelist.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

iterable=['/home/gleray/Work/2016-10-Radius-project/data/days/'+str(i)+'.h5' for i in datelist]

data,ids=read_format(iterable[0])
# check fro outliers
outliers=Kmpp(data,K=10,outlierT=3.5)[1]
for i in outliers:
	del ids[i]
	del data[i]

# initialize in parallel
pool=Pool(processes=6)
Iterations=[K]*10
res=pool.map(clustering_init,Iterations)

for date in range(1,5):
	print(date)
	data=read_format(iterable[date],ids=ids)
	res=np.append(res,pool.map(clustering_init,Iterations))

#WCSS,assigned,centroids,dist=res[np.argmin([min(res[i][0]) for i in range(15)])]

# create a probability distance matrix
prob=np.zeros(shape=(len(data),len(data)))
for i in range(50):                                                                                               
	for k in range(K):                                                        
		for x,y in itertools.product(np.where(res[1+(i*4)]==k)[0],np.where(res[1+(i*4)]==k)[0]):
			prob[x,y]+=1

prob=scipy.spatial.distance.squareform(1-prob/50)


# get the best split into around 15 set
hac=linkage(prob,'ward')


# get assigned centroids and dist
assigned=fcluster(hac,16,criterion='maxclust')-1
centroids=compute_means(data,assigned,16)
dist=assign_points(data,centroids,16)[1]

plot_meanCF(data,assigned,datelist[0])

# create list of distance from each points centroids
listdist=np.array([item for sublist in [dist[i,np.where(assigned==i)[0]] for i in range(max(assigned)+1)] for item in sublist])

# generate the density 
x=np.linspace(0,math.ceil(max(listdist)),1000)
gaussiandist=stats.gaussian_kde(listdist)(x)
thetax=[x[np.min(np.where(gaussiandist<0.001))]]

distcentroids=assign_points(centroids,centroids,max(assigned)+1)[1]
listdistcentroids=np.array([item for sublist in distcentroids for item in sublist])
xc=np.linspace(0,math.ceil(max(listdistcentroids)),1000)
gaussiandistcentroids=stats.gaussian_kde(listdistcentroids)(xc)

# split until the density of theta=0.001 is lower than 3 (arbitrary)
print(thetax)
deltaThetax=2
while deltaThetax>1:
	#plt.plot(x,gaussiandistcentroids)
	#plt.axhline(y=0.001,xmin=0,xmax=max(listdist),c="blue",linewidth=0.5,zorder=0)

	clusterSplit=np.unique(assigned[np.where(listdist>=thetax[-1])[0]])

	for toSplit in clusterSplit:
		#print(toSplit)
		dataSplit=np.array([data[i] for i in np.where(assigned==toSplit)[0]])
		centreSplit=np.array([dataSplit[i] for i in init_centers(dataSplit,K=2)])
		assignedSplit,centroidsSplit,distSlipt=clustering(dataSplit,centreSplit)[1:]
		assignedSplit[np.where(assignedSplit==1)[0]]=max(assigned)+1
		assignedSplit[np.where(assignedSplit==0)[0]]=toSplit
		index=np.where(assigned==toSplit)[0]
		for i in range(len(index)):
			#print(i)
			assigned[index[i]]=assignedSplit[i]


	centroids=compute_means(data,assigned,max(assigned)+1)
	dist=assign_points(data,centroids,max(assigned)+1)[1]

	listdist=np.array([item for sublist in [dist[i,np.where(assigned==i)[0]] for i in range(max(assigned)+1)] for item in sublist])

	x=np.linspace(0,math.ceil(max(listdist)),1000)
	gaussiandist=stats.gaussian_kde(listdist)(x)
	thetax=np.append(thetax,x[np.min(np.where(gaussiandist<0.001))])
	print(thetax)

	#distcentroids=assign_points(centroids,centroids,max(assigned)+1)[1]
	#listdistcentroids=np.array([item for sublist in distcentroids for item in sublist])
	#xc=np.linspace(0,math.ceil(max(listdistcentroids)),1000)
	#gaussiandistcentroids=stats.gaussian_kde(listdistcentroids)(xc)
	plot_meanCF(data,assigned,datelist[4])
	deltaThetax=thetax[(len(thetax)-2)]-thetax[-1]



data=read_format(iterable[7],ids=ids)
K=max(assigned)+1
for k in range(K):
	if k==0:
		distTemp=euclidean_dist(data,centroids[k])
	else:
		distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[k])))

dist=np.stack((dist,bda*dist+distTemp))
assigned=np.vstack((assigned,np.argmin(dist[-1],axis=0)))
centroids=np.stack((centroids,compute_means(data,assigned[-1],K)))


for date in range(8,12):
	data=read_format(iterable[date],ids=ids)

	for k in range(K):
		if k==0:
			distTemp=euclidean_dist(data,centroids[-1][k])
		else:
			distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[-1][k])))

	lastDist=bda*dist[-1]+distTemp
	dist=np.append(dist,lastDist[None,:,:],axis=0)
	assigned=np.vstack((assigned,np.argmin(dist[-1],axis=0)))
	lastCentroids=np.array(compute_means(data,assigned[-1],K))
	centroids=np.append(centroids,lastCentroids[None,:,:],axis=0)




# format output of online K-means for sankey graph R
count=[]
for j in range(5):
	count.append({})                                                                                   
	for i in itertools.product(range((j)*53,(j+1)*53),range((j+1)*53,(j+2)*53)):
		count[j][i]=0                                                                                           
	for c1,c2 in zip(assigned[j]+j*53,assigned[j+1]+(j+1)*53): 
		count[j][(c1,c2)]+=1





countToWrite=np.column_stack((list(chain(*count)),list(chain(*[count[i].values() for i in range(5)]))))


with open('sankey.csv', 'wb') as output_file:

    dict_writer = csv.writer(output_file, delimiter=';', quoting=csv.QUOTE_NONE)
    dict_writer.writerows(np.array([row for row in countToWrite if row[2]!=0]))

#np.array([row for row in countToWrite if row[2]!=0])
import sys
sys.setrecursionlimit(10000)
from fastcluster import linkage
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(hac)
plt.savefig('./graphs/HAC-week.pdf',bbox_inches='tight')






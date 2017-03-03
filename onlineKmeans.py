import random
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
import itertools
import matplotlib.pyplot as plt
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
			if (np.sum(store['refPower'][np.where(ids==id)[0]]*store['data/'+str(id)])) and (np.var(store['data/'+str(id)])!=0):
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

def init_centers(data,K):
	'''
	randomly initialize the  K centers
	'''
	return([[random.randint(0,len(data)-1)] for k in range(K)])

def euclidean_dist(x,y,weight=False):
	'''
	calculate the euclidean distance with or without weight between 1 matrice (or vector) and a vector (cannot be a matrice)
	'''
	if weight:
		#print y
		weight=exponential_forgetting(len(y)/24)
		dist=np.nansum(weight*(np.array(x,dtype=np.float)-np.array(y,dtype=np.float))**2,axis=1)
		#print dist
	else:
		dist=np.nansum((np.array(x,dtype=np.float)-np.array(y,dtype=np.float))**2,axis=1)

	return(dist)

def exponential_forgetting(window):
	'''
	generate a vector of weight calculated based on the exponential forgeting (need to adjust the slope with a coeff)
	'''
	return(list(itertools.chain.from_iterable([[0.99**(window-i)]*24 for i in range(window)])))

def assign_points(data,centroids,K):
	'''
	assign the points (time series) to the closest centroid
	'''
	for k in range(K):
		if k==0:
			dist=euclidean_dist(data,centroids[k],weight=True)
		else:
			dist=np.vstack((dist,euclidean_dist(data,centroids[k])))

	return([np.argmin(dist,axis=0),dist]) # for each consumer in data affect it to the closest cluster.

def  compute_means(data,assigned,K):
	'''
	calculate the average for each cluster
	'''
	return([np.nanmean(np.array([data[i] for i in np.where(assigned==k)[0]],dtype=np.float),axis=0) for k in range(K)])

def objective_function(data,centroids,assigned,K):
	'''
	caclculate the Within Cluster Sum of Square at eahc iteration
	'''
	return(np.nansum([np.nansum([(np.array(data[i],dtype=np.float)-np.array(centroids[k],dtype=np.float))**2 for i in np.where(assigned==k)[0]]) for k in range(K)]))

def clustering(K):
	'''
	overall K-means function looping over up to 300 iterations

	'''
	# randomly pick the starting points
	#centroids=init_centers(data,K)
	# K means++
	centres,outliers=Kmpp(data,K)
	centroids=[data[i] for i in centres]
	WCSS=[]
	# iterate up to 300 loop; stop if the Winthin sum of square is not reducing anymore (<1.e-4)
	for i in xrange(300):
		#assign points to a cluster (center defined by centroid)
		assigned,dist=assign_points(data,centroids,K)
		#recalculate the centroid
		centroids=compute_means(data,assigned,K)
		#check convergence
		WCSS.append(objective_function(data,centroids,assigned,K))
		if len(WCSS)>2:
			if (WCSS[len(WCSS)-2]-WCSS[-1])<1.e-4 and (WCSS[len(WCSS)-2]-WCSS[-1])>=0:
				break
	return([WCSS,assigned,centroids,dist])

def Kmpp(data,K):
	'''
	K means++ seeding; find the most extreme point of the data to start as centers (unless they are isolated)
	'''
	# first point randomly seeded
	centroids=[random.randint(0,len(data)-1)]
	outliers=[]
	k=1
	while k<K: # get the k farthest points as cluster centres.
		
		if k==1:
			dist=euclidean_dist(data,data[centroids[-1]])	
			centroids.append(np.argmax(dist))
			k+=1
		else:
			disttemp=euclidean_dist(data,data[centroids[-1]])
			# if a center is too far from the closest point, it is classified as outlier and cannot be seeded
			if sorted(disttemp)[2]>3.5:
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

	return(centroids,outliers)

def main(data,K,Ninit=30):
	'''
	main function running different initialization (as the random seeding may get it stuck in a local optimum) 
	'''
	pool=Pool(processes=6)
	Iterations=[K]*Ninit
	res=pool.map(clustering,Iterations)

	return(res)

def main_pseudo_online(dateStart,nDays=7,window=5,shift=24,K=10,Ninit=15):

	dateStart='2015-01-12'
	nDays=7
	window=5
	shift=24
	K=10

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
	count1213={}
	for i in itertools.product(range(20),range(20)):
		count1213[i]=0


	for c1,c2 in zip(assigned[0],assigned[1]):
		count1213[(c1,c2)]+=1



def main_online(dateStart,nDays=5,K=20,bda=0.8,Ninit=30):
#dateStart='2015-01-12'
# nDays=5
# bda=0.8
# K=20
	datelist = pd.date_range(pd.to_datetime(dateStart,format='%Y-%m-%d'), periods=nDays)
	datelist = datelist.format(formatter=lambda x: x.strftime('%Y-%m-%d'))

	iterable=['/data/days/'+str(i)+'.h5' for i in datelist]


	for t in range(nDays):
		if t==0:
			#Initialize phase, uses the data from the 1st date to create the clusters using Kmpp
			# start Ninit different initialization and take the best one
			data,ids=read_format(iterable[t])
			pool=Pool(processes=6)
			Iterations=[K]*Ninit
			res=pool.map(clustering,Iterations)
			WCSS,assigned,centroids,dist=res[np.argmin([min(res[i][0]) for i in range(30)])]
			plotmeanCF(data,assigned,datelist[t])

		else:
			# for each iteration read the data and calculate the distance from centers then add the 
			data=read_format(iterable[t],ids=ids)
			for k in range(K):
				if k==0:
					distTemp=euclidean_dist(data,centroids[-1][k])
				else:
					distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[-1][k])))

			dist=[dist]+[bda*dist[-1]+distTemp]

			assigned=np.vstack((assigned,np.argmin(dist,axis=0)))
			centroids=[centroids]+[compute_means(data,assigned[-1],K)]

			plotmeanCF(data,assigned[-1],datelist[t])

	return(dist,assigned,centroids)





data,ids=read_format(iterable[0])
pool=Pool(processes=6)
Iterations=[K]*30
res=pool.map(clustering,Iterations)
WCSS,assigned,centroids,dist=res[np.argmin([min(res[i][0]) for i in range(30)])]
plotmeanCF(data,assigned,datelist[0])

data=read_format(iterable[3],ids=ids)
for k in range(K):
	if k==0:
		distTemp=euclidean_dist(data,centroids[-1][k])
	else:
		distTemp=np.vstack((distTemp,euclidean_dist(data,centroids[-1][k])))

dist=bda*dist+distTemp 

assigned=np.vstack((assigned,np.argmin(dist,axis=0)))
centroids=[centroids]+[compute_means(data,assigned[-1],K)]

plotmeanCF(data,assigned[-1],datelist[3])

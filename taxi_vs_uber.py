from pyspark import SparkContext
from pyspark.sql import SQLContext

import os
import sys
import csv
import urllib2
import datetime
import pandas as pd
import numpy as np
import scipy
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import reverse_geocoder as rg

from scipy import stats
from sklearn.linear_model import Lasso
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def internet_on():
	try:
		urllib2.urlopen('http://216.58.192.142', timeout=1)
		return True
	except urllib2.URLError as err: 
		return False
	
def getlocation(x):
	point_coord = (float(x[0]) , float(x[1]))
	this_location = rg.search(point_coord, mode = 1)
	print(this_location)
	if ( this_location[0]['admin2'] == 'Queens County'):
		return 'Queens'
	else :
		return this_location[0]['name']
		
def extractTaxi(partId, records):
	if partId==0:
		records.next()
	reader = csv.reader(records)
	for row in reader:
		(pickup, boro) = (row[1].split(" ")[0], getlocation((row[6],row[5])))
		if boro in ['The Bronx', 'Brooklyn', 'Queens', 'Manhattan', 'Staten Island']:
			yield ((boro, pickup) , 1)
		continue
			
def extractUber(partId, records):
	if partId==0:
		records.next()
	import csv
	reader = csv.reader(records)
	for row in reader:
		(pickup, boro) = (row[0].split(" ")[0],  getlocation((row[1],row[2])))
		if boro in ['The Bronx', 'Brooklyn', 'Queens', 'Manhattan', 'Staten Island']:
			yield ((boro, pickup) , 1)
		continue
	
# gets data given a key
def get_data(data, i):
	# returns ALL values
	if i == -1:
		return data.values().map(lambda x: list(zip(*x)[1])).collect()
	# returns ALL dates
	if i == -2:
		return data.values().map(lambda x: zip(*x)[0]).collect()
	# returns all days
	if i == -3:
		return data.values().map(lambda x: zip(*x)[0]) \
			.map(lambda dates : map(lambda date: int(datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%d")), dates)).collect()
	return list(data.values().zipWithIndex().filter(lambda (key,index) : index == i).map(lambda x: zip(*x[0])[1]).collect()[0])
	
	
##########################################################################
def get_vb(key):

	# evaluate the histogram
	tvb = list(np.histogram(get_data(trdd, key), bins=30))
	uvb = list(np.histogram(get_data(urdd, key), bins=30))
  
	#evaluate the cumulative
	tvb[0] = np.cumsum(tvb[0])
	uvb[0] = np.cumsum(uvb[0])
	
	return tvb, uvb


def get_growth(data, key):
	data = sc.parallelize(data.collect()[key][1])
	df = sqlc.createDataFrame(data, ["date", "value"])
	my_window = Window.partitionBy().orderBy("date")

	df = df.withColumn("prev_value", F.lag(df.value).over(my_window))
	df = df.withColumn("diff", F.when(F.isnull(((df.value - df.prev_value)/df.prev_value)*100), 0)
								  .otherwise((df.value - df.prev_value)/df.prev_value)*100)
	return df.rdd.map(lambda x: x.date.encode("utf-8")).collect(), df.rdd.map(lambda x: x.diff).collect()	
	
	
if __name__ == "__main__":
	sc = SparkContext()
	sqlc = SQLContext(sc)
	taxi_aug14 = sc.textFile('yellow_tripdata_2014-08.csv', use_unicode=False).filter(lambda x: x != "").cache()
	uber_aug14 = sc.textFile('uber-raw-data-aug14.csv', use_unicode=False).cache()
	
	if not internet_on():
		print 'internet not on!!!'
		sys.exit()
	
	#############################################################################
	trdd = taxi_aug14.mapPartitionsWithIndex(extractTaxi)\
			.reduceByKey(lambda x, y: x+y)\
			.sortBy(lambda x:(x[0][0], x[0][1]))\
			.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
			.reduceByKey(lambda x, y: (x+y)) \
			.map(lambda x: ('Bronx',x[1]) if x[0] == 'The Bronx' else x) \
			.sortBy(lambda x:(x[0]))


	urdd = uber_aug14.mapPartitionsWithIndex(extractUber)\
			.reduceByKey(lambda x, y: x+y)\
			.map(lambda x: ((x[0][0], datetime.datetime.strptime(x[0][1], "%m/%d/%Y").strftime("%Y-%m-%d")), x[1]))\
			.sortBy(lambda x:(x[0][0], x[0][1]))\
			.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
			.reduceByKey(lambda x, y: (x+y)) \
			.map(lambda x: ('Bronx',x[1]) if x[0] == 'The Bronx' else x) \
			.sortBy(lambda x:(x[0]))             
	
	trdd.saveAsTextFile('taxi_data')
	urdd.saveAsTextFile('uber_data')
	
	# don't use index 2 in the actual database because it is 'outside_nyc'
	lboro = urdd.map(lambda x: x[0]).collect()
	nboro = len(trdd.map(lambda x: x[0]).collect())
		
	tvbs = []
	uvbs = []
	for i in range(nboro):
		tvb, uvb = get_vb(i)
		tvbs.append(tvb)
		uvbs.append(uvb)
	
	sc.parallelize(tvbs).saveAsTextFile('taxi_cumulative_data')
	sc.parallelize(uvbs).saveAsTextFile('uber_cumulative_data')
	
	###############################################################
	datetimes = list(range(len(get_data(urdd, 0))))
	
	tdata = []
	udata = []
	for i in range(nboro):
		tdata.append(get_growth(trdd, i)[1])
		udata.append(get_growth(urdd, i)[1])
	sc.parallelize(tdata).saveAsTextFile('taxi_growth_data')
	sc.parallelize(udata).saveAsTextFile('uber_growth_data')
			
	################################################################
	prval = []
	for i in range(nboro):
		x = scipy.array(get_data(urdd, i))
		y = scipy.array(get_data(urdd, i))
		prval.append(stats.linregress(x,y))
	sc.parallelize(prval).saveAsTextFile('pr_value_data')
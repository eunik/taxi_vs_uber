from pyspark import SparkContext
from pyspark.sql import SQLContext

# imports
import csv
import pandas as pd
import numpy as np
import scipy
import datetime
import fiona
import pyproj
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point

from scipy import stats
from sklearn.linear_model import Lasso
from pyspark.sql import functions as F
from pyspark.sql.window import Window
		
def getlocation(x):
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)
    gLoc = Point(proj(float(x[0]) , float(x[1])))
    for i in range(len(boro_range)):
        if(gLoc.within(boro_range[i])):
            return lboro[i]
		
def extractTaxi(partId, records):
    if partId==0:
        records.next()
    reader = csv.reader(records)
    for row in reader:
        (pickup, boro) = (row[1].split(" ")[0], getlocation((row[5],row[6])))
        if boro in ['Bronx', 'Brooklyn','Manhattan', 'Queens', 'Staten Island']:
            yield ((boro, pickup) , 1)

			
def extractUber(partId, records):
    if partId==0:
        records.next()
    reader = csv.reader(records)
    for row in reader:
        (pickup, boro) = (row[0].split(" ")[0],  getlocation((row[2],row[1])))
        if boro in ['Bronx', 'Brooklyn', 'Queens', 'Manhattan', 'Staten Island']:
            yield ((boro, pickup) , 1)
	
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
def normalize(data):
    total = float(sum(map(abs,data)))
    if total:
        return [y / total for y in data]
    return data

def get_vb(key):
    # evaluate the histogram
    tvb = list(np.histogram(normalize(get_data(trdd, key)), bins=31))
    uvb = list(np.histogram(normalize(get_data(urdd, key)), bins=31))
   
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
	boro_shape = fiona.open('nyu_2451_34490.shp')
	
	#############################################################################
	lboro = ['Bronx', 'Brooklyn','Manhattan', 'Queens', 'Staten Island']
	boro_range = [[],[],[],[],[]]
	for i in range(5):
		boro_range[i] = shape(boro_shape[i]['geometry'])
	
	##############################################################################
	
	trdd = taxi_aug14.mapPartitionsWithIndex(extractTaxi)\
                .reduceByKey(lambda x, y: x+y)\
                .sortBy(lambda x:(x[0][0], x[0][1]))\
                .map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
                .reduceByKey(lambda x, y: (x+y)) \
                .sortBy(lambda x:(x[0]))

	urdd = uber_aug14.mapPartitionsWithIndex(extractUber)\
                .reduceByKey(lambda x, y: x+y)\
                .map(lambda x: ((x[0][0], datetime.datetime.strptime(x[0][1], "%m/%d/%Y").strftime("%Y-%m-%d")), x[1]))\
                .sortBy(lambda x:(x[0][0], x[0][1]))\
                .map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
                .reduceByKey(lambda x, y: (x+y)) \
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
	tdata = []
	udata = []
	for i in range(nboro):
		tdata.append(normalize(get_growth(trdd, i)[1]))
		udata.append(normalize(get_growth(urdd, i)[1]))
	sc.parallelize(tdata).saveAsTextFile('taxi_growth_data')
	sc.parallelize(udata).saveAsTextFile('uber_growth_data')
			
	################################################################
	prval = []

	for i in range(nboro):
		x = scipy.array(get_data(trdd, i))
		y = scipy.array(get_data(urdd, i))
		prval.append(stats.linregress(x,y))
	sc.parallelize(prval).saveAsTextFile('pr_value_data')

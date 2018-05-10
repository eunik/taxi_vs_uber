from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
import numpy as np
from geopy.distance import great_circle
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import datetime
import csv
from sklearn.linear_model import Lasso
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import scipy
from scipy import stats
import numpy as np


def get_closest_borough(latitude,longitude,max_dist = 20):
	global boroughDict
	borough_distances = {borough:great_circle(boroughDict[borough],(latitude,longitude)).miles for borough in boroughDict}
	min_borough = min(borough_distances, key=borough_distances.get)
	if borough_distances[min_borough] < max_dist:
		return min_borough 
	else:
		return "outside_nyc"
		
def extractTaxi(partId, records):
	if partId==0:
		records.next()
	reader = csv.reader(records)
	for row in reader:
		(pickup, boro) = (row[1].split(" ")[0], get_closest_borough(row[6],row[5]))
		yield ((boro, pickup) , 1)
			
def extractUber(partId, records):
	if partId==0:
		records.next()
	import csv
	reader = csv.reader(records)
	for row in reader:
		(pickup, boro) = (row[0].split(" ")[0],  get_closest_borough(row[1],row[2]))
		yield ((boro, pickup), 1)
	# gets data given a key
	
def get_data(data, key):
	# returns ALL values
	if key == -1:
		return data.values().map(lambda x: list(zip(*x)[1])).collect()
	# returns ALL dates
	if key == -2:
		return data.values().map(lambda x: zip(*x)[0]).collect()[4]
	data = zip(*data.collect()[key][1])[1]
	if data:
		return data
	print "None found"
	return []
	
	
##########################################################################
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
	queensCenter = ((40.800760+40.542920)/2,(-73.700272-73.962616)/2)
	brookCenter = ((40.739877+40.57042)/2,(-73.864754-74.04344)/2)
	bronxCenter = ((40.915255+40.785743)/2,(-73.765274-73.933406)/2)
	manhattanCenter = ((40.874663+40.701293)/2,(-73.910759-74.018721)/2)
	siCenter = ((40.651812+40.477399)/2,(-74.034547-74.259090)/2)

	boroughDict = {}
	boroughDict["queens"] = queensCenter
	boroughDict["brooklyn"] = brookCenter
	boroughDict["bronx"] = bronxCenter
	boroughDict["manhattan"] = manhattanCenter
	boroughDict["staten"] = siCenter
	# don't use index 2 in the actual database because it is 'outside_nyc'
	lboro = ['bronx', 'brooklyn', 'outside_nyc', 'queens', 'manhattan', 'staten']
	
	#############################################################################
	trdd = taxi_aug14.mapPartitionsWithIndex(extractTaxi)\
				.reduceByKey(lambda x, y: x+y)\
				.sortBy(lambda x:(x[0][0], x[0][1]))\
				.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
				.reduceByKey(lambda x, y: (x+y))


	urdd = uber_aug14.mapPartitionsWithIndex(extractUber)\
					.reduceByKey(lambda x, y: x+y)\
					.map(lambda x: ((x[0][0], datetime.datetime.strptime(x[0][1], "%m/%d/%Y").strftime("%Y-%m-%d")), x[1]))\
					.sortBy(lambda x:(x[0][0], x[0][1]))\
					.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
					.reduceByKey(lambda x, y: (x+y))             

					
	uvb = [[],[],[],[],[]]
	tvb = [[],[],[],[],[]]
	# evaluate the histogram
	tvb[0] = list(np.histogram(get_data(trdd, 0), bins=30))
	tvb[1] = list(np.histogram(get_data(trdd, 1), bins=30))
	tvb[2] = list(np.histogram(get_data(trdd, 3), bins=30))
	tvb[3] = list(np.histogram(get_data(trdd, 4), bins=30))
	tvb[4] = list(np.histogram(get_data(trdd, 5), bins=30))

	uvb[0] = list(np.histogram(get_data(urdd, 0), bins=30))
	uvb[1] = list(np.histogram(get_data(urdd, 1), bins=30))
	uvb[2] = list(np.histogram(get_data(urdd, 3), bins=30))
	uvb[3] = list(np.histogram(get_data(urdd, 4), bins=30))
	uvb[4] = list(np.histogram(get_data(urdd, 5), bins=30))

	#evaluate the cumulative
	tvb[0][0] = np.cumsum(tvb[0][0])
	tvb[1][0] = np.cumsum(tvb[1][0])
	tvb[2][0] = np.cumsum(tvb[2][0])
	tvb[3][0] = np.cumsum(tvb[3][0])
	tvb[4][0] = np.cumsum(tvb[4][0])

	uvb[0][0] = np.cumsum(uvb[0][0])
	uvb[1][0] = np.cumsum(uvb[1][0])
	uvb[2][0] = np.cumsum(uvb[2][0])
	uvb[3][0] = np.cumsum(uvb[3][0])
	uvb[4][0] = np.cumsum(uvb[4][0])
	
	sc.parallelize(tvb).saveAsTextFile('taxi_cumulative_data')
	sc.parallelize(uvb).saveAsTextFile('uber_cumulative_data')
	
	###############################################################
	datetimes = list(range(len(get_data(urdd, 0))))
	
	tdata = []
	udata = []
	for i in range(6):
		if i != 2:
			tdata.append(get_growth(trdd, i)[1])
			udata.append(get_growth(urdd, i)[1])
	sc.parallelize(tdata).saveAsTextFile('taxi_growth_data')
	sc.parallelize(udata).saveAsTextFile('uber_growth_data')
			
	################################################################
	
	prval = []
	for i in range(6):
		if i != 2:
			x = scipy.array(get_data(trdd, i))
			y = scipy.array(get_data(urdd, i))
			prval.append(stats.linregress(x,y))
	sc.parallelize(prval).saveAsTextFile('pr_value_data')
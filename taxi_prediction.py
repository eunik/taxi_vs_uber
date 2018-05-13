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
from sklearn.model_selection import train_test_split
import urllib2
import shapefile
from shapely.geometry import shape, Point

def getlocation(x):
    import pyproj
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
        if boro in lboro:
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

if __name__ == "__main__":
    #sc = SparkContext()
    taxi_aug14 = sc.textFile('taxi2014augest.csv' , use_unicode=False).filter(lambda x: x != "").cache()
	boro_shape = shapefile.Reader('../nyu_2451_34490/nyu_2451_34490.shp')
    #uber_aug14 = sc.textFile('uber-raw-data-aug14.csv', use_unicode=False).cache()

    lboro = ['Bronx', 'Brooklyn','Manhattan', 'Queens', 'Staten Island']
	boro_range = [[],[],[],[],[]]
	for i in range(5):
		boro_range[i] = shape(boro_shape.shapeRecords()[i].shape.__geo_interface__)
    columns = ['boro', 'days_in_month', 'number_of_pickup']
    dic_boro = {'Bronx': 0, 'Brooklyn':1, 'Manhattan':2, 'Queens': 3, 'Staten Island': 4}


    # urdd = uber_aug14.mapPartitionsWithIndex(extractUber)\
                # .reduceByKey(lambda x, y: x+y)\
                # .map(lambda x: ((x[0][0], datetime.datetime.strptime(x[0][1], "%m/%d/%Y").strftime("%Y-%m-%d")), x[1]))\
                # .sortBy(lambda x:(x[0][0], x[0][1]))\
                # .map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
                # .reduceByKey(lambda x, y: (x+y))   
    trdd_table = taxi_aug14.mapPartitionsWithIndex(extractTaxi)\
                .reduceByKey(lambda x, y: x+y)\
                .sortBy(lambda x:(x[0][0], x[0][1]))\
                .map(lambda x: (dic_boro[x[0][0]], 
                                int(datetime.datetime.strptime(x[0][1], "%Y-%m-%d").strftime("%d"))
                                ,x[1]))
	df_trdd = trdd_table.collect()
	df_boro = [[],[],[],[],[]]
	X = [[],[],[],[],[]]
	y =[[],[],[],[],[]]
	X_train=[[],[],[],[],[]]
	X_test=[[],[],[],[],[]]
	y_train=[[],[],[],[],[]] 
	y_test=[[],[],[],[],[]]
	predition=[[],[],[],[],[]]
	for i in range(5):
		for item in df_trdd:
			if item[0] == i:
				df_boro[i].append(item)
		df_boro[i] = pd.DataFrame(df_boro[i], columns=columns)
	for i in range(5):
		X[i] = df_boro[i][['boro', 'days_in_month']].values
		y[i] = df_boro[i][['number_of_pickup']].values
		X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.4, random_state=1)
	lasso_r = Lasso()
	for i in range(5):
		if X_train[i].any() and y_train[i].any():
			lasso_r.fit(X_train[i], y_train[i])
			predition[i] = (lboro[i], lasso_r.predict(df_boro[i][['boro', 'days_in_month']].values))
			sc.parallelize(predition[i]).saveAsTextFile(lboro[i])
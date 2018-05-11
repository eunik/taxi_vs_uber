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
import reverse_geocoder as rg
import urllib2

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

if __name__ == "__main__":
    #sc = SparkContext()
    taxi_aug14 = sc.textFile('taxi2014augest.csv' , use_unicode=False).filter(lambda x: x != "").cache()
    #uber_aug14 = sc.textFile('uber-raw-data-aug14.csv', use_unicode=False).cache()

    lboro = {0: 'The Bronx', 1: 'Brooklyn', 2: 'Queens', 3: 'Manhattan',  4:'Staten Island'}
    columns = ['boro', 'days_in_month', 'number_of_pickup']
    dic_boro = {'The Bronx': 0, 'Brooklyn':1, 'Queens': 2, 'Manhattan':3, 'Staten Island': 4}


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
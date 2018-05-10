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
	
	
if __name__ == "__main__":
	sc = SparkContext()
	taxi_aug14 = sc.textFile('taxi2014augest.csv' , use_unicode=False).filter(lambda x: x != "").cache()
	#uber_aug14 = sc.textFile('uber-raw-data-aug14.csv', use_unicode=False).cache()
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
	lboro = ['bronx', 'brooklyn', 'queens', 'manhattan', 'staten']
	columns = ['boro', 'days_in_month', 'number_of_pickup']
	dic_boro = {'bronx': 0, 'brooklyn': 1, 'manhattan': 2, 'outside_nyc': 3, 'queens': 4, 'staten': 5}
	boro = ['bronx', 'brooklyn', 'manhattan', 'outside_nyc', 'queens', 'staten']
				
	
	# urdd = uber_aug14.mapPartitionsWithIndex(extractUber)\
                # .reduceByKey(lambda x, y: x+y)\
                # .map(lambda x: ((x[0][0], datetime.datetime.strptime(x[0][1], "%m/%d/%Y").strftime("%Y-%m-%d")), x[1]))\
                # .sortBy(lambda x:(x[0][0], x[0][1]))\
                # .map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
                # .reduceByKey(lambda x, y: (x+y))   
	trdd_table = taxi_aug14.mapPartitionsWithIndex(extractUber)\
                .reduceByKey(lambda x, y: x+y)\
                .sortBy(lambda x:(x[0][0], x[0][1]))\
                .map(lambda x: (dic_boro[x[0][0]], int(datetime.datetime.strptime(x[0][1], "%Y-%m-%d").strftime("%d")),x[1]))
	df_trdd = trdd_table.collect()
	df_boro = [[],[],[],[],[],[]]
	X = [[],[],[],[],[],[]]
	y =[[],[],[],[],[],[]]
	X_train=[[],[],[],[],[],[]]
	X_test=[[],[],[],[],[],[]]
	y_train=[[],[],[],[],[],[]] 
	y_test=[[],[],[],[],[],[]]
	predition=[[],[],[],[],[],[]]	
	for i in range(6):
		for item in df_trdd:
			if item[0] == i:
				df_boro[i].append(item)
		df_boro[i] = pd.DataFrame(df_boro[i], columns=columns)
	for i in range(6):
		X[i] = df_boro[i][['boro', 'days_in_month']].values
		y[i] = df_boro[i][['number_of_pickup']].values
		X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X[i], y[i], test_size=0.4, random_state=1)
	lasso_r = Lasso()
	for i in range(6):
		lasso_r.fit(X_train[i], y_train[i])
		predition[i] = (boro[i], lasso_r.predict(df_boro[i][['boro', 'days_in_month']].values))
		predition[i].rdd.saveAsTextFile(boro[i])

	
	
	
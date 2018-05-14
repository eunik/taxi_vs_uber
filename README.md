# taxi_vs_uber

This project is purely for educational purposes, all data used are available at NYC open-data.  
This project coresponds to Yellow and Uber Taxi service.  
Hypothesis: There is an inverse correlation between NY Yellow and Uber Taxis in August 2014 for respective boroughs.  
Hypothesis: There will be a decrease of service in the days after the month of August 2014 for respective boroughs.


## Getting Started
Make sure you have pyspark available with hadoop to run the command:
```
spark-submit --num-executors 10 --executor-cores 5 --files nyu_2451_34490/nyu_2451_34490.shp,nyu_2451_34490/nyu_2451_34490.shx taxi_vs_uber.py
```

## Prerequisites
The following libraries are needed:
* python-2.7
* pandas
* numpy
* scipy
* fiona
* pyproj
* scikit-learn
* matplotlib
* shapely
* geopandas
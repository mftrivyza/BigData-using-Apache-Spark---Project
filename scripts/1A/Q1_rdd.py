from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import csv
import time
from datetime import datetime

spark = SparkSession.builder.appName("Q1_rdd").getOrCreate()
sc = spark.sparkContext
 
def getData(arg):
    line = arg.split(",")
    start_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    hour = "{:02d}".format(start_time.hour)
    longitude = float(line[3])
    latitude = float(line[4])
    return hour, (longitude, latitude, 1)

def filterData(arg):
    line = arg.split(",")
    start_time = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
    start_longitude, start_latitude = float(line[3]), float(line[4])
    end_longitude, end_latitude = float(line[5]), float(line[6])
    cost = float(line[7])
    return ((start_time < end_time) and (start_longitude > -80) and (end_longitude > -80) and (start_longitude < -70) and 
            (end_longitude < -70) and (start_latitude > 40) and (end_latitude > 40) and (start_latitude < 46) and (end_latitude < 46)
            and ((start_longitude != end_longitude) and (start_latitude != end_latitude)) and cost > 0)

# Q1 using Map-Reduce
start_time_mapreduce = time.time()

result = sc.textFile("hdfs://master:9000/yellow_tripdata_1m.csv")\
            .filter(lambda x: (filterData(x)))\
            .map(lambda x: (getData(x)))\
            .reduceByKey(lambda x,y : (x[0]+y[0], x[1]+y[1], x[2]+y[2]))\
            .map(lambda x: (x[0], x[1][0]/x[1][2], x[1][1]/x[1][2]))\
            .sortBy(lambda x: x[0]).collect()
          
print("HourOfDay    |   Longitude    |   Latitude")
for x in result:
    print(x)
print("Time of Q1 using Map-Reduce with csv is: %s seconds" % (time.time() - start_time_mapreduce)) 


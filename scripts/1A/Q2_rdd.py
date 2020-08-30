from pyspark import SparkConf, SparkContext
import os
import csv
from datetime import datetime
import math
from math import radians, cos, sin, asin, sqrt
from operator import itemgetter
from functools import partial
import time
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Q2_rdd").getOrCreate()
sc = spark.sparkContext

def haversine(long1, lat1, long2, lat2):
    a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((long2-long1)/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = 6371 * c
    return d

def getData(arg):
    line = arg.split(",")
    id_ = line[0]
    datetime_start = datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")
    datetime_end = datetime.strptime(line[2], "%Y-%m-%d %H:%M:%S")
    duration = (datetime_end - datetime_start).total_seconds() / 60.0
    distance = haversine(float(line[3]), float(line[4]), float(line[5]), float(line[6]))
    return id_, [distance, duration]

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

start_time_mapreduce = time.time()

yellow_tripvendors_1m = sc.textFile("hdfs://master:9000/yellow_tripvendors_1m.csv")\
                          .map(lambda x: x.split(","))

result = sc.textFile("hdfs://master:9000/yellow_tripdata_1m.csv")\
           .filter(lambda x: (filterData(x)))\
           .map(lambda x: (getData(x)))\
           .join(yellow_tripvendors_1m)\
           .map(lambda x: (x[1][1], x[1][0][0], x[1][0][1]))\
           .map(lambda x: (x[0], x))\
           .reduceByKey(lambda x, y: x if x[1] > y[1] else y)\
           .sortByKey()\
           .values().collect()

print("Vendor    |   Distance    |   Duration")
for x in result:
    print(x)
print("Time of Q2 using Map-Reduce with csv is: %s seconds" % (time.time() - start_time_mapreduce))

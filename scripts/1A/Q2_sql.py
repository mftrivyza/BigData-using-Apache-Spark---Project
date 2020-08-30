from pyspark import SparkConf, SparkContext
import os
import csv
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
import time
import math
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.functions import col, udf, desc
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType, TimestampType, FloatType, LongType
from pyspark.sql import SparkSession, Window

spark = SparkSession.builder.appName("Q2_sql").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

# Q2 using SQL
start_time_SQL = time.time()

schema_tripdata = StructType([StructField("ID", StringType(), False),
               	              StructField("Start_Datetime", StringType(), False),
                              StructField("End_Datetime", StringType(), False),
                              StructField("Start_Longitude", FloatType(), False),
                              StructField("Start_Latitude", FloatType(), False),
                              StructField("End_Longitude", FloatType(), False),
                              StructField("End_Latitude", FloatType(), False),
                              StructField("Cost", FloatType(), False)
                             ])

yellow_tripdata_1m = spark.read.format("csv").schema(schema_tripdata)\
                          .options(header='false', inferSchema='true')\
                          .load("hdfs://master:9000/yellow_tripdata_1m.csv")

schema_tripvendors = StructType([StructField("ID", StringType()),
                                 StructField("Vendor", StringType())
                               ])

yellow_tripvendors_1m = spark.read.format("csv").schema(schema_tripvendors)\
                             .options(header='false', inferSchema='true')\
                             .load("hdfs://master:9000/yellow_tripvendors_1m.csv")
  
yellow_tripdata_1m = yellow_tripdata_1m.where((to_timestamp(col('Start_Datetime')) < to_timestamp(col("End_Datetime"))) &\
					     ((col('Start_Longitude') != col("End_Longitude")) &\
                                             (col('Start_Latitude') != col("End_Latitude"))) &\
                                             (col('Start_Longitude') > -80) &\
                                             (col('Start_Longitude') < -70) &\
                                             (col('Start_Latitude') > 40) &\
                                             (col('Start_Latitude') < 46) &\
                                             (col('End_Longitude') > -80) &\
                                             (col('End_Longitude') < -70) &\
                                             (col('End_Latitude') > 40) &\
                                             (col('End_Latitude') < 46) &\
                                             (col('Cost') > 0))

#yellow_tripdata_1m.printSchema()
yellow_tripdata_1m = yellow_tripdata_1m.withColumn("Duration", ((unix_timestamp(col("End_Datetime")) - 
                                                                  unix_timestamp(col("Start_Datetime")))/60))\
                                       .withColumn("Diff_Longitude", col("End_Longitude") - col("Start_Longitude"))\
                                       .withColumn("Diff_Latitude", col("End_Latitude") - col("Start_Latitude"))\
                                       .withColumn("a", F.pow(F.sin(col("Diff_Latitude")/2),2) +\
                                                        F.cos(col("Start_Latitude"))*F.cos(col("End_Latitude"))*F.pow(F.sin(col("Diff_Longitude")/2),2))\
                                       .withColumn("Distance", 2 * 6371 * F.atan2(F.sqrt(col("a")), F.sqrt(1.0 - col("a"))))\
                                       .drop("Diff_Longitude").drop("Diff_Latitude").drop("Start_Datetime")\
                                       .drop("End_Datetime").drop("Start_Longitude").drop("Start_Latitude")\
                                       .drop("End_Longitude").drop("End_Latitude").drop("a").drop("Cost")

yellow_trip_joined = yellow_tripdata_1m.join(yellow_tripvendors_1m, "ID", "inner").drop("ID")
yellow_trip_joined.createOrReplaceTempView("yellow_trip_joined")

window = Window.partitionBy("Vendor")
res = yellow_trip_joined.withColumn("Max_Distance", F.max("Distance").over(window))\
                        .where(col("Distance") == col("Max_Distance"))\
                        .drop("Max_Distance").select(["Vendor", "Distance", "Duration"]) 

res.show()
print("Time of Q2 using SQL with csv is: %s seconds" % (time.time() - start_time_SQL)) 

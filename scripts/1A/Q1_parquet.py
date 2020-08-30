from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import time
from datetime import datetime

spark = SparkSession.builder.appName("Q1_parquet").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

schema_tripdata = StructType([StructField("ID", StringType(), False),
              		      StructField("Start_Datetime", StringType(), False),
			      StructField("End_Datetime", StringType(), False),
			      StructField("Start_Longitude", FloatType(), False),
			      StructField("Start_Latitude", FloatType(), False),
                              StructField("End_Longitude", FloatType(), False),
                              StructField("End_Latitude", FloatType(), False),
                              StructField("Cost", FloatType(), False)
                            ])

# Q1 using SQL with Parquet 
yellow_tripdata_1m = spark.read.format('csv').schema(schema_tripdata)\
                          .options(header='false', inferSchema='true')\
                          .load("hdfs://master:9000/yellow_tripdata_1m.csv")

# DataFrames can be saved as Parquet files, maintaining the schema information.
start_time_write_parquet = time.time()
yellow_tripdata_1m.write.parquet("hdfs://master:9000/yellow_tripdata_1m.parquet")
print("Time to write parquet is: %s seconds" % (time.time() - start_time_write_parquet))

# Read in the Parquet file created above.
# Parquet files are self-describing so the schema is preserved.
# The result of loading a parquet file is also a DataFrame.
start_time_parquet = time.time()
yellow_tripdata_1m = sqlContext.read.parquet("hdfs://master:9000/yellow_tripdata_1m.parquet")
 
# Parquet files can also be used to create a temporary view and then used in SQL statements.
yellow_tripdata_1m.createOrReplaceTempView("yellow_tripdata_1m")
res = spark.sql("""SELECT hour(to_timestamp(Start_Datetime)) AS Hour, avg(Start_Longitude) AS Longitude, avg(Start_Latitude) AS Latitude 
                   FROM yellow_tripdata_1m 
                   WHERE ((to_timestamp(Start_Datetime) < to_timestamp(End_Datetime)) AND ((Start_Longitude != End_Longitude) AND 
                   (Start_Latitude != End_Latitude)) AND Cost > 0 AND
                   (Start_Longitude > -80) AND (Start_Longitude < -70) AND (Start_Latitude > 40) AND (Start_Latitude < 46) AND
                   (End_Longitude > -80) AND (End_Longitude < -70) AND (End_Latitude > 40) AND (End_Latitude < 46))
                   GROUP BY Hour 
                   ORDER BY Hour ASC""") 

res.show(24)                 
print("Time of Q1 using SQL with parquet is: %s seconds" % (time.time() - start_time_parquet))

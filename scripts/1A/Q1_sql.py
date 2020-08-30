from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time
from datetime import datetime

spark = SparkSession.builder.appName("Q1_sql").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

#Q1 using SQL
start_time_SQL = time.time()

yellow_tripdata_1m = spark.read.format("csv").options(header='false', inferSchema='true')\
            .load("hdfs://master:9000/yellow_tripdata_1m.csv")
yellow_tripdata_1m.createOrReplaceTempView("yellow_tripdata_1m")

res = spark.sql("""SELECT hour(to_timestamp(_c1)) AS Hour, avg(_c3) AS Longitude, avg(_c4) AS Latitude 
                   FROM yellow_tripdata_1m 
                   WHERE ((to_timestamp(_c1) < to_timestamp(_c2)) AND ((_c3 != _c5) AND (_c4 != _c6)) AND 
                   (_c3 > -80) AND (_c3 < -70) AND (_c4 > 40) AND (_c4 < 46) AND
                   (_c5 > -80) AND (_c5 < -70) AND (_c6 > 40) AND (_c6 < 46) AND _c7 > 0)
                   GROUP BY Hour 
                   ORDER BY Hour ASC""") 
res.show(24)
print("Time of Q1 using SQL with csv is: %s seconds" % (time.time() - start_time_SQL))

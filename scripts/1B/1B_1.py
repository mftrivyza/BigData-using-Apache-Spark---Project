from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.conf import SparkConf
import time

spark = SparkSession.builder.appName("1B_1").getOrCreate()                                                                                 
sc = spark.sparkContext
sqlContext = SQLContext(sc) 

start_time = time.time()

yellow_tripdata_1m = sqlContext.read.parquet("hdfs://master:9000/yellow_tripdata_1m.parquet")
yellow_tripvendors_1m = sqlContext.read.parquet("hdfs://master:9000/yellow_tripvendors_1m.parquet") 

yellow_trip_joined = yellow_tripdata_1m.join(yellow_tripvendors_1m.limit(50), "ID", "inner")
yellow_trip_joined.explain()
yellow_trip_joined.show(50)

print("Execution time: %s seconds" %(time.time() - start_time))

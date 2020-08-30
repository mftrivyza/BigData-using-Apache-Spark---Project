from __future__ import print_function
import os
from datetime import datetime
import math
from math import radians, cos, sin, asin, sqrt, log
from operator import itemgetter
from functools import partial
import time

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.ml.linalg import SparseVector 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))
eng_stopwords.add('xxxx')
eng_stopwords.add('xxxxxxxx')
eng_stopwords.add('xxxxxxxxxxxx')
eng_stopwords.add('')

spark = SparkSession.builder.appName("ML").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)

def filterData(arg):
    line = arg.split(",")
    if len(line) == 3:
        date = line[0]
        productCategory = line[1]
        complaints = line[2]
        return (date.startswith("201") and bool(complaints and complaints.strip()))
    else:
        return False

def getData(arg):
    line = arg.split(",")
    date = line[0]
    productCategory = line[1]
    complaints = line[2].lower()
    return productCategory, complaints

customer_complaints = sc.textFile("hdfs://master:9000/customer_complaints.csv")\
                        .filter(filterData).map(getData).cache()
#print(customer_complaints.take(5))

unique_words = customer_complaints.flatMap(lambda x : x[1].split(" "))\
                                  .map(lambda x: re.sub('[^a-zA-Z]+', '', x))\
                                  .filter(lambda x: x.lower() not in eng_stopwords)\
                                  .map(lambda x : (x, 1))\
                                  .reduceByKey(lambda x, y: x + y)\
                                  .sortBy(lambda x : x[1], ascending = False)\
                                  .map(lambda x : x[0])

lexicon_size = 150 
#print(unique_words.take(lexicon_size))
unique_words = unique_words.take(lexicon_size)
broad_com_words = sc.broadcast(unique_words)

customer_complaints = customer_complaints.map(lambda x : (x[0], x[1].split(" ")))\
                        .map(lambda x : (x[0], [y for y in x[1] if y in broad_com_words.value]))\
                        .filter(lambda x : len(x[1]) != 0)\
                        .zipWithIndex()
                        # Output Tuple : ((string_label, list_of_sentence_words_in_lexicon), sentence_index)               
#print(customer_complaints.take(5))

number_of_complaints = customer_complaints.count()
idf = customer_complaints.flatMap(lambda x : [(y, 1) for y in set(x[0][1])])\
                         .reduceByKey(lambda x, y : x + y)\
                         .map(lambda x : (x[0], math.log(number_of_complaints/x[1])))
#print(idf.take(5))

customer_complaints = customer_complaints.flatMap(lambda x : [((y, x[0][0], x[1], len(x[0][1])), 1) for y in x[0][1]]
                                         # Output Tuple : ((word, string_label, sentence_index, sentence_length), 1)
                                         ).reduceByKey(lambda x, y : x + y
                                         # Output Tuple : ((word, string_label, sentence_index, sentence_length), word_count_in_sentence)
                                         ).map(lambda x : ((x[0][0], (x[0][1], x[0][2], x[1]/x[0][3], broad_com_words.value.index(x[0][0]))))
                                         # Output Tuple : (word, (string_label, sentence_index, word_frequency_in_sentence, word_index_in_lexicon))
                                         ).join(idf
                                         # Output Tuple : (word, ((string_label, sentence_index, word_frequency_in_sentence, word_index_in_lexicon), idf))
                                         ).map(lambda x : ((x[0], x[1][0][0], x[1][0][1]), (x[1][0][2]*x[1][1], x[1][0][3]))
                                         # Output Tuple : ((word, string_label, sentence_index), (tf*idf, word_index_in_lexicon))
                                         ).map(lambda x : ((x[0][2], x[0][1]), [(x[1][1], x[1][0])])
                                         # Output Tuple : ((sentence_index, string_label), [(word_index_in_lexicon, tfidf_in_sentence)])
                                         ).reduceByKey(lambda x, y : x + y)\
                                         .map(lambda x : (x[0][1], sorted(x[1], key = lambda y : y[0])))\
                                         .map(lambda x : (x[0], SparseVector(lexicon_size, [y[0] for y in x[1]], [y[1] for y in x[1]])))
                                         # Output Tuple : (string_label, SparseVector(lexicon_size, list_of(word_index_in_lexicon), list_of(tfidf_in_sentence)))
print(customer_complaints.take(5))

# Metatrepoume to RDD se dataframe gia na ekpaideusoume montelo tis vivliothikis SparkML kai gia diki mas dieukolinsi 
# dinoume ta katallila onomata stis stiles tou dataframe
customer_complaints_DF = customer_complaints.toDF(["string_label", "features"])

# Mesw StringIndexer Metasxhmatizoume ta string labels se akeraious 
stringIndexer = StringIndexer(inputCol="string_label", outputCol="label")
stringIndexer.setHandleInvalid("skip")
stringIndexerModel = stringIndexer.fit(customer_complaints_DF)
customer_complaints_DF = stringIndexerModel.transform(customer_complaints_DF)

#customer_complaints_DF.show(15)
customer_complaints_DF.groupBy("label").count().show()

# Diaxwrismos se train kai test set 
# Taking 70% of all labels into training set
train = customer_complaints_DF.sampleBy("label", fractions={0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75, 9: 0.75, 
							    10: 0.75, 11: 0.75, 12: 0.75, 13: 0.75, 14: 0.75, 15: 0.75, 16: 0.75, 17: 0.75}, seed = 1234)
# Subtracting 'train' from original 'customer_complaints_DF' to get test set
test = customer_complaints_DF.subtract(train)
# Checking distributions of all labels in train and test sets after sampling 
train.groupBy("label").count().show()
test.groupBy("label").count().show()

# specify layers for the neural network:
# input layer of size lexicon_size (features), one intermediate of size (lexicon_size+18)//2
# and output of size 18 (classes)
layers = [lexicon_size, (lexicon_size+18)//2, 18]

# Orismos montelou
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

# Execute ml part twice, one for not cached trainset and one for cached trainset
for i in range(0,2):
    if i == 0:
        print("No cached trainset")
        start_time = time.time()
        # Ekpaideusi sto train set kai aksiologisi sto test set
        # Fit the model
        model = trainer.fit(train)
        time_no_cache = time.time() - start_time
        print("Time of fit: %s seconds" %time_no_cache)
        # compute accuracy on the test set
        # Kanoume transform panw sto montelo to test set kai pairnoume mia nea stili sto test dataframe pou perilambanei ta predictions
        result = model.transform(test)
        # Kratame ta pragmatika labels kai ta predictions
        predictionAndLabels = result.select("prediction", "label") 
        # Orizoume enan evaluator pou mporei na upologisei to accuracy
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy") 
        # Ypologizoume kai tupwnoume to accuracy score me vash ta predictions/labels pou apomonwsame nwritera
        print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
    if i == 1:
        print("Cached trainset")
        train = train.cache()
        start_time = time.time()
        model = trainer.fit(train)
        time_cache = time.time() - start_time
        print("Time of fit: %s seconds" %time_cache)
        result = model.transform(test)
        predictionAndLabels = result.select("prediction", "label")
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
        print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

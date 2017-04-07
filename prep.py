#!/usr/bin/env python
# taste_profile_usercat_120k.txt
# usrid -> username
# ### afecb94d7a30492b9880c5f0698f07adc01ed62b_tmp_catalog --> CAERPSJ1332EA3A080

# train_triplets.txt
# userId songId count
# b80344d063b5ccb3212f76538f3d9e43d87dca9e        SODDNQT12A6D4F5F7E      5

# unique_tracks.txt
# track id <SEP> song id<SEP>artist name<SEP>song title
# TRMMMRZ128F4265EB4<SEP>SOEEHEY12CF5F88FB4<SEP>Aerosmith<SEP>I'm Ready
import sys
import timeit
import itertools
import hashlib
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


def hashUser(line):
    fields = line.strip().split("\t")
    return hash(fields[0])

def hashSongs(val):
    fields = val.strip().split("\t")
    return (fields[1], hash(fields[1]), fields[2], fields[0])

conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executor.memory", "2g")
      
sc = SparkContext(conf=conf)


 
# userIds = sc.textFile('./datasets/train_triplets.txt').map(lambda x: x[0]).distinct().map(hashUser)
outputRDD = sc.textFile('./datasets/subset.txt').keyBy(hashUser).mapValues(hashSongs)


outputRDD.coalesce(1).saveAsTextFile("./datasets/sub_user_tastes")
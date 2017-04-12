#!/usr/bin/env python3
# This file used to preprocess play count data
# turn string id to numerical id 
from pyspark import SparkConf, SparkContext
# line: 
# b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5
# (8881628083222163775, ('SOVQEYZ12A8C1379D8', 8476353328107704408, '1', 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'))
def extractUserId(line):
    fields = line.strip().split("\t")
    return fields[0]

def extractSongId(line):
    fields = line.strip().split("\t")
    return fields[1]

def extractSongId2(line):
    return line[1].split('\t')[1]


conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executor.memory", "4g")
      
sc = SparkContext(conf=conf)

input_rdd = sc.textFile('./datasets/train_triplets.txt')
user_mapping_rdd = input_rdd.keyBy(extractUserId).keys().distinct().zipWithIndex()
song_mapping_rdd = input_rdd.keyBy(extractSongId).keys().distinct().zipWithIndex()
# [('b80344d063b5ccb3212f76538f3d9e43d87dca9e', 0), ('8937134734f869debcab8f23d77465b4caaa85df', 1)]
# [('SODDNQT12A6D4F5F7E', 0), ('SOCNMUH12A6D4F6E6D', 1), ('SODACBL12A8C13C273', 2), ('SOFRDND12A58A7D6C5', 3), ('SOJQOIK12AF72A0AAF', 4), ('SOJCGJJ12A8AE48B5D', 5)]
 
temp_rdd1 = user_mapping_rdd.join(input_rdd.keyBy(extractUserId)).values()
temp_rdd2 = temp_rdd1.keyBy(extractSongId2).join(song_mapping_rdd).values()

temp_rdd2.coalesce(1).saveAsTextFile("./datasets/user_tastes")

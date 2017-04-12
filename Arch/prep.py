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


from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext

# line: 
# b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5
# (8881628083222163775, ('SOVQEYZ12A8C1379D8', 8476353328107704408, '1', 'b80344d063b5ccb3212f76538f3d9e43d87dca9e'))
def extractUserId(line):
    fields = line.strip().split(",")
    return fields[0]

def extractSongsId(line):
    fields = line.strip().split(",")
    return fields[1]

conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executor.memory", "2g")
      
sc = SparkContext(conf=conf)


 
# userIds = sc.textFile('./datasets/train_triplets.txt').map(lambda x: x[0]).distinct().map(hashUser)
# outputRDD = sc.textFile('./datasets/train_triplets.txt').keyBy(hashUser).mapValues(hashSongs)
# outputRDD = sc.textFile('./datasets/train_triplets.txt').keyBy(extractUserId)
outputRDD1 = sc.textFile('./datasets/subset.txt').keyBy(extractUserId)
outputRDD2 = outputRDD1.keys().distinct().zipWithIndex()
outputRDD3 = outputRDD2.join(outputRDD1)
outputRDD4 = outputRDD3.values()
# outputRDD1 ('b80344d063b5ccb3212f76538f3d9e43d87dca9e', 'b80344d063b5ccb3212f76538f3d9e43d87dca9e,SOCNMUH12A6D4F6E6D,1'),
print(outputRDD1.collect())
print(outputRDD2.collect())
print(outputRDD3.collect())
# (0, 'b80344d063b5ccb3212f76538f3d9e43d87dca9e,SOCNMUH12A6D4F6E6D,1'), (0, 'b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODACBL12A8C13C273,1'), 
# (0, 'b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5'), (1, '8937134734f869debcab8f23d77465b4caaa85df,SOFRDND12A58A7D6C5,5'), 
# (1, '8937134734f869debcab8f23d77465b4caaa85df,SOJCGJJ12A8AE48B5D,5'), (1, '8937134734f869debcab8f23d77465b4caaa85df,SOJQOIK12AF72A0AAF,5')
# outputRDD.coalesce(1).saveAsTextFile("./datasets/user_tastes")

# SODDNQT12A6D4F5F7E,(b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5)
outputSongRDD1 = sc.textFile('./datasets/subset.txt').keyBy(extractSongsId)
# SODDNQT12A6D4F5F7E,1
outputSongRDD2 = outputSongRDD1.keys().distinct().zipWithIndex()
# SODDNQT12A6D4F5F7E,(1,(b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5))
outputSongRDD3 = outputSongRDD2.join(outputSongRDD1)
outputSongRDD4 = outputSongRDD3.values()
# (1,(b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,5))
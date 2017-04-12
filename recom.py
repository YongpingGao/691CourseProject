#!/usr/bin/env python3

import sys
import timeit
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


def parseRating(line):
    """
    Parses a listening count record in musicData format: userId <tab> songId <tab> count
    """
    line = line.replace('(', '').replace(')', '').replace(' ', '').replace('\'', '').split(',')
    b_feedback = 1 if int(line[3]) >= 3 else 0
    return (int(line[0]), int(line[4]), b_feedback)

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print("File {0} does not exist.".format(ratingsFile))
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print("No ratings provided.")
        sys.exit(1)
    else:
        return ratings

# TRMMMKQ128F92EBCB5<SEP>SOOLRHW12A8C142643<SEP>Kreator<SEP>All of the same blood
def parseSong(line):
    """
    Parses a song record in musicData format: track id <SEP> song id<SEP>artist name<SEP>song title
    """
    fields = line.strip().split("<SEP>")
    return (fields[1], fields[3])

def parseSongMapping(line):
    fields = line.replace('(', '').replace(')', '').replace(' ', '').replace('\'', '').split(',')
    return (fields[0], fields[1])

if __name__ == "__main__":


    mySongsPlayed = [
        (0, 16521,  1),
        (0, 132453, 1),
        (0, 151792, 1),
        (0, 153114, 1),
        (0, 200274, 2),
        (0, 129553, 1),
        (0, 251209, 1),
        (0, 252088, 2),
        (0, 216398, 2),
        (0, 289909, 1),
        (0, 362800, 2)
    ]

    # set up environment
    conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executorEnv.PYTHONHASHSEED","123")
    sc = SparkContext(conf=conf)

    # songs = (1, Before He Kissed Me)
    # songmaping = ('SOFXVZW12AB018181E', 17)
    # movies is an RDD of (songId, songTitle)
    songids = sc.textFile("s3n://spark-project-file/user_tastes/unique_tracks.txt").map(parseSong)
    song_mapping = sc.textFile("s3n://spark-project-file/user_tastes/song_mapping/part-00000").map(parseSongMapping)


    # (17, Before He Kissed Me)
    songs = dict(song_mapping.join(songids).values().collect())
    # load model
    bestModel = MatrixFactorizationModel.load(sc, "./RecommenderModel")
    # load personal ratings
    myRatingsRDD = sc.parallelize(mySongsPlayed, 1)

    # make personalized recommendations
    myRatedSongIds = set([x[1] for x in mySongsPlayed])
    candidates = sc.parallelize([m for m in songs if m not in myRatedSongIds])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]
    print("UserId 0' s play records: ")

    for song in mySongsPlayed:
        print("\"{0}\" is played {1} times".format(songs.get(str(song[1]), "None"), song[2]))

    print("Songs recommended for userid 0:")
    for i in range(len(recommendations)):
        print("{0:2d}: {1:s}".format(i + 1, songs.get(str(recommendations[i][1]))))
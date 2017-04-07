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
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
    """
    Parses a listening count record in musicData format: userId <tab> songId <tab> count
    """
    fields = line.strip().split('\t')
    return (fields[0], fields[1], int(fields[2]))

def parseSong(line):
    """
    Parses a song record in musicData format: track id <SEP> song id<SEP>artist name<SEP>song title
    """
    fields = line.strip().split("<SEP>")
    return (fields[1], fields[3])

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

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])).join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":
    # if (len(sys.argv) != 3):
    #     print("Usage: [usb root directory]/spark/bin/spark-submit --driver-memory 2g " + \
    #       "recom.py musicDataDir personalRatingsFile")
    #     sys.exit(1)

    # set up environment

    start = timeit.default_timer()

    conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # load personal ratings
    # myRatings = loadRatings(sys.argv[2])
    # myRatingsRDD = sc.parallelize(myRatings, 1)
    
    # load ratings and movie titles

    musicDataDir = sys.argv[1]

    # ratings is an RDD of (userId, songId, count)
    ratings = sc.textFile(join(musicDataDir, "train_triplets.txt")).map(parseRating)

    # movies is an RDD of (songId, songTitle)
    songs = dict(sc.textFile(join(musicDataDir, "unique_tracks.txt")).map(parseSong).collect())


    numRatings = ratings.count()
    numUsers = ratings.map(lambda r: r[0]).distinct().count()
    numSongs = ratings.map(lambda r: r[1]).distinct().count()


    print ("Got {0} ratings from {1} users on {2} songs.".format(numRatings, numUsers, numSongs))
    print("------------------------------------------------------------------------------------------")
   
   
   
    # numPartitions = 4
    # training = ratings.filter(lambda x: x[0] < 6).values().union(myRatingsRDD).repartition(numPartitions).cache()
    # validation = ratings.filter(lambda x: x[0] >= 6 and x[0] < 8).values().repartition(numPartitions).cache()
    # test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    # numTraining = training.count()
    # numValidation = validation.count()
    # numTest = test.count()

    # print("Training: {0}, validation: {1}, test: {2}".format(numTraining, numValidation, numTest))
    # print("------------------------------------------------------------------------------------------")
    # ranks = [8, 12]
    # lambdas = [0.1, 10.0]
    # numIters = [10, 20]
    # bestModel = None
    # bestValidationRmse = float("inf")
    # bestRank = 0
    # bestLambda = -1.0
    # bestNumIter = -1

    # for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    #     model = ALS.train(training, rank, numIter, lmbda)
    #     validationRmse = computeRmse(model, validation, numValidation)
    #     print("RMSE (validation) = {0:f} for the model trained with rank = {1:d}, lambda = {2:.1f}, and numIter = {3:d}.".format(validationRmse, rank, lmbda, numIter))
    #     if (validationRmse < bestValidationRmse):
    #         bestModel = model
    #         bestValidationRmse = validationRmse
    #         bestRank = rank
    #         bestLambda = lmbda
    #         bestNumIter = numIter

    # testRmse = computeRmse(bestModel, test, numTest)

    # # evaluate the best model on the test set
    # print("The best model was trained with rank = {0:d} and lambda = {1:.1f}, and numIter = {2:d}, and its RMSE on the test set is {3:f}.".format(bestRank, bestLambda, bestNumIter, testRmse))

    # # compare the best model with a naive baseline that always returns the mean rating
    # meanRating = training.union(validation).map(lambda x: x[2]).mean()
    # baselineRmse = sqrt(test.map(lambda x: (meanRating - x[2]) ** 2).reduce(add) / numTest)
    # improvement = (baselineRmse - testRmse) / baselineRmse * 100
    # print("The best model improves the baseline by {0:.2f}%.".format(improvement))

    # # make personalized recommendations

    # myRatedMovieIds = set([x[1] for x in myRatings])
    # candidates = sc.parallelize([m for m in movies if m not in myRatedMovieIds])
    # predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    # recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    # print("Movies recommended for you:")
    # for i in range(len(recommendations)):
    #     print("{0:2d}: {1:s}".format(i + 1, movies[recommendations[i][1]]))

    # clean up
    sc.stop()

    stop = timeit.default_timer()

    print("Time elaps: {0}".format(stop - start))
#!/usr/bin/env python3
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
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel

# ((0, 'b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,3'), 0),
# 0,b80344d063b5ccb3212f76538f3d9e43d87dca9e,SODDNQT12A6D4F5F7E,3,0,

def parseRating(line):
    """
    Parses a listening count record in musicData format: userId <tab> songId <tab> count
    """
    line = line.replace('(', '').replace(')', '').replace(' ', '').replace('\'', '').split(',')
    b_feedback = 1 if int(line[3]) >= 3 else 0
    return (int(line[0]), int(line[4]), b_feedback)

# computeRmse(model, validation, numValidation)
def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])).join(data.map(lambda x: ((x[0], x[1]), x[2]))).values()

    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":

    # set up environment
    conf = SparkConf() \
      .setAppName("Music Recommender") \
      .set("spark.executorEnv.PYTHONHASHSEED","123")
    sc = SparkContext(conf=conf)

    ratings = sc.textFile("s3n://spark-project-file/user_tastes/part-00000").cache().map(parseRating)

    numRatings = ratings.count()
    numUsers = ratings.map(lambda r: r[0]).distinct().count()
    numSongs = ratings.map(lambda r: r[1]).distinct().count()
    print ("Got {0} ratings from {1} users on {2} songs.".format(numRatings, numUsers, numSongs))

    (training, validation, test) = ratings.randomSplit([0.6, 0.2, 0.2])

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print("Training: {0}, validation: {1}, test: {2}".format(numTraining, numValidation, numTest))

    ranks = [5, 10]
    lambdas = [0.1, 0.01]
    numIters = [5, 10]
    alphas = [0.01, 0.1]

    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    bestAlpha = -1.0

    for rank, lmbda, numIter, alpha in itertools.product(ranks, lambdas, numIters, alphas):
        # Build model
        model = ALS.trainImplicit(training, rank, numIter, lmbda, alpha=alpha)
        validationRmse = computeRmse(model, validation, numValidation)
        print("RMSE (validation) = {0:f} for the model trained with rank = {1:d}, lambda = {2:.2f}, numIter = {3:d}, and alpha = {4:.1f}".format(validationRmse, rank, lmbda, numIter, alpha))
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
            bestAlpha = alpha
    testRmse = computeRmse(bestModel, test, numTest)

    # evaluate the best model on the test set
    print("The best model was trained with rank = {0:d} and lambda = {1:.2f}, numIter = {2:d}, and alpha = {3: 1f} and its RMSE on the test set is {4:f}.".format(bestRank, bestLambda, bestNumIter, bestAlpha, testRmse))


    # Save and load model
    modelDir = "./RecommenderModel"
    bestModel.save(sc, modelDir)
    print("Best model is saved to {0}".format(modelDir))
    
    # clean up
    sc.stop()
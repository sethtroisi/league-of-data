'''
import argparse
import functools
import matplotlib.pyplot as pyplot
import random
import re
import sklearn.metrics
import time

from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split
from Util import *
import GraphModelStats
import TFFeaturize

import tensorflow as tf


def getArgParse():
    parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

    parser.add_argument(
        '-i', '--input-file',
        type=str,
        default='features.json',
        help='Input match file (produced by Seth or GameParser.py)')

    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=-1,
        help='Numbers of games to load (default -1 = all)')

    parser.add_argument(
        '-p', '--holdback',
        type=int,
        default=15,
        help='percent of games to holdback for testing/validation')

    parser.add_argument(
        '-v', '--verbose',
        type=int,
        default=0,
        help='how much data to print (0 = little, 1 = most, 3 = everything)')

    return parser


def filterMaxBlock(blockNum, games, goals):
    blockStart = blockNum * TFFeaturize.SECONDS_PER_BLOCK

    inds, gas, gos = [], [], []
    for i, (ga, go) in enumerate(zip(games, goals)):
        if blockStart > ga['debug']['duration']:
            continue

        inds.append(i)
        gas.append(ga)
        gos.append(go)

    return inds, gas, gos

featurizerTime = 0
trainTime = 0
def gameToPDF(args, games, *, featuresToExport = None, blockNum = 0, training = False):
    global featurizerTime, trainTime

    if training and args.verbose >= 2:
        print ("\tfeaturizing {} games".format(len(games)))

    T0 = time.time()

    frames = []
    for index, game in enumerate(games):
        # NOTE for some models it might make sense to use gameTime = game['debug']['duration'] or 7200
        gameTime = blockNum * TFFeaturize.SECONDS_PER_BLOCK

        gameData = TFFeaturize.parseGame(index, game, gameTime)
        frames.append(gameData)

    T1 = time.time()
    featurizerTime += T1 - T0

    if training and args.verbose >= 1:
        print ("\t\tFeaturize Timing: {:.1f}".format(T1 - T0))

    if training:
        allColumns = set()
        for frame in frames:
            allColumns.update(frame.keys())

    if training:
        return frames, allColumns
    return frames

def getPrediction(args, classifiers, featuresUses, testGames, blockNum, testGoals):
    blockIndex = min(blockNum, len(classifiers) - 1)
    classifier = classifiers[blockIndex]
    featuresUsed = featuresUses[blockIndex]

    df = gameToPDF(args, testGames, blockNum = blockNum, featuresToExport = featuresUsed)
    modelGuess = classifier.predict_proba(
        input_fn = functools.partial(inputFn, featuresUsed, df, testGoals))

    for testGoal in testGoals:
        probs = next(modelGuess)

        # This is due to the sorting of [False, True].
        BProb, AProb = probs
        correct = (AProb > 0.5) == testGoal
        yield correct, [BProb, AProb]


def main(args):
    global featurizerTime, trainTime

    if args.verbose == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose == 1:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose >= 2:
        tf.logging.set_verbosity(tf.logging.INFO)

    T0 = time.time()

    MAX_BLOCKS = int(3600 // TFFeaturize.SECONDS_PER_BLOCK) + 1

    games, goals = TFFeaturize.getRawGameData(args.input_file, args.num_games)

    T1 = time.time()
    loadTime = T1 - T0

    trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
        games,
        goals,
        test_size = args.holdback / 100,
        random_state = 42)

    del games, goals

    if args.verbose == 0:
        print ("Training games: {}, Testing holdback: {}".format(
            len(trainingGames), len(testingGames)))

    assert len(trainingGames) == len(trainingGoals)
    assert len(testingGames) == len(testingGoals)

    T2 = time.time()
    splitTime = T2 - T1

    classifiers, featuresUses = buildClassifier(
        args,
        MAX_BLOCKS,
        trainingGames, trainingGoals, testingGames, testingGoals)

    T3 = time.time()
    innerTrainTime = T3 - T2

    # Variables about testGames.
    times = [(b * TFFeaturize.SECONDS_PER_BLOCK) / 60 for b in range(MAX_BLOCKS)]
    samples = [0 for b in range(MAX_BLOCKS)]
    corrects = [0 for b in range(MAX_BLOCKS)]
    # Averages over data (calculated after all data).
    ratios = [0 for b in range(MAX_BLOCKS)]
    logLosses = [0 for b in range(MAX_BLOCKS)]
    # Per Game stats.
    testWinProbs = [[] for a in range(len(testingGames))]

    for blockNum in range(MAX_BLOCKS):
        gameIs, testingBlockGames, testingBlockGoals = \
            filterMaxBlock(blockNum, testingGames, testingGoals)

        # Do all predictions at the sametime (needed because of how predict reloads the model each time).
        preditions = getPrediction(args, classifiers, featuresUses, testingBlockGames, blockNum, testingBlockGoals)

        for gameI, predition in zip(gameIs, preditions):
            correct, gamePredictions = predition

            # store data to graph
            samples[blockNum] += 1
            corrects[blockNum] += 1 if correct else 0
            testWinProbs[gameI].append(gamePredictions)

    for blockNum in range(MAX_BLOCKS):
        if samples[blockNum] > 0:
            ratios[blockNum] = corrects[blockNum] / samples[blockNum]

            goals = []
            predictions = []
            for testGoal, gamePredictions in zip(testingGoals, testWinProbs):
                if len(gamePredictions) > blockNum:
                    goals.append(testGoal)
                    predictions.append(gamePredictions[blockNum])

            logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions, labels = [True, False])

    T4 = time.time()
    statsTime = T4 - T3

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
        GraphModelStats.stats(times, samples, corrects, ratios, logLosses)
        GraphModelStats.plotData(times, samples, corrects, ratios, logLosses)
        GraphModelStats.plotGame(times, testingGoals, testWinProbs)

    T5 = time.time()
    viewTime = T5 - T4

    print ("Timings:")
    print ("\tloadTime: {:.3f}".format(loadTime))
    print ("\tsplitTime: {:.3f}".format(splitTime))
    print ("\ttrainTime: {:.3f}".format(innerTrainTime))
    print ("\tstatsTime: {:.3f}".format(statsTime))
    print ("\tviewTime: {:.3f}".format(viewTime))
    print ()
    print ("\tfeaturizerTime: {:.3f}".format(featurizerTime))
    print ("\ttrainTime: {:.3f}".format(trainTime))


if __name__ == '__main__':
    args = getArgParse().parse_args()
    main(args)
'''
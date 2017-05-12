import argparse
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as pyplot
import random

from matplotlib.widgets import Slider
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from BoolFeaturize import *
from Util import *
import GraphModelStats

def getArgParse():
  parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='features.json',
      help='Input match file (produced by Seth or GameParser.py)')

  # TODO(sethtroisi): Add and utilize a flag for verbosity.

  return parser


def buildClassifiers(numBlocks, trainGoals, trainGames, vectorizer):
  #SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
  #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
  #       loss='hinge', n_iter=2, n_jobs=1, penalty='l2', power_t=0.5,
  #       random_state=None, shuffle=False, verbose=True, warm_start=False)
  #clf = SGDClassifier(loss="log", penalty="l2", n_iter=3000, shuffle=True,
  #  alpha = 0.02, verbose=False)

  clfs = []
  for blockNum in range(min(21, numBlocks)):
    time = blockNum * SECONDS_PER_BLOCK

#    clf = SGDClassifier(loss="log", penalty="l2", n_iter=3000, shuffle=True,
#        alpha = 0.02, verbose=False)

#    '''
    clf = MLPClassifier(
        solver='adam',
        max_iter = 150,
        alpha = 0.5,
        learning_rate_init = 0.001,
        hidden_layer_sizes = (10, 4),
#        early_stopping = True,
#        validation_fraction = 0.1,
        verbose = True)
    #'''

    subTrainGoals= []
    subTrainFeatures = []
    for goal, game in zip(trainGoals, trainGames):
      duration = game['debug']['duration']
      if duration < time:
        continue

      subTrainGoals.append(goal)
      gameFeatures = parseGameToFeatures(game, time)
      subTrainFeatures.append(gameFeatures)

    if len(subTrainGoals) <= 0:
      break

    print ("training on {} datums (block {})".format(len(subTrainGoals), blockNum))
    sparseFeatures = vectorizer.transform(subTrainFeatures)
    clf.fit(sparseFeatures, subTrainGoals)
    clfs.append(clf)

#    print ("clf {:2d}: loss: {:.3f} after {:4d} iters".format(
#      blockNum, clf.loss_, clf.n_iter_))


  #print ("With training set size: {} games {} features - {} nnz".format(
  #    len(trainGoals), trainFeatures.shape[1], trainFeatures.nnz))


  #print (clf.coef_)
  #print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
  #    clf.intercept_[0], 100 * trainGoals.count(True) / len(trainGoals)))
  #print ()

  return clfs


def getPrediction(classifiers, gameI, blockNum, testGoal, testFeature):
  if blockNum >= len(classifiers):
    return None, None    
  modelGuesses = classifiers[blockNum].predict_proba(testFeature)

  # This is due to the sorting of [False, True].
  BProb, AProb = modelGuesses[0]
  correct = (AProb > 0.5) == testGoal
  return correct, [BProb, AProb]



def main(args):
    MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

    games, goals, vectorizer, features = getGamesData(args.input_file)

    trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
        games,
        goals,
        test_size = 0.15,
        random_state = 42)

    print ("Training games: {}, Testing holdback: {}".format(
      len(trainingGames), len(testingGames)))
    assert len(trainingGames) == len(trainingGoals)
    assert len(testingGames) == len(testingGoals)

    classifiers = buildClassifiers(MAX_BLOCKS, trainingGoals, trainingGames, vectorizer)

    # Variables about testGames.
    times = [(b * SECONDS_PER_BLOCK) / 60 for b in range(MAX_BLOCKS)]
    samples = [0 for b in range(MAX_BLOCKS)]
    corrects = [0 for b in range(MAX_BLOCKS)]
    # Averages over data (calculated after all data).
    ratios = [0 for b in range(MAX_BLOCKS)]
    logLosses = [0 for b in range(MAX_BLOCKS)]
    # Per Game stats.
    testGoals = []
    testWinProbs = []

    for gameI, (game, tGoal) in enumerate(zip(testingGames, testingGoals)):
      duration = game['debug']['duration']
      goal = game['goal']
      assert tGoal == goal

      predictions = []
      # TODO(sethtroisi): determine this point algorimically as 80% for game end.
      # TODO(sethtroisi): alternatively truncate when samples < 50.
      for blockNum in range(MAX_BLOCKS):
        time = blockNum * SECONDS_PER_BLOCK

        # TODO(sethtroisi): remove games that have ended.
        if duration < time:
          break

        gameFeatures = parseGameToFeatures(game, time)

        sparse = vectorizer.transform(gameFeatures)

        correct, gamePredictions = getPrediction(classifiers, gameI, blockNum, goal, sparse)

        if correct == None:
          continue

        # store data to graph
        samples[blockNum] += 1
        corrects[blockNum] += 1 if correct else 0
        predictions.append(gamePredictions)

      testGoals.append(goal)
      testWinProbs.append(predictions)

    for blockNum in range(MAX_BLOCKS):
      if samples[blockNum] > 0:
        ratios[blockNum] = corrects[blockNum] / samples[blockNum]

        goals = []
        predictions = []
        for testGoal, gamePredictions in zip(testGoals, testWinProbs):
          if len(gamePredictions) > blockNum:
            goals.append(testGoal)
            predictions.append(gamePredictions[blockNum])

        if len(set(goals)) <= 1:
          break
        logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions)

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
      GraphModelStats.stats(times, samples, corrects, ratios, logLosses)
      GraphModelStats.plotData(times, samples, corrects, ratios, logLosses)
      GraphModelStats.plotGame(times, testGoals, testWinProbs)


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

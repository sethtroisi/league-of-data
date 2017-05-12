import argparse
import functools

import matplotlib.pyplot as pyplot
import random
import pandas
import sklearn.metrics
import tensorflow as tf
import time

from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split
from BoolFeaturize import *
from Util import *
import GraphModelStats
import TFFeaturize

def getArgParse():
  parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='features.json',
      help='Input match file (produced by Seth or GameParser.py)')

  # TODO(sethtroisi): Add and utilize a flag for verbosity.

  return parser


allColumns = []
def gameToPDF(games, *, blockNum = 0, training = False):
  global allColumns

  if training:
    print ("featurizing {} games".format(len(games)))


  frames = []
  for index, game in enumerate(games):
    if training:
      time = game['debug']['duration']
    else:
      time = blockNum * SECONDS_PER_BLOCK

    gameFrame = TFFeaturize.parseGameToPD(index, game, time)
    frames.append(gameFrame)

  if training:
    print ("joining {} games".format(len(games)))

  test = pandas.concat(frames).fillna(0)

  if training:
    allColumns = list(test.columns.values)
    print ("saving {} feature columns".format(len(allColumns)))
  else:
    curCols = set(test.columns.values)
    for col in allColumns:
      if col not in curCols:
        test[col] = 0
    
#  print ("df shape:", test.shape)
#  print ("allColumns:", len(allColumns))
  return test


def inputFn(df, goals = None):
  global allColumns
  featureCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in allColumns}
  print ("input:", df.shape, len(allColumns))
  if goals == None:
    return  featureCols
  labels = tf.constant(goals, shape=[len(goals), 1])
  return featureCols, labels


def buildClassifier(trainGoals, trainGames):
  global allColumns
  df = gameToPDF(trainGames, training = True)

  params = {
    'dropout': 0.7,
    'learningRate': 0.001,
    'steps': 250
  }

  featureColumns = [tf.contrib.layers.real_valued_column(k) for k in allColumns]
  print ("featureColumns:", len(featureColumns))

  optimizer = tf.train.AdamOptimizer(learning_rate = params['learningRate'])
  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns = featureColumns,
      hidden_units = [20, 10],
      n_classes = 2,
      dropout = params['dropout'],
      optimizer = optimizer,
  )

  tf.logging.set_verbosity(tf.logging.INFO)

  classifier.fit(
      input_fn = functools.partial(inputFn, df, trainGoals),
      steps = params['steps'])
  
  return classifier


def getPrediction(classifier, testGames, blockNum, testGoals):
  df = gameToPDF(testGames, blockNum = blockNum)
  modelGuess = classifier.predict_proba(
      input_fn = functools.partial(inputFn, df, testGoals))

  for testGoal in testGoals:
    probs = next(modelGuess)
    
    # This is due to the sorting of [False, True].
    BProb, AProb = probs
    correct = (AProb > 0.5) == testGoal
    yield correct, [BProb, AProb]


def main(args):
  MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

  games, goals = TFFeaturize.getRawGameData(args.input_file)
  trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
      games,
      goals,
      test_size = 0.15,
      random_state = 42)
  del games, goals

  print ("Training games: {}, Testing holdback: {}".format(
    len(trainingGames), len(testingGames)))
  assert len(trainingGames) == len(trainingGoals)
  assert len(testingGames) == len(testingGoals)

  classifier = buildClassifier(trainingGoals, trainingGames)

  # Variables about testGames.
  times = [(b * SECONDS_PER_BLOCK) / 60 for b in range(MAX_BLOCKS)]
  samples = [0 for b in range(MAX_BLOCKS)]
  corrects = [0 for b in range(MAX_BLOCKS)]
  # Averages over data (calculated after all data).
  ratios = [0 for b in range(MAX_BLOCKS)]
  logLosses = [0 for b in range(MAX_BLOCKS)]
  # Per Game stats.
  testWinProbs = [[] for a in range(len(testingGames))]

  for blockNum in range(MAX_BLOCKS):
    time = blockNum * SECONDS_PER_BLOCK

    gameIs = []
    partialGames = []
    partialGoals = []
    for gameI, game in enumerate(testingGames):
      duration = game['debug']['duration']
      if duration < time:
        continue

      gameIs.append(gameI)
      partialGames.append(game)
      partialGoals.append(game['goal'])

    # Do all predictions at the sametime (needed because of how predict reloads the model each time).
    preditions = getPrediction(classifier, partialGames, blockNum, partialGoals)
    for gameI, partialGoal, predition in zip(gameIs, partialGoals, preditions):
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

  # If data was tabulated on the testingData print stats about it.
  if len(times) > 0:
    GraphModelStats.stats(times, samples, corrects, ratios, logLosses)
    GraphModelStats.plotData(times, samples, corrects, ratios, logLosses)
    GraphModelStats.plotGame(times, testingGoals, testWinProbs)


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

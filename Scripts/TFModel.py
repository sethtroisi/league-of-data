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

def getArgParse():
  parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='features.json',
      help='Input match file (produced by Seth or GameParser.py)')

  # TODO(sethtroisi): Add and utilize a flag for verbosity.

  return parser


vectorizer = DictVectorizer(sparse = False)
def gameToPDF(games, *, blockNum = 0, training = False):
  features = []
  for game in games:
    if training:
      time = game['debug']['duration']
    else:
      time = blockNum * SECONDS_PER_BLOCK

    gameFeatures = parseGameToFeatures(game, time)
    features.append(gameFeatures)
    assert len(gameFeatures) > 0, time
  assert len(features) > 0

  if training:
    print ("Rebuilding feature transformer")
    sparse = vectorizer.fit_transform(features)
    print ("training on {} datums, {} features".format(
        len(game),
        len(vectorizer.get_feature_names())))
  else:
    sparse = vectorizer.transform(features)

  return pandas.DataFrame(
      data = sparse,
      index = range(len(games)),
      columns = vectorizer.get_feature_names())


def inputFn(df, goals = None):
  featureCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in df.columns.values}
  if goals == None:
    return  featureCols
  labels = tf.constant(goals, shape=[len(goals), 1])
  return featureCols, labels


def buildClassifier(trainGoals, trainGames):
  df = gameToPDF(trainGames, training = True)
  features = vectorizer.get_feature_names()

  featureColumns = [tf.contrib.layers.real_valued_column(k) for k in features]

  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns = featureColumns,
      hidden_units = [10, 10],
      n_classes = 2)
     # model_dir = "/tmp/lodm/")


  tf.logging.set_verbosity(tf.logging.INFO)

  classifier.fit(
      input_fn = functools.partial(inputFn, df, trainGoals),
      steps = 500)
  
#    print ("clf {:2d}: loss: {:.3f} after {:4d} iters".format(
#      blockNum, clf.loss_, clf.n_iter_))

  return classifier


''' Gragabe that didn't work :(
def getPredictorFunction(classifier, warmupGame):
  game = warmupGame
  blockNum = 10
  def updatingInputFn():
    print ("Hi!", len(str(game)))
    while game != None:
      df = gameToPDF([warmupGame], blockNum = 10)
#      game = None
      return inputFn(df, None)

  modelGuess = classifier.predict_proba(
      input_fn = updatingInputFn,
      as_iterable=True)

#  print (modelGuess)

  def predict(testGame, testBlockNum, testGoal):
    # changes game which will now be consumed by updatingInputFn
    game = testGame
    blockNum = testBlockNum

    print ("hi")
    probs = next(modelGuess)
    
    # This is due to the sorting of [False, True].
    BProb, AProb = probs
    correct = (AProb > 0.5) == testGoal
    return correct, [BProb, AProb]

  return predict
  '''


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

  games, goals, allFeatures = getRawGameData(args.input_file)
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
        break

      gameIs.append(gameI)
      partialGames.append(game)
      partialGoals.append(game['goal'])

    preditions = getPrediction(classifier, partialGames, blockNum, partialGoals)
    for gameI, partialGoal, predition in zip(gameIs, partialGoals, preditions):
      correct, gamePredictions = predition

      # store data to graph
      samples[blockNum] += 1
      corrects[blockNum] += 1 if correct else 0
      testWinProbs[gameI].append(gamePredictions)

  for blockNum in range(1, MAX_BLOCKS):
    if samples[blockNum] > 0:
      ratios[blockNum] = corrects[blockNum] / samples[blockNum]

      goals = []
      predictions = []
      for testGoal, gamePredictions in zip(testingGoals, testWinProbs):
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
    GraphModelStats.plotGame(times, testingGoals, testWinProbs)


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

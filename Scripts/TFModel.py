import argparse
import functools
import matplotlib.pyplot as pyplot
import pandas
import random
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
  blockStart = blockNum * SECONDS_PER_BLOCK

  inds, gas, gos = [], [], []
  for i, (ga, go) in enumerate(zip(games, goals)):
    if blockStart > ga['debug']['duration']:
      continue
  
    inds.append(i)
    gas.append(ga)
    gos.append(go)

  return inds, gas, gos


featurizerTime = 0
pandasTime = 0
trainTime = 0
def gameToPDF(args, games, *, columnsToExport = None, blockNum = 0, training = False):
  global featurizerTime, pandasTime, trainTime

  if training and args.verbose >= 2:
    print ("\tfeaturizing {} games".format(len(games)))


  T0 = time.time()

  frames = []
  for index, game in enumerate(games):
    if training:
      gameTime = game['debug']['duration']
    else:
      gameTime = blockNum * SECONDS_PER_BLOCK

    gameFrame = TFFeaturize.parseGameToPD(index, game, gameTime)
    frames.append(gameFrame)

  T1 = time.time()

  if training and args.verbose >= 2:
    print ("\tjoining {} games".format(len(games)))

  test = pandas.concat(frames).fillna(0)

  T2 = time.time()

  if training:
    allColumns = list(test.columns.values)
    #print ("saving {} feature columns".format(len(allColumns)))
    #print (test.describe())
    
  else:
    curCols = set(test.columns.values)
    for col in columnsToExport:
      if col not in curCols:
        test[col] = 0

  T3 = time.time()
  featurizerTime += T1 - T0
  pandasTime += T3 - T1
  if training and args.verbose >= 1:
    print ("\t\tFeaturize Timing: {:.2f} build, {:.2f} concat, {:2f} fill".format(
        T1 - T0, T2 - T1, T3 - T2))

  if training:
    return test, allColumns
  return test


def inputFn(columnsUsed, df, goals = None):
  featureCols = {k: tf.constant(df[k].values, shape=[df[k].size, 1], dtype='int32') for k in columnsUsed}
  #print ("input:", df.shape, len(columnsUsed))
  if goals == None:
    return  featureCols
  labels = tf.constant(goals, shape=[len(goals), 1], dtype='int32')
  return featureCols, labels


def buildClassifier(args, numBlocks, trainGames, trainGoals):
  global featurizerTime, pandasTime, trainTime

  params = {
    'dropout': 0.7,
    'learningRate': 0.0001,
    'steps': 5000
  }

  classifiers = []
  columnUses = []
  
  for blockNum in range(numBlocks):
    blockTime = blockNum * SECONDS_PER_BLOCK
    usableIndexes, usableGames, usableGoals = filterMaxBlock(blockNum, trainGames, trainGoals)

    if (args.verbose >= 1):
      print ("\ttraining block {} on {} games".format(blockNum, len(usableGames)))
        
    if len(usableGames) == 0:
      break
  
    trainDF, columnsUsed = gameToPDF(args, usableGames, blockNum = blockNum, training = True)

    featureColumns = [
        tf.contrib.layers.real_valued_column(k) for k in columnsUsed
    ]
# tf.contrib.layers.embedding_column(tf.contrib.layers.sparse_column_with_integerized_feature(k, 100), dimension = 20)

    T0 = time.time()

    optimizer = tf.train.AdamOptimizer(learning_rate = params['learningRate'])
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns = featureColumns,
        hidden_units = [20, 10],
        n_classes = 2,
        dropout = params['dropout'],
        optimizer = optimizer,
    )

    classifier.fit(
        input_fn = functools.partial(inputFn, columnsUsed, trainDF, usableGoals),
        steps = params['steps'])

    T1 = time.time()
    trainTime += T1 - T0

    print ("\t", blockNum, len(columnsUsed), classifier != None)

    classifiers.append(classifier)
    columnUses.append(columnsUsed)

  return classifiers, columnUses


def getPrediction(args, classifiers, columnUses, testGames, blockNum, testGoals):
  blockIndex = min(blockNum, len(classifiers) - 1)
  classifier = classifiers[blockIndex]
  columnsUsed = columnUses[blockIndex]
  
  df = gameToPDF(args, testGames, blockNum = blockNum, columnsToExport = columnsUsed)
  modelGuess = classifier.predict_proba(
      input_fn = functools.partial(inputFn, columnsUsed, df, testGoals))

  for testGoal in testGoals:
    probs = next(modelGuess)
    
    # This is due to the sorting of [False, True].
    BProb, AProb = probs
    correct = (AProb > 0.5) == testGoal
    yield correct, [BProb, AProb]


def main(args):
  global featurizerTime, pandasTime, trainTime

  if args.verbose == 0:
    tf.logging.set_verbosity(tf.logging.ERROR)
  elif args.verbose >= 1:
    tf.logging.set_verbosity(tf.logging.ERROR)
  elif (args.verbose >= 2):
    tf.logging.set_verbosity(tf.logging.INFO)

  
  T0 = time.time()

  MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

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

  classifiers, columnUses = \
      buildClassifier(args, MAX_BLOCKS, trainingGames, trainingGoals)

  T3 = time.time()
  innerTrainTime = T3 - T2

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
    blockTime = blockNum * SECONDS_PER_BLOCK

    gameIs, testingBlockGames, testingBlockGoals = \
        filterMaxBlock(blockNum, testingGames, testingGoals)

    # Do all predictions at the sametime (needed because of how predict reloads the model each time).
    preditions = getPrediction(args, classifiers, columnUses, testingBlockGames, blockNum, testingBlockGoals)

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
  print ("\tpandasTime: {:.3f}".format(pandasTime))


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

import argparse
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as pyplot
import random

from matplotlib.widgets import Slider

from sklearn.ensemble import GradientBoostingClassifier

from collections import defaultdict

from OrdinalFeaturize import *
from Util import *


def getArgParse():
  parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='features.json',
      help='Input match file (produced by Seth or GameParser.py)')

  parser.add_argument(
      '-r', '--randomize',
      action="store_true",
      help='Select training / testing data randomly from input?')

  # TODO(sethtroisi): Add and utilize a flag for verbosity.

  return parser


# Plot general data about accuracy, logloss, number of samples.
def plotData(times, samples, corrects, ratios, logLosses):
  fig, (axis1, axis2, axis3) = pyplot.subplots(3, 1)
  fig.subplots_adjust(hspace = 0.6)

  # Common styling 'Patch' for text
  props = dict(boxstyle='round', facecolor='#abcdef', alpha=0.5)

  # Upper graph of prediction power.
  axis1.plot(times, ratios)
  axis1.set_title('Correct Predictions')
  axis1.set_xlabel('time (m)')
  axis1.set_ylabel('correctness')

  bestAccuracy = max(ratios[:len(ratios) * 2 // 3])
  time = times[ratios.index(bestAccuracy)]
  accuracyText = '{:.3f} (@{:2.0f}m)'.format(bestAccuracy, time)
  axis3.text(
      time / max(times), 0.1,
      accuracyText, transform=axis1.transAxes, fontsize=14,
      bbox=props,
      verticalalignment='bottom', horizontalalignment='center')

  # Middle graph of log loss.
  axis2.plot(times, logLosses)
  axis2.set_title('Log Loss')
  axis2.set_xlabel('time (m)')
  axis2.set_ylabel('loss (log)')
  axis2.set_ylim([0, 1.5])

  minLogLoss = min(logLosses[:len(logLosses) * 2 // 3])
  time = times[logLosses.index(minLogLoss)]
  logLossText = '{:.3f} (@{:2.0f}m)'.format(minLogLoss, time)
  axis2.text(
      time / max(times), 0.7,
      logLossText, transform=axis2.transAxes, fontsize=14,
      bbox=props,
      verticalalignment='bottom', horizontalalignment='center')

  # Lower graph of sample data.
  incorrects = [s - c for s, c in zip(samples, corrects)]

  axis3.plot(times, samples, 'b',
             times, corrects, 'g',
             times, incorrects, 'r')
  axis3.set_title('Number of samples')
  axis3.set_xlabel('time (m)')
  axis3.set_ylabel('samples')

  pyplot.show()


# Plot game predictions vs time.
def plotGame(times, results, winPredictions):
  fig, (axis1, axis2) = pyplot.subplots(2, 1)
  fig.subplots_adjust(hspace = 0.65)

  # Note: I didn't have luck with subplots(3, 1) and resizing so I used this.
  sliderAxis = pyplot.axes(
      [0.125, 0.44, 0.75, 0.05],
      axisbg='lightgoldenrodyellow')

  resultColors = {True:'g', False:'r'}

  # For every game print prediction through out the game.
  for gi in results.keys():
    result = results[gi]
    gamePredictions = winPredictions[gi]
    blocks = len(gamePredictions)
    color = resultColors[result]
    axis1.plot(times[:blocks], gamePredictions, color, alpha = 0.1)

  axis1.set_title('Predictions of win rate across the game')
  axis1.set_xlabel('time (m)')
  axis1.set_ylabel('prediction confidence')

  # At X minutes print confidences.
  sliderTime = Slider(sliderAxis, 'Time', 0, 60, valinit=20)

  percents = [p / 100 for p in range(100 + 1)]

  def plotConfidentAtTime(requestedTime):
    ti = min([(abs(requestedTime - t), i) for i,t in enumerate(times)])[1]

    cdfTrue = [0] * len(percents)
    cdfFalse = [0] * len(percents)
    for gi in results.keys():
      result = results[gi]
      gamePredictions = winPredictions[gi]
      if len(gamePredictions) <= ti:
        continue

      prediction = gamePredictions[ti]
      for pi, percent in enumerate(percents):
        if percent > prediction:
          break
        if result:
          cdfTrue[pi] += 1
        else:
          cdfFalse[pi] += 1

    axis2.cla();

    axis2.plot(percents, cdfTrue, resultColors[True])
    axis2.plot(percents, cdfFalse[::-1], resultColors[False])

    axis2.set_xlabel('confidence')
    axis2.set_ylabel('count of games')

    axis2.set_ylim([0, max(cdfTrue[0], cdfFalse[0]) + 1])

    fig.canvas.draw_idle()

  plotConfidentAtTime(20)
  sliderTime.on_changed(plotConfidentAtTime)

  pyplot.show()


def stats(times, samples, corrects, ratios, logLosses):
  startBlock = timeToBlock(10 * 60)
  endBlock = timeToBlock(40 * 60)

  sumLosses = sum(logLosses[startBlock:endBlock+1])
  totalSamples = sum(samples[startBlock:endBlock+1])
  totalCorrect = sum(corrects[startBlock:endBlock+1])
  totalIncorrect = totalSamples - totalCorrect
  mediumRatio = np.median(ratios[startBlock:endBlock+1])

  print ()
  print ("Global Stats 10 to 40 minutes")
  print ()
  print ("Sum LogLoss: {:.3f}".format(sumLosses))
  print ("Correct Predictions:", totalCorrect)
  print ("Incorrect Predictions:", totalIncorrect)
  print ("Global Ratio: {:2.1f}".format(100 * totalCorrect / totalSamples))
  print ("Mean Ratio: {:2.1f}".format(100 * mediumRatio))


def buildClassifier(trainGoals, trainFeatures):
  #GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100,
  #    subsample=1.0, min_samples_split=2, min_samples_leaf=1,
  #    min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None,
  #    max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False)
  clf = GradientBoostingClassifier(verbose=True)

  #print ("With training set size: {} games {} features - {} nnz".format(
  #    len(trainGoals), trainFeatures.shape[1], trainFeatures.nnz))

  clf.fit(trainFeatures, trainGoals)

  #print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
  #    clf.intercept_[0], 100 * trainGoals.count(True) / len(trainGoals)))
  print ()

  return clf


def getPrediction(classifier, testGoal, testFeature):
  modelGuesses = classifier.predict_proba(testFeature)

  # This is due to the sorting of [False, True].
  BProb, AProb = modelGuesses[0]

  correct = (AProb > 0.5) == testGoal

  return correct, AProb


def seperate(args, games, goals, features):
  holdBackPercent = 50
  sampleSize = len(games)
  trainingSize = sampleSize - (holdBackPercent * sampleSize) // 100

  trainingGoals = []
  trainingFeatures = []
  testingGames = []

  if args.randomize:
    trainingNums = set(random.sample(range(sampleSize), trainingSize))
  else:
    trainingNums = set(range(trainingSize))

  for i in range(sampleSize):
    if i in trainingNums:
      trainingGoals.append(goals[i])
    else:
      testingGames.append(games[i])

  # features is a scipy.sparse so use the column sample directly on it
  #trainingFeatures = features[sorted(trainingNums)]
  trainingFeatures = [features[i] for i in sorted(trainingNums)]

  return (trainingGoals, trainingFeatures, testingGames)


def main(args):
    MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

    games, goals, features = getGamesData(args.input_file)

    trainingGoals, trainingFeatures, testingGames = \
        seperate(args, games, goals, features)

    # Variables about testGames.
    times = [(b * SECONDS_PER_BLOCK) / 60 for b in range(MAX_BLOCKS)]
    samples = [0 for b in range(MAX_BLOCKS)]
    corrects = [0 for b in range(MAX_BLOCKS)]
    # Averages over data (calculated after all data).
    ratios = [0 for b in range(MAX_BLOCKS)]
    logLosses = [0 for b in range(MAX_BLOCKS)]
    # Per Game stats.
    testGoals = {}
    winPredictions = defaultdict(list)

    classifier = buildClassifier(trainingGoals, trainingFeatures)
    for blockNum in range(MAX_BLOCKS):
      time = blockNum * SECONDS_PER_BLOCK

      # TODO rebult classifier here.

      for gi, game in enumerate(testingGames):
        duration = game['debug']['duration']
        goal = game['goal']


        # TODO(sethtroisi): remove games that have ended.
        if duration < time:
          continue

        gameFeatures = parseGameToFeatures(game, time)

        correct, prediction = getPrediction(classifier, goal, gameFeatures)

        # store data to graph
        samples[blockNum] += 1
        corrects[blockNum] += 1 if correct else 0

        testGoals[gi] = goal
        winPredictions[gi].append(prediction)

    '''
    for blockNum in range(MAX_BLOCKS):
      if samples[blockNum] > 0:
        ratios[blockNum] = corrects[blockNum] / samples[blockNum]

        goals = []
        predictions = []
        for testGoal, gamePredictions in zip(testGoals, winPredictions):
          if len(gamePredictions) > blockNum:
            goals.append(testGoal)
            predictions.append(gamePredictions[blockNum])

        logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions)

    # Use the model to make some simple predictions.
    # TODO(sethtroisi): move this under a flag.
    #predict(classifier, vectorizer)
    '''

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
      stats(times, samples, corrects, ratios, logLosses)
      #plotData(times, samples, corrects, ratios, logLosses)
      plotGame(times, testGoals, winPredictions)

if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

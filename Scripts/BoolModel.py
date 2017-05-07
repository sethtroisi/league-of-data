import argparse
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as pyplot
import random

from matplotlib.widgets import Slider
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from BoolFeaturize import *
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
  for result, gamePredictions in zip(results, winPredictions):
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
    for result, gamePredictions in zip(results, winPredictions):
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


def buildClassifiers(numBlocks, trainGoals, trainGames, vectorizer, testFoo):
  #SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
  #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
  #       loss='hinge', n_iter=2, n_jobs=1, penalty='l2', power_t=0.5,
  #       random_state=None, shuffle=False, verbose=True, warm_start=False)
  #clf = SGDClassifier(loss="log", penalty="l2", n_iter=3000, shuffle=True,
  #  alpha = 0.02, verbose=False)

  clfs = []
  for blockNum in range(numBlocks):
    time = blockNum * SECONDS_PER_BLOCK
    clf = MLPClassifier(
        solver='lbfgs',
        random_state=1,
        max_iter=5000,
        hidden_layer_sizes=(10,10))

    subTrainGoals= []
    subTrainFeatures = []
    for goal, game in zip(trainGoals, trainGames):
      duration = game['debug']['duration']
      if duration < time:
        continue

      subTrainGoals.append(goal)
      gameFeatures = parseGameToFeatures(game, time)
      subTrainFeatures.append(gameFeatures)

    if len(subTrainGoals) <= 1:
      break

    vectorizer = DictVectorizer(sparse=True)
    sparseFeatures = vectorizer.fit_transform(subTrainFeatures)

    clf.fit(sparseFeatures, subTrainGoals)
    clfs.append(clf)

  #print ("With training set size: {} games {} features - {} nnz".format(
  #    len(trainGoals), trainFeatures.shape[1], trainFeatures.nnz))


  #print (clf.coef_)
  #print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
  #    clf.intercept_[0], 100 * trainGoals.count(True) / len(trainGoals)))
  #print ()

  return clfs


def getPrediction(classifiers, blockNum, testGoal, testFeature):
  modelGuesses = classifiers[blockNum].predict_proba(testFeature)

  # This is due to the sorting of [False, True].
  BProb, AProb = modelGuesses[0]

  correct = (AProb > 0.5) == testGoal

  return correct, AProb


def seperate(args, games, goals):
  holdBackPercent = 50
  sampleSize = len(games)
  trainingSize = sampleSize - (holdBackPercent * sampleSize) // 100

  trainingGoals = []
  trainingGames = []
  testingGames = []

  if args.randomize:
    trainingNums = set(random.sample(range(sampleSize), trainingSize))
  else:
    trainingNums = set(range(trainingSize))

  for i in range(sampleSize):
    if i in trainingNums:
      trainingGoals.append(goals[i])
      trainingGames.append(games[i])
    else:
      testingGames.append(games[i])

  return (trainingGoals, trainingGames, testingGames)


def main(args):
    MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

    games, goals, vectorizer, features = getGamesData(args.input_file)

    trainingGoals, trainingGames, testingGames = \
        seperate(args, games, goals)

    classifiers = buildClassifiers(MAX_BLOCKS, trainingGoals, trainingGames, vectorizer, features)

    # Variables about testGames.
    times = [(b * SECONDS_PER_BLOCK) / 60 for b in range(MAX_BLOCKS)]
    samples = [0 for b in range(MAX_BLOCKS)]
    corrects = [0 for b in range(MAX_BLOCKS)]
    # Averages over data (calculated after all data).
    ratios = [0 for b in range(MAX_BLOCKS)]
    logLosses = [0 for b in range(MAX_BLOCKS)]
    # Per Game stats.
    testGoals = []
    winPredictions = []

    for game in testingGames:
      duration = game['debug']['duration']
      goal = game['goal']

      predictions = []
      # TODO(sethtroisi): determine this point algorimically as 80% for game end.
      # TODO(sethtroisi): alternatively truncate when samples < 50.
      for blockNum in range(MAX_BLOCKS):
        time = blockNum * SECONDS_PER_BLOCK

        # TODO(sethtroisi): remove games that have ended.
        if duration < time:
          continue

        gameFeatures = parseGameToFeatures(game, time)

        sparse = vectorizer.transform(gameFeatures)

        correct, prediction = getPrediction(classifiers, blockNum, goal, sparse)

        # store data to graph
        samples[blockNum] += 1
        corrects[blockNum] += 1 if correct else 0
        predictions.append(prediction)

      testGoals.append(goal)
      winPredictions.append(predictions)

    for blockNum in range(MAX_BLOCKS):
      if samples[blockNum] > 0:
        ratios[blockNum] = corrects[blockNum] / samples[blockNum]

        goals = []
        predictions = []
        for testGoal, gamePredictions in zip(testGoals, winPredictions):
          if len(gamePredictions) > blockNum:
            goals.append(testGoal)
            predictions.append(gamePredictions[blockNum])

        if len(goals) <= 1:
          break
        logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions)

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
      stats(times, samples, corrects, ratios, logLosses)
      plotData(times, samples, corrects, ratios, logLosses)
      plotGame(times, testGoals, winPredictions)


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

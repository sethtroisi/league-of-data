import argparse
import numpy as np
import matplotlib.pyplot as pyplot
import random
import sklearn.metrics
import tensorflow as tf

from matplotlib.widgets import Slider
from sklearn.model_selection import train_test_split

from TFFeaturize import *
from Util import *


def getArgParse():
  parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='features.json',
      help='Input match file (produced by Seth or GameParser.py)')

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
  maxLogLoss = max(1.4, min(10, 1.2 * max(logLosses)))

  axis2.plot(times, logLosses)
  axis2.set_title('Log Loss')
  axis2.set_xlabel('time (m)')
  axis2.set_ylabel('loss (log)')
  axis2.set_ylim([0, maxLogLoss])

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
  axis2_2 = axis2.twinx()

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
    axis1.plot(times[:blocks], [gP[1] for gP in gamePredictions], color, alpha = 0.1)

  axis1.set_title('Predictions of win rate across the game')
  axis1.set_xlabel('time (m)')
  axis1.set_ylabel('prediction confidence')

  # At X minutes print confidences.
  sliderTime = Slider(sliderAxis, 'Time', 0, 60, valinit=20)

  percentBuckets = 100
  percents = [p / percentBuckets for p in range(percentBuckets + 1)]

  def plotConfidentAtTime(requestedTime):
    ti = min([(abs(requestedTime - t), i) for i,t in enumerate(times)])[1]

    cdfTrue = [0] * len(percents)
    cdfFalse = [0] * len(percents)
    pdfTrue = [0] * len(percents)
    pdfFalse = [0] * len(percents)

    for gameResult, gamePredictions in zip(results, winPredictions):
      if len(gamePredictions) <= ti:
        continue

      prediction = gamePredictions[ti][1]
      for pi, percent in enumerate(percents):
        if percent > prediction:
          break
        if gameResult:
          cdfTrue[pi] += 1
        else:
          cdfFalse[pi] += 1

      bucket = int(percentBuckets * prediction)
      if gameResult:
        pdfTrue[bucket] += 1
      else:
        # ~ is a fun trick to get the negative index (0 => -1, 1 => -2, ...) of an item
        pdfFalse[~bucket] += 1
 
    axis2.cla();
    axis2_2.cla();   

    axis2.plot(percents, cdfTrue, color = resultColors[True], alpha = 0.9)
    axis2.plot(percents, cdfFalse, color = resultColors[False], alpha = 0.9)

    axis2_2.bar(percents, pdfTrue,  width = 0.008, color = resultColors[True],  alpha = 0.5)
    axis2_2.bar(percents, pdfFalse, width = 0.008, color = resultColors[False], alpha = 0.5)

    axis2.set_xlabel('confidence')
    axis2.set_ylabel('count of games (cdf)')
    axis2_2.set_ylabel('count of games (pdf)')

    axis2.set_xlim([0, 1]);
    axis2_2.set_xlim([0, 1]);

#    axis2.set_ylim([0, max(cdfTrue[0], cdfFalse[0]) + 1])
#    axis2_2.set_ylim([0, max(max(pdfTrue), max(pdfFalse)) + 1]])

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


def buildClassifier(trainGoals, trainGames):

  subTrainGoals= []
  subTrainFeatures = []
  for goal, game in zip(trainGoals, trainGames):
    duration = game['debug']['duration']

    subTrainGoals.append(goal)
    gameFeatures = parseGameToFeatures(game, duration + 1)
    subTrainFeatures.append(gameFeatures)

  columns = ["gold"]
  df = pandas.DataFrame(columns = columns)


  assert len(subTrainGoals) > 0

  print ("training on {} datums".format(len(subTrainGoals)))
  columns = ["gold"]
  df = pandas.DataFrame(columns = columns)


  featureColumns = [
    tf.contrib.layers.
  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns = featureColumns,
      hidden_units = [10, 10],
      n_classes = 1))


#  def inputFn(dataFrame):
#    featureCols = {label: valueList}
#    labels = subTrainGoals
#    return featureCols, labels
#  classifier.fit(input_fn=inputFn, steps = 5000)
  
  classifier.fit(
      x = np.array(subTrainFeatures, dtype = int64),
      y = np.array(subTrainGoals,    dtype = int64),
      steps = 5000)

#    print ("clf {:2d}: loss: {:.3f} after {:4d} iters".format(
#      blockNum, clf.loss_, clf.n_iter_))

  return classifier


def getPrediction(classifier, blockNum, testGoal, testFeature):
  guess = classifier.predict(testFeature),
  modelGuesses = classifier.predict_proba(testFeature)

  print (guess, "\t", modelGuesses)

  # This is due to the sorting of [False, True].
  BProb, AProb = modelGuesses[0]
  correct = (AProb > 0.5) == testGoal
  return correct, [BProb, AProb]



def main(args):
    MAX_BLOCKS = int(3600 // SECONDS_PER_BLOCK) + 1

    games, goals = getRawGameData(args.input_file)

    trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
        games,
        goals,
        test_size = 0.15,
        random_state = 42)

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

        correct, gamePredictions = getPrediction(classifier, blockNum, goal, sparse)

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
      stats(times, samples, corrects, ratios, logLosses)
      plotData(times, samples, corrects, ratios, logLosses)
      plotGame(times, testGoals, testWinProbs)


if __name__ == '__main__':
  args = getArgParse().parse_args()
  main(args)

from Featurize import *

from sklearn.linear_model import SGDClassifier
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as pyplot

# TODO(sethtroisi): add flag parsing to this file to display verbose.


def buildClassifier(trainGoals, trainFeatures):
  #SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
  #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
  #       loss='hinge', n_iter=2, n_jobs=1, penalty='l2', power_t=0.5,
  #       random_state=None, shuffle=False, verbose=True, warm_start=False)
  clf = SGDClassifier(loss="log", penalty="l2", n_iter=5000, shuffle=True,
    alpha = 0.01, verbose = False)

  clf.fit(trainFeatures, trainGoals)

  print ("With training set size: {} features".format(
      len(trainGoals), trainFeatures.shape))

  #print ("Score:", clf.score(trainFeatures, trainGoals))
  #print ()

  #print (clf.coef_)
  print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
      clf.intercept_[0], 100 * trainGoals.count(True) / len(trainGoals)))
  print ()

  return clf


def testClassifier(classifier, testGoals, testFeatures):
  modelGoals = classifier.predict_proba(testFeatures)

  samples = len(testGoals)

  corrects = 0
  predictA = 0
  predictB = 0
  for modelGuess, testResult in zip(modelGoals, testGoals):
    BProb, AProb = modelGuess # this is due to the sorting of [False, True]

    correct = (AProb > 0.5) == testResult
    corrects += correct

    predictA += AProb > 0.5
    predictB += BProb > 0.5

#  print ("Predict A: {}, B: {}".format(predictA, predictB))
#  print ("True A: {}, B: {}".format(
#      testGoals.count(True), testGoals.count(False)))
#  print ()

  print ("Correctness: {}/{} = {:2.1f}".format(
      corrects, samples, 100 * corrects / samples))
#  print ()

  logLoss = sklearn.metrics.log_loss(testGoals, modelGoals)
#  print ("log loss: {:.4f}".format(logLoss))
#  print ("\t(lower is better, null model is .6912)")
#  print ()
#  print ()

  return corrects, samples - corrects, logLoss


def seperate(games, goals, features):
  holdBackPercent = 25
  sampleSize = len(goals)
  holdBackAmount = (holdBackPercent * sampleSize) // 100

  trainingGoals = goals[:-holdBackAmount]
  trainingFeatures = features[:-holdBackAmount]

  testingGames = games[-holdBackAmount:]

  return (trainingGoals, trainingFeatures, testingGames)


# MAIN CODE
times = []
samples = []
corrects = []
incorrects = []
testingSize = []
ratios = []
logLosses = []

games, goals, vectorizer, features = getGamesData()
data = seperate(games, goals, features)

trainingGoals, trainingFeatures, testingGames  = data

classifier = buildClassifier(trainingGoals, trainingFeatures)

# TODO(sethtroisi): this code needs to be rewritten to support sampling at block X.
for blockNum in range((60 * 60) // SECONDS_PER_BLOCK):
  goals = []
  featuresList = []

  time = blockNum * SECONDS_PER_BLOCK
  for game in testingGames:
    # TODO(sethtroisi): remove games that have ended.
    duration = game['features']['duration']
    if duration < time:
      continue

    rawFeatures, goal = parseGameToFeatures(game, time)

    goals.append(goal)
    featuresList.append(rawFeatures)

  sparse = vectorizer.transform(featuresList)

  correct, incorrect, logLoss = \
      testClassifier(classifier, goals, sparse)

  # store data to graph
  times.append(time / 60)
  samples.append(len(goals))

  corrects.append(correct)
  incorrects.append(incorrect)
  testingSize.append(correct + incorrect)

  ratios.append(correct / (correct + incorrect))

  logLosses.append(logLoss)

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
axis2.set_ylim([0,1])

minLogLoss = min(logLosses)
time = times[logLosses.index(minLogLoss)]
logLossText = '{:.3f} (@{:2.0f}m)'.format(minLogLoss, time)
axis2.text(
    time / max(times), 0.7,
    logLossText, transform=axis2.transAxes, fontsize=14,
    bbox=props,
    verticalalignment='bottom', horizontalalignment='center')


# Lower graph of sample data.
axis3.plot(times, samples, 'b',
           times, testingSize, 'b',
           times, corrects, 'g',
           times, incorrects, 'r')
axis3.set_title('Number of samples')
axis3.set_xlabel('time (m)')
axis3.set_ylabel('samples')

pyplot.show()

# '''

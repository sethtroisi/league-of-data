from Featurize import *

from sklearn.linear_model import SGDClassifier
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as pyplot


def trainModel(trainGoals, trainFeatures):
  #SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
  #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
  #       loss='hinge', n_iter=2, n_jobs=1, penalty='l2', power_t=0.5,
  #       random_state=None, shuffle=False, verbose=True, warm_start=False)
  clf = SGDClassifier(loss="log", penalty="l2", n_iter=1000, shuffle=True,
    alpha = 0.01, verbose = False)

  clf.fit(trainFeatures, trainGoals)

  #print ("Score:", clf.score(trainFeatures, trainGoals))
  #print ()

  #print (clf.coef_)
  #print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
  #    clf.intercept_[0], 100 * trainGoals.count(True) / len(trainGoals)))
  #print ()

  return clf;


def testModel(trainGoals, trainFeatures, testGoals, testFeatures):
   # TODO(sethtroisi): add flag parsing to this file to display verbose.
#  print ("With training set size: {} x {} features".format(
#      len(trainGoals), len(trainFeatures[0])))

#  print ("With testing set size: {} x {} features".format(
#      len(testGoals), len(testFeatures[0])))
#  print ()

  classifier = trainModel(trainGoals, trainFeatures)
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
  print ()

#  print ("log loss: {:.4f}".format(
#      sklearn.metrics.log_loss(testGoals, modelGoals)))
#  print ("\t(lower is better, null model is .6912)")
#  print ()
#  print ()

  return corrects, samples - corrects

def seperateToTrainingAndTest(goals, blocks):
  holdBackPercent = 25
  sampleSize = len(goals)
  holdBackAmount = (holdBackPercent * sampleSize) // 100

  trainGoals = goals[:-holdBackAmount]
  trainBlocks = blocks[:-holdBackAmount]

  testGoals = goals[-holdBackAmount:]
  testBlocks = blocks[-holdBackAmount:]

  return (trainGoals, trainBlocks, testGoals, testBlocks)


# MAIN CODE
times = []
samples = []
corrects = []
incorrects = []
testingSize = []
ratios = []

goals, matches = getTrainingAndTestData()
for blockNum in range(100):
  blockGoals = []
  blockFeatures = []

  for goal, blocks in zip(goals, matches):
    if len(blocks) > blockNum:
      blockGoals.append(goal)
      blockFeatures.append(blocks[blockNum])

  if len(blockGoals) < 50:
    break

  trainGoals, trainFeatures, testGoals, testFeatures = \
    seperateToTrainingAndTest(blockGoals, blockFeatures)

  correct, incorrect = testModel(trainGoals, trainFeatures, testGoals, testFeatures)

  # store data to graph
  times.append(blockNum * SECONDS_PER_BLOCK / 60)
  samples.append(len(blockGoals))

  corrects.append(correct)
  incorrects.append(incorrect)
  testingSize.append(correct + incorrect)

  ratios.append(correct / (correct + incorrect))

fig, (axis1, axis2) = pyplot.subplots(2, 1)
fig.subplots_adjust(hspace = 0.45)

axis1.plot(times, ratios)
axis1.set_title('Correct Predictions')
axis1.set_xlabel('time (m)')
axis1.set_ylabel('correctness')

axis2.plot(times, samples, 'b',
           times, testingSize, 'b',
           times, corrects, 'g',
           times, incorrects, 'r')
axis2.set_title('Number of samples')
axis2.set_xlabel('time (m)')
axis2.set_ylabel('samples')

pyplot.show()

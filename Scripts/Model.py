import json
import math

from sklearn.linear_model import SGDClassifier


DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'


def loadOutputFile():
  features = []
  goals = []

  f = open(OUTPUT_FILE)
  for line in f.readlines():
    parsed = json.loads(line)

    gameFeatures = parsed['features']

    firstDragon = gameFeatures[:2]
    dragonTime = gameFeatures[2] // 1000
    assert 0 < dragonTime < 2*60*60 or dragonTime == 10 ** 7
    for i in range(7, 12):
      firstDragon.append(dragonTime < 2 ** i)

    firstTower = gameFeatures[3:5]

    feature = firstDragon + firstTower
    assert (feature.count(True) + feature.count(False)) == len(feature)
    features.append(feature)

    goal = parsed['goal']
    assert goal in (True, False)
    goals.append(goal)

  f.close()


  sampleSize = len(goals)
  holdBackPercent = 20
  holdBackSize = (sampleSize * holdBackPercent) // 100

  print ("loaded {} games".format(len(goals)))
  print ()

  trainingFeatures = features[:-holdBackSize]
  trainingGoals = goals[:-holdBackSize]

  testFeatures = features[-holdBackSize:]
  testGoals = goals[-holdBackSize:]

  return [trainingFeatures, trainingGoals, testFeatures, testGoals]


def trainModel(trainingFeatures, trainingGoals, testFeatures):
  #SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
  #       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
  #       loss='hinge', n_iter=2, n_jobs=1, penalty='l2', power_t=0.5,
  #       random_state=None, shuffle=False, verbose=True, warm_start=False)
  clf = SGDClassifier(loss="log", penalty="l2", n_iter=1000, shuffle=True,
    alpha = 0.01, verbose = False)

  clf.fit(trainingFeatures, trainingGoals)

  print ("Score:", clf.score(trainingFeatures, trainingGoals))
  print ()

  print (clf.coef_)
  print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
      clf.intercept_[0], 100 * trainingGoals.count(True) / len(trainingGoals)))
  print ()

  predictions = clf.predict_proba(testFeatures)

  return predictions


trainFeatures, trainGoals, testFeatures, testGoals = loadOutputFile()

print ("With training set size: {} x {} features".format(
    len(trainGoals), len(trainFeatures[0])))

print ("With test set size: {} x {} features".format(
    len(testGoals), len(testFeatures[0])))
print ()

modelGoals = trainModel(trainFeatures, trainGoals, testFeatures)

samples = len(testGoals)

corrects = 0
predictA = 0
predictB = 0
logLoss = 0
for modelGuess, testResult in zip(modelGoals, testGoals):
  if testResult:
    logLoss += (1 - modelGuess[1])
  else:
    logLoss += (1 - modelGuess[0])

  correct = (modelGuess[1] > 0.5) == testResult
  corrects += correct

  predictA += modelGuess[1] > 0.5
  predictB += modelGuess[0] > 0.5

print ("Predict A: {}, B: {}".format(predictA, predictB))

print ("Correctness: {}/{} = {:2.1f}".format(
    corrects, samples, 100 * corrects / samples))

print ("log loss: {:.3f}".format(logLoss / samples))

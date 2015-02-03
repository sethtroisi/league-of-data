import json
from sklearn.linear_model import SGDClassifier
import random

DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'


def loadOutputFile():
  features = []
  goals = []

  f = open(OUTPUT_FILE)
  for line in f.readlines():
    parsed = json.loads(line)

    gameFeatures = parsed['features']

    boolVersion = gameFeatures[:2]

    dragonTime = gameFeatures[2] // 1000

    assert 0 < dragonTime < 2*60*60 or dragonTime == 10 ** 7
    for i in range(7, 12):
      boolVersion.append(dragonTime < 2 ** i)

    features.append(boolVersion)

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


  print (clf.coef_)
  print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
      clf.intercept_[0], 100 * trainingGoals.count(True) / len(trainingGoals)))
  print ()

  predictions = clf.predict_proba(testFeatures)

  sumProp = sum([max(prob) for prob in predictions])
  normalized = sumProp / len(testFeatures)
  print ("sum prob: {:4f}".format(normalized))

  return predictions


trainFeatures, trainGoals, testFeatures, testGoals = loadOutputFile()

print ("With training set size: {} x {} features".format(
    len(trainGoals), len(trainFeatures[0])))

print ("With test set size: {} x {} features".format(
    len(testGoals), len(testFeatures[0])))
print ()

modelGoals = trainModel(trainFeatures, trainGoals, testFeatures)

correctPredictions = [(modelGuess[1] > 0.5) == testResult
    for modelGuess, testResult in zip(modelGoals, testGoals)]

corrects = correctPredictions.count(True)
testSamples = len(correctPredictions)

print ("Correctness: {}/{} = {:2.1f}".format(
  corrects, testSamples, 100 * corrects / testSamples))

print ("Predict A: {}, B: {}".format(
    len([i for i in modelGoals if i[0] > 0.5]),
    len([i for i in modelGoals if i[1] > 0.5])))

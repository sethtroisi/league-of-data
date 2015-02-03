import json
from sklearn.linear_model import SGDClassifier
import random

#X = [[0., 0.], [0., 1.], [0., 2.], [0., 3.]]
#y = [0, 0, 1, 1]
#for y in range(0, 50):
#  test = [0., y / 10.0]
#  print (test)
#  print (clf.predict([test]), clf.decision_function([test]))

DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'


def loadOutputFile():
  features = []
  goals = []

  f = open(OUTPUT_FILE)
  for line in f.readlines():
    parsed = json.loads(line)

    gameFeatures = parsed['features']
    intVersion = []
    for feature in gameFeatures:
      intVersion.append(1 if feature else 0)

    intVersion = intVersion[:2]
    features.append(intVersion)

    goal = parsed['goal']
    assert goal in (True, False)
    goals.append(1 if goal else 0)

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
  clf = SGDClassifier(loss="log", penalty="l2", n_iter=20, verbose=True)

  clf.fit(trainingFeatures, trainingGoals)

  print (clf.coef_)
  print ("intercept: {:4.3f}, TrueProp: {:3.1f}%".format(
      clf.intercept_[0], 100 * trainingGoals.count(True) / len(trainingGoals)))
  print ()
  #print (clf.predict_proba(testFeatures))
  #print (clf.decision_function(testFeatures))

  return clf.predict_proba(testFeatures)


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

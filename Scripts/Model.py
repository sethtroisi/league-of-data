import json
from sklearn.linear_model import SGDClassifier

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
      if feature == True:
        intVersion.append(1)
      elif feature == False:
        intVersion.append(-1)
      else:
        intVersion.append(feature)

    features.append(intVersion)
    goals.append(parsed['goal'])

  f.close()


  sampleSize = len(goals)
  holdBackPercent = 20
  holdBackSize = (sampleSize * holdBackPercent) // 100

  print ("loaded {} games".format(len(goals)))
 
  trainingFeatures = features[:-holdBackSize]
  trainingGoals = goals[:-holdBackSize]

  testFeatures = features[-holdBackSize:]
  testGoals = goals[-holdBackSize:]

  return [trainingFeatures, trainingGoals, testFeatures, testGoals]


def trainModel(trainingFeatures, trainingGoals, testFeatures):
  clf = SGDClassifier(loss="hinge", penalty="l2")

  clf.fit(trainingFeatures, trainingGoals)

  print ("train?")
  SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
         fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
         loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
         random_state=None, shuffle=False, verbose=True, warm_start=False)

  return clf.predict(testFeatures)


trainFeatures, trainGoals, testFeatures, testGoals = loadOutputFile()

print ("With training set size: {} x {} features".format(
    len(trainGoals), len(trainFeatures[0])))

print ("With test set size: {} x {} features".format(
    len(testGoals), len(testFeatures[0])))

modelGoals = trainModel(trainFeatures, trainGoals, testFeatures)

correctPredictions = [modelGuess == testResult
    for modelGuess, testResult in zip(modelGoals, testGoals)]
corrects = correctPredictions.count(True)
testSamples = len(correctPredictions)

print ("Correctness: {}/{} = {:2.1f}".format(
  corrects, testSamples, 100 * corrects / testSamples))

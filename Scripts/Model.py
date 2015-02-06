from Featurize import *

from sklearn.linear_model import SGDClassifier

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


trainFeatures, trainGoals, testFeatures, testGoals = getTrainingAndTestData()

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
    logLoss += -math.log(modelGuess[0])
  else:
    logLoss += -math.log(modelGuess[1])

  correct = (modelGuess[1] > 0.5) == testResult
  corrects += correct

  predictA += modelGuess[1] > 0.5
  predictB += modelGuess[0] > 0.5

print ("Predict A: {}, B: {}".format(predictA, predictB))

print ("Correctness: {}/{} = {:2.1f}".format(
    corrects, samples, 100 * corrects / samples))

print ("log loss: {:.3f}".format(logLoss / samples))
print ("\thigher is better, null model is .697")

import json
import math

from sklearn.linear_model import SGDClassifier


DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'


def parseGame(parsed):
  gameFeatures = parsed['features']

  firstDragon = gameFeatures[:2]
  dragonTime = gameFeatures[2] // 1000
  assert 0 < dragonTime < 2*60*60 or dragonTime == 10 ** 7
  for i in range(7, 12):
    firstDragon.append(dragonTime < 2 ** i)

  firstTower = gameFeatures[3:5]

  features = firstDragon + firstTower
  
  goal = parsed['goal']

  return features, goal


def loadOutputFile():
  features = []
  goals = []

  dataFile = open(OUTPUT_FILE)
  for line in dataFile.readlines():
    parsed = json.loads(line)

    matchFeatures, goal = parseGame(parsed)

    assert all([f in (True, False) for f in matchFeatures])
    assert goal in (True, False)

    features.append(matchFeatures)
    goals.append(goal)

  dataFile.close()

  return goals, features


def getTrainingAndTestData():
  goals, features = loadOutputFile()

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



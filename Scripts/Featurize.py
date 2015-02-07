import json
import math

from sklearn.linear_model import SGDClassifier


DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'


# Creates several features from the first dragon (team, time)
def firstDragonFeatures(dragons):
  firstDragon = [False] * (2 + 5)
  if len(dragons) > 0:
    dragonTime, isTeamOne = dragons[0]
    firstDragon[0] = isTeamOne
    firstDragon[1] = not isTeamOne

    assert 0 < dragonTime < 2*60*60 or dragonTime == 10 ** 7
    for i in range(5):
      firstDragon[2 + i] = dragonTime < 2 ** (7 + i)
  return firstDragon


# Creates several features about total dragon taken (team, count)
def dragonFeatures(dragons):
  lastDragon = [False] * 2
  takenDragons = [False] * (2 * 5)

  dragonsA, dragonsB = 0, 0
  for dragon in dragons[:1]:
    dragonTime, isTeamOne = dragon

    if isTeamOne:
      lastDragon = [True, False]

      dragonsA += 1
      if dragonsA <= 5:
        takenDragons[dragonsA - 1] = True
    else:
      lastDragon = [False, True]

      dragonsB += 1
      if dragonsB <= 5:
        takenDragons[5 + dragonsB - 1] = True

  return lastDragon + takenDragons


# Creates several features from towers (team, position)
def towerFeatures(towers):
  takenTowers = [False] * (2 * 3 * 4)

  # Note: only use the first n tower to avoid overfitting
  # TODO(sethtroisi): block on time to avoid overfitting
  for towerData in towers[:2]:
    towerTime, towerNum = towerData
    takenTowers[towerNum] = True

  return takenTowers


def parseGame(parsed):
  gameFeatures = parsed['features']

  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']

  features = []
  features += dragonFeatures(dragons)
  features += towerFeatures(towers)

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

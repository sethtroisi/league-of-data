import json
import math

from sklearn.linear_model import SGDClassifier


DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'

SECONDS_PER_BLOCK = 2 * 60

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
def dragonFeatures(dragons, sampleTime):
  lastDragon = [False] * 2
  takenDragons = [False] * (2 * 5)

  dragonsA, dragonsB = 0, 0
  for dragon in dragons:
    dragonTime, isTeamOne = dragon
    if dragonTime > sampleTime:
      break

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
def towerFeatures(towers, sampleTime):
  takenTowers = [False] * (2 * 3 * 4)

  # Note: only use the first n tower to avoid overfitting
  # TODO(sethtroisi): block on time to avoid overfitting
  for towerData in towers:
    towerTime, towerNum = towerData
    if towerTime > sampleTime:
      break

    takenTowers[towerNum] = True

  return takenTowers


def parseGameToBlocks(parsed):
  gameFeatures = parsed['features']

  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']

  blocks = []

  duration = gameFeatures['duration']

  totalBlocks = duration // SECONDS_PER_BLOCK
  for blockNum in range(totalBlocks + 1):
    time = blockNum * SECONDS_PER_BLOCK

    features = []
    features += dragonFeatures(dragons, time)
#    features += towerFeatures(towers, time)

#    print (blockNum, features.count(True))

    blocks.append(features)

  goal = parsed['goal']
  return blocks, goal


def loadOutputFile():
  matches = []
  goals = []

  dataFile = open(OUTPUT_FILE)
  for line in dataFile.readlines():
    parsed = json.loads(line)

    matchBlocks, goal = parseGameToBlocks(parsed)

    assert all([f in (True, False)
        for features in matchBlocks for f in features])
    assert goal in (True, False)

    matches.append(matchBlocks)
    goals.append(goal)

  dataFile.close()

  return goals, matches


def getTrainingAndTestData():
  goals, matches = loadOutputFile()

  sampleSize = len(goals)
  print ("loaded {} games".format(sampleSize))

  lastBlock = max(len(blocks) for blocks in matches)
  numBlocks = [0] * lastBlock
  for blocks in matches:
    numBlocks[len(blocks) - 1] += 1

  print ("Games with X blocks: {}".format(numBlocks))
  print ("totalBlocks: {}".format(sum(len(blocks) for blocks in matches)))
  print ()

  return goals, matches

import json
import math

from sklearn.feature_extraction import DictVectorizer

DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'

SECONDS_PER_BLOCK = 2 * 60

def timeToBlock(time):
  # I think it's more correct to return the block it's happening in.
  # IE event (T = 0) = 0, (0 < T <= 5) = 1
  return (time - 1) // SECONDS_PER_BLOCK + 1

## Creates several features from the first dragon (team, time)
#def firstDragonFeatures(dragons):
#  firstDragon = [False] * (2 + 5)
#  if len(dragons) > 0:
#    dragonTime, isTeamOne = dragons[0]
#    firstDragon[0] = isTeamOne
#    firstDragon[1] = not isTeamOne
#
#    assert 0 < dragonTime < 2*60*60 or dragonTime == 10 ** 7
#    for i in range(5):
#      firstDragon[2 + i] = dragonTime < 2 ** (7 + i)
#  return firstDragon


# Creates several features about total dragon taken (team, count)
def dragonFeatures(dragons, sampleTime):
  features = {}

  dragonsA, dragonsB = 0, 0
  for dragon in dragons:
    dragonTime, isTeamOne = dragon
    if dragonTime > sampleTime:
      break

    timeBlock = timeToBlock(dragonTime)

    if isTeamOne:
      dragonsA += 1
      features['dragon-a-{}-{}'.format(timeBlock, dragonsA)] = True
    else:
      dragonsB += 1
      features['dragon-b-{}-{}'.format(timeBlock, dragonsB)] = True

  return features


# Creates several features from towers (team, position)
def towerFeatures(towers, sampleTime):
  features = {}

  for towerData in towers:
    towerTime, towerNum = towerData
    if towerTime > sampleTime:
      break

    timeBlock = timeToBlock(towerTime)


    features['towers-{}-{}'.format(timeBlock, towerNum)] = True

  return features


def parseGameToFeatures(parsed, time=100000):
  gameFeatures = parsed['features']

  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']

  features = {}

  # TODO(sethtroisi): add feature
  #duration = gameFeatures['duration']
  #totalBlocks = duration // SECONDS_PER_BLOCK

  features.update(dragonFeatures(dragons, time))
  features.update(towerFeatures(towers, time))

  goal = parsed['goal']
  return features, goal


def loadOutputFile():
  games = []
  featuresList = []
  goals = []

  dataFile = open(OUTPUT_FILE)
  for line in dataFile.readlines():
    parsed = json.loads(line)

    gameFeatures, goal = parseGameToFeatures(parsed)

    for k, v in gameFeatures.items():
        assert v in (True, False) and type(k) == str
    assert goal in (True, False)

    games.append(parsed)
    featuresList.append(gameFeatures)
    goals.append(goal)

  dataFile.close()

  return games, goals, featuresList


def getGamesData():
  games, goals, featuresList = loadOutputFile()

  sampleSize = len(goals)
  print ("Loaded {} games".format(sampleSize))

  vectorizer = DictVectorizer(sparse=True)

  sparseFeatures = vectorizer.fit_transform(featuresList)

  print ('Data size: {}'.format(sparseFeatures.shape))
  print ('Number non-zero: {}'.format(sparseFeatures.getnnz()))
  print ()

  return games, goals, vectorizer, sparseFeatures

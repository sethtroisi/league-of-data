import json
import math
import re

from collections import Counter

from sklearn.feature_extraction import DictVectorizer


DATA_DIR = '../Data/'
OUTPUT_FILE = DATA_DIR + 'output.txt'

SECONDS_PER_BLOCK = 2 * 60
GOLD_DELTA_BLOCK = 2000

def timeToBlock(time):
  # I think it's more correct to return the block it's happening in.
  # IE event (T = 0) = 0, (0 < T <= 5) = 1
  # This for sure will be a source of off by one errors be wary.
  return (time - 1) // SECONDS_PER_BLOCK + 1


# Create features about dragons taken (team, count)
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
      features['dragon_a_{}_{}'.format(timeBlock, dragonsA)] = True
    else:
      dragonsB += 1
      features['dragon_b_{}_{}'.format(timeBlock, dragonsB)] = True

  return features


# Create features from towers (team, position)
def towerFeatures(towers, sampleTime):
  features = {}

  for towerData in towers:
    towerTime, towerNum = towerData
    if towerTime > sampleTime:
      break

    blockDestroyed = timeToBlock(towerTime)
    features['towers_{}_{}'.format(blockDestroyed, towerNum)] = True

  return features


# Creates features from gold values (delta)
def goldFeatures(gold, sampleTime):
  features = {}

  # TODO(sethtroisi): verify gold use fencpost problem
  lastBlock = timeToBlock(sampleTime)
  for blockNum in range(1, lastBlock-1):
    blockGold = gold.get(str(blockNum), None)
    if not blockGold:
      break

    teamAGold = 0
    teamBGold = 0
    for pId, totalGold in blockGold.items():
      pId = int(pId)

      assert 1 <= pId <= 10
      if 1 <= pId <= 5:
        teamAGold += totalGold
      else:
        teamBGold += totalGold

    delta = teamAGold - teamBGold
    blockedGold = GOLD_DELTA_BLOCK * (delta // GOLD_DELTA_BLOCK)

    feature = 'gold_delta_{}_{}k'.format(
      blockNum, blockedGold // 1000)
    features[feature] = True

  return features

def parseGameToFeatures(parsed, time=None):
  gameFeatures = parsed['features']

  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']
  gold = gameFeatures['gold']

  features = {}

  if time == None:
    duration = gameFeatures['duration']
    time = duration + SECONDS_PER_BLOCK
    # TODO(sethtroisi): add feature without overfitting somehow.
    #lastBlock = timeToBlock(duration);

  features.update(dragonFeatures(dragons, time))
  features.update(towerFeatures(towers, time))
  features.update(goldFeatures(gold, time))

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

def generateFeatureData(featuresList):
  # Note: This is to help find sparse features or to produce cool graphs.
  # It's more one than the other. Take a guess which.

  features = Counter()
  for gameFeatures in featuresList:
    features.update(gameFeatures)

  split = lambda f: re.split('[_]', f)
  splitFeatures = map(split, features.keys())
  baseFeatures = sorted(set([f[0] for f in splitFeatures]))

  for dimension in range(0, 4+1):
    partialFeatures = Counter()
    for f, c in features.items():
      partial = '_'.join(split(f)[:dimension])
      partialFeatures[partial] += c

    print ("With {} dimensions {} features:".format(
        dimension, len(partialFeatures)))

    common = partialFeatures.most_common()
    if len(common) > 10:

      common = common[:5] + [("...", "...")] + common[-5:]

    for pf, c in common:
      print("\t{} x {}".format(pf, c))
    print ()
    #TODO(sethtroisi): cluster features with <= X count back into base names.
    # ie 'gold_belta_34_4k' + 'gold-delta_28_2k' => 'gold_delta' x 2


def getGamesData():
  games, goals, featuresList = loadOutputFile()

  sampleSize = len(goals)
  print ("Loaded {} games".format(sampleSize))

  vectorizer = DictVectorizer(sparse=True)

  sparseFeatures = vectorizer.fit_transform(featuresList)

  print ('Data size: {}'.format(sparseFeatures.shape))
  print ('Number non-zero: {}'.format(sparseFeatures.getnnz()))
  print ()

  generateFeatureData(featuresList)

  return games, goals, vectorizer, sparseFeatures

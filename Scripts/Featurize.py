import json
import re

from collections import Counter
from sklearn.feature_extraction import DictVectorizer

from Util import *


SECONDS_PER_BLOCK = 2 * 60
GOLD_DELTA_BLOCK = 2000


def timeToBlock(time):
  # I think it's more correct to return the block it's happening in.
  # IE event (T = 0) = 0, (0 < T <= 5) = 1
  # This for sure will be a source of off by one errors be wary.
  return (time - 1) // SECONDS_PER_BLOCK + 1


# Create a feature that counts how many events of the type have happened.
def countedFeature(name, events, sampleTime, order=False, verus=True):
  features = set()

  eventOrder = ''
  for event in events:
    eventTime, isTeamOne = event[:2]
    assert 0 <= eventTime <= 10000
    assert isTeamOne in (True, False)

    if eventTime > sampleTime:
      break

    teamLetter = 'A' if isTeamOne else 'B'
    eventOrder += teamLetter
    features.add('{}_{}_{}'.format(
        name, teamLetter, eventOrder.count(teamLetter)))

    if order:
      features.add('{}_order_{}'.format(
            name, eventOrder))

  if verus:
    features.add('{}_{}_to_{}'.format(
          name, eventOrder.count('A'), eventOrder.count('B')))

  return features


# Create features from champs
def champFeature(champs, sampleTime):
  features = set()

  teamAChamps, teamBChamps = champs

  lastBlock = timeToBlock(sampleTime)
  for block in range(0, lastBlock + 1):
    for team, teamChamps in (('A', teamAChamps), ('B', teamBChamps)):
      for champ in teamChamps:
        features.add('champ_{}_{}_{}'.format(team, champ, block))

  return features


# Create features from towers (team, position)
def towerFeatures(towers, sampleTime):
  features = set()

  towersA, towersB = 0, 0
  for towerData in towers:
    towerTime, isTeamA, towerNum = towerData
    if towerTime > sampleTime:
      break

    if teamATowerKill(towerNum):
      towersA += 1
    else:
      towersB += 1

    timeBlock = timeToBlock(towerTime)
    features.add('towers_{}'.format(towerNum))
    features.add('towers_{}_{}'.format(timeBlock, towerNum))

  features.add('towerskilled_{}_{}'.format(towersA, towersB))

  return features


# Creates features from gold values (delta)
def goldFeatures(gold, sampleTime):
  features = set()

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

    for thousands in range(1, 100):
      if thousands * 1000 < teamAGold:
        features.add('gold_a_{}k'.format(thousands))

      if thousands * 1000 < teamBGold:
        features.add('gold_b_{}k'.format(thousands))

    delta = teamAGold - teamBGold
    blockedGold = GOLD_DELTA_BLOCK * (delta // GOLD_DELTA_BLOCK)
    features.add('gold_delta_{}_{}k'.format(blockNum, blockedGold // 1000))

  return features

def parseGameToFeatures(parsed, time=None):
  gameFeatures = parsed['features']

  # Gold
  gold = gameFeatures['gold']

  # Objectives
  barons = gameFeatures['barons']
  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']
  inhibs = gameFeatures['inhibs']

  # Champions
  champs = gameFeatures['champs']

  # Other
  pinkWards = gameFeatures['pinkWards']
  yellowWardsA = gameFeatures['stealthWards2Min']
  yellowWardsB = gameFeatures['stealthWards3Min']
  yellowWardsCombined = sorted(yellowWardsA + yellowWardsB)

  features = set()

  if time == None:
    duration = parsed['debug']['duration']
    time = duration + SECONDS_PER_BLOCK

  features.update(towerFeatures(towers, time))
  features.update(goldFeatures(gold, time))

  features.update(countedFeature('barons', barons, time))
  features.update(countedFeature('dragons', dragons, time))

  features.update(countedFeature('inhibs', inhibs, time))

  # TODO(sethtroisi): investigate why this increases log loss.
  features.update(champFeature(champs, time))

  features.update(countedFeature(
      'pinkwards', pinkWards, time,
      order = False, verus = False))
  features.update(countedFeature(
      'yellowwards', yellowWardsCombined, time,
      order = False, verus = False))

  # Model expects results in a dictionary format so map features to True.
  return dict((f, True) for f in features)


def loadOutputFile(fileName):
  games = []
  featuresList = []
  goals = []

  outputData = loadJsonFile(fileName)
  for data in outputData:
    goal = data['goal']
    gameFeatures = parseGameToFeatures(data)

    for k, v in gameFeatures.items():
        assert v in (True, False) and type(k) == str
    assert goal in (True, False)

    games.append(data)
    featuresList.append(gameFeatures)
    goals.append(goal)

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


def getGamesData(fileName):
  games, goals, featuresList = loadOutputFile(fileName)

  sampleSize = len(goals)
  print ("Loaded {} games".format(sampleSize))

  vectorizer = DictVectorizer(sparse=True)

  sparseFeatures = vectorizer.fit_transform(featuresList)

  print ('Data size: {}'.format(sparseFeatures.shape))
  print ('Number non-zero: {}'.format(sparseFeatures.getnnz()))
  print ()

  # TODO(sethtroisi): add this under a flag.
  #generateFeatureData(featuresList)

  return games, goals, vectorizer, sparseFeatures

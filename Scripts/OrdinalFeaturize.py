import json
import re

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
  '''features = set()

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

  return features'''


# Create features from champs
def champFeature(champs, sampleTime):
  '''features = set()

  teamAChamps, teamBChamps = champs

  lastBlock = timeToBlock(sampleTime)
  for block in range(0, lastBlock + 1):
    for team, teamChamps in (('A', teamAChamps), ('B', teamBChamps)):
      for champ in teamChamps:
        features.add('champ_{}_{}_{}'.format(team, champ, block))

  return features'''


# Create features from towers (team, position)
def towerFeatures(towers, sampleTime):
  features = []

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
    #features['towers_{}'.format(towerNum)] = towerTime

  features.append(towersA) #['towers_killed_a'] = towersA
  features.append(towersB) #['towers_killed_b'] = towersB

  return features


# Creates features from gold values (delta)
def goldFeatures(gold, sampleTime):
  '''features = set()

  # TODO(sethtroisi): verify gold use fencpost problem
  lastBlock = timeToBlock(sampleTime)
  for blockNum in range(1, lastBlock):
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

  return features'''

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
  #pinkWards = gameFeatures['pinkWards']
  #yellowWardsA = gameFeatures['stealthWards2Min']
  #yellowWardsB = gameFeatures['stealthWards3Min']
  #yellowWardsCombined = sorted(yellowWardsA + yellowWardsB)

  features = []

  if time == None:
    duration = parsed['debug']['duration']
    time = duration + SECONDS_PER_BLOCK

  features += towerFeatures(towers, time)
  #features.update(goldFeatures(gold, time))

  #features.update(countedFeature('barons', barons, time))
  #features.update(countedFeature('dragons', dragons, time))

  #features.update(countedFeature('inhibs', inhibs, time))

  # TODO(sethtroisi): investigate why this increases log loss.
  #features.update(champFeature(champs, time))

  #features.update(countedFeature(
  #    'pinkwards', pinkWards, time,
  #    order = False, verus = False))
  #features.update(countedFeature(
  #    'yellowwards', yellowWardsCombined, time,
  #    order = False, verus = False))

  # Model expects results in a dictionary format so map features to True.
  return features


def loadOutputFile(fileName):
  games = []
  featuresList = []
  goals = []

  outputData = loadJsonFile(fileName)
  for data in outputData:
    goal = data['goal']
    gameFeatures = parseGameToFeatures(data)

    #for k, v in gameFeatures.items():
    #    assert type(v) == int  and type(k) == str
    assert goal in (True, False)

    games.append(data)
    featuresList.append(gameFeatures)
    goals.append(goal)

  return games, goals, featuresList


def getGamesData(fileName):
  games, goals, featuresList = loadOutputFile(fileName)

  sampleSize = len(goals)
  print ("Loaded {} games".format(sampleSize))

  #print ('Data size: {}'.format(sparseFeatures.shape))
  #print ('Number non-zero: {}'.format(sparseFeatures.getnnz()))
  print ()

  # TODO(sethtroisi): add this under a flag.
  #generateFeatureData(featuresList)

  return games, goals, featuresList

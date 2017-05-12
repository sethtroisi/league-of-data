import json
import re
import pandas

from collections import Counter

from Util import *


SECONDS_PER_BLOCK = 2 * 60
GOLD_DELTA_BUCKET_SIZE = 2000


def timeToBlock(time):
  # I think it's more correct to return the block it's happening in.
  # IE event (T = 0) = 0, (0 < T <= 5) = 1
  # This for sure will be a source of off by one errors be wary.
  return (time - 1) // SECONDS_PER_BLOCK + 1


# Create a feature that counts how many events of the type have happened.
def countedFeature(df, name, events, sampleTime, verus=True):
  counts = [0, 0]
  for event in events:
    eventTime, isTeamOne = event[:2]
    assert 0 <= eventTime <= 10000
    assert isTeamOne in (True, False)

    if eventTime > sampleTime:
      break

    counts[isTeamOne] += 1
    feature = '{}_{}_{}'.format(name, 'A' if isTeamOne else 'B', counts[isTeamOne])
    df.set_value(0, feature, 1.0)

  if verus:
    feature = '{}_{}_to_{}'.format(name, counts[0], counts[1])
    df.set_value(0, feature, 1.0)


# Create features from champs
def champFeature(champs, sampleTime):
  features = set()

  teamAChamps, teamBChamps = champs

  #lastBlock = timeToBlock(sampleTime)
  #for block in range(0, lastBlock + 1):
  #  for team, teamChamps in (('A', teamAChamps), ('B', teamBChamps)):
  #    for champ in teamChamps:
  #      features.add('champ_{}_{}_{}'.format(team, champ, block))

  for team, teamChamps in (('A', teamAChamps), ('B', teamBChamps)):
    for champ in teamChamps:
      # TODO remove the strip after GameParser finishs.
      features.add('c_{}_{}'.format(team, filter(str.isalnum, champ)))

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
    features.add('t_d_{}'.format(towerNum))
    features.add('t_d_at_{}_{}'.format(timeBlock, towerNum))

  features.add('t_d_{}_{}'.format(towersA, towersB))

  return features


# Creates features from gold values (delta)
def goldFeatures(df, gold, sampleTime):
  lastBlock = timeToBlock(sampleTime)
  for blockNum in range(lastBlock+1):
    blockGold = gold.get(str(blockNum), None)
    if not blockGold:
      continue

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
        df.set_value(0, 'g_a_{}k'.format(thousands), 1.0)

      if thousands * 1000 < teamBGold:
        df.set_value(0, 'g_b_{}k'.format(thousands), 1.0)

    deltaSign = teamAGold > teamBGold
    delta = abs(teamAGold - teamBGold)
    bucketsOfGold = delta // GOLD_DELTA_BUCKET_SIZE
    for bucket in range(bucketsOfGold):
      feature = 'g_d_{}_{}_{}'.format(
          blockNum,
          'p' if deltaSign else 'n', 
          bucket * GOLD_DELTA_BUCKET_SIZE)
      df.set_value(0, feature, 1.0)


def parseGameToPD(index, parsed, time=None):
  if time == None:
    duration = parsed['debug']['duration']
    time = duration

  gameFeatures = parsed['features']

  # Gold
  gold = gameFeatures['gold']

  # Objectives
  #barons = gameFeatures['barons']
  #dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']
  inhibs = gameFeatures['inhibs']

  # Champions
  #champs = gameFeatures['champs']

  # Other

  df = pandas.DataFrame()


  goldFeatures(df, gold, time)

  #features.update(towerFeatures(towers, time))
  countedFeature(df, 'inhibs', inhibs, time)

  #features.update(countedFeature('barons', barons, time))
  #features.update(countedFeature('dragons', dragons, time))

  #features.update(champFeature(champs, time))

  df.index = [index]
  return df

def loadOutputFile(fileName):
  games = []
  goals = []

  outputData = loadJsonFile(fileName)
  for dataI, data in enumerate(outputData):
    goal = data['goal']
    
    #if dataI > 1000:
    #  break

    assert goal in (True, False)

    games.append(data)
    goals.append(goal)
  return games, goals


def getRawGameData(fileName):
  # Consider passing in a variable to limit number loaded
  games, goals = loadOutputFile(fileName)
  print ("Loaded {} games".format(len(goals)))
  return games, goals


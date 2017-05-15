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
def towerFeatures(df, towers, sampleTime):
  features = set()

#  for tower in range(0, 24):
#    df.set_value(0, 'tower_{}_destroyed_at',     

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
  lastBlockNum = max(b for b in map(int, gold.keys()) if b <= lastBlock)

  '''
  blockGold = gold.get(str(lastBlockNum), None)
  assert blockGold, gold.keys()

  teamAGold = 0
  teamBGold = 0
  for pId, totalGold in blockGold.items():
    pId = int(pId)

    assert 1 <= pId <= 10
    if 1 <= pId <= 5:
      teamAGold += totalGold
    else:
      teamBGold += totalGold

  df.set_value(0, 'gold_a', teamAGold // 1000)
  df.set_value(0, 'gold_b', teamBGold // 1000)
  df.set_value(0, 'gold_a_adv', max(0, teamAGold - teamBGold) // 1000) 
  df.set_value(0, 'gold_b_adv', max(0, teamBGold - teamAGold) // 1000) 

  '''
  for blockNum in range(lastBlock+1):
    teamAGold = 0
    teamBGold = 0

    blockGold = gold.get(str(blockNum), None)
    if blockGold:
      for pId, totalGold in blockGold.items():
        pId = int(pId)

        assert 1 <= pId <= 10
        if 1 <= pId <= 5:
          teamAGold += totalGold
        else:
          teamBGold += totalGold

    # Each team gets ~3k more gold every 2 minutes, makes vars ~ (0, 1.5]
    normalizeFactor = 3000 * (blockNum + 1)
    df.set_value(0, 'gold_a_block_{}'.format(blockNum), teamAGold / normalizeFactor)
    df.set_value(0, 'gold_b_block_{}'.format(blockNum), teamBGold / normalizeFactor)
    
    # A huge win is +15k gold at 40 minutes so maybe ~1k every 2 minutes, to get [-2, +2]
    deltaGold = teamAGold - teamBGold
    normalizeFactor = 1000 * (blockNum + 1)
    df.set_value(0, 'gold_a_adv_block_{}'.format(blockNum), deltaGold / normalizeFactor) 
  

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

  df = pandas.DataFrame()# columns = columnGuess)
  #print (columnGuess, df)


  goldFeatures(df, gold, time)

  towerFeatures(df, towers, time)
  countedFeature(df, 'inhibs', inhibs, time)

  #features.update(countedFeature('barons', barons, time))
  #features.update(countedFeature('dragons', dragons, time))

  #features.update(champFeature(champs, time))

  df.index = [index]
  return df

def loadOutputFile(fileName, numGames):
  games = []
  goals = []

  outputData = loadJsonFile(fileName)
  for dataI, data in enumerate(outputData):
    if data['debug']['duration'] < 600:
      # Filtering remakes and stuff
      continue
  
    if len(games) == numGames:
      break

    goal = data['goal']
    assert goal in (True, False)

    games.append(data)
    goals.append(goal)
  return games, goals


def getRawGameData(fileName, numGames):
  # Consider passing in a variable to limit number loaded
  games, goals = loadOutputFile(fileName, numGames)
  print ("Loaded {} games".format(len(goals)))
  return games, goals


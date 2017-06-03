import json
import re

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
    eventTime, isTeamA = event[:2]
    assert 0 <= eventTime <= 10000
    assert isTeamA in (True, False)

    if eventTime > sampleTime:
      break

    counts[isTeamA] += 1
    feature = '{}_{}_{}'.format(name, 'A' if isTeamA else 'B', counts[isTeamA])
    df[feature] = 1.0

  if verus:
    feature = '{}_{}_to_{}'.format(name, counts[0], counts[1])
    df[feature] = 1.0


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
#  for tower in range(0, 24):
#    df.set_value(0, 'tower_{}_destroyed_at',     

  towersA, towersB = 0, 0
  for towerData in towers:
    towerTime, isTeamA, towerNum = towerData
    if towerTime > sampleTime:
      break

    if isTeamA:
      towersA += 1
      if towersA == 1:
        df['first_tower_A'] = towerTime / 1800
      df['last_tower_A'] = towerTime / 1800
    else:
      towersB += 1
      if towersB == 1:
        df['first_tower_B'] = towerTime / 1800
      df['last_tower_B'] = towerTime / 1800

    # TODO figure out how to default other values to infinite or something
    df['tower_{}_destroyed'.format(towerNum)] = 1
    df['tower_{}_destroyed_at'.format(towerNum)] = towerTime / 1800

  df['towers_destroyed_A'] = towersA
  df['towers_destroyed_B'] = towersB


def dragonFeatures(df, dragons, sampleTime):
  features = set()

  dragonsA = []
  dragonsB = []

  for dragonI, dragon in enumerate(dragons, 1):
    dragonTime, name, isTeamA = dragon
    if dragonTime > sampleTime:
      break

    if isTeamA:
      df['last_dragon_A'] = dragonTime / 1800
      dragonsA.append(name)
    else:
      df['last_dragon_B'] = dragonTime / 1800
      dragonsB.append(name)
    
    df['dragon_{}_taken_at'.format(dragonI)] = dragonTime / 1800
    df['dragon_{}_taken_by'.format(dragonI)] = 1 if isTeamA else -1

    # TODO last dragon_a
    
  df["dragon_taken_A"] = len(dragonsA)
  df["dragon_taken_B"] = len(dragonsB)            

  for dType in set(dragonsA + dragonsB):
    name = dType.lower()
    df["dragon_A_" + name] = dragonsA.count(dType)
    df["dragon_B_" + name] = dragonsB.count(dType)

  return features

# Creates features from gold values (delta)
def goldFeatures(df, gold, sampleTime):
  lastBlock = timeToBlock(sampleTime)
  lastBlockNum = max(b for b in map(int, gold.keys()) if b <= lastBlock)

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
    df['gold_a_block_{}'.format(blockNum)] = teamAGold / normalizeFactor
    df['gold_b_block_{}'.format(blockNum)] = teamBGold / normalizeFactor
    
    # A huge win is +15k gold at 40 minutes so maybe ~1k every 2 minutes, to get [-2, +2]
    deltaGold = teamAGold - teamBGold
    normalizeFactor = 1000 * (blockNum + 1)
    df['gold_a_adv_block_{}'.format(blockNum)] = deltaGold / normalizeFactor


# Create a feature that counts how many events of the type have happened.
def countedFeature(df, name, events, sampleTime, verus=True):
  counts = [0, 0]
  for event in events:
    eventTime, isTeamA = event[:2]
    assert 0 <= eventTime <= 10000
    assert isTeamA in (True, False)

    if eventTime > sampleTime:
      break

    counts[isTeamA] += 1
    feature = '{}_{}_{}'.format(name, 'A' if isTeamA else 'B', counts[isTeamA])
    df[feature] = 1.0

  if verus:
    feature = '{}_{}_to_{}'.format(name, counts[0], counts[1])
    df[feature] = 1.0


def parseGame(index, parsed, time):
  if time == None:
    assert False

  gameFeatures = parsed['features']

  # Gold
  gold = gameFeatures['gold']

  # Objectives
  barons = gameFeatures['barons']
  dragons = gameFeatures['dragons']
  towers = gameFeatures['towers']
  inhibs = gameFeatures['inhibs']

  # Champions
  #champs = gameFeatures['champs']


  # Data that ML will see

  data = {}
  data['current_time'] = time / 3600

  goldFeatures(data, gold, time)
  towerFeatures(data, towers, time)
  dragonFeatures(data, dragons, time)

  countedFeature(data, 'inhibs', inhibs, time)
  countedFeature(data, 'barons', barons, time)

  #features.update(champFeature(champs, time))

  return data


def rankOrdering(rank):
  order = {
    'BRONZE': 0,
    'SILVER': 1,
    'UNRANKED': 1,
    'GOLD': 2,
    'PLATINUM': 3,
    'DIAMOND': 4,
    'CHALLENGER': 5,
    'MASTER': 6
  }
  assert rank in order, "Unknown Rank: '{}'".format(rank)
  return order[rank]


def getRawGameData(args):
  fileName = args.input_file
  numGames = args.num_games
  rank = args.rank

  games = []
  goals = []

  filtered = 0

  requiredRank = rankOrdering(rank) 

  outputData = loadJsonFile(fileName)
  for dataI, data in enumerate(outputData):
    if data['debug']['duration'] < 600:
      # Filtering remakes and stuff
      continue
  
    # TODO consider removing surrender games
  
    # Filter out low rank games
    lowerRanked = len([1 for c in data['features']['champs'] if rankOrdering(c['approxRank']) < requiredRank])
    if lowerRanked >= 2:
      filtered += 1
      continue

    goal = data['goal']
    assert goal in (True, False)

    games.append(data)
    goals.append(goal)

    if len(games) == numGames:
      break

  print ("Loaded {} games (filtereed {})".format(
      len(goals), filtered))
  return games, goals


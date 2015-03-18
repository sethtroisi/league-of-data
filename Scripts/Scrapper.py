import json
import random
import time
import urllib.request

# TODO(riechelp): convert all * imports to named imports.
from Util import *

# API REFERENCE
# https://developer.riotgames.com/api/methods
#
# Lots of information here
# https://developer.riotgames.com/docs/game-constants
#
# Game API
# https://developer.riotgames.com/api/methods#!/933/3233
#
# Match API (this is what we want most)
# https://developer.riotgames.com/api/methods#!/929/3214

API_KEY = '38940d99-0f69-4dfd-a2ea-5e9de11552a3'
BASE_URL = 'https://na.api.pvp.net/api/lol/'
KEY_PARAM = 'api_key={}'.format(API_KEY)
SLEEP_TIME = 1.5
GAMES_PER_SUMMONER = 3

def buildUrl(apiPath, params = []):
  urlParams = '&'.join([KEY_PARAM] + params)
  return BASE_URL + apiPath + '?' + urlParams


def getParsedResponse(url):
  # TODO(sethtroisi): move rate limiting here?
  response = urllib.request.urlopen(url)
  data = response.read()
  # TODO(sethtroisi): Verify utf-8 is correct assumption.
  stringData = data.decode('utf-8')
  return json.loads(stringData)


# Get a list of summeronIds from a list of names
# Ex: ['inkaruga', 'kingvash']
def getSummonerId(names):
  apiFormat = 'na/v1.4/summoner/by-name/{summonerNames}'
  joinedNames = ','.join(names)
  apiPath = apiFormat.format(summonerNames = joinedNames)
  url = buildUrl(apiPath)
  time.sleep(SLEEP_TIME)

  parsed = getParsedResponse(url)

  ids = {}
  for name in names:
    ids[name] = parsed[name]['id']
  return ids


def getMatchHistory(summonerId):
  apiFormat = 'na/v2.2/matchhistory/{summonerId}'
  apiPath = apiFormat.format(summonerId = summonerId)
  url = buildUrl(apiPath)
  time.sleep(SLEEP_TIME)

  return getParsedResponse(url)


def getMatch(matchId):
  apiFormat= 'na/v2.2/match/{matchId}'
  apiPath = apiFormat.format(matchId = matchId)
  url = buildUrl(apiPath, ['includeTimeline=True'])
  time.sleep(SLEEP_TIME)

  return getParsedResponse(url)


def getSummonerMatches(summonerId):
  # TODO(sethtroisi): Use rankedQueues query param.
  history = getMatchHistory(summonerId)
  saveName = 'matchHistory/getMatchHistory-{}'.format(summonerId)
  #writeJsonFile(saveName, history)
  #history = loadJsonFile(saveName)

  matches = history['matches']
  assert len(matches) > 0

  fellowSummoners = {}
  fullMatches = {}
  for match in matches[:GAMES_PER_SUMMONER]:
    matchId = match['matchId']
    queueType = match['queueType']
    region = match['region']
    season = match['season']

    # TODO(sethtroisi): count number of filtered games (by filter).
    if queueType != 'RANKED_SOLO_5x5':
      continue

    if region != 'NA':
      continue

    if season != 'SEASON2015':
      continue

    print ('\tFetching match (id: {})'.format(matchId))
    fullMatch = getMatch(matchId)
    matchSaveName = 'matches/getMatch-{}'.format(matchId)
    writeJsonFile(matchSaveName, fullMatch)
    #fullMatch = loadJsonFile(matchSaveName)
    fullMatches[matchId] = fullMatch

    # Seems to only return information on self? duo?
    #otherParticipants = match['participantIdentities']
    otherParticipants = fullMatch['participantIdentities']
    for participant in otherParticipants:
      player = participant['player']
      summonerId = player['summonerId']
      summonerName = player['summonerName']
      fellowSummoners[summonerName] = summonerId

  return fullMatches, fellowSummoners


def main():
  # Id -> Name (all seen summoners)
  summoners = {}

  # Id -> Name (not yet processed) (only add if not in summoners)
  unvisited = {}

  # MatchId -> Match.
  matches = {}

  seedNames = ['inkaruga', 'kingvash']
  seedIds = getSummonerId(seedNames)
  #writeJsonFile('example-getSummonerId', seedIds)
  #seedIds = loadJsonFile('example-getSummonerId')

  for name, sumId in seedIds.items():
    summoners[sumId] = name
    unvisited[sumId] = name

  while len(matches) < 5000:
    newId = random.choice(list(unvisited.keys()))
    newName = unvisited[newId]
    # Remove from the list of unprocessed
    del unvisited[newId]

    print ('Exploying \'{}\'(id: {}) ({} of {} unexplored) ({} games)'.format(
        newName, newId, len(unvisited), len(summoners), len(matches)))

    newMatches, fellowSummoners = getSummonerMatches(newId)

    # TODO(sethtroisi): make this unique games/summoners
    print ('\tAdded {} games, {} summoners'.format(
        len(newMatches), len(fellowSummoners)))

    matches.update(newMatches)
    for fName, fId in fellowSummoners.items():
      if fId not in summoners:
        summoners[fId] = fName
        unvisited[fId] = fName


if __name__ == '__main__':
  main()

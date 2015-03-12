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

API_KEY = "38940d99-0f69-4dfd-a2ea-5e9de11552a3"
BASE_URL = 'https://na.api.pvp.net/api/lol/'
HASH_PARAMS = 'api_key={}'.format(API_KEY)


def buildUrl(apiPath):
  return BASE_URL + apiPath + '?' + HASH_PARAMS


def getParsedResponse(url):
  # TODO(sethtroisi): move rate limiting here?
  response = urllib.request.urlopen(url)
  data = response.read()
  # TODO(sethtroisi): Verify utf-8 is correct assumption.
  stringData = data.decode("utf-8")
  return json.loads(stringData)


# Get a list of summeronIds from a list of names
# Ex: ['inkaruga', 'kingvash']
def getSummonerId(names):
  apiFormat = 'na/v1.4/summoner/by-name/{summonerNames}'
  joinedNames = ','.join(names)
  apiPath = apiFormat.format(summonerNames = joinedNames)
  url = buildUrl(apiPath)
  time.sleep(1)

  parsed = getParsedResponse(url)

  ids = {}
  for name in names:
    ids[name] = parsed[name]['id']
  return ids


def getMatchHistory(summonerId):
  apiFormat = 'na/v2.2/matchhistory/{summonerId}'
  apiPath = apiFormat.format(summonerId = summonerId)
  url = buildUrl(apiPath)
  time.sleep(1)

  return getParsedResponse(url)


def getMatch(matchId):
  apiFormat= 'na/v2.2/match/{matchId}'
  apiPath = apiFormat.format(matchId = matchId)
  url = buildUrl(apiPath)
  time.sleep(1)

  return getParsedResponse(url)


def getChampIds():
  apiPrefix = 'static-data/na/v1.2/champion/{champId}'

  champMap = {}
  for champ in range(1, 270):
    try:
      time.sleep(0.5)
      apiPath = apiPrefix + str(champ)
      url = buildUrl(apiPath)

      parsed = getParsedResponse(url)
      name = parsed['name']

      print (parsed, name)

      champMap[name] = champ
    except urllib.request.HTTPError:
      print ("failed to find:", champ)

  print (sorted(champMap.items()))
  print ("{} champions found".format(len(champMap)))
  return champMap


def getSummonerMatches(summonerId):
  # TODO(sethtroisi): Use rankedQueues query param.
  history = getMatchHistory(summonerId)
  writeJsonFile('example-getMatchHistory', history)
  #history = loadJsonFile('example-getMatchHistory')

  matches = history['matches']
  assert len(matches) > 0

  fellowSummoners = {}
  fullMatches = {}
  for match in matches[:2]:
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

    print ("fetching/saving match (id: {})".format(matchId))
    fullMatch = getMatch(matchId)
    writeJsonFile('example-getMatch-{}'.format(matchId), fullMatch)
    #fullMatch = loadJsonFile('example-getMatch-{}'.format(matchId))
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
  #seedIds = getSummonerId(seedName)
  #writeJsonFile('example-getSummonerId', seedIds
  seedIds = loadJsonFile('example-getSummonerId')

  for name, sumId in seedIds.items():
    summoners[sumId] = name
    unvisited[sumId] = name

  while len(matches) < 15:
    newId = random.choice(list(unvisited.keys()))
    newName = unvisited[newId]
    # Remove from the list of unprocessed
    del unvisited[newId]

    print ("Exploying {} ({}) ({} summoners {} visited {} games)".format(
        newName, newId, len(summoners), len(unvisited), len(matches)))

    newMatches, fellowSummoners = getSummonerMatches(newId)

    matches.update(newMatches)
    for fName, fId in fellowSummoners.items():
      if fId not in summoners:
        summoners[fId] = fName
        unvisited[fId] = fName


if __name__ == "__main__":
  main()

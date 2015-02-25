import json
import urllib.request
import time
#import IPython
#IPython.start_ipython(argv=[])

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
  time.sleep(0.5)

  parsed = getParsedResponse(url)

  ids = {}
  for name in names:
    ids[name] = parsed[name]['id']
  return ids


#summonerIds = getSummonerId(['inkaruga', 'kingvash'])
summonerIds = {'inkaruga': 22809484, 'kingvash': 25226531}
print (summonerIds)

summonerId = summonerIds[0]


def getMatchHistory(summonerId):
  apiFormat = 'na/v2.2/matchhistory/{summonerId}'
  apiPath = apiFormat.format(summonerId = summonerId)
  url = buildUrl(apiPath)
  time.sleep(0.5)

  parsed = getParsedResponse(url)
  print (len(parsed))
  return parsed


def getMatch(matchId):
  apiFormat= 'na/v2.2/match/{matchId}'
  apiPath = apiFormat.format(matchId = matchId)
  url = buildUrl(apiPath)
  time.sleep(0.5)

  parsed = getParsedResponse(url)
  print (len(parsed))


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

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
  response = urllib.request.urlopen(url)
  data = response.read()
  # TODO(sethtroisi): verify utf-8 is correct assumption.
  stringData = data.decode("utf-8")
  return json.loads(stringData)


def getChampIds():
  apiPrefix = 'static-data/na/v1.2/champion/'

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


getChampIds()

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

baseUrl = 'https://na.api.pvp.net/api/lol/'
partial = 'static-data/na/v1.2/champion/'
hashParam = '?api_key=38940d99-0f69-4dfd-a2ea-5e9de11552a3'

champMap = {}
for champ in range(265, 270):
  try:
    time.sleep(0.5)
    url = baseUrl + partial + str(champ) + hashParam
    print (url)
    response = urllib.request.urlopen(url)
    data = response.read()
    stringData = data.decode("utf-8")
    parsed = json.loads(stringData)
    name = parsed['name']
    print (parsed, name)
    champMap[name] = champ
  except:
    print ("failed to find:", champ)

print (champMap)

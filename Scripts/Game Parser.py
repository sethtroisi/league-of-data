import json
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

DATA_DIR = '../Data/'

FILE_NAME = DATA_DIR + 'matches{}.json'
OUTPUT_FILE = DATA_DIR + 'output.txt'


# takes in a match json object and returns features about it.
def parseGameRough(match):
#  print ("keys:", match.keys())

  teamInfo = match['participants']

  # for now we return champtionId, firstDragon team, firstDragon time
  championIds = [p['championId'] for p in teamInfo]
  participantIds = [p['participantId'] for p in teamInfo]
  teamOne = participantIds[:5]
  teamTwo = participantIds[5:]

  teamOneChampIds = championIds[:5]
  teamTwoChampIds = championIds[5:]
 
  teamOneChampFeatures = []
  teamTwoChampFeatures = []
  for champ, champId in getChamps():
      teamOneChampFeatures.append(champId in teamOneChampIds)
      teamTwoChampFeatures.append(champId in teamTwoChampIds)
 
  assert teamOneChampFeatures.count(True) == 5
  assert teamTwoChampFeatures.count(True) == 5
#  print ("Champions: {}".format(championIds))
#  champNames = list(map(championIdToName, championIds))
#  print ("Names: {}".format(champNames))

  dragons = []
  frames = match['timeline']['frames']
  for frame in frames:
    events = frame.get('events', [])
    for event in events:
      monsterType = event.get('monsterType', None)
      if monsterType == 'DRAGON':
        time = event['timestamp']
        killer = event['killerId']
#        print ("killer {} @{:.0f}s".format(
#          champNames[killer-1], time / 1000))
        dragons.append((time, killer))

  firstDragonFeature = [False, False, 10 ** 10]
  if len(dragons) > 0:
    firstDragonFeature[0] = dragons[0][1] in teamOne
    firstDragonFeature[1] = dragons[0][1] in teamTwo
    firstDragonFeature[2] = dragons[0][0]

  features = firstDragonFeature
#    teamOneChampFeatures + \
#    teamTwoChampFeatures

  result = match['teams'][0]['winner']
  return result, features


# MAIN CODE

output = open(OUTPUT_FILE, mode='w')

gameNum = 0
for fileNumber in range(1, 11):
  fileData = loadFile(FILE_NAME.format(fileNumber))
  parsed = json.loads(fileData)

  games = parsed['matches']
  for game in games:
    result, features = parseGameRough(game)
    jsonObject = {'goal': result, 'features': features}
    jsonString = json.dumps(jsonObject)

    output.write(jsonString + '\n')

#    print ('{} <= {}'.format(result, features))
    gameNum += 1

print ("parsed {} games".format(gameNum))

output.close()

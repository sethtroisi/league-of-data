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
  champNames = list(map(championIdToName, championIds))
#  print ("Champions: {}".format(championIds))
#  print ("Names: {}".format(champNames))

  dragons = []
  towers = []
  frames = match['timeline']['frames']
  for frame in frames:
    events = frame.get('events', [])
    for event in events:
      monsterType = event.get('monsterType', None)
      if monsterType == 'DRAGON':
        time = event['timestamp'] // 1000
        killer = event['killerId']
        isTeamOne = killer in teamOne
        dragons.append((time, isTeamOne))

      buildingType = event.get('buildingType', None)
      if buildingType == 'TOWER_BUILDING':
        time = event['timestamp'] // 1000
        killer = event['killerId']
        towerType = event['towerType']
        laneType = event['laneType']
        isTeamOne = event['teamId'] == 100

        if towerType == 'FOUNTAIN_TURRET':
          # TODO(sehtroisi): figure out what causes this.
          continue

        towerNum = getTowerNumber(isTeamOne, laneType, towerType)

#        print ("killer {} @{:.0f}s: ({} x {} x {}) = {}".format(
#          champNames[killer - 1], time / 1000,
#          isTeamOne, laneType, towerType, towerNum))

        # TODO(sethtroisi): Stop mid nexus turret double count.
        #assert all(tNum != towerNum for t, k, tNum in towers)
        towers.append((time, towerNum))

  features = dict()
  features['dragons'] = dragons
  features['towers'] = towers

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

output.close()

print ("parsed {} games".format(gameNum))

example = jsonString[:70] + ('...' if (len(jsonString) > 70) else '')
print ("example line: '{}'".format(example))

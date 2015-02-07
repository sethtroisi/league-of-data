import argparse
import json
import random
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


def getArgParse():
  parser = argparse.ArgumentParser(description='Parses Games and produces features.')
  parser.add_argument(
      '-f', '--full-examples',
      action="store_true",
      help='print full examples')
  parser.add_argument(
      '-e', '--examples',
      type=int,
      default=2,
      help='how many examples to print')
  parser.add_argument(
      '-l', '--limited',
      action="store_true",
      help='run over only the first match file')
  parser.add_argument(
      '-n', '--dry-run',
      action="store_true",
      help='don\'t write output file instead print to screen')
  return parser


# takes in a match json object and returns features about it.
def parseGameRough(match):
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
  features['duration'] = match['matchDuration']

  result = match['teams'][0]['winner']
  return result, features


def main(args):

  gameNum = 0
  outputData = []

  lastFile = 1 if args.limited else 10
  for fileNumber in range(1, lastFile + 1):
    fileData = loadFile(FILE_NAME.format(fileNumber))
    parsed = json.loads(fileData)

    games = parsed['matches']
    for game in games:
      result, features = parseGameRough(game)
      jsonObject = {'goal': result, 'features': features}
      jsonString = json.dumps(jsonObject)

      outputData.append(jsonString)
      gameNum += 1

  print ("parsed {} games".format(gameNum))

  if not args.dry_run:
    output = open(OUTPUT_FILE, mode='w')
    for line in outputData:
        output.write(line + '\n')
    output.close()

    print ("wrote games to output file ('{}')".format(OUTPUT_FILE))
  else:
    for line, data in enumerate(outputData, 1):
      print ('{}: {}'.format(line, data))

  exampleLines = random.sample(range(gameNum), args.examples)
  for exampleLine in exampleLines:
    game = outputData[exampleLine]
    if args.full_examples:
      example = game
    else:
      example = game[:70] + ('..' if len(jsonString) > 70 else '')

    print ()
    print ("line {}: '{}'".format(exampleLine, example))


# START CODE HERE
args = getArgParse().parse_args()
main(args)

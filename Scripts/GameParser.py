import argparse
import json
import random
from Featurize import *
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
  assert teamOne == [1, 2, 3, 4, 5]
  assert teamTwo == [6, 7, 8, 9, 10]

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
  barons = []
  towers = []
  inhibs = []
  pinkWards = []
  stealthWards3Min = []
  stealthWards2Min = []
  gold = {}
  frames = match['timeline']['frames']
  for frame in frames:
    frameTime = frame['timestamp'] // 1000
    blockNum = timeToBlock(frameTime)

    frameGold = {}
    # NOTE: frames appear to be 60seconds
    gold[blockNum+1] = frameGold
    for pId, pFrame in frame['participantFrames'].items():
      # TODO(sethtroisi): use item gold instead of totalGold
      frameGold[pId] = pFrame['totalGold']

    events = frame.get('events', [])
    for event in events:
      monsterType = event.get('monsterType', None)
      if monsterType:
        time = event['timestamp'] // 1000
        killer = event['killerId']
        isTeamOne = killer in teamOne
        if monsterType == 'DRAGON':
          dragons.append((time, isTeamOne))
        elif monsterType == 'BARON_NASHOR':
          barons.append((time, isTeamOne))
        #Red/blue buffs aren't recorded here as specified in API

      wardEvent = event.get('eventType', None)
      if wardEvent == 'WARD_PLACED':
        time = event['timestamp'] // 1000
        wardType = event['wardType']
        isTeamOne = event['creatorId'] <= 5 
        if wardType == 'VISION_WARD':
          pinkWards.append((time, isTeamOne))
        elif wardType == 'YELLOW_TRINKET':
          stealthWards2Min.append((time, isTeamOne))
        elif wardType in ('YELLOW_TRINKET_UPGRADE', 'SIGHT_WARD'):
          stealthWards3Min.append((time, isTeamOne))
        #unhandled case: TEEMO_MUSHROOM

      buildingType = event.get('buildingType', None)
      if buildingType == 'TOWER_BUILDING':
        time = event['timestamp'] // 1000
        killer = event['killerId']
        towerType = event['towerType']
        laneType = event['laneType']
        isTeamOneTower = event['teamId'] == 100

        if towerType == 'FOUNTAIN_TURRET':
          # TODO(sehtroisi): figure out what causes this.
          continue

        towerNum = getTowerNumber(isTeamOneTower, laneType, towerType)

        #print ("killer {}({}) @{:.0f}s: ({} x {} x {}) = {}".format(
        #  champNames[killer - 1], killer, time,
        #  isTeamOneTower, laneType, towerType, towerNum))

        # TODO(sethtroisi): Stop mid nexus turret double count.
        #assert all(tNum != towerNum for t, k, tNum in towers)
        towers.append((time, towerNum))

      elif buildingType == 'INHIBITOR_BUILDING':
        time = event['timestamp'] // 1000
        killer = event['killerId']
        laneType = event['laneType']
        isTeamOneInhib = event['teamId'] == 100
        inhibNum = getInhibNumber(isTeamOneInhib, laneType)
        inhibs.append((time, inhibNum))

  features = dict()
  features['dragons'] = dragons
  features['towers'] = towers
  features['gold'] = gold
  features['pinkWards'] = pinkWards
  features['stealthWards2Min'] = stealthWards2Min
  features['stealthWards3Min'] = stealthWards3Min
  features['barons'] = barons
  features['inhibs'] = inhibs

  # TODO(sethtroisi): plumb debug info instead of reusing features.
  features['duration'] = match['matchDuration']

  result = match['teams'][0]['winner']
  return result, features

def main(args):

  gameNum = 0
  outputData = []

  lastFile = 1 if args.limited else 10
  for fileNumber in range(1, lastFile + 1):
    parsed = loadJsonFile('matches{}.json'.format(fileNumber))

    games = parsed['matches']
    for game in games:
      result, features = parseGameRough(game)
      data = {'goal': result, 'features': features}
      outputData.append(data)
      gameNum += 1

  chars = len(str(outputData))

  print ("parsed {} games".format(gameNum))
  print ("~{} chars ~{:.1f}MB, ~{:.1f}KB/game".format(
      chars, chars / 10 ** 6, chars /(10 ** 3 * gameNum)))
  print ()

  if not args.dry_run:
    writeJsonFile('output.txt', outputData)

  exampleLines = random.sample(range(gameNum), args.examples)
  for exampleLine in sorted(exampleLines):
    gameStr = str(outputData[exampleLine])
    if args.full_examples:
      example = gameStr
    else:
      example = gameStr[:70] + ('..' if len(gameStr) > 70 else '')

    print ()
    print ("line {}: {}".format(exampleLine, example))


# START CODE HERE
args = getArgParse().parse_args()
main(args)

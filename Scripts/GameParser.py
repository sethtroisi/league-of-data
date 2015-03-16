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
      help='Print full examples')

  parser.add_argument(
      '-e', '--examples',
      type=int,
      default=2,
      help='How many examples to print')

  parser.add_argument(
      '-i', '--input-file',
      type=str,
      default='matchesRiotSampleSmall.json',
      help='Input match file (produced by Seth or Coalesce.py)')

  parser.add_argument(
      '-o', '--output-file',
      type=str,
      default='featuresRiotSampleSmall.json',
      help='Output feature file (consumed by Model.py / Featurize.py)')

  parser.add_argument(
      '-n', '--dry-run',
      action="store_true",
      help='Don\'t write output file instead print to screen')
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
      time = event.get('timestamp', None)
      if time:
        time //= 1000

      if monsterType:
        killer = event['killerId']
        isTeamOne = killer in teamOne
        if monsterType == 'DRAGON':
          dragons.append((time, isTeamOne))
        elif monsterType == 'BARON_NASHOR':
          barons.append((time, isTeamOne))
        #Red/blue buffs aren't recorded here as specified in API

      buildingType = event.get('buildingType', None)
      if buildingType == 'TOWER_BUILDING':
        killer = event['killerId']
        towerType = event['towerType']
        laneType = event['laneType']
        isTeamOneTower = event['teamId'] == 100

        if towerType == 'FOUNTAIN_TURRET':
          # NOTE: Azir turrets are coded as turrets.
          continue

        towerNum = getTowerNumber(isTeamOneTower, laneType, towerType)
        assert isTeamOneTower == teamATowerKill(towerNum)

        #print ("killer {}({}) @{:.0f}s: ({} x {} x {}) = {}".format(
        #  champNames[killer - 1], killer, time,
        #  isTeamOneTower, laneType, towerType, towerNum))

        # TODO(sethtroisi): Stop mid nexus turret double count.
        #assert all(tNum != towerNum for t, k, tNum in towers)
        towers.append((time, isTeamOneTower, towerNum))

      elif buildingType == 'INHIBITOR_BUILDING':
        killer = event['killerId']
        laneType = event['laneType']
        isTeamOneInhib = event['teamId'] == 100
        inhibNum = getInhibNumber(isTeamOneInhib, laneType)
        inhibs.append((time, isTeamOneInhib, inhibNum))

      wardEvent = event.get('eventType', None)
      if wardEvent == 'WARD_PLACED':
        wardType = event['wardType']
        isTeamOne = event['creatorId'] <= 5
        if wardType == 'VISION_WARD':
          pinkWards.append((time, isTeamOne))
        elif wardType == 'YELLOW_TRINKET':
          stealthWards2Min.append((time, isTeamOne))
        elif wardType in ('YELLOW_TRINKET_UPGRADE', 'SIGHT_WARD'):
          stealthWards3Min.append((time, isTeamOne))
        #unhandled case: TEEMO_MUSHROOM

  features = dict()
  features['dragons'] = dragons
  features['barons'] = barons
  features['towers'] = towers
  features['inhibs'] = inhibs
  features['pinkWards'] = pinkWards
  features['stealthWards2Min'] = stealthWards2Min
  features['stealthWards3Min'] = stealthWards3Min
  features['gold'] = gold

  # TODO(sethtroisi): plumb debug info instead of reusing features.
  features['duration'] = match['matchDuration']

  result = match['teams'][0]['winner']
  return result, features


def main(args):

  gameNum = 0
  outputData = []

  games = loadJsonFile(args.input_file)
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
    writeJsonFile(args.output_file, outputData)

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

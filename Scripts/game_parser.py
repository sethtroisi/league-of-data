import argparse
import random
import time

import util

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
        '-c', '--count',
        type=int,
        default=-1,
        help='Number of games to output (and read in)')

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
        default='matchesAll.json',
        help='Input match file (produced by Seth or Coalesce.py)')

    parser.add_argument(
        '-o', '--output-file',
        type=str,
        default='features.json',
        help='Output feature file (consumed by Model.py / Featurize.py)')

    parser.add_argument(
        '-n', '--dry-run',
        action="store_true",
        help='Don\'t write output file instead print to screen')
    return parser


# takes in a match json object and returns features about it.
def parseGameRough(match, timeline):
    teamInfo = match['participants']

    teamOne = []
    champs = []
    for champI, participant in enumerate(teamInfo, 1):
        pId = int(participant['participantId'])
        isTeamOne = participant['teamId'] == 100
        assert pId == champI, "{} != {}".format(pId, champI)
        assert (1 <= pId <= 5) == isTeamOne, "{} != {}".format(pId, isTeamOne)

        if isTeamOne:
            teamOne.append(pId)

        champ = {}
        champs.append(champ)

        champId = participant['championId']
        champ['isTeamOne'] = isTeamOne

        role = participant['timeline']['role']
        lane = participant['timeline']['lane']
        possibleRoles = ['NONE', 'SOLO', 'DUO', 'DUO_CARRY', 'DUO_SUPPORT']
        possibleLanes = ['BOTTOM', 'JUNGLE', 'MIDDLE', 'TOP']
        assert role in possibleRoles, role
        assert lane in possibleLanes, lane

        position = 10 * possibleLanes.index(lane) + possibleRoles.index(role)
        champ['role'] = role
        champ['lane'] = lane
        champ['positionIndex'] = position

        champ['championId'] = champId
#        champ['champion'] = Util.championIdToName(champId)
        champ['spell1'] = participant['spell1Id']
        champ['spell2'] = participant['spell2Id']
        champ['approxRank'] = participant['highestAchievedSeasonTier']

    dragons = []
    barons = []
    towers = []
    inhibs = []
    gold = {}
    frames = timeline['frames']
    for frame in frames:
        frameTime = frame['timestamp'] // 1000
        blockNum = util.timeToBlock(frameTime)

        frameGold = {}
        # NOTE: frames appear to be 60 seconds
        gold[blockNum] = frameGold
        for pId, pFrame in frame['participantFrames'].items():
            # TODO use item gold instead of totalGold
            frameGold[pId] = pFrame['totalGold']

        events = frame.get('events', [])
        for event in events:
            monsterType = event.get('monsterType', None)
            gameTime = event.get('timestamp', None)
            if gameTime:
                gameTime //= 1000

            if monsterType:
                killer = event['killerId']
                isTeamOne = killer in teamOne
                if monsterType == 'DRAGON':
                    subType = event['monsterSubType']
                    assert '_DRAGON' in subType
                    commonName = subType.replace('_DRAGON', '')

                    dragons.append((gameTime, commonName, isTeamOne))
                elif monsterType == 'BARON_NASHOR':
                    barons.append((gameTime, isTeamOne))
                # TODO elder dragon?
                # Red/blue buffs aren't recorded here as specified in API

            buildingType = event.get('buildingType', None)
            if buildingType == 'TOWER_BUILDING':
                # killer = event['killerId']
                towerType = event['towerType']
                laneType = event['laneType']
                isTeamOneTower = event['teamId'] == 100

                # TODO verify this is still true (2017-10)
                if towerType == 'FOUNTAIN_TURRET':
                    # NOTE: Azir turrets are coded as turrets.
                    continue

                towerNum = util.getTowerNumber(isTeamOneTower, laneType, towerType)
                assert isTeamOneTower == util.teamATowerKill(towerNum)

                # print ("killer {}({}) @{:.0f}s: ({} x {} x {}) = {}".format(
                #  champNames[killer - 1], killer, time,
                #  isTeamOneTower, laneType, towerType, towerNum))

                # TODO Stop mid nexus turret double count.
                # assert all(tNum != towerNum for t, k, tNum in towers)
                towers.append((gameTime, isTeamOneTower, towerNum))

            elif buildingType == 'INHIBITOR_BUILDING':
                # killer = event['killerId']
                laneType = event['laneType']
                isTeamOneInhib = event['teamId'] == 100
                inhibNum = util.getInhibNumber(isTeamOneInhib, laneType)
                inhibs.append((gameTime, isTeamOneInhib, inhibNum))

            # wardEvent = event.get('eventType', None)
            # if wardEvent == 'WARD_PLACED':
                # wardType = event['wardType']
                # isTeamOne = event['creatorId'] <= 5
                # TODO save and record ward events

    features = dict()
    features['champs'] = champs
    features['dragons'] = dragons
    features['barons'] = barons
    features['towers'] = towers
    features['inhibs'] = inhibs
    # features['wards'] = wards

    features['gold'] = gold

    debug = dict()
    debug['duration'] = match['gameDuration']
    debug['matchId'] = match['gameId']

    rawResult = match['teams'][0]['win']
    assert rawResult in ('Win', 'Fail'), rawResult
    result = rawResult == 'Win'

    parsed = dict()
    parsed['features'] = features
    parsed['debug'] = debug
    parsed['goal'] = result

    return parsed


def main(args):
    T0 = time.time()

    gameNum = 0
    outputData = []

    inFile = util.loadJsonFile(args.input_file)
    items = len(inFile)
    print ("{} has {} items".format(args.input_file, items))
    printEvery = items // 15

    for t in inFile:
        assert type(t) == list
        assert len(t) == 2

        # If you had an error on this line re-run Coalesce.py
        match = util.loadJsonFile(t[0])
        timeline = util.loadJsonFile(t[1])

        parsed = parseGameRough(match, timeline)
        outputData.append(parsed)
        gameNum += 1

        if gameNum % printEvery == 0:
            print ("parsed {} of {} ({:0.0f}%) ({:.2f}s)".format(
                    gameNum, items, 100 * gameNum / items, time.time() - T0))

        if gameNum == args.count:
            print ("Stopping after {} games (like you asked with -c)".format(args.count))
            break

    # Remove any ordering effect from game number
    random.shuffle(outputData)

    chars = len(str(outputData))

    print ("parsed {} games".format(gameNum))
    print ("~{} chars ~{:.1f}MB, ~{:.1f}KB/game".format(
        chars, chars / 10 ** 6, chars / (10 ** 3 * gameNum)))
    print ()

    exampleLines = random.sample(range(gameNum), args.examples)
    for exampleLine in sorted(exampleLines):
        gameStr = str(outputData[exampleLine])
        if args.full_examples:
            example = gameStr
        else:
            example = gameStr[:70] + ('..' if len(gameStr) > 70 else '')

        print ()
        print ("line {}: {}".format(exampleLine, example))

    if args.examples > 0:
        util.writeJsonFile('example-feature.json', outputData[exampleLines[0]])

    if not args.dry_run:
        util.writeJsonFile(args.output_file, outputData)


# START CODE HERE
if __name__ == "__main__":
    parsedArgs = getArgParse().parse_args()
    main(parsedArgs)

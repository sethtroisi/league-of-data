import argparse
import math
import random
import time

from collections import defaultdict

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


def isSurrender(byTeamOne, result, towers):
    # This bit isn't saved in match or timeline so take a guess

    # Can't surrender and win
    if byTeamOne == result:
        return False

    towerNums = [towerNum for _, _, towerNum in towers]

    # Check if the tower of the team hypothetically surrendering is destroyed
    nexusTowerMid = util.getTowerNumber(byTeamOne, "MID_LANE", "NEXUS_TURRET")
    nexusTowerTop = util.getTowerNumber(byTeamOne, "TOP_LANE", "NEXUS_TURRET")

    # Lost but nexus towers weren't destroyed => Surrendered
    if nexusTowerMid not in towerNums or nexusTowerTop not in towerNums:
        return True

    # Lost, Nexus Towers were destroyed => assume it was a normal loss.
    return False


# takes in a match json object and returns features about it.
def parseGameRough(match, timeline):
    teamInfo = match['participants']
    matchId = match['gameId']

    teamOne = set()
    champs = []
    champLookup = {}
    for champI, participant in enumerate(teamInfo, 1):
        pId = int(participant['participantId'])
        isTeamOne = participant['teamId'] == 100
        assert pId == champI, "{} != {}".format(pId, champI)
        assert (1 <= pId <= 5) == isTeamOne, "{} != {}".format(pId, isTeamOne)

        if isTeamOne:
            teamOne.add(pId)

        champ = {}
        champs.append(champ)
        champLookup[pId] = champ

        champId = participant['championId']
        champ['pId'] = pId
        champ['isTeamOne'] = isTeamOne

        pTimeline = participant['timeline']
        role = util.getAndValidate(pTimeline, 'role', ['SOLO', 'DUO', 'DUO_CARRY', 'DUO_SUPPORT', 'NONE'])
        lane = util.getAndValidate(pTimeline, 'lane', ['BOTTOM', 'JUNGLE', 'MIDDLE', 'TOP'])

        champ['role'] = role
        champ['lane'] = lane
        champ['position'] = util.guessPosition(champ)

        # TODO Take a guess at lane swap
        # TODO consider filtering lane swap games

        champ['championId'] = champId
#        champ['champion'] = Util.championIdToName(champId)
        champ['spell1'] = participant['spell1Id']
        champ['spell2'] = participant['spell2Id']
        champ['approxRank'] = participant['highestAchievedSeasonTier']

    dragons = []
    barons = []
    towers = []
    inhibs = []
    frameStats = defaultdict(
        lambda: defaultdict(lambda: [None for _ in range(10)])
    )

    frames = timeline['frames']
    lastBlockNum = -1
    for frameI, frame in enumerate(frames):
        frameTime = frame['timestamp'] // 1000
        blockNum = util.timeToBlock(frameTime)
        assert blockNum == lastBlockNum or blockNum == (lastBlockNum + 1)
        lastBlockNum = blockNum

        # NOTE: frames are every ~60 seconds (they have XYZ millis)
        for pId, pFrame in frame['participantFrames'].items():
            pId = int(pId)
            isTeamOne = pId in teamOne
            listPId = pId - 1
            assert 0 <= listPId <= 9, listPId
            # TODO use item gold instead of totalGold
            frameStats['gold'][blockNum][listPId] = pFrame['totalGold']
            frameStats['farm'][blockNum][listPId] = pFrame['minionsKilled']
            frameStats['jungleFarm'][blockNum][listPId] = pFrame['jungleMinionsKilled']

            # Disconnected is [0,1] predicting how likely they aren't connected
            frameStats['disconnect'][blockNum][listPId] = 0

            if frameI + 1 == len(frames):
                # Most of the time position is not in the last frame
                # assert 'position' not in pFrame
                continue

            position = pFrame['position']
            onPedestal = util.onPedestal(isTeamOne, position)
            if frameI == 0:
                assert onPedestal, "{} {}".format(isTeamOne, position)
            elif blockNum >= 2:
                assert frameTime >= 120, frameTime
                lastPFrame = frames[blockNum - 1]['participantFrames'][str(pId)]
                def deltaPFrame(key):
                    return pFrame[key] - lastPFrame[key]

                sameLoc = util.distance(position, lastPFrame['position']) < 25
                xpDelta = deltaPFrame('xp')

                totalFarm = pFrame['minionsKilled'] + pFrame['jungleMinionsKilled']
                farmDelta = deltaPFrame('minionsKilled') + deltaPFrame('jungleMinionsKilled')
                isSupport = champLookup[pId]['position'] == 'SUPPORT' or (totalFarm < 25)

                # TODO different logic for support
                zScore = 1.6 * sameLoc + \
                         0.8 * onPedestal + \
                         0.7 * (not isSupport and farmDelta < 3) + \
                         0.6 * (xpDelta < 150)

                prop = 1 - math.exp(-zScore ** 2)
                assert 0 <= prop <= 1, "{} -> {}".format(zScore, prop)
                frameStats['disconnect'][blockNum][listPId] = prop


            # TODO other frame features (Level, XP, inventory gold, ...)

        events = frame.get('events', [])
        for event in events:
            monsterType = event.get('monsterType', None)
            gameTime = event.get('timestamp', None)
            if gameTime:
                gameTime //= 1000

            def isTeamOne():
                return 100 == util.getAndValidate(event, 'teamId', (100, 200))

            def killerId():
                return util.getAndValidate(event, 'killerId', range(11))  # 0 means minion

            if monsterType:
                killer = killerId()
                isTeamOne = killer in teamOne
                if monsterType == 'DRAGON':
                    dragonType = event['monsterSubType']
                    assert '_DRAGON' in dragonType
                    commonName = dragonType.replace('_DRAGON', '')
                    dragons.append((gameTime, commonName, isTeamOne))

                elif monsterType == 'BARON_NASHOR':
                    barons.append((gameTime, isTeamOne))

                # TODO 'RIFTHERALD' and 'ELDER_DRAGON' (handled by DRAGON code?)

            buildingType = event.get('buildingType', None)
            if buildingType == 'TOWER_BUILDING':
                towerType = event['towerType']
                laneType = event['laneType']
                isTeamOneTowerDestroyed = isTeamOne()

                # Deduplicate MID_LANE + NEXUS_TURRET.
                if towerType == 'NEXUS_TURRET':
                    yCord = util.getAndValidate(event['position'], 'y', (1807, 2270, 12612, 13084))
                    isTopNexus = yCord in (2270, 13084)
                    if isTopNexus:
                        laneType = "TOP_LANE"

                towerNum = util.getTowerNumber(isTeamOneTowerDestroyed, laneType, towerType)
                assert isTeamOneTowerDestroyed != util.teamATowerKill(towerNum)

                assert all(tNum != towerNum for t, k, tNum in towers), "{} new: {}".format(towers, event)
                towers.append((gameTime, isTeamOneTowerDestroyed, towerNum))

            elif buildingType == 'INHIBITOR_BUILDING':
                # killer = event['killerId']
                laneType = event['laneType']
                isTeamOneInhib = isTeamOne()
                inhibNum = util.getInhibNumber(isTeamOneInhib, laneType)
                inhibs.append((gameTime, isTeamOneInhib, inhibNum))

            # wardEvent = event.get('eventType', None)
            # if wardEvent == 'WARD_PLACED':
                # wardType = event['wardType']
                # isTeamOne = event['creatorId'] <= 5
                # TODO save and record ward events

            # TODO KILLS, LEVEL UP, SKILL ORDER (?)

    features = dict()
    features['champs'] = champs
    features['dragons'] = dragons
    features['barons'] = barons
    features['towers'] = towers
    features['inhibs'] = inhibs
    # features['wards'] = wards

    # Verify key is present and convert defaultdict to dictionary.
    features['gold'] = dict(frameStats['gold'])
    features['farm'] = dict(frameStats['farm'])
    features['jungleFarm'] = dict(frameStats['jungleFarm'])
    features['disconnect'] = dict(frameStats['disconnect'])

    rawResult = match['teams'][0]['win']
    assert rawResult in ('Win', 'Fail'), rawResult
    result = rawResult == 'Win'

    debug = dict()
    debug['duration'] = match['gameDuration']
    debug['matchId'] = match['gameId']
    debug['TeamASurrendered'] = isSurrender(True, result, towers)
    debug['TeamBSurrendered'] = isSurrender(False, result, towers)
    debug['surrendered'] = debug['TeamASurrendered'] or debug['TeamBSurrendered']

    parsed = dict()
    parsed['features'] = features
    parsed['debug'] = debug
    parsed['goal'] = result

    return parsed


def debugStats(args, outputData, T0):
    numberOfGames = len(outputData)

    numberOfSurrenders = sum([parsed['debug']['surrendered'] for parsed in outputData])
    percentSurrenders = 100 * numberOfSurrenders / numberOfGames
    print("\t{:2.1f}% games were surrenders".format(percentSurrenders))
    assert numberOfSurrenders > 0 and percentSurrenders <= 25

    numberOfBlocks = 0
    countOfDisconnect = 0
    sumDisconnect = 0
    for parsed in outputData:
        for blockDisconnect in parsed['features']['disconnect'].values():
            for playerDisconnect in blockDisconnect:
                numberOfBlocks += 1
                sumDisconnect += playerDisconnect
                countOfDisconnect += playerDisconnect > 0.25
    percentDisconnect = 100 * countOfDisconnect / numberOfBlocks
    print("\t{}/{} = {:.2f}% disconnect, {:.3f} prop".format(
        countOfDisconnect, numberOfBlocks, percentDisconnect, sumDisconnect / numberOfBlocks))
    assert 0 < percentDisconnect < 10

    totalTime = time.time() - T0
    chars = len(str(outputData))
    print ()
    print("Parsed {} games:".format(numberOfGames))
    print("\t~{:.1f} MB, ~{:.1f} KB/game".format(
        chars / 10 ** 6, chars / (10 ** 3 * numberOfGames)))
    print("\t{:.1f} seconds, ~{:.0f} games/second".format(
        totalTime,  numberOfGames / totalTime))
    print()


def main(args):
    T0 = time.time()

    outputData = []

    inFile = util.loadJsonFile(args.input_file)
    # Remove any ordering effect from game number
    random.shuffle(inFile)

    toProcess = len(inFile)
    print("{} has {} items".format(args.input_file, toProcess))
    if 0 < args.count < toProcess:
        toProcess = args.count
        print("\tonly parsing {}".format(args.count))
        print()

    printEvery = toProcess // 15
    for gameNum, fileNames in enumerate(inFile, 1):
        assert type(fileNames) == list
        assert len(fileNames) == 2

        # If you had an error on this line re-run Coalesce.py
        match = util.loadJsonFile(fileNames[0])
        timeline = util.loadJsonFile(fileNames[1])

        parsed = parseGameRough(match, timeline)

        outputData.append(parsed)

        if gameNum % printEvery == 0:
            print ("parsed {} of {} ({:0.0f}%) ({:.2f}s)".format(
                gameNum, toProcess, 100 * gameNum / toProcess, time.time() - T0))

        if gameNum == args.count:
            print ()
            print ("Stopping after {} games (like you asked with -c)".format(args.count))
            break

    debugStats(args, outputData, T0)

    exampleLines = random.sample(range(toProcess), args.examples)
    for exampleLine in sorted(exampleLines):
        gameStr = str(outputData[exampleLine])
        example = util.abbreviateString(gameStr, 10000 if args.full_examples else 70)
        print()
        print("line {}: {}".format(exampleLine, example))

    if args.examples > 0:
        util.writeJsonFile('example-feature.json', outputData[exampleLines[0]])

    if not args.dry_run:
        util.writeJsonFile(args.output_file, outputData)

# START CODE HERE
if __name__ == "__main__":
    parsedArgs = getArgParse().parse_args()
    main(parsedArgs)

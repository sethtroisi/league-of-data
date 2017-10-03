#!/usr/bin/env python3

import datetime
import json
import os
import random
import re
import time
import urllib.request

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

API_KEYS = [
    "RGAPI-5a926470-e10e-42dd-8021-ba986dfd4626"  # Main
]

START_TIME_FILTER = int(datetime.datetime(2017, 9, 15).timestamp()) * 1000
REGIONS = ("NA", "NA1")
SEASON = 9

QUEUE_ID_TO_FOLDER = {
    4: "ranked",
    420: "ranked",  # SOLO
    440: "ranked",  # RANKED_FLEX
    65: "aram"
}

TIER_EXPLORE_ODDS = {
    "UNRANKED":   0.1,
    "BRONZE":     0.05,
    "SILVER":     0.1,
    "GOLD":       0.2,
    "PLATINUM":   0.3,
    "DIAMOND":    0.6,
    "CHALLENGER": 0.7,
    "MASTER":     0.9
}

GAMES_PER_SUMMONER = 3

'''    CONSTANTS     '''

BASE_URL = "https://na1.api.riotgames.com/lol/"
KEY_PARAM = "api_key={apiKey}"

RATE_LIMITS_RAW = "100:120,20:1"
# noinspection PyTypeChecker
RATE_LIMITS = dict([map(int, l.split(":")[::-1]) for l in RATE_LIMITS_RAW.split(",")])

MIN_SLEEP_TIME = max((time / count for time, count in RATE_LIMITS.items()))
print ("SLEEP_TIME: {:.2f}".format(MIN_SLEEP_TIME))
print ("RATE_LIMITS: {}".format(
    ", ".join(map(lambda l: "{} per {}".format(l[1], l[0]), RATE_LIMITS.items()))))
print ()


def loadExistingMatches(matchIds):
    directory = "../Data/"
    allFiles = set()
    for root, dirs, files in os.walk(directory, followlinks=True):
        allFiles.update(files)

    regex = re.compile('^getMatch-([0-9]*)$')
    for fileName in allFiles:
        match = regex.match(fileName)
        if match:
            matchId = match.group(1)
            matchIds.add(matchId)

    print ("\t{} pre-existing matches".format(len(matchIds)))
    print ()


def buildUrl(apiPath, params=None):
    params = params or []
    urlParams = "&".join([KEY_PARAM] + params)
    return BASE_URL + apiPath + "?" + urlParams


def getParsedResponse(url):
    if not hasattr(getParsedResponse, "keyUsed"):
        getParsedResponse.keyUsed = dict((key, 0) for key in API_KEYS)
    lastUsed, apiKey = min((u, k) for k, u in getParsedResponse.keyUsed.items())

    timeToWait = (lastUsed + MIN_SLEEP_TIME) - time.time()
    if timeToWait > 0.01:
        time.sleep(timeToWait)

    getParsedResponse.keyUsed[apiKey] = time.time()

    url = url.format(apiKey=apiKey)
    response = urllib.request.urlopen(url)
    data = response.read()
    stringData = data.decode("utf-8")

    # Wait to respect rate limits passed in header
    # "X-App-Rate-Limit":        "100:120, 20:1",
    # "X-App-Rate-Limit-Count": "   5:120,  1:1",
    assert RATE_LIMITS_RAW == response.getheader("X-App-Rate-Limit")
    rawCounts = response.getheader("X-App-Rate-Limit-Count")
    parsedRates = list(map(lambda r: list(map(int, r.split(":"))), rawCounts.split(",")))
    for count, timeLimit in parsedRates:
        # Aim for 75% of limits
        limit = RATE_LIMITS[timeLimit]
        if count > 0.75 * limit:
            closeness = limit - count
            timeToWait = timeLimit / max(2, closeness ** 1.4)
            print("\t\tNearing rateLimit ({} of {} per {}s) waiting {:.2f}s)".format(
                count, limit, timeLimit, timeToWait))
            time.sleep(timeToWait)

    return json.loads(stringData)


def getSummonerAccountId(name):
    apiFormat = "summoner/v3/summoners/by-name/{summonerName}"

    apiPath = apiFormat.format(summonerName=name.replace(" ", ""))
    url = buildUrl(apiPath)
    parsed = getParsedResponse(url)

    return parsed["accountId"]


def getMatchHistory(summonerId):
    apiFormat = "match/v3/matchlists/by-account/{summonerId}/recent"
    apiPath = apiFormat.format(summonerId=summonerId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getMatch(matchId):
    apiFormat = "match/v3/matches/{matchId}"
    apiPath = apiFormat.format(matchId=matchId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getTimeline(matchId):
    apiFormat = "match/v3/timelines/by-match/{matchId}"
    apiPath = apiFormat.format(matchId=matchId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getSummonerMatches(summonerId, matchIds):
    # TODO: Use rankedQueues query param.
    history = getMatchHistory(summonerId)

    matches = history["matches"]
    assert len(matches) > 0

    fellowSummoners = set()

    # Games are stored in increasing chronological order.
    saved = 0
    for match in matches[:GAMES_PER_SUMMONER]:
        matchId = match["gameId"]
        queueType = match["queue"]
        region = match["platformId"]
        season = match["season"]
        timestamp = match["timestamp"]

        if matchId in matchIds:
            print ("\t\tskipping already fetched match")
            continue

        if timestamp <= START_TIME_FILTER:
            print ("\t\tskipping old timestamp:", timestamp)
            continue

        if region not in REGIONS:
            print ("\t\tskipping non NA region:", region)
            continue

        if season != SEASON:
            print ("\t\tskipping season:", season)
            continue

        if queueType not in QUEUE_ID_TO_FOLDER:
            print ("\t\tskipping queue:", queueType)
            continue
        folder = QUEUE_ID_TO_FOLDER[queueType]

        print ("\tFetching match (id: {})".format(matchId))
        fullMatch = getMatch(matchId)
        matchSaveName = "matches/{}/getMatch-{}".format(folder, matchId)
        util.writeJsonFile(matchSaveName, fullMatch)

        timeline = getTimeline(matchId)
        matchSaveName = "matches/{}/getTimeline-{}".format(folder, matchId)
        util.writeJsonFile(matchSaveName, timeline)

        saved += 1
        matchIds.add(matchId)

        otherParticipants = fullMatch["participantIdentities"]
        for participant in otherParticipants:
            assert "player" in participant, "Old non-ranked game?"
            player = participant["player"]
            platformId = player["platformId"]
            accountId = player["accountId"]
            summonerName = player["summonerName"]

            # Some players have transferred profiles EUW to NA which we ignore because accountId doesn't work
            # and I haven"t investigated if currentAccountId always works. NA1 vs NA is okay.
            currentAccountId = player["currentAccountId"]
            if accountId != currentAccountId:
                print("\t\tNot adding \"{}\" has mismatch account: {} vs {}".format(
                    summonerName, accountId, currentAccountId))
                continue

            currentPlatformId = player["currentPlatformId"]
            if not (currentPlatformId.startswith(platformId) or platformId.startswith(currentPlatformId)):
                print ("\t\tNot adding \"{}\" has mismatched platform: {} vs {}".format(
                    summonerName, platformId, currentPlatformId))
                continue

            participantId = participant["participantId"]
            for gameData in fullMatch["participants"]:
                if gameData["participantId"] == participantId:
                    seasonTier = gameData["highestAchievedSeasonTier"]
                    break
            else:
                assert False, "participant not found in participants"
            fellowSummoners.add((summonerName, accountId, seasonTier))

    return saved, fellowSummoners


def main():
    T0 = time.time()

    # Id -> Name (all seen summoners)
    summoners = {}

    # Id -> Name (not yet processed) (only add if not in summoners)
    unvisited = {}

    # Id -> Highest achieved season tier
    tier = {}

    matchIds = set()
    loadExistingMatches(matchIds)

    seedNames = ["inkaruga", "kingvash", "siren swag", "falco36"]
    seedIds = {}
    for name in seedNames:
        seedIds[name] = getSummonerAccountId(name)

    for name, sumId in seedIds.items():
        summoners[sumId] = name
        unvisited[sumId] = name
        tier[sumId] = "BRONZE"  # Causes the system to prefer other players

    added = 0
    successes = 0
    fails = 0
    recentStatus = []
    while added < 20000:
        # Prefer exploring higher ELO players.
        newId = random.choice(list(unvisited.keys()))

        newTier = tier[newId]
        if random.random() > TIER_EXPLORE_ODDS[newTier]:
            continue

        newName = unvisited[newId]
        # Remove from the list of unprocessed
        del unvisited[newId]

        print ("Exploring \"{}\", id: {}, tier: {}\t({} of {} unexplored) ({} games collected = {:.1f}/hr)".format(
            newName, newId, newTier,
            len(unvisited), len(summoners), added,
            added / ((time.time() - T0) / 3600)))

        try:
            matchesAdded, fellowSummoners = getSummonerMatches(newId, matchIds)
            added += matchesAdded
            successes += 1
            recentStatus = [True] + recentStatus[:9]
        except Exception as e:
            fails += 1
            recentStatus = [False] + recentStatus[:9]

            print ()
            print ("FAIL({} to {}) (with id {}):\n\"{}\"".format(
                fails, successes, newId, e))

            if 30 * (fails - 1) > successes:
                print ("breaking from {} fails".format(fails))
                return

            if not any(recentStatus):
                print ("ALL RECENT STATUS FAILED", recentStatus)
                return
            continue
        except KeyboardInterrupt as e:
            print ("Breaking from KeyboardInterrupt:", e)
            break

        newSums = 0
        for fName, fId, fTier in fellowSummoners:
            if fId not in summoners:
                newSums += 1
                summoners[fId] = fName
                unvisited[fId] = fName
                tier[fId] = fTier

        print ("\tFound {} new summoners from {} games with {} participants".format(
            newSums, matchesAdded, len(fellowSummoners)))

    T1 = time.time()
    runTimeHours = (T1 - T0) / 3600
    print ("Loaded {} games in {:.2f} hours = {:.1f} games/hr".format(
        added, runTimeHours, added / runTimeHours))


if __name__ == "__main__":
    main()

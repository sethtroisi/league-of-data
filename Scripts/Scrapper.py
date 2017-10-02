#!/usr/bin/env python3

import datetime
import random
import time
import urllib.request
import socket

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

API_KEYS = [
    'RGAPI-5a926470-e10e-42dd-8021-ba986dfd4626' # Main
]

START_TIME_FILTER = int(datetime.datetime(2017, 9, 15).timestamp()) * 1000
REGIONS = ('NA', 'NA1')
SEASON = 9

QUEUE_ID_TO_FOLDER = {
    4: "ranked",
    420: "ranked", # SOLO
    440: "ranked", # RANKED_FLEX
    65: "aram"
}

GAMES_PER_SUMMONER = 3

###### CONSTANTS ######

BASE_URL = 'https://na1.api.riotgames.com/lol/'
KEY_PARAM = 'api_key={apiKey}'

RATE_LIMITS_RAW = "100:120,20:1"
RATE_LIMITS = dict([map(int, l.split(":")[::-1]) for l in RATE_LIMITS_RAW.split(",")])

MIN_SLEEP_TIME = max((time / count for time, count in RATE_LIMITS.items()))
print ("SLEEP_TIME: {:.2f}".format(MIN_SLEEP_TIME))
print ("RATE_LIMITS: {}".format(
    ", ".join(map(lambda l: "{} per {}".format(l[1], l[0]), RATE_LIMITS.items()))))
print ()


def buildUrl(apiPath, params = None):
    params = params or []
    urlParams = '&'.join([KEY_PARAM] + params)
    return BASE_URL + apiPath + '?' + urlParams


def getParsedResponse(url):
    if not hasattr(getParsedResponse, 'keyUsed'):
      getParsedResponse.keyUsed = dict((key, 0) for key in API_KEYS)
    lastUsed, apiKey = min((u, k) for k, u in getParsedResponse.keyUsed.items())

    timeToWait = (lastUsed + MIN_SLEEP_TIME) - time.time()
    if timeToWait > 0.01:
        time.sleep(timeToWait)

    getParsedResponse.keyUsed[apiKey] = time.time()

    url = url.format(apiKey = apiKey)
    response = urllib.request.urlopen(url)
    data = response.read()
    stringData = data.decode('utf-8')

    # Wait to respect rate limits passed in header
    # "X-App-Rate-Limit":        "100:120, 20:1",
    # "X-App-Rate-Limit-Count": "   5:120,  1:1",
    assert RATE_LIMITS_RAW == response.getheader('X-App-Rate-Limit')
    rawCounts = response.getheader('X-App-Rate-Limit-Count')
    parsedRates = list(map(lambda r: list(map(int, r.split(":"))), rawCounts.split(",")))
    for count, timeLimit in parsedRates:
        # Aim for 75% of limits
        limit = RATE_LIMITS[timeLimit]
        if count > 0.75 * limit:
            closeness = limit - count
            timeToWait = timeLimit / max(2, closeness ** 1.4)
            print("\t\tNearing rateLimit({} of {} per {}s) waiting {:.2f}s)".format(
                count, limit, timeLimit, timeToWait))
            time.sleep(timeToWait)

    return json.loads(stringData)


def getSummonerAccountId(name):
    apiFormat = 'summoner/v3/summoners/by-name/{summonerName}'

    apiPath = apiFormat.format(summonerName = name.replace(' ', ''))
    url = buildUrl(apiPath)
    parsed = getParsedResponse(url)

    return parsed['accountId']


def getMatchHistory(summonerId):
    apiFormat = 'match/v3/matchlists/by-account/{summonerId}/recent'
    apiPath = apiFormat.format(summonerId = summonerId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getMatch(matchId):
    apiFormat= 'match/v3/matches/{matchId}'
    apiPath = apiFormat.format(matchId = matchId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getTimeline(matchId):
    apiFormat = 'match/v3/timelines/by-match/{matchId}'
    apiPath = apiFormat.format(matchId = matchId)
    url = buildUrl(apiPath)
    return getParsedResponse(url)


def getSummonerMatches(summonerId):
    # TODO(sethtroisi): Use rankedQueues query param.
    history = getMatchHistory(summonerId)

    matches = history['matches']
    assert len(matches) > 0

    fellowSummoners = {}
    matchIds = set()

    # Games are stored in increasing chronological order.
    for match in matches[:GAMES_PER_SUMMONER]:
        matchId = match['gameId']
        queueType = match['queue']
        region = match['platformId']
        season = match['season']
        timestamp = match['timestamp']

        if queueType not in QUEUE_ID_TO_FOLDER:
            print ("\t\tskipping queue:", queueType)
            continue
        folder = QUEUE_ID_TO_FOLDER[queueType]

        if timestamp <= START_TIME_FILTER:
            print ("\t\tskipping old timestamp:", timestamp)
            continue

        if region not in REGIONS:
            print ("\t\tskipping non NA region:", region)
            continue

        if season != SEASON:
            print ("\t\tskipping season:", season)
            continue

        print ('\tFetching match (id: {})'.format(matchId))
        fullMatch = getMatch(matchId)
        matchSaveName = 'matches/{}/getMatch-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, fullMatch)

        timeline = getTimeline(matchId)
        matchSaveName = 'matches/{}/getTimeline-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, timeline)

        matchIds.add(matchId)

        # Seems to only return information on self? duo?
        #otherParticipants = match['participantIdentities']
        otherParticipants = fullMatch['participantIdentities']
        for participant in otherParticipants:
            if "player" not in participant:
                continue # non-ranked game???
            player = participant['player']
            platformId = player['platformId']
            accountId = player['accountId']
            summonerName = player['summonerName']

            # Some players have transferred profiles EUW to NA which we ignore because accountId doesn't work
            # and I haven't investigated if currentAccountId always works. NA1 vs NA is okay.
            currentAccountId = player['currentAccountId']
            if accountId != currentAccountId:
                print("\t\tMismatch account: {} vs {}".format(accountId, currentAccountId))

            currentPlatformId = player['currentPlatformId']
            if not (currentPlatformId.startswith(platformId) or platformId.startswith(currentPlatformId)):
                print ("\t\tSkipping mismatched platform: {} vs {}".format(
                    platformId, currentPlatformId))
                continue

            fellowSummoners[summonerName] = accountId

    return matchIds, fellowSummoners


def main():
    # Id -> Name (all seen summoners)
    summoners = {}

    # Id -> Name (not yet processed) (only add if not in summoners)
    unvisited = {}

    matchIds = set()

    seedNames = ['inkaruga', 'kingvash', 'siren swag', 'falco36']
    seedIds = {}
    for name in seedNames:
        seedIds[name] = getSummonerAccountId(name)

    for name, sumId in seedIds.items():
        summoners[sumId] = name
        unvisited[sumId] = name

    successes = 0
    fails = 0
    recentStatus = []
    while len(matchIds) < 20000:
        newId = random.choice(list(unvisited.keys()))
        newName = unvisited[newId]
        # Remove from the list of unprocessed
        del unvisited[newId]

        print ('Exploring \'{}\' (id: {}) ({} of {} unexplored) ({} games)'.format(
            newName, newId, len(unvisited), len(summoners), len(matchIds)))

        try:
            newMatchIds, fellowSummoners = getSummonerMatches(newId)
            successes += 1
            recentStatus = [True] + recentStatus[:9]
        except Exception as e:
            fails += 1
            recentStatus = [False] + recentStatus[:9]

            print ()
            print ("FAIL({} to {}) (with id {}):\n'{}'".format(
                fails, successes, newId, e))

            if 30 * (fails - 1) > successes:
                print ("breaking from {} fails".format(fails))
                return

            if not any(recentStatus):
                print ("ALL RECENT STATUS FAILED", recentStatus)
                return

            continue

        # TODO(sethtroisi): make this unique games/summoners
        print ('\tAdded {} games, {} summoners'.format(
            len(newMatchIds), len(fellowSummoners)))

        matchIds.update(newMatchIds)
        for fName, fId in fellowSummoners.items():
            if fId not in summoners:
                summoners[fId] = fName
                unvisited[fId] = fName

if __name__ == '__main__':
    main()

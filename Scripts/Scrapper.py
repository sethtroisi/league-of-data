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

START_TIME_FILTER = int(datetime.datetime(2017, 9, 15).timestamp()) * 1000

QUEUE_ID_TO_FOLDER = {
    4: "ranked",
    420: "ranked", # SOLO
    440: "ranked", # RANKED_FLEX
    65: "aram"
}

API_KEYS = [
   'RGAPI-ae274dc2-4ce8-4778-a46c-6c29f0f47375' # Main
]

BASE_URL = 'https://na1.api.riotgames.com/lol/'
KEY_PARAM = 'api_key={apiKey}'

SLEEP_TIME = 1.3
GAMES_PER_SUMMONER = 3

socket.setdefaulttimeout(2.0)


def buildUrl(apiPath, params = None):
    params = params or []
    urlParams = '&'.join([KEY_PARAM] + params)
    return BASE_URL + apiPath + '?' + urlParams


keyUsed = dict((key, 0) for key in API_KEYS)
def getParsedResponse(url):
    lastUsed, apiKey = min((u, k) for k, u in keyUsed.items())

    timeToWait = (lastUsed + SLEEP_TIME) - time.time()
    if timeToWait > 0.01:
        time.sleep(timeToWait)

    keyUsed[apiKey] = time.time()

    url = url.format(apiKey = apiKey)
    print ("\t\t", url)
    response = urllib.request.urlopen(url)
    data = response.read()
    stringData = data.decode('utf-8')

    # save  "X-App-Rate-Limit-Count": "5:120,1:1",
    # and   "X-App-Rate-Limit": "100:120,20:1",

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
    #saveName = 'matchHistory/getMatchHistory-{}'.format(summonerId)
    #writeJsonFile(saveName, history)
    #history = loadJsonFile(saveName)

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
            print ("bad queue:", queueType)
            continue
        folder = QUEUE_ID_TO_FOLDER[queueType]

        if timestamp <= START_TIME_FILTER:
            print ("old game pre START_TIME_FILTER:", timestamp)
            continue

        if matchId <= 2500000000:
            print ("old game (pre 2017/04):", matchId)
            continue

        if region != 'NA1':
            print ("bad region:", region)
            continue

        if season != 9:
            print ("bad season:", season)
            continue

        print ('\tFetching match (id: {})'.format(matchId))
        fullMatch = getMatch(matchId)
        matchSaveName = '{}/matches/getMatch-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, fullMatch)

        timeline = getTimeline(matchId)
        matchSaveName = '{}/matches/getTimeline-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, timeline)

        matchIds.add(matchId)

        # Seems to only return information on self? duo?
        #otherParticipants = match['participantIdentities']
        otherParticipants = fullMatch['participantIdentities']
        for participant in otherParticipants:
            if "player" not in participant:
                continue # non-ranked game
            player = participant['player']
            platformId = player['platformId']
            accountId = player['accountId']
            summonerName = player['summonerName']

            if platformId != player['currentPlatformId'] or accountId != player['currentAccountId']:
                print ("\t\tMismatch platform or account skipping")
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
            print ("FAIL({} of {}) (with id {}):\n'{}'".format(
                fails, fails + successes, newId, e))

            if 30 * (fails - 1) > len(matchIds):
                print ("breaking from {} fails".format(fails))
                return

            if not all(recentStatus):
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

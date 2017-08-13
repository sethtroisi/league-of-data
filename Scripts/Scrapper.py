#!/usr/bin/env python3

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

QUEUE_ID_TO_FOLDER = {
    4: "ranked",
    420: "ranked", # SOLO
    440: "ranked", # RANKED_FLEX
    65: "aram"
}

API_KEYS = [
   'RGAPI-1f514c25-76a4-440e-b12a-8dded48436c5' # KingVash
#    'RGAPI-81e88d9f-2847-48af-966c-fbf02102ccd3',
#    'RGAPI-aedb5983-6170-4b41-9dd6-05336d24b345',
]
BASE_URL = 'https://na1.api.riotgames.com/lol/'
KEY_PARAM = 'api_key={apiKey}'

SLEEP_TIME = 2
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

    response = urllib.request.urlopen(url)
    data = response.read()
    stringData = data.decode('utf-8')
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
    fullMatches = {}
    fullTimeline = {}

    # Games are stored in increasing chronological order.
    for match in matches[:GAMES_PER_SUMMONER]:
        matchId = match['gameId']
        queueType = match['queue']
        region = match['platformId']
        season = match['season']

        # TODO(sethtroisi): consider filtering on matchCreation time also.

        if queueType not in QUEUE_ID_TO_FOLDER:
            print ("bad queue:", queueType)
            continue
        folder = QUEUE_ID_TO_FOLDER[queueType]


        if region != 'NA1':
            print ("bad region:", region)
            continue


        if season != 9:
            print ("bad season:", season)
            continue

        if matchId <= 2500000000:
            print ("old game (pre 2017/04):", matchId)
            continue

        print ('\tFetching match (id: {})'.format(matchId))
        fullMatch = getMatch(matchId)
        matchSaveName = '{}/matches/getMatch-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, fullMatch)
        #fullMatch = loadJsonFile(matchSaveName)
#        fullMatches[matchId] = fullMatch

        timeline = getTimeline(matchId)
        matchSaveName = '{}/matches/getTimeline-{}'.format(folder, matchId)
        writeJsonFile(matchSaveName, timeline)
#        fullTimeline[matchId] = timeline

        matchIds.add(matchId)

        # Seems to only return information on self? duo?
        #otherParticipants = match['participantIdentities']
        otherParticipants = fullMatch['participantIdentities']
        for participant in otherParticipants:
            if "player" not in participant:
                continue # non-ranked game
            player = participant['player']
            summonerId = player['accountId']
            summonerName = player['summonerName']
            fellowSummoners[summonerName] = summonerId

    return matchIds, fellowSummoners, fullMatches, fullTimeline


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

    fails = 0
    while len(matchIds) < 20000:
        newId = random.choice(list(unvisited.keys()))
        newName = unvisited[newId]
        # Remove from the list of unprocessed
        del unvisited[newId]

        print ('Exploring \'{}\' (id: {}) ({} of {} unexplored) ({} games)'.format(
            newName, newId, len(unvisited), len(summoners), len(matchIds)))

#        try:
        newMatchIds, fellowSummoners, newMatches, newTimelines = \
            getSummonerMatches(newId)
#        except Exception as e:
#            print ("FAIL: '{}'".format(e))
#            fails += 1
#            if 30 * (fails - 1) > len(matchIds):
#                print ("breaking from {} fails".format(fails))
#                return
#            continue

        # TODO(sethtroisi): make this unique games/summoners
        print ('\tAdded {} games, {} summoners'.format(
            len(newMatches), len(fellowSummoners)))

        matchIds.update(newMatchIds)
        for fName, fId in fellowSummoners.items():
            if fId not in summoners:
                summoners[fId] = fName
                unvisited[fId] = fName

if __name__ == '__main__':
    main()

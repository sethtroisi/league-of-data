import json
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

API_KEY = '38940d99-0f69-4dfd-a2ea-5e9de11552a3'
BASE_URL = 'https://na1.api.riotgames.com/lol/'
KEY_PARAM = 'api_key={}'.format(API_KEY)
SLEEP_TIME = 1.0
GAMES_PER_SUMMONER = 3

socket.setdefaulttimeout(1.0)

def buildUrl(apiPath, params = []):
	urlParams = '&'.join([KEY_PARAM] + params)
	return BASE_URL + apiPath + '?' + urlParams


def getParsedResponse(url):
	time.sleep(SLEEP_TIME)

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
	print ("MATCH URL:", url)
	return getParsedResponse(url)


def getMatch(matchId):
	apiFormat= 'match/v3/matches/{matchId}'
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
	fullMatches = {}

	# Games are stored in increasing chronological order.
	for match in matches[:GAMES_PER_SUMMONER]:
		matchId = match['gameId']
		queueType = match['queue']
		region = match['platformId']
		season = match['season']

		# TODO(sethtroisi): consider filtering on matchCreation time also.

		# TODO(sethtroisi): count number of filtered games (by filter).
		if queueType not in [420,440]:
			print ("bad queue:", queueType)
			continue

		if region != 'NA1':
			print ("bad region:", region)
			continue


		if season != 8:
			print ("bad season:", season)
			continue


		print ('\tFetching match (id: {})'.format(matchId))
		fullMatch = getMatch(matchId)
		matchSaveName = 'matches/getMatch-{}'.format(matchId)
		writeJsonFile(matchSaveName, fullMatch)
		#fullMatch = loadJsonFile(matchSaveName)
		fullMatches[matchId] = fullMatch

		# Seems to only return information on self? duo?
		#otherParticipants = match['participantIdentities']
		otherParticipants = fullMatch['participantIdentities']
		for participant in otherParticipants:
			player = participant['player']
			summonerId = player['accountId']
			summonerName = player['summonerName']
			fellowSummoners[summonerName] = summonerId

	return fullMatches, fellowSummoners


def main():
	# Id -> Name (all seen summoners)
	summoners = {}

	# Id -> Name (not yet processed) (only add if not in summoners)
	unvisited = {}

	# MatchId -> Match.
	matches = {}

	seedNames = ['inkaruga', 'kingvash', 'siren swag']
	seedIds = {}
	for name in seedNames:
		seedIds[name] = getSummonerAccountId(name)

	for name, sumId in seedIds.items():
		summoners[sumId] = name
		unvisited[sumId] = name

	fails = 0
	while len(matches) < 15000:
		newId = random.choice(list(unvisited.keys()))
		newName = unvisited[newId]
		# Remove from the list of unprocessed
		del unvisited[newId]

		print ('Exploring \'{}\' (id: {}) ({} of {} unexplored) ({} games)'.format(
				newName, newId, len(unvisited), len(summoners), len(matches)))

		try:
			newMatches, fellowSummoners = getSummonerMatches(newId)
		except Exception as e:
			print ("FAIL: '{}'".format(e))
			fails += 1
			if fails > len(matches):
				print ("breaking from {} fails".format(fails))
				return

		# TODO(sethtroisi): make this unique games/summoners
		print ('\tAdded {} games, {} summoners'.format(
				len(newMatches), len(fellowSummoners)))

		matches.update(newMatches)
		for fName, fId in fellowSummoners.items():
			if fId not in summoners:
				summoners[fId] = fName
				unvisited[fId] = fName


if __name__ == '__main__':
	main()

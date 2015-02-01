import json
#import IPython
#IPython.start_ipython(argv=[])

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

def loadFile(fileName):
  file = open(fileName, encoding='latin1')
  fileData = ''.join(file.readlines())
  file.close()
  return fileData


def getChamps():
  return [('Annie', 1), ('Olaf', 2), ('Galio', 3), ('Twisted Fate', 4), ('Xin Zhao', 5), ('Urgot', 6), ('LeBlanc', 7), ('Vladimir', 8), ('Fiddlesticks', 9), ('Kayle', 10), ('Master Yi', 11), ('Alistar', 12), ('Ryze', 13), ('Sion', 14), ('Sivir', 15), ('Soraka', 16), ('Teemo', 17), ('Tristana', 18), ('Warwick', 19), ('Nunu', 20), ('Miss Fortune', 21), ('Ashe', 22), ('Tryndamere', 23), ('Jax', 24), ('Morgana', 25), ('Zilean', 26), ('Singed', 27), ('Evelynn', 28), ('Twitch', 29), ('Karthus', 30), ('Cho\'Gath', 31), ('Amumu', 32), ('Rammus', 33), ('Anivia', 34), ('Shaco', 35), ('Dr. Mundo', 36), ('Sona', 37), ('Kassadin', 38), ('Irelia', 39), ('Janna', 40), ('Gangplank', 41), ('Corki', 42), ('Karma', 43), ('Taric', 44), ('Veigar', 45), ('Trundle', 48), ('Swain', 50), ('Caitlyn', 51), ('Blitzcrank', 53), ('Malphite', 54), ('Katarina', 55), ('Nocturne', 56), ('Maokai', 57), ('Renekton', 58), ('Jarvan IV', 59), ('Elise', 60), ('Orianna', 61), ('Wukong', 62), ('Brand', 63), ('Lee Sin', 64), ('Vayne', 67), ('Rumble', 68), ('Cassiopeia', 69), ('Skarner', 72), ('Heimerdinger', 74), ('Nasus', 75), ('Nidalee', 76), ('Udyr', 77), ('Poppy', 78), ('Gragas', 79), ('Pantheon', 80), ('Ezreal', 81), ('Mordekaiser', 82), ('Yorick', 83), ('Akali', 84), ('Kennen', 85), ('Garen', 86), ('Leona', 89), ('Malzahar', 90), ('Talon', 91), ('Riven', 92), ('Kog\'Maw', 96), ('Shen', 98), ('Lux', 99), ('Xerath', 101), ('Shyvana', 102), ('Ahri', 103), ('Graves', 104), ('Fizz', 105), ('Volibear', 106), ('Rengar', 107), ('Varus', 110), ('Nautilus', 111), ('Viktor', 112), ('Sejuani', 113), ('Fiora', 114), ('Ziggs', 115), ('Lulu', 117), ('Draven', 119), ('Hecarim', 120), ('Kha\'Zix', 121), ('Darius', 122), ('Jayce', 126), ('Lissandra', 127), ('Diana', 131), ('Quinn', 133), ('Syndra', 134), ('Zyra', 143), ('Gnar', 150), ('Zac', 154), ('Yasuo', 157), ('Vel\'Koz', 161), ('Braum', 201), ('Jinx', 222), ('Lucian', 236), ('Zed', 238), ('Vi', 254), ('Aatrox', 266), ('Nami', 267), ('Azir', 268), ('Thresh', 412), ('Rek\'Sai', 421), ('Kalista', 429)]


def championIdToName(champId):
  for k,v in getChamps():
    if v == champId:
      return k
  print (champId)
  return str(champId)

# takes in a match json object and returns features about it.
def parseGameRough(match):
#  print ("keys:", match.keys())

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
#  print ("Champions: {}".format(championIds))
#  champNames = list(map(championIdToName, championIds))
#  print ("Names: {}".format(champNames))

#  dragons = []
#  frames = match['timeline']['frames']
#  for frame in frames:
#    events = frame.get('events', [])
#    for event in events:
#      monsterType = event.get('monsterType', None)
#      if monsterType == 'DRAGON':
#        time = event['timestamp']
#        killer = event['killerId']
#        print ("killer {} @{:.0f}s".format(
#          champNames[killer-1], time / 1000))
#        dragons.append((time, killer in teamOne))

  result = match['teams'][0]['winner']
  return result, teamOneChampFeatures + teamTwoChampFeatures


# MAIN CODE

output = open(OUTPUT_FILE, mode='w')

gameNum = 0
for fileNumber in range(1, 11):
  fileData = loadFile(FILE_NAME.format(fileNumber))
  parsed = json.loads(fileData)

  games = parsed['matches']
  for game in games:
    result, features = parseGameRough(game)
    jsonObject = {'goal': result, 'features': features}
    jsonString = json.dumps(jsonObject)

    output.write(jsonString + '\n')

    print ('{} <= {}'.format(result, features))
    gameNum += 1

output.close()

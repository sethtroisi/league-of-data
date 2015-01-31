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


def championIdToName(champId):
  CHAMP_MAP = {'Shen': 98, 'Gangplank': 41, 'Yorick': 83, 'Sona': 37, 'Tristana': 18, 'Viktor': 112, 'Darius': 122, 'Olaf': 2, 'Malzahar': 90, 'Hecarim': 120, 'Orianna': 61, 'Wukong': 62, 'Trundle': 48, 'Pantheon': 80, 'Miss Fortune': 21, 'Singed': 27, 'Brand': 63, 'Ashe': 22, 'Nunu': 20, 'Corki': 42, 'Rengar': 107, 'Kassadin': 38, 'Nocturne': 56, 'Ziggs': 115, 'Graves': 104, 'Draven': 119, 'Swain': 50, 'Anivia': 34, 'Shaco': 35, 'Karthus': 30, 'Soraka': 16, 'Heimerdinger': 74, 'Mordekaiser': 82, 'Vladimir': 8, 'Sivir': 15, 'LeBlanc': 7, 'Poppy': 78, 'Nasus': 75, 'Fizz': 105, 'Katarina': 55, 'Quinn': 133, 'Maokai': 57, 'Sejuani': 113, 'Diana': 131, 'Xin Zhao': 5, "Kog'Maw": 96, 'Gragas': 79, 'Renekton': 58, 'Vayne': 67, 'Teemo': 17, 'Fiora': 114, "Cho'Gath": 31, 'Warwick': 19, 'Garen': 86, 'Karma': 43, 'Taric': 44, 'Shyvana': 102, 'Blitzcrank': 53, 'Cassiopeia': 69, 'Lissandra': 127, 'Jax': 24, 'Caitlyn': 51, 'Evelynn': 28, 'Lee Sin': 64, 'Varus': 110, 'Akali': 84, 'Ezreal': 81, 'Urgot': 6, 'Talon': 91, 'Annie': 1, 'Kayle': 10, 'Lux': 99, 'Janna': 40, 'Udyr': 77, 'Lulu': 117, 'Nautilus': 111, 'Galio': 3, 'Irelia': 39, 'Xerath': 101, 'Volibear': 106, 'Malphite': 54, 'Morgana': 25, 'Rammus': 33, 'Jarvan IV': 59, 'Zilean': 26, 'Dr. Mundo': 36, 'Ryze': 13, 'Alistar': 12, 'Leona': 89, 'Elise': 60, 'Twisted Fate': 4, 'Nidalee': 76, 'Rumble': 68, 'Tryndamere': 23, 'Twitch': 29, 'Syndra': 134, 'Jayce': 126, 'Fiddlesticks': 9, 'Amumu': 32, 'Riven': 92, 'Skarner': 72, 'Kennen': 85, 'Veigar': 45, 'Master Yi': 11, 'Ahri': 103, 'Sion': 14, "Kha'Zix": 121, 'Gnar': 150, 'Zac': 154, 'Yasuo': 157, 'Zyra': 143, "Vel'Koz": 161, 'Vi': 254, "Rek'Sai": 421, 'Kalista': 429, 'Thresh': 412, 'Zed': 238, 'Jinx': 222, 'Lucian': 236, 'Braum': 201, 'Nami': 267, 'Azir': 268, 'Aatrox': 266}
  for k,v in CHAMP_MAP.items():
    if v == champId:
      return k
  print (champId)
  return str(champId)

# takes in a match json object and returns a couple of metrics about it
def parseGameRough(match):
#  print ("keys:", match.keys())

  teamInfo = match['participants']

  # for now we return champtionId, firstDragon team, firstDragon time
  championIds = [p['championId'] for p in teamInfo]
  participantIds = [p['participantId'] for p in teamInfo]
  teamOne = participantIds[:5]
  teamTwo = participantIds[5:]
  
#  print ("Champions: {}".format(championIds))
  champNames = list(map(championIdToName, championIds))
#  print ("Names: {}".format(champNames))

  dragons = []
  frames = match['timeline']['frames']
  for frame in frames:
    events = frame.get('events', [])
    for event in events:
      monsterType = event.get('monsterType', None)
      if monsterType == 'DRAGON':
        time = event['timestamp']
        killer = event['killerId']
#        print ("killer {} @{:.0f}s".format(
#          champNames[killer-1], time / 1000))
        dragons.append((time, killer in teamOne))

  result = match['teams'][0]['winner']
  return result, champNames, dragons


# MAIN CODE

output = open(OUTPUT_FILE, mode='w')

gameNum = 0
for fileNumber in range(1, 11):
  fileData = loadFile(FILE_NAME.format(fileNumber))
  parsed = json.loads(fileData)

  games = parsed['matches']
  for game in games:
    metrics = parseGameRough(game)
    output.write(str(metrics))
    print (gameNum, metrics)
    gameNum += 1

output.close()

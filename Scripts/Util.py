import json
import os.path

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


def getDataDirFileName(name):
  if name.startswith('../'):
    return name

  DATA_DIR = '../Data/'
  return os.path.join(DATA_DIR, name)


def loadJsonFile(name):
  fileName = getDataDirFileName(name)
  with open(fileName) as f:
    fileData = f.read()
  return json.loads(fileData)


def getJsonString(data):
  return json.dumps(data)


def writeJsonFile(name, data):
  fileName = getDataDirFileName(name)
  with open(fileName, mode = 'w') as f:
    json.dump(data, f, indent=1)


champNameMapping = [('Annie', 1), ('Olaf', 2), ('Galio', 3), ('Twisted Fate', 4), ('Xin Zhao', 5), ('Urgot', 6), ('LeBlanc', 7), ('Vladimir', 8), ('Fiddlesticks', 9), ('Kayle', 10), ('Master Yi', 11), ('Alistar', 12), ('Ryze', 13), ('Sion', 14), ('Sivir', 15), ('Soraka', 16), ('Teemo', 17), ('Tristana', 18), ('Warwick', 19), ('Nunu', 20), ('Miss Fortune', 21), ('Ashe', 22), ('Tryndamere', 23), ('Jax', 24), ('Morgana', 25), ('Zilean', 26), ('Singed', 27), ('Evelynn', 28), ('Twitch', 29), ('Karthus', 30), ('Cho\'Gath', 31), ('Amumu', 32), ('Rammus', 33), ('Anivia', 34), ('Shaco', 35), ('Dr. Mundo', 36), ('Sona', 37), ('Kassadin', 38), ('Irelia', 39), ('Janna', 40), ('Gangplank', 41), ('Corki', 42), ('Karma', 43), ('Taric', 44), ('Veigar', 45), ('Trundle', 48), ('Swain', 50), ('Caitlyn', 51), ('Blitzcrank', 53), ('Malphite', 54), ('Katarina', 55), ('Nocturne', 56), ('Maokai', 57), ('Renekton', 58), ('Jarvan IV', 59), ('Elise', 60), ('Orianna', 61), ('Wukong', 62), ('Brand', 63), ('Lee Sin', 64), ('Vayne', 67), ('Rumble', 68), ('Cassiopeia', 69), ('Skarner', 72), ('Heimerdinger', 74), ('Nasus', 75), ('Nidalee', 76), ('Udyr', 77), ('Poppy', 78), ('Gragas', 79), ('Pantheon', 80), ('Ezreal', 81), ('Mordekaiser', 82), ('Yorick', 83), ('Akali', 84), ('Kennen', 85), ('Garen', 86), ('Leona', 89), ('Malzahar', 90), ('Talon', 91), ('Riven', 92), ('Kog\'Maw', 96), ('Shen', 98), ('Lux', 99), ('Xerath', 101), ('Shyvana', 102), ('Ahri', 103), ('Graves', 104), ('Fizz', 105), ('Volibear', 106), ('Rengar', 107), ('Varus', 110), ('Nautilus', 111), ('Viktor', 112), ('Sejuani', 113), ('Fiora', 114), ('Ziggs', 115), ('Lulu', 117), ('Draven', 119), ('Hecarim', 120), ('Kha\'Zix', 121), ('Darius', 122), ('Jayce', 126), ('Lissandra', 127), ('Diana', 131), ('Quinn', 133), ('Syndra', 134), ('Zyra', 143), ('Gnar', 150), ('Zac', 154), ('Yasuo', 157), ('Vel\'Koz', 161), ('Braum', 201), ('Jinx', 222), ('Lucian', 236), ('Zed', 238), ('Vi', 254), ('Aatrox', 266), ('Nami', 267), ('Azir', 268), ('Thresh', 412), ('Rek\'Sai', 421), ('Kalista', 429)]
def getChamps():
  return champNameMapping



def championIdToName(champId):
  for k,v in getChamps():
    if v == champId:
      return k
  return 'unknown-champion-{}'.format(champId)


def getTowerNumber(isTeamOneTower, lane, tower):
  lanes = ('BOT_LANE', 'MID_LANE', 'TOP_LANE')
  towers = ('OUTER_TURRET', 'INNER_TURRET', 'BASE_TURRET', 'NEXUS_TURRET')

  # TODO(sethtroisi): figure out how to deal with both NEXUS_TOWER being in MID_LANE

  assert isTeamOneTower in (False, True)
  assert lane in lanes
  assert tower in towers

  return len(towers) * len(lanes) * (isTeamOneTower == False) + \
      lanes.index(lane) * len(towers) + \
      towers.index(tower)

def teamATowerKill(towerNum):
  lanes = 3
  towers = 4
  return towerNum < (lanes * towers)

def getInhibNumber(isTeamOneInhib, lane):
  lanes = ('BOT_LANE', 'MID_LANE', 'TOP_LANE')

  assert isTeamOneInhib in (False, True)
  assert lane in lanes

  return len(lanes) * (isTeamOneInhib == False) + \
      lanes.index(lane)

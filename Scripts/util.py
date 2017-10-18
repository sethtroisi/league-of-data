import json
import re
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

SECONDS_PER_BLOCK = 2 * 60

DRAGON_NAMES = ["air", "water", "earth", "fire"]

SUMMONER_SPELLS_BY_ID = {
    1: "Cleanse", 3: "Exhaust", 4: "Flash", 6: "Ghost", 7: "Heal", 11: "Smite",
    12: "Teleport", 13: "Clarity", 14: "Ignite", 21: "Barrier"
}


def timeToBlock(time):
    # I think it's more correct to return the block it's happening in.
    # IE event (T = 0) = 0, (0 < T <= 5) = 1
    # This for sure will be a source of off by one errors be wary.
    return (time - 1) // SECONDS_PER_BLOCK + 1


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
    with open(fileName, mode='w') as f:
        json.dump(data, f, indent=1)


def getAndValidate(data, key, validValues):
    value = data.get(key, None)
    assert value in validValues, "{} not in ({})".format(value, validValues)
    return value


def rankOrdering(rank):
    order = {
        'BRONZE': 0,
        'SILVER': 1,
        'UNRANKED': 1,
        'GOLD': 2,
        'PLATINUM': 3,
        'DIAMOND': 4,
        'CHALLENGER': 5,
        'MASTER': 6
    }
    assert rank.upper() in order, "Unknown Rank: '{}'".format(rank)
    return order[rank]


# noinspection SpellCheckingInspection
def spellIdToName(spellId):
    return SUMMONER_SPELLS_BY_ID[spellId].lower()


# noinspection SpellCheckingInspection
champIdToName = {
    1: "Annie", 2: "Olaf", 3: "Galio", 4: "Twisted Fate", 5: "Xin Zhao", 6: "Urgot", 7: "LeBlanc", 8: "Vladimir",
    9: "Fiddlesticks", 10: "Kayle", 11: "Master Yi", 12: "Alistar", 13: "Ryze", 14: "Sion", 15: "Sivir", 16: "Soraka",
    17: "Teemo", 18: "Tristana", 19: "Warwick", 20: "Nunu", 21: "Miss Fortune", 22: "Ashe", 23: "Tryndamere", 24: "Jax",
    25: "Morgana", 26: "Zilean", 27: "Singed", 28: "Evelynn", 29: "Twitch", 30: "Karthus", 31: "Cho'Gath", 32: "Amumu",
    33: "Rammus", 34: "Anivia", 35: "Shaco", 36: "Dr. Mundo", 37: "Sona", 38: "Kassadin", 39: "Irelia", 40: "Janna",
    41: "Gangplank", 42: "Corki", 43: "Karma", 44: "Taric", 45: "Veigar", 48: "Trundle", 50: "Swain", 51: "Caitlyn",
    53: "Blitzcrank", 54: "Malphite", 55: "Katarina", 56: "Nocturne", 57: "Maokai", 58: "Renekton", 59: "Jarvan IV",
    60: "Elise", 61: "Orianna", 62: "Wukong", 63: "Brand", 64: "Lee Sin", 67: "Vayne", 68: "Rumble", 69: "Cassiopeia",
    72: "Skarner", 74: "Heimerdinger", 75: "Nasus", 76: "Nidalee", 77: "Udyr", 78: "Poppy", 79: "Gragas",
    80: "Pantheon", 81: "Ezreal", 82: "Mordekaiser", 83: "Yorick", 84: "Akali", 85: "Kennen", 86: "Garen", 89: "Leona",
    90: "Malzahar", 91: "Talon", 92: "Riven", 96: "Kog'Maw", 98: "Shen", 99: "Lux", 101: "Xerath", 102: "Shyvana",
    103: "Ahri", 104: "Graves", 105: "Fizz", 106: "Volibear", 107: "Rengar", 110: "Varus", 111: "Nautilus",
    112: "Viktor", 113: "Sejuani", 114: "Fiora", 115: "Ziggs", 117: "Lulu", 119: "Draven", 120: "Hecarim",
    121: "Kha'Zix", 122: "Darius", 126: "Jayce", 127: "Lissandra", 131: "Diana", 133: "Quinn",
    134: "Syndra", 136: "Aurelion Sol", 141: "Kayn", 143: "Zyra", 150: "Gnar", 154: "Zac", 157: "Yasuo",
    161: "Vel'Koz", 163: "Taliyah", 164: "Camille", 201: "Braum", 202: "Jhin", 203: "Kindred", 222: "Jinx",
    223: "Tahm Kench", 236: "Lucian", 238: "Zed", 240: "Kled", 245: "Ekko", 254: "Vi", 266: "Aatrox", 267: "Nami",
    268: "Azir", 412: "Thresh", 420: "Illaoi", 421: "Rek'Sai", 427: "Ivern", 429: "Kalista", 432: "Bard",
    497: "Rakan", 498: "Xayah", 516: "Ornn"
}


def championIdToName(champId):
    return champIdToName.get(champId)  # , 'unknown-champion-{}'.format(champId))


idToMinimized = {champId: sorted(champIdToName.keys()).index(champId) for champId in champIdToName.keys()}


def minimizedChampId(champId):
    return idToMinimized[champId]


def getTowerNumber(isTeamOneTower, lane, tower):
    LANES = ('BOT_LANE', 'MID_LANE', 'TOP_LANE')
    TOWERS = ('OUTER_TURRET', 'INNER_TURRET', 'BASE_TURRET', 'NEXUS_TURRET')

    # NOTE: One of the NEXUS_TURRETS is moved to TOP_LANE to prevent duplicate towers.

    assert isTeamOneTower in (False, True)
    assert lane in LANES
    assert tower in TOWERS

    return len(TOWERS) * len(LANES) * (isTeamOneTower is False) + \
        LANES.index(lane) * len(TOWERS) + TOWERS.index(tower)


def teamATowerKill(towerNum):
    lanes = 3
    towers = 4
    return towerNum < lanes * towers


def getInhibNumber(isTeamOneInhib, lane):
    lanes = ('BOT_LANE', 'MID_LANE', 'TOP_LANE')

    assert isTeamOneInhib in (False, True)
    assert lane in lanes

    return len(lanes) * (isTeamOneInhib is False) + lanes.index(lane)


def guessPosition(champ):
    role = champ['role']
    lane = champ['lane']

    assert lane in ("JUNGLE", "BOTTOM", "MIDDLE", "TOP"), lane
    assert role in ("SOLO", "CARRY", "DUO", "DUO_CARRY", "DUO_SUPPORT", "NONE"), role

    if lane == "JUNGLE" and role == "NONE":
        return "JUNGLE"

    if lane == "MIDDLE" and role == "SOLO":
        return "MID"

    if lane == "TOP" and role == "SOLO":
        # maybe try to account for lane swap?
        return "TOP"

    if lane == "BOTTOM":
        if "SUPPORT" in role:
            return "SUPPORT"
        if "CARRY" in role:
            return "ADC"

        if role == "DUO":
            # suppressing some of the error below, not sure what position this is
            return "OTHER"

    # print ("ERROR unknown position:", lane, role, "\t", champ)
    return "OTHER"


def compressFeatureList(featuresUsed):
    savedToken = "(?<=_)({})(?=$|_)"

    def patternRe(pattern):
        return re.compile(savedToken.format(pattern), re.I)

    def listRe(items):
        return re.compile(savedToken.format("|".join(items)), re.I)

    compressedList = set()
    for feature in featuresUsed:
        feature = patternRe("[0-9]+").sub("<X>", feature)
        feature = patternRe("[AB]").sub("<team>", feature)
        feature = listRe(DRAGON_NAMES).sub("<dragon>", feature)
        feature = listRe(SUMMONER_SPELLS_BY_ID.values()).sub("<summoner_spell>", feature)

        compressedList.add(feature)

    compressedFeatures = ", ".join(sorted(compressedList))
    return len(compressedList), compressedFeatures

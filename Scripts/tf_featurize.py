import Util

from collections import defaultdict

GOLD_DELTA_BUCKET_SIZE = 2000


# Create a feature that counts how many events of the type have happened.
def countedFeature(df, name, events, sampleTime, verus=True):
    counts = [0, 0]
    for event in events:
        eventTime, isTeamA = event[:2]
        assert 0 <= eventTime <= 10000
        assert isTeamA in (True, False)

        if eventTime > sampleTime:
            break

        counts[isTeamA] += 1
        feature = '{}_{}_{}'.format(name, 'A' if isTeamA else 'B', counts[isTeamA])
        df[feature] = 1.0

    if verus:
        feature = '{}_{}_to_{}'.format(name, counts[0], counts[1])
        df[feature] = 1.0


# Create features from champs
def champFeature(data, champs):
    #assert False, "None of these seemed to decrease log loss :("

    ranks = defaultdict(int)
    summoners = defaultdict(int)
    for playerI, champ in enumerate(champs):
        champId = champ['championId']
        isTeamA = champ['isTeamOne']

        champion = Util.championIdToName(champId)
        minchampId = Util.minimizedChampId(champId)

        position = Util.guessPosition(champ)

        spell1 = champ['spell1']
        spell2 = champ['spell2']

        summoners[(isTeamA, spell1)] += 1
        summoners[(isTeamA, spell2)] += 1

        rank = champ['approxRank']
        ranks[(isTeamA, rank)] += 1

        data['embedding_team_{}_player_{}_champion'.format('A' if isTeamA else 'B', playerI)] = minchampId
        data['embedding_team_{}_position_{}_champion'.format('A' if isTeamA else 'B', position)] = minchampId
#        data['team_{}_has_champion_{}'.format('A' if isTeamA else 'B', champId)] = 1

    for (isTeamA, spellId), count in summoners.items():
        spellName = Util.spellIdToName(spellId)
        data['team_spells_{}_{}s'.format('A' if isTeamA else 'B', spellName)] = count

    sumRank = 0
    for (isTeamA, rank), count in ranks.items():
        sumRank += (1 if isTeamA else -1) * count * Util.rankOrdering(rank)
        data['team_ranks_{}_{}s'.format('A' if isTeamA else 'B', rank)] = float(count)
    data['rank_sum_diff'] = sumRank


# Create features from towers (team, position)
def towerFeatures(df, towers, sampleTime):
#  for tower in range(0, 24):
#        df.set_value(0, 'tower_{}_destroyed_at',

    towersA, towersB = 0, 0
    for towerData in towers:
        towerTime, isTeamA, towerNum = towerData
        if towerTime > sampleTime:
            break

        if isTeamA:
            towersA += 1
            if towersA == 1:
                df['first_tower_A'] = towerTime / 1800
            df['last_tower_A'] = towerTime / 1800
        else:
            towersB += 1
            if towersB == 1:
                df['first_tower_B'] = towerTime / 1800
            df['last_tower_B'] = towerTime / 1800

        # TODO figure out how to default other values to infinite or something
        df['tower_{}_destroyed'.format(towerNum)] = 1
        df['tower_{}_destroyed_at'.format(towerNum)] = towerTime / 1800

    df['towers_destroyed_A'] = towersA
    df['towers_destroyed_B'] = towersB


def dragonFeatures(df, dragons, sampleTime):
    features = set()

    dragonsA = []
    dragonsB = []

    for dragonI, dragon in enumerate(dragons, 1):
        dragonTime, name, isTeamA = dragon
        if dragonTime > sampleTime:
            break

        if isTeamA:
            df['last_dragon_A'] = dragonTime / 1800
            dragonsA.append(name)
        else:
            df['last_dragon_B'] = dragonTime / 1800
            dragonsB.append(name)

        df['dragon_{}_taken_at'.format(dragonI)] = dragonTime / 1800
        df['dragon_{}_taken_by'.format(dragonI)] = 1 if isTeamA else -1

    df["dragon_taken_A"] = len(dragonsA)
    df["dragon_taken_B"] = len(dragonsB)

    for dType in set(dragonsA + dragonsB):
        name = dType.lower()
        df["dragon_A_" + name] = dragonsA.count(dType)
        df["dragon_B_" + name] = dragonsB.count(dType)

    return features


# Creates features from gold values (delta)
def goldFeatures(df, gold, sampleTime):
    lastBlock = Util.timeToBlock(sampleTime)
    lastBlockNum = max(b for b in map(int, gold.keys()) if b <= lastBlock)

#   Explore adding each positions' gold / adv
#    position = Util.guessPosition(champ)

    for blockNum in range(lastBlock+1):
        teamAGold = 0
        teamBGold = 0

        blockGold = gold.get(str(blockNum), None)
        if blockGold:
            for pId, totalGold in blockGold.items():
                pId = int(pId)

                assert 1 <= pId <= 10
                if 1 <= pId <= 5:
                    teamAGold += totalGold
                else:
                    teamBGold += totalGold

        # Each team gets ~3k more gold every 2 minutes, makes vars ~ [0.5, 2]
        normalizeFactor = 3000 * (blockNum + 1)
        df['gold_block_{}_a'.format(blockNum)] = teamAGold / normalizeFactor
        df['gold_block_{}_b'.format(blockNum)] = teamBGold / normalizeFactor

        # Normalized in TFModel
        deltaGold = teamAGold - teamBGold
        df['gold_adv_block_{}_a'.format(blockNum)] = max(0, deltaGold)
        df['gold_adv_block_{}_b'.format(blockNum)] = max(0, -deltaGold)


def parseGame(parsed, time):
    if time is None:
        assert False

    gameFeatures = parsed['features']

    # Gold
    gold = gameFeatures['gold']

    # Objectives
    barons = gameFeatures['barons']
    dragons = gameFeatures['dragons']
    towers = gameFeatures['towers']
    inhibs = gameFeatures['inhibs']

    # Champions
    champs = gameFeatures['champs']

    # Data that ML will see

    data = dict()
    data['current_time'] = time / 3600

    champFeature(data, champs)

    goldFeatures(data, gold, time)
    towerFeatures(data, towers, time)
    dragonFeatures(data, dragons, time)

    countedFeature(data, 'inhibs', inhibs, time)
    countedFeature(data, 'barons', barons, time)

    return data


def getRawGameData(args):
    fileName = args.input_file
    numGames = args.num_games
    rank = args.rank

    games = []
    goals = []

    requiredRank = Util.rankOrdering(rank)

    outputData = Util.loadJsonFile(fileName)
    for dataI, data in enumerate(outputData):
        if data['debug']['duration'] < 600:
            # Filtering remakes and stuff
            continue

        # TODO consider removing surrender games

        # Filter out low rank games
        lowerRanked = len([1 for c in data['features']['champs'] if Util.rankOrdering(c['approxRank']) < requiredRank])
        if lowerRanked >= 2:
            continue

        #t = data['features']['champs']
        #if not any(s['championId'] == 11 for s in t):
        #    continue

        #index = min([i for i, s in enumerate(t) if s['championId'] == 11])
        #yiWin = (index <= 4) == data['goal']
        #import random
        #if not (random.random() < 0.2 or yiWin):
        #    continue



        goal = data['goal']
        assert goal in (True, False)

        games.append(data)
        goals.append(goal)

        if len(games) == numGames:
            break

    print ("Loaded {} games (filtereed {})".format(
        len(goals), len(outputData) - len(goals)))
    return games, goals
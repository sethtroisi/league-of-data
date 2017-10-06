from collections import defaultdict

import util

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


def champFeature(data, champs):
    ranks = defaultdict(int)
    summoners = defaultdict(int)
    for playerI, champ in enumerate(champs):
        champId = champ['championId']
        isTeamA = champ['isTeamOne']

        # champion = util.championIdToName(champId)
        minChampId = util.minimizedChampId(champId)

        position = util.guessPosition(champ)

        spell1 = champ['spell1']
        spell2 = champ['spell2']

        summoners[(isTeamA, spell1)] += 1
        summoners[(isTeamA, spell2)] += 1

        rank = champ['approxRank']
        ranks[(isTeamA, rank)] += 1

#        data['embedding_team_{}_player_{}_champion'.format('A' if isTeamA else 'B', playerI)] = minChampId
#        data['embedding_team_{}_position_{}_champion'.format('A' if isTeamA else 'B', position)] = minChampId
#        data['team_{}_has_champion_{}'.format('A' if isTeamA else 'B', champId)] = 1

    for (isTeamA, spellId), count in summoners.items():
        spellName = util.spellIdToName(spellId)
        data['team_spells_{}_{}s'.format('A' if isTeamA else 'B', spellName)] = count

    sumRank = 0
    for (isTeamA, rank), count in ranks.items():
        sumRank += (1 if isTeamA else -1) * count * util.rankOrdering(rank)
        data['team_ranks_{}_{}s'.format('A' if isTeamA else 'B', rank)] = float(count)
    data['rank_sum_diff'] = sumRank


def towerFeatures(df, towers, sampleTime):
    # Experiment with setting to far future
    # for tower in range(0, 24):
    #    df.set_value(0, 'tower_{}_destroyed_at', 2)

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


def goldFeatures(df, gold, sampleTime):
    lastBlock = util.timeToBlock(sampleTime)

    # Explore adding each positions' gold / adv
    # position = Util.guessPosition(champ)

    for blockNum in range(lastBlock+1):
        blockGold = gold.get(str(blockNum))

        assert len(blockGold) == 10
        playersAGold = blockGold[:5]
        playersBGold = blockGold[5:]
        assert len(playersAGold) == len(playersBGold) == 5

        # Each player gets ~500 / 2 minutes, team gets ~3k / 2 minutes. Normalize features to ~ [0.5, 2].
        playerNormalizeFactor = 600 * (blockNum + 1)
        teamNormalizeFactor = 5 * playerNormalizeFactor

        playersAGold.sort(reverse=True)
        playersBGold.sort(reverse=True)
        for richIndex, (goldA, goldB) in enumerate(zip(playersAGold, playersBGold), 1):
            df['gold_{}_richest_A_block_{}'.format(richIndex, blockNum)] = goldA / playerNormalizeFactor
            df['gold_{}_richest_B_block_{}'.format(richIndex, blockNum)] = goldB / playerNormalizeFactor

        teamAGold = sum(playersAGold)
        teamBGold = sum(playersBGold)
        df['gold_block_{}_A'.format(blockNum)] = teamAGold / teamNormalizeFactor
        df['gold_block_{}_B'.format(blockNum)] = teamBGold / teamNormalizeFactor

        # Normalized in TFModel
        deltaGold = teamAGold - teamBGold
        df['gold_adv_block_{}_A'.format(blockNum)] = max(0, deltaGold)
        df['gold_adv_block_{}_B'.format(blockNum)] = max(0, -deltaGold)


def farmFeatures(df, farm, jungleFarm, sampleTime):
    # TODO add jungle farm somewhere
    lastBlock = util.timeToBlock(sampleTime)

    # Explore adding some Jungle features.
    # position = Util.guessPosition(champ)

    for blockNum in range(lastBlock+1):
        blockFarm = farm.get(str(blockNum))

        assert len(blockFarm) == 10
        playersAFarm = blockFarm[:5]
        playersBFarm = blockFarm[5:]
        assert len(playersAFarm) == len(playersBFarm) == 5

        # Each player gets ~16 / 2 minutes, team gets ~80 / 2 minutes. Normalize features to ~ [0.5, 2].
        playerNormalizeFactor = 16 * (blockNum + 1)
        teamNormalizeFactor = 4 * playerNormalizeFactor

        playersAFarm.sort(reverse=True)
        playersBFarm.sort(reverse=True)
        for farmedIndex, (farmA, farmB) in enumerate(zip(playersAFarm, playersBFarm), 1):
            df['farmed_{}_best_A_block_{}'.format(farmedIndex, blockNum)] = farmA / playerNormalizeFactor
            df['farmed_{}_best_B_block_{}'.format(farmedIndex, blockNum)] = farmB / playerNormalizeFactor

        teamAFarm = sum(playersAFarm)
        teamBFarm = sum(playersBFarm)
        df['farm_block_{}_A'.format(blockNum)] = teamAFarm / teamNormalizeFactor
        df['farm_block_{}_B'.format(blockNum)] = teamBFarm / teamNormalizeFactor

        deltaFarm = teamAFarm - teamBFarm
        df['farm_adv_block_{}_A'.format(blockNum)] = max(0, deltaFarm)
        df['farm_adv_block_{}_B'.format(blockNum)] = max(0, -deltaFarm)


def parseGame(parsed, time):
    if time is None:
        assert False

    gameFeatures = parsed['features']

    # What you are: Champions
    champs = gameFeatures['champs']

    # What you have: Gold
    gold = gameFeatures['gold']
    farm = gameFeatures['farm']
    jungleFarm = gameFeatures['jungleFarm']

    # What you did: Objectives
    barons = gameFeatures['barons']
    dragons = gameFeatures['dragons']
    towers = gameFeatures['towers']
    inhibs = gameFeatures['inhibs']

    # Data that ML will see
    data = dict()
    data['current_time'] = time / 3600

    champFeature(data, champs)

    goldFeatures(data, gold, time)
    farmFeatures(data, farm, jungleFarm, time)

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

    requiredRank = util.rankOrdering(rank)

    outputData = util.loadJsonFile(fileName)
    for dataI, data in enumerate(outputData):
        if data['debug']['duration'] < 600:
            # Filtering remakes and stuff
            continue

        # TODO consider removing surrender games

        # Filter out low rank games
        lowerRanked = len([1 for c in data['features']['champs'] if util.rankOrdering(c['approxRank']) < requiredRank])
        if lowerRanked >= 2:
            continue

        goal = data['goal']
        assert goal in (True, False)

        games.append(data)
        goals.append(goal)

        if len(games) == numGames:
            break

    print ("Loaded {} games (filtered {})".format(
        len(goals), len(outputData) - len(goals)))
    return games, goals

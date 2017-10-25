from collections import defaultdict

import util

GOLD_DELTA_BUCKET_SIZE = 2000


def splitBlockFeature(featureByPid):
    assert len(featureByPid) == 10
    playersA = featureByPid[:5]
    playersB = featureByPid[5:]
    assert len(playersA) == len(playersB) == 5
    return playersA, playersB


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




def champFeature(df, champs):
    ranks = defaultdict(int)
    summoners = defaultdict(int)
    for champ in champs:
        champId = champ['championId']
        # pId = champ['pId']
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

        # df['embedding_team_{}_player_{}_champion'.format('A' if isTeamA else 'B', playerI)] = minChampId
        df['embedding_team_{}_position_{}_champion'.format('A' if isTeamA else 'B', position)] = minChampId
        df['team_{}_has_champion_{}'.format('A' if isTeamA else 'B', champId)] = 1

    for (isTeamA, spellId), count in summoners.items():
        spellName = util.spellIdToName(spellId)
        df['team_spells_{}_{}s'.format('A' if isTeamA else 'B', spellName)] = count

    sumRank = 0
    for (isTeamA, rank), count in ranks.items():
        sumRank += (1 if isTeamA else -1) * count * util.rankOrdering(rank)
        df['team_ranks_{}_{}s'.format('A' if isTeamA else 'B', rank)] = float(count)
    df['rank_sum_diff'] = sumRank


def towerFeatures(df, towers, sampleTime):
    # Note: Awkwardly [TowersB, TowersA] because of index of [True] = [1]
    towersDestroyed = [0, 0]
    for towerdf in towers:
        towerTime, isTeamA, towerNum = towerdf
        if towerTime > sampleTime:
            break
        team = 'A' if isTeamA else 'B'

        # Experimented with setting to "far future" (5) and saw no improvement over defaulting to 0
        df['tower_{}_destroyed'.format(towerNum)] = 1
        df['tower_{}_destroyed_at'.format(towerNum)] = towerTime / 1800

        towersDestroyed[isTeamA] += 1

        # TODO: investigate why trinary "1 if isTeamA else -1" (0 = neither), isn't better here.
        nthTower = sum(towersDestroyed)
        df['{}_nth_tower_destroyed_by'.format(nthTower)] = isTeamA
        df['{}_nth_tower_destroyed_at'.format(nthTower)] = towerTime / 1800

        df['last_tower_destroyed_by'] = isTeamA
        df['last_tower_destroyed_at'.format(team)] = towerTime / 1800

    df['towers_destroyed_A'] = towersDestroyed[True]
    df['towers_destroyed_B'] = towersDestroyed[False]

    # TODO consider adding per block counts.


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


def disconnectFeatures(df, disconnect, sampleTime):
    lastBlock = util.timeToBlock(sampleTime)

    for blockNum in range(lastBlock+1):
        blockDisconnects = disconnect[str(blockNum)]
        playersADisc, playersBDisc = splitBlockFeature(blockDisconnects)

        # TODO refactor to function and helper
        df['num_disconnects_{}_A'.format(blockNum)] = len([disc for disc in playersADisc if disc > 0.25])
        df['num_disconnects_{}_B'.format(blockNum)] = len([disc for disc in playersBDisc if disc > 0.25])

        df['sum_disconnects_{}_A'.format(blockNum)] = sum(playersADisc)
        df['sum_disconnects_{}_B'.format(blockNum)] = sum(playersBDisc)


def goldFeatures(df, gold, champs, sampleTime):
    lastBlock = util.timeToBlock(sampleTime)

    for blockNum in range(lastBlock+1):
        blockGold = gold[str(blockNum)]
        playersAGold, playersBGold = splitBlockFeature(blockGold)

        # Each player gets ~500 / 2 minutes, team gets ~3k / 2 minutes. Normalize features to ~ [0.5, 2].
        playerNormalizeFactor = 600 * (blockNum + 1)
        teamNormalizeFactor = 5 * playerNormalizeFactor

        # 3K lead at 20 minutes => ~[0, 3]
        advantageNormalizeFactor = 300 * (blockNum + 1)

        playersAGold.sort(reverse=True)
        playersBGold.sort(reverse=True)

        teamAGold = sum(playersAGold)
        teamBGold = sum(playersBGold)
        df['gold_block_{}_A'.format(blockNum)] = teamAGold / teamNormalizeFactor
        df['gold_block_{}_B'.format(blockNum)] = teamBGold / teamNormalizeFactor

        # Normalized delta to ~ [0, 3]
        deltaGold = (teamAGold - teamBGold) / advantageNormalizeFactor
        df['gold_adv_block_{}_A'.format(blockNum)] = max(0.0, deltaGold)
        df['gold_adv_block_{}_B'.format(blockNum)] = max(0.0, -deltaGold)


        for richIndex, (goldA, goldB) in enumerate(zip(playersAGold, playersBGold), 1):
            df['gold_{}_richest_A_block_{}'.format(richIndex, blockNum)] = goldA / playerNormalizeFactor
            df['gold_{}_richest_B_block_{}'.format(richIndex, blockNum)] = goldB / playerNormalizeFactor

        for champ in champs:
            isTeamA = champ['isTeamOne']
            pId = util.getAndValidate(champ, 'pId', range(1,11))
            position = util.guessPosition(champ)

            positionGold = blockGold[pId - 1] / playerNormalizeFactor
            team = 'A' if isTeamA else 'B'
            df['gold_{}_block_{}_{}'.format(position, blockNum, team)] = positionGold


def farmFeatures(df, farm, jungleFarm, sampleTime):
    lastBlock = util.timeToBlock(sampleTime)

    for blockNum in range(lastBlock+1):
        blockFarm = farm[str(blockNum)]
        blockJungle = jungleFarm[str(blockNum)]
        playersAFarm, playersBFarm = splitBlockFeature(blockFarm)
        playersAJungle, playersBJungle = splitBlockFeature(blockJungle)

        # Each player gets ~16 / 2 minutes, team gets ~80 / 2 minutes. Normalize features to ~ [0.5, 2].
        playerNormalizeFactor = 16 * (blockNum + 1)
        teamNormalizeFactor = 4 * playerNormalizeFactor
        advantageNormalizeFactor = 20 * (blockNum + 1)

        playersAFarm.sort(reverse=True)
        playersBFarm.sort(reverse=True)
        for farmedIndex, (farmA, farmB) in enumerate(zip(playersAFarm, playersBFarm), 1):
            df['farmed_{}_best_A_block_{}'.format(farmedIndex, blockNum)] = farmA / playerNormalizeFactor
            df['farmed_{}_best_B_block_{}'.format(farmedIndex, blockNum)] = farmB / playerNormalizeFactor

        teamAFarm = sum(playersAFarm)
        teamBFarm = sum(playersBFarm)
        df['farm_block_{}_A'.format(blockNum)] = teamAFarm / teamNormalizeFactor
        df['farm_block_{}_B'.format(blockNum)] = teamBFarm / teamNormalizeFactor

        deltaFarm = (teamAFarm - teamBFarm) / advantageNormalizeFactor
        df['farm_adv_block_{}_A'.format(blockNum)] = max(0.0, deltaFarm)
        df['farm_adv_block_{}_B'.format(blockNum)] = max(0.0, -deltaFarm)

        teamAJungle = sum(playersAJungle)
        teamBJungle = sum(playersBJungle)
        # Team only gets 1 player worth of jungle per block.
        df['jungle_block_{}_A'.format(blockNum)] = teamAJungle / playerNormalizeFactor
        df['jungle_block_{}_B'.format(blockNum)] = teamBJungle / playerNormalizeFactor

        deltaJungle = (teamAJungle - teamBJungle) / advantageNormalizeFactor
        df['jungle_adv_block_{}_A'.format(blockNum)] = max(0.0, deltaJungle)
        df['jungle_adv_block_{}_B'.format(blockNum)] = max(0.0, -deltaJungle)



def parseGame(parsed, time):
    if time is None:
        assert False

    gameFeatures = parsed['features']

    # What you are: Champions
    champs = gameFeatures['champs']

    disconnect = gameFeatures['disconnect']

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
    featureData = dict()
    featureData['current_time'] = time / 3600

    champFeature(featureData, champs)

    # Gold and Tower are the highest valued predictors
    goldFeatures(featureData, gold, champs, time)
    farmFeatures(featureData, farm, jungleFarm, time)
    towerFeatures(featureData, towers, time)

    dragonFeatures(featureData, dragons, time)
    countedFeature(featureData, 'inhibs', inhibs, time)

    # Not available pre 20.
    # countedFeature(data, 'barons', barons, time)

    # Doesn't seem to help
    # disconnectFeatures(data, disconnect, time)

    return featureData


def getRawGameData(args):
    fileName = args.input_file
    numGames = args.num_games

    requiredRank = util.rankOrdering(args.rank)

    filtered = defaultdict(int)
    if args.filter_weird_games:
        print()
        print("FILTERING WEIRD_GAMES")
        print("INVALIDATES STATS PROBABLY")
        print("INVESTIGATE IF YOU ARE USING THIS")
        print()

    games = []
    goals = []

    outputData = util.loadJsonFile(fileName)
    for dataI, data in enumerate(outputData):
        # Filtering remakes and stuff (sudo valid beacuse know at t ~= 0)
        if data['debug']['duration'] < 600:
            filtered['short_game'] += 1
            continue

        # Filter out low rank games (valid because know at t = 0)
        lowerRanked = len([1 for c in data['features']['champs'] if util.rankOrdering(c['approxRank']) < requiredRank])
        if lowerRanked >= 2:
            filtered['rank'] += 1
            continue

        if args.filter_weird_games:
            if data['debug']['surrendered']:
                filtered['surrendered'] += 1
                continue

            anyOtherPositions = False
            for champ in data['features']['champs']:
                position = util.guessPosition(champ)
                if position == "OTHER":
                    anyOtherPositions = True
                    break
            if anyOtherPositions:
                filtered['bad_position'] += 1
                continue

        goal = data['goal']
        assert goal in (True, False)

        games.append(data)
        goals.append(goal)

        if len(games) == numGames:
            break

    filterCount = len(outputData) - len(goals)
    assert filterCount == sum(filtered.values())

    print("Loaded {} games (filtered {} = {:.1f}%)".format(
        len(goals), filterCount, 100 * filterCount / len(outputData)))

    for reason, count in sorted(filtered.items()):
        print("\t{}: {}".format(reason, count))
    print()

    return games, goals

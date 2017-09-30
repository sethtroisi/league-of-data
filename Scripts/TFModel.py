import argparse
import datetime
import itertools
import functools
import sklearn.metrics
import time

from collections import defaultdict
from sklearn.model_selection import train_test_split

import GraphModelStats
import TFFeaturize
import Util


import tensorflow as tf


def getArgParse():
    parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

    parser.add_argument(
        '-i', '--input-file',
        type=str,
        default='features.json',
        help='Input match file (produced by Seth or GameParser.py)')

    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=-1,
        help='Numbers of games to load (default -1 = all)')

    parser.add_argument(
        '-b', '--blocks',
        type=str,
        default="all",
        help='blocks to test (all, 5, 13, 2-10)')

    parser.add_argument(
        '-r', '--rank',
        type=str,
        default='GOLD',
        help='Filter out all games from ranks below this')

    parser.add_argument(
        '-p', '--holdback',
        type=int,
        default=15,
        help='percent of games to holdback for testing/validation')

    parser.add_argument(
        '-v', '--verbose',
        type=int,
        default=0,
        help='how much data to print (0 = little, 1 = most, 3 = everything)')

    return parser


def processBlocks(args):
    lowBlock = 0
    highBlock = int(3600 // Util.SECONDS_PER_BLOCK)

    rawBlock = args.blocks
    if rawBlock == "all":
        pass
    elif '-' in rawBlock:
        lowBlock, highBlock = list(map(int, rawBlock.split("-")))
    else:
        assert rawBlock.isdigit(), "block not digits [{}]".format(rawBlock)
        lowBlock = highBlock = int(rawBlock)

    return list(range(lowBlock, highBlock + 1))


def filterMaxBlock(blockNum, games, goals):
    blockStart = blockNum * Util.SECONDS_PER_BLOCK

    indexes = []
    useableGames = []
    useableGoals = []
    for i, (game, goal) in enumerate(zip(games, goals)):
        if blockStart > game['debug']['duration']:
            continue

        indexes.append(i)
        useableGames.append(game)
        useableGoals.append(goal)

    return indexes, useableGames, useableGoals


def featuresToColumns(features):
    requiredToBeFound = {"real": 10, "gold": 2, "embedding": 0,}
    columnTypes = defaultdict(int)

    columns = []
    for feature in features:
        if feature.startswith("gold_adv_"):
            columnTypes["gold"] += 1

            realColumn = tf.contrib.layers.real_valued_column(feature)

            # TODO can I assert this value is positive?

            column = tf.contrib.layers.bucketized_column(
                source_column=realColumn,
                boundaries= [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000])

        elif feature.startswith("embedding_"):
            columnTypes["embedding"] += 1
            assert feature.endswith("champion"), "\"{}\" requires setup for embedding".format(feature)

            sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
                feature,
                bucket_size = 150)

            # shared_columns = tf.contrib.layers.shared_embedding_columns(
            #     [sparse_column],
            #     shared_embedding_name="champion_embedding",
            #     dimension = 10,
            #     combiner="mean")
            # print (shared_columns)
            #
            # column = shared_columns[0]
            column = tf.contrib.layers.embedding_column(
                sparse_column,
                dimension = 20,
                combiner = "mean")
        else:
            columnTypes["real"] += 1
            column = tf.contrib.layers.real_valued_column(feature)

        columns.append(column)

    for type, count in requiredToBeFound.items():
        assert columnTypes[type] >= count, "{} had {} not >= {}".format(type, columnTypes[type], count)

    return columns

featurizerTime = 0
trainTime = 0
def gameToFeatures(args, games, goals, blockNum, *, training = False):
    global featurizerTime, trainTime

    # Filter out games that ended already
    indexes, games, goals = filterMaxBlock(blockNum, games, goals)

    T0 = time.time()

    gameFeatureSets = []
    for index, game in enumerate(games):
        #if training:
        #  gameTime = game['debug']['duration']
        #else:
        gameTime = blockNum * Util.SECONDS_PER_BLOCK

        gameData = TFFeaturize.parseGame(game, gameTime)
        gameFeatureSets.append(gameData)

    T1 = time.time()
    featurizerTime += T1 - T0

    if training and args.verbose >= 1:
        print ("\t\tFeaturize Timing: {:.1f}".format(T1 - T0))

    if training:
        allColumns = set()
        for featureSet in gameFeatureSets:
            allColumns.update(featureSet.keys())
        return gameFeatureSets, goals, allColumns

    return gameFeatureSets, goals

def featureNameToType(feature):
    if feature.startswith("embedding_"):
        return tf.int32
    return tf.float16


def inputFn(featuresUsed, data, goals = None):
    featureCols = {
        feature: tf.constant(
            [d.get(feature, 0) for d in data],
            shape = [len(data), 1],
            dtype = featureNameToType(feature),
        )
        for feature in featuresUsed
    }

    if goals is None:
        return featureCols
    labels = tf.constant(goals, shape=[len(goals), 1], dtype='float16')
    return featureCols, labels

def learningRateFn(params):
    learningRate = tf.train.exponential_decay(
        learning_rate = params['learningRate'],
        global_step = tf.contrib.framework.get_or_create_global_step(),
        decay_steps = 1000,
        decay_rate = .70,
        staircase = True)
    learningRate = params['learningRate']

#    tf.summary.scalar("learning_rate/learning_rate", learningRate)

#    assert 0.000001 <= learningRate <= .001, "stuff .0001 seems fairly reasonable"
#    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)

    assert 0.0001 <= learningRate < 0.3, "Fails to learn anything (or converge quickly) outside this range"
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate = learningRate,
#        l1_regularization_strength = params['regularization'],
        l2_regularization_strength = params['regularization'],
    )

#    assert 0.001 <= learningRate < 0.3, "Fails to learn anything (or converge quickly) outside this range"
#    optimizer = tf.train.ProximalGradientDescentOptimizer(
#        learning_rate = learningRate,
#        l1_regularization_strength = params['regularization'],
#        l2_regularization_strength = params['regularization'],
#    )

    return optimizer


def buildClassifier(args, blocks, trainGames, trainGoals, testGames, testGoals):
    global featurizerTime, trainTime

    # Over eighty briefly with
    # ('dropout', 0.0), ('learningRate', 0.02), ('steps', 100000), ('hiddenUnits', [600, 800, 400, 300, 20]), ('regularization', 0.01)

#    constParams = {
#        'modelName': 'exploring',
#        'dropout': 0.00,
#        'regularization': 0.01,
#        'learningRate': 0.01,
#        'hiddenUnits': [400, 500, 300, 200, 20],
#        'earlyStoppingRounds': 2000,
#        'steps': 100000,
#    }

    constParams = {
        'modelName': 'exploring',
        'dropout': 0.00,
        'regularization': 0.01,
        'learningRate': 0.01,
        'hiddenUnits': [10, 10],
#        'earlyStoppingRounds': 2000,
        'steps': 200000,
    }


    gridSearchParams = [
#        ('dropout', [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
#        ('regularization', [0.0001, 0.001, 0.01, 0.1, 1.0]),
#        ('learningRate', [0.005, 0.007, 0.01, 0.013, 0.017]),
    ]

    classifiers = {}
    featuresUses = {}

    for blockNum in blocks:
        bestOfGridSearch = (1000, None, None)

        for gridSearchValues in itertools.product(*[values for name, values in gridSearchParams]):
            gridSearchInstanceParams = dict(zip([name for name, v in gridSearchParams], gridSearchValues))
            duplicateKeys = constParams.keys() & gridSearchInstanceParams.keys()
            assert len(duplicateKeys) == 0, duplicateKeys
            params = {**constParams, **gridSearchInstanceParams}

            blockTrainFeatureSets, blockTrainGoals, featuresUsed = gameToFeatures(
                args, trainGames, trainGoals, blockNum, training = True)

            blockTestFeatureSets, blockTestGoals = gameToFeatures(
                args, testGames, testGoals, blockNum, training = False)

            if len(blockTrainFeatureSets) == 0:
                break

            if args.verbose >= 1:
                print ("\ttraining block {} on {} games".format(blockNum, len(blockTrainFeatureSets)))

            if args.verbose >= 1:
                numberOfCompressedFeatures, compressedPretty = \
                    Util.compressFeatureList(featuresUsed)
                if numberOfCompressedFeatures < 40 or args.verbose >= 2:
                    print ("\t{} features: {}".format(
                        numberOfCompressedFeatures, compressedPretty))

            featureColumns = featuresToColumns(featuresUsed)

            T0 = time.time()

            runTime = datetime.datetime.now().strftime("%m_%d_%H_%M")
            gridSearchName = "".join("-{}={}".format(name, value) for name, value in
                                     sorted(gridSearchInstanceParams.items()))
            modelName = params['modelName'] + gridSearchName
            modelDir = "/tmp/tmp-tf-lol/exploring/{}/b{}/model_{}".format(runTime, blockNum, modelName)
            print ("Saving in", modelDir)
            print ("\t", params.items(), "\n")

            classifier = tf.contrib.learn.DNNClassifier(
    #        classifier = DnnClassifier.DNNClassifier( DO NOT SUBMIT
                hidden_units = params['hiddenUnits'],
                feature_columns = featureColumns,
                model_dir = modelDir,
                n_classes = 2,
                dropout = params['dropout'],
                optimizer = functools.partial(learningRateFn, params),
                config = tf.contrib.learn.RunConfig(
                    save_summary_steps = 200,
                    save_checkpoints_steps = 500,
                ),
            )

    #        validationMetrics = {
    #            "accurary": tf.contrib.metrics.streaming_accuracy,
    #            "auc": tf.contrib.metrics.streaming_auc,
    #        }
            validationMonitor = tf.contrib.learn.monitors.ValidationMonitor(
                input_fn = functools.partial(
                    inputFn, featuresUsed, blockTestFeatureSets, blockTestGoals),
                eval_steps = 1,
                every_n_steps = 1,
    #            metrics = validationMetrics,
                early_stopping_metric = "loss",
                early_stopping_rounds = params.get('earlyStoppingRounds', params['steps']),
                name = "validation_mn",
            )

            if args.verbose >= 2:
                print ()
                print ()

            classifier.fit(
                input_fn = functools.partial(
                    inputFn, featuresUsed, blockTrainFeatureSets, blockTrainGoals),
                monitors = [validationMonitor],
                steps = params['steps'],
            )

    #        classifier.evaluate(
    #            input_fn = functools.partial(
    #                inputFn, featuresUsed, blockTestFeatureSets, blockTestGoals),
    #            steps = 1,
    #            name="eval_at_the_end_of_time",
    #        )

            if args.verbose >= 1:
                for v in range(args.verbose):
                    print ()
                    print ()

            T1 = time.time()
            trainTime += T1 - T0

            # Determine the best of the grid search
            loss = 10
            if loss < bestOfGridSearch[0]:
                bestOfGridSearch = (loss, classifier, featuresUsed)

        # Best of the grid search
        loss, classifier, featuresUsed = bestOfGridSearch

        classifiers[blockNum] = classifier
        featuresUses[blockNum] = featuresUsed
    return classifiers, featuresUses


#allprobs = set()
#featureGroups = set()
def getPrediction(args, classifiers, featuresUses, testGames, testGoals, blockNum):
    assert blockNum in classifiers and blockNum in featuresUses
    classifier = classifiers[blockNum]
    featuresUsed = featuresUses[blockNum]

    predictFeatureSets, predictGoalsEmpty = gameToFeatures(
        args, testGames, testGoals, blockNum)

    modelGuess = classifier.predict_proba(
        input_fn = functools.partial(inputFn, featuresUsed, predictFeatureSets))

    for i, (probs, testGoal) in enumerate(zip(modelGuess, testGoals)):
    #     key = tuple(sorted(predictFeatureSets[i].keys()))
    #     if key not in featureGroups or probs[0] not in allprobs:
    #         tfFeatures = inputFn(featuresUsed, [predictFeatureSets[i]])
    #
    #         print ("\t", probs, testGoal, i, "\t", key,
    #                "\t", predictFeatureSets[i])
    #         featureGroups.add(key)
    #         allprobs.add(probs[0])
    #

        # This is due to the sorting of [False, True].
        BProb, AProb = probs
        correct = (AProb > 0.5) == testGoal
        yield correct, [BProb, AProb]


def main(args):
    global featurizerTime, trainTime

    if args.verbose == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose == 1:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose >= 2:
        tf.logging.set_verbosity(tf.logging.INFO)


    T0 = time.time()

    blocks = processBlocks(args)
    games, goals = TFFeaturize.getRawGameData(args)

    T1 = time.time()
    loadTime = T1 - T0

    trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
        games,
        goals,
        test_size = args.holdback / 100,
        random_state = 42)
    del games, goals

    if args.verbose == 0:
        print ("Training games: {}, Testing holdback: {}".format(
            len(trainingGames), len(testingGames)))
    assert len(trainingGames) == len(trainingGoals)
    assert len(testingGames) == len(testingGoals)



    # testFeatures = ['team_A_has_champion_11', 'team_B_has_champion_11']
    # if testFeatures:
    #     featureRecall = defaultdict(lambda : [0,0])
    #     for game, goal in zip(trainingGames, trainingGoals):
    #         features = TFFeaturize.parseGame(game, 3600)
    #         featureRecall["bias"][goal] += 1
    #         for testFeature in testFeatures:
    #             featureRecall[testFeature][goal] += 1
    #
    #
    #     for feature, results in sorted(featureRecall.items()):
    #         print ("\t{} - {} - {:.1f}%".format(feature, results, 100 * results[1] / sum(results)))
    #     print ()

    T2 = time.time()

    classifiers, featuresUses = buildClassifier(
        args,
        blocks,
        trainingGames,
        trainingGoals,
        testingGames,
        testingGoals
    )

    T3 = time.time()
    innerTrainTime = T3 - T2

    maxBlock = max(blocks)

    # Variables about testGames.
    times = [(b * Util.SECONDS_PER_BLOCK) / 60 for b in range(maxBlock + 1)]
    samples = [0 for b in range(maxBlock + 1)]
    corrects = [0 for b in range(maxBlock + 1)]

    # Averages over data (calculated after all data).
    ratios = [0 for b in range(maxBlock + 1)]
    logLosses = [0 for b in range(maxBlock + 1)]

    # Per Game stats.
    testWinProbs = [[] for game in testingGames]

    for blockNum in blocks:
        gameIs, testingBlockGames, testingBlockGoals = filterMaxBlock(
            blockNum, testingGames, testingGoals)

        # Do all predictions at the same time (needed because of how predict reloads the model each time).
        predictions = getPrediction(args, classifiers, featuresUses, testingBlockGames, testingBlockGoals, blockNum)

        for gameI, prediction in zip(gameIs, predictions):
            correct, gamePrediction = prediction

            # store data to graph
            samples[blockNum] += 1
            corrects[blockNum] += 1 if correct else 0

            paddingNeeded = blockNum - len(testWinProbs[gameI]) + 1
            if paddingNeeded > 0:
                testWinProbs[gameI] += [[0.5, 0.5]] * paddingNeeded
            testWinProbs[gameI][blockNum] = gamePrediction

    for blockNum in blocks:
        if samples[blockNum] > 20:
            ratios[blockNum] = corrects[blockNum] / samples[blockNum]

            goals = []
            predictions = []
            for testGoal, gamePredictions in zip(testingGoals, testWinProbs):
                if len(gamePredictions) > blockNum:
                    goals.append(testGoal)
                    predictions.append(gamePredictions[blockNum])

            logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions, labels = [True, False])

    T4 = time.time()
    statsTime = T4 - T3

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
        GraphModelStats.stats(blocks, times, samples, corrects, ratios, logLosses)
        GraphModelStats.plotData(blocks, times, samples, corrects, ratios, logLosses)
        GraphModelStats.plotGame(max(blocks), times, testingGoals, testWinProbs)

    T5 = time.time()
    viewTime = T5 - T4

    print ("Timings:")
    print ("\tloadTime: {:.3f}".format(loadTime))
    print ("\ttrainTime: {:.3f}".format(innerTrainTime))
    print ("\tstatsTime: {:.3f}".format(statsTime))
    print ("\tviewTime: {:.3f}".format(viewTime))
    print ()
    print ("\tfeaturizerTime: {:.3f}".format(featurizerTime))
    print ("\ttrainTime: {:.3f}".format(trainTime))

if __name__ == '__main__':
    args = getArgParse().parse_args()
    main(args)

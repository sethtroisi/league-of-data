import argparse
import datetime
import itertools
import functools
import sklearn.metrics
import time
import tensorflow as tf

from collections import defaultdict
from sklearn.model_selection import train_test_split

import graph_model_stats
import tf_featurize
import util
import my_classifier

def getArgParse():
    parser = argparse.ArgumentParser(description='Takes features and models outcomes.')

    parser.add_argument(
        '-v', '--verbose',
        type=int,
        default=0,
        help='how much data to print (0 = little, 1 = most, 3 = everything)')

    parser.add_argument(
        '-i', '--input-file',
        type=str,
        default='features.json',
        help='Input match file (produced by game_parser.py)')

    parser.add_argument(
        '--model-dir',
        type=str,
        default='/tmp/tmp-tf-lol/',
        help='Model directory base folder')

    parser.add_argument(
        '--panda-debug',
        action="store_true",
        help='drop to interactive prompt with pandas to debug data')

    parser.add_argument(
        '-n', '--num-games',
        type=int,
        default=-1,
        help='Numbers of games to load (default -1 = all)')

    parser.add_argument(
        '-p', '--holdback',
        type=int,
        default=15,
        help='percent of games to holdback for testing/validation')

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
        '--filter-weird-games',
        action="store_true",
        help='Filter out games which are non-standard (lane swap, surrenders, ...)')

    return parser


def processBlocks(args):
    lowBlock = 0
    highBlock = int(3600 // util.SECONDS_PER_BLOCK)

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
    blockStart = blockNum * util.SECONDS_PER_BLOCK

    indexes = []
    usableGames = []
    usableGoals = []
    for i, (game, goal) in enumerate(zip(games, goals)):
        if blockStart > game['debug']['duration']:
            continue

        indexes.append(i)
        usableGames.append(game)
        usableGoals.append(goal)

    return indexes, usableGames, usableGoals


def featuresToColumns(features):
    requiredToBeFound = {"real": 10, "gold": 2, "embedding": 0, }
    columnTypes = defaultdict(int)

    columns = []
    for feature in features:
        if feature.startswith("embedding_"):
            columnTypes["embedding"] += 1
            assert feature.endswith("champion"), "\"{}\" requires setup for embedding".format(feature)

            sparse_column = tf.contrib.layers.sparse_column_with_integerized_feature(
                feature,
                bucket_size=150)

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
                dimension=10,
                combiner="mean")
        else:
            columnTypes["real"] += 1
            column = tf.contrib.layers.real_valued_column(feature)

        columns.append(column)

    #for name, count in requiredToBeFound.items():
    #    assert columnTypes[name] >= count, "{} had {} not >= {}".format(name, columnTypes[name], count)

    return columns


featurizeTime = 0
trainTime = 0


def gameToFeatures(args, games, goals, blockNum, *, training):
    global featurizeTime, trainTime

    # Filter out games that ended already
    indexes, games, goals = filterMaxBlock(blockNum, games, goals)

    T0 = time.time()

    gameFeatureSets = []
    for index, game in enumerate(games):
        gameTime = blockNum * util.SECONDS_PER_BLOCK

        gameData = tf_featurize.parseGame(game, gameTime)
        gameFeatureSets.append(gameData)

    T1 = time.time()
    featurizeTime += T1 - T0

    if training and args.verbose >= 1:
        print("\t\tFeaturize Timing: {:.1f}".format(T1 - T0))

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


def inputFn(featuresUsed, data, goals=None):
    featureCols = {
        feature: tf.constant(
            [d.get(feature, 0) for d in data],
            shape=[len(data), 1],
            dtype=featureNameToType(feature),
        )
        for feature in featuresUsed
    }

    if goals is None:
        return featureCols

    labels = tf.constant(goals, shape=[len(goals), 1], dtype='float16')
    return featureCols, labels


def optimizerFn(params):
    learningRate = tf.train.exponential_decay(
        learning_rate=params['learningRate'],
        global_step=tf.contrib.framework.get_or_create_global_step(),
        decay_steps=1000,
        decay_rate=.9,
        staircase=True)
    #    learningRate = params['learningRate']

    tf.summary.scalar("learning_rate/learning_rate", learningRate)

    '''
    assert 0.000001 <= learningRate <= .001, "stuff .0001 seems fairly reasonable"
    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
    # '''

    # assert 0.0001 <= learningRate < 0.3, "Fails to learn anything (or converge quickly) outside this range"
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=learningRate,
        l1_regularization_strength=params['l1_regularization'],
        l2_regularization_strength = params['l2_regularization'],
    )

    '''
    assert 0.001 <= learningRate < 0.3, "Fails to learn anything (or converge quickly) outside this range"
    optimizer = tf.train.ProximalGradientDescentOptimizer(
        learning_rate = learningRate,
        l1_regularization_strength = params['l1_regularization'],
        l2_regularization_strength = params['l2_regularization'],
    )
    # '''

    return optimizer

def buildClassifier(args, blocks, trainGames, trainGoals, testGames, testGoals):
    global featurizeTime, trainTime

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    '''
     Over eighty briefly with
     ('dropout', 0.0),
     ('learningRate', 0.02),
     ('steps', 100000),
     ('hiddenUnits', [600, 800, 400, 300, 20]),
     ('regularization', 0.01)
    '''

    constParams = {
        'modelName': 'exploring',

        # ML hyperparams
        'learningRate': 0.001,
        'dropout': 0.00,
        'l1_regularization': 0.00001,
        'l2_regularization': 0.00003,
        'hiddenUnits': [50, 50, 50, 50, 50, 50],
        'steps': 6100,

        # Also controls how often eval_validation data is calculated
        'saveCheckpointSteps': 1000,
        'earlyStoppingRounds': 2000,
    }

    gridSearchParams = [
        # ('dropout', [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
#        ('l1_regularization', [0.001, 0.0001, 0.00001]),
#        ('l2_regularization', [0.01, 0.001, 0.0001]),
#        ('learningRate', [0.02, 0.01, 0.005]),
    ]

    classifiers = {}
    featureSets = {}

    for blockNum in blocks:
        bestOfGridSearch = (1000, None, None)

        for gridSearchValues in itertools.product(*[values for name, values in gridSearchParams]):
            gridSearchInstanceParams = dict(zip([name for name, v in gridSearchParams], gridSearchValues))
            duplicateKeys = constParams.keys() & gridSearchInstanceParams.keys()
            assert len(duplicateKeys) == 0, duplicateKeys
            params = {**constParams, **gridSearchInstanceParams}

            blockTrainFeatureSets, blockTrainGoals, featuresUsed = gameToFeatures(
                args, trainGames, trainGoals, blockNum, training=True)

            blockTestFeatureSets, blockTestGoals = gameToFeatures(
                args, testGames, testGoals, blockNum, training=False)

            if len(blockTrainFeatureSets) == 0:
                break

            if args.verbose >= 1:
                print("\ttraining block {} on {} games".format(blockNum, len(blockTrainFeatureSets)))
                print()

            if args.verbose >= 1:
                countCompressed, compressedPretty = util.compressFeatureList(featuresUsed)
                if countCompressed < 40 or args.verbose >= 2:
                    print("\t{} features, {} compressed: {}".format(
                        len(featuresUsed), countCompressed, compressedPretty))
                    print ()
            featureColumns = featuresToColumns(featuresUsed)

            T0 = time.time()

            runTime = datetime.datetime.now().strftime("%m_%d_%H_%M")
            gridSearchName = "".join("-{}={}".format(name, value) for name, value in
                                     sorted(gridSearchInstanceParams.items()))
            modelName = params['modelName'] + gridSearchName
            modelDir = args.model_dir + "/{}/b{}/model_{}".format(runTime, blockNum, modelName)
            print("Saving in", modelDir)
            for hyperparam in params.items():
                print("\t", hyperparam)
            print()

            # classifier = tf.contrib.learn.DNNClassifier(
            #     hidden_units=params['hiddenUnits'],
            #     feature_columns=featureColumns,
            #     model_dir=modelDir,
            #     n_classes=2,
            #     dropout=params['dropout'],
            #     optimizer=functools.partial(optimizerFn, params),
            #     config=tf.contrib.learn.RunConfig(
            #         save_summary_steps=200,
            #         save_checkpoints_steps=params['saveCheckpointSteps'],
            #     ),
            # )

            classifier = my_classifier.MyClassifier(
                hidden_units=params['hiddenUnits'],
                feature_columns=featureColumns,
                model_dir=modelDir,
                n_classes=2,
                dropout=params['dropout'],
                optimizer=functools.partial(optimizerFn, params),
                config=tf.contrib.learn.RunConfig(
                    save_summary_steps=200,
                    save_checkpoints_steps=params['saveCheckpointSteps'],
                ),
            )

            validationMonitor = tf.contrib.learn.monitors.ValidationMonitor(
                input_fn=functools.partial(
                    inputFn, featuresUsed, blockTestFeatureSets, blockTestGoals),
                eval_steps=1,
                every_n_steps=1,
                early_stopping_metric="loss",
                early_stopping_rounds=params.get('earlyStoppingRounds', params['steps']),
                name="validation_mn",
            )

            if args.verbose >= 2:
                print()
                print()

            classifier.fit(
                input_fn=functools.partial(
                    inputFn, featuresUsed, blockTrainFeatureSets, blockTrainGoals),
                monitors=[validationMonitor],
                steps=params['steps'],
            )

            if args.verbose >= 1:
                for v in range(args.verbose):
                    print()
                    print()

            T1 = time.time()
            trainTime += T1 - T0

            # Determine the best of the grid search
            loss = 10
            if loss < bestOfGridSearch[0]:
                bestOfGridSearch = (loss, classifier, featuresUsed)

        # Best of the grid search
        loss, classifier, featuresUsed = bestOfGridSearch

        classifiers[blockNum] = classifier
        featureSets[blockNum] = featuresUsed
    return classifiers, featureSets


def getPrediction(args, classifiers, featureSets, testGames, testGoals, blockNum):
    assert blockNum in classifiers and blockNum in featureSets
    classifier = classifiers[blockNum]
    featuresUsed = featureSets[blockNum]

    predictFeatureSets, predictGoalsEmpty = gameToFeatures(
        args, testGames, testGoals, blockNum, training=False)

    modelGuess = classifier.predict_proba(
        input_fn=functools.partial(inputFn, featuresUsed, predictFeatureSets))

    for i, (probs, testGoal) in enumerate(zip(modelGuess, testGoals)):
        # This is due to the sorting of [False, True].
        BProb, AProb = probs
        correct = (AProb > 0.5) == testGoal
        yield correct, [BProb, AProb]


def buildGraphData(args, blocks, testingGames, testingGoals, classifiers, featureSets):
    maxBlock = max(blocks)

    # Variables about testGames.
    times = [(b * util.SECONDS_PER_BLOCK) / 60 for b in range(maxBlock + 1)]
    samples = [0] * (maxBlock + 1)
    corrects = [0] * (maxBlock + 1)

    # Averages over data (calculated after all data).
    ratios = [0] * (maxBlock + 1)
    logLosses = [0] * (maxBlock + 1)

    # Per Game stats.
    testWinProbs = [[] for _ in testingGames]

    for blockNum in blocks:
        gameIs, testingBlockGames, testingBlockGoals = filterMaxBlock(
            blockNum, testingGames, testingGoals)

        # Do all predictions at the same time (needed because of how predict reloads the model each time).
        predictions = getPrediction(args, classifiers, featureSets, testingBlockGames, testingBlockGoals, blockNum)

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

            logLosses[blockNum] = sklearn.metrics.log_loss(goals, predictions, labels=[True, False])

    return times, samples, corrects, ratios, logLosses, testWinProbs


def main(args):
    global featurizeTime, trainTime

    if args.verbose == 0:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose == 1:
        tf.logging.set_verbosity(tf.logging.ERROR)
    elif args.verbose >= 2:
        tf.logging.set_verbosity(tf.logging.INFO)

    T0 = time.time()

    blocks = processBlocks(args)
    games, goals = tf_featurize.getRawGameData(args)

    T1 = time.time()
    loadTime = T1 - T0

    trainingGames, testingGames, trainingGoals, testingGoals = train_test_split(
        games,
        goals,
        test_size=args.holdback / 100,
        random_state=42)
    del games, goals

    if args.verbose == 0:
        print("Training games: {}, Testing holdback: {}".format(
            len(trainingGames), len(testingGames)))
    assert len(trainingGames) == len(trainingGoals)
    assert len(testingGames) == len(testingGoals)

    if args.panda_debug:
        for blockNum in blocks:
            blockTrainFeatureSets, blockTrainGoals, featuresUsed = gameToFeatures(
                args, trainingGames, trainingGoals, blockNum, training=True)

            countCompressed, compressedPretty = util.compressFeatureList(featuresUsed)
            print("\t{} features, {} compressed: {}".format(
                    len(featuresUsed), countCompressed, compressedPretty))
            print()

            print("Panda Debugging block", blockNum)
            print("\tdata loading into \"df\"")
            print("\t.columns .describe() .fillna(0) are common")
            print()
            print()

            '''
            pd.set_option('display.max_columns', None)
            df.columns
            df.describe()
            Counter(df['team_spells_B_teleports'].fillna(0).tolist())
            '''

            import pandas as pd
            df = pd.DataFrame(blockTrainFeatureSets)
            print (df.columns)

            import IPython
            IPython.embed()

            return

    T2 = time.time()

    classifiers, featureSets = buildClassifier(
        args,
        blocks,
        trainingGames,
        trainingGoals,
        testingGames,
        testingGoals
    )

    T3 = time.time()
    innerTrainTime = T3 - T2

    times, samples, corrects, ratios, logLosses, testWinProbs = \
        buildGraphData(args, blocks, testingGames, testingGoals, classifiers, featureSets)

    T4 = time.time()
    statsTime = T4 - T3

    # If data was tabulated on the testingData print stats about it.
    if len(times) > 0:
        graph_model_stats.stats(blocks, samples, corrects, ratios, logLosses)
        graph_model_stats.plotData(blocks, times, samples, corrects, ratios, logLosses)
        graph_model_stats.plotGame(blocks, times, samples, corrects, ratios, logLosses, testingGoals, testWinProbs)

    T5 = time.time()
    viewTime = T5 - T4

    print("Timings:")
    print("\tloadTime: {:.3f}".format(loadTime))
    print("\ttrainTime: {:.3f}".format(innerTrainTime))
    print("\tstatsTime: {:.3f}".format(statsTime))
    print("\tviewTime: {:.3f}".format(viewTime))
    print()
    print("\tfeaturizeTime: {:.3f}".format(featurizeTime))
    print("\ttrainTime: {:.3f}".format(trainTime))


if __name__ == '__main__':
    parsedArgs = getArgParse().parse_args()
    main(parsedArgs)

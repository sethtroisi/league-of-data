import argparse
import functools
import re
import sklearn.metrics
import time
import datetime

from sklearn.model_selection import train_test_split
import GraphModelStats
import TFFeaturize
import Util
import DnnClassifier


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
    columns = []
    for feature in features:
        if not feature.startswith("embedding_"):
            column = tf.contrib.layers.real_valued_column(feature)
        else:
            column = tf.contrib.layers.embedding_column(
                tf.contrib.layers.sparse_column_with_integerized_feature(
                    feature,
                    bucket_size = 150),
                dimension = 4, # * 10 for every champion
                combiner = "mean")
        columns.append(column)
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
        decay_rate = .95,
        staircase = True)

    tf.summary.scalar("learning_rate/learning_rate", learningRate)

    optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)

#    optimizer = tf.train.ProximalAdagradOptimizer(
#        learning_rate = learningRate,
#        l1_regularization_strength = params['regularization'],
#        l2_regularization_strength = params['regularization'],
#    )
    return optimizer


def buildClassifier(args, blocks, trainGames, trainGoals, testGames, testGoals):
    global featurizerTime, trainTime

    params = {
        'modelName': 'exploring',
        'dropout': 0.9,
        'regularization': 0.0003,
        'learningRate': 0.001,
        'hiddenUnits': [150, 100, 5],
        'earlyStoppingRounds': 5000,
        'steps': 30000,
    }

    classifiers = {}
    featuresUses = {}

    for blockNum in blocks:
        blockTrainFeatureSets, blockTrainGoals, featuresUsed = gameToFeatures(
            args, trainGames, trainGoals, blockNum, training = True)

        blockTestFeatureSets, blockTestGoals = gameToFeatures(
            args, testGames, testGoals, blockNum, training = False)


        if len(blockTrainFeatureSets) == 0:
            break

        if args.verbose >= 1:
            print ("\ttraining block {} on {} games".format(blockNum, len(blockTrainFeatureSets)))


        if args.verbose >= 1:
            regex = re.compile('_([ABb0-9]{1,2})(?=$|_)', re.I)
            def replacement(m):
                text = m.group(1)
                if text in 'abAB':
                    return '_<team>'
                elif text.isnumeric():
                    return '_<X>'
                else:
                    print ("Bad sub:", text, m.groups())
                    return m.group()

            compressedList = \
                set(map(functools.partial(regex.sub, replacement), featuresUsed))

            compressedFeatures = ""
            if len(compressedList) < 40 or args.verbose >= 2:
                compressedFeatures = ", ".join(sorted(compressedList))

            print ("\t{} features: {}".format(
                len(featuresUsed), compressedFeatures))

        featureColumns = featuresToColumns(featuresUsed)

        T0 = time.time()

        runName = datetime.datetime.now().strftime("%m_%d_%H_%M")
        modelDir = "/tmp/tmp-tf-lol/exploring/{}-{}/model".format(runName, blockNum)
        print ("Saving in", modelDir)

#        classifier = tf.contrib.learn.DNNClassifier(
        classifier = DnnClassifier.DNNClassifier(
            hidden_units = params['hiddenUnits'],
            feature_columns = featureColumns,
            model_dir = modelDir,
            n_classes = 2,
            dropout = params['dropout'],
            gradient_clip_norm = 0.5,
            optimizer = functools.partial(learningRateFn, params),
            config = tf.contrib.learn.RunConfig(
                save_checkpoints_steps = 99,
                save_checkpoints_secs = None
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
            every_n_steps = 100,
#            metrics = validationMetrics,
            early_stopping_metric = "loss",
            early_stopping_rounds = params['earlyStoppingRounds'],
            name = "validation_monitor",
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

        classifier.evaluate(
            input_fn = functools.partial(
                inputFn, featuresUsed, blockTestFeatureSets, blockTestGoals),
            steps = 1,
            name="eval_at_the_end_of_time",
        )

        if args.verbose >= 1:
            for v in range(args.verbose):
                print ()
                print ()

        T1 = time.time()
        trainTime += T1 - T0

        classifiers[blockNum] = classifier
        featuresUses[blockNum] = featuresUsed
    return classifiers, featuresUses


def getPrediction(args, classifiers, featuresUses, testGames, testGoals, blockNum):
    assert blockNum in classifiers and blockNum in featuresUses
    classifier = classifiers[blockNum]
    featuresUsed = featuresUses[blockNum]

    predictFeatureSets, predictGoalsEmpty = gameToFeatures(
        args, testGames, testGoals, blockNum)

    modelGuess = classifier.predict_proba(
        input_fn = functools.partial(inputFn, featuresUsed, predictFeatureSets))

    for probs, testGoal in zip(modelGuess, testGoals):

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
    testWinProbs = [
        [[0.5, 0.5] for block in range(min(maxBlock, Util.timeToBlock(game['debug']['duration'])) + 1)]
        for game in testingGames
    ]

    # TODO try setting samples for all blocks (if I can)

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

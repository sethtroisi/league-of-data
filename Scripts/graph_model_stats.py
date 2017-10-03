import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.widgets import Slider
import random

import util


# Plot general data about accuracy, logloss, number of samples.
def plotData(blocks, times, samples, corrects, ratios, logLosses):
    fig, (axis1, axis2, axis3) = pyplot.subplots(3, 1)
    fig.subplots_adjust(hspace=0.6)

    # Common styling 'Patch' for text
    props = dict(boxstyle='round', facecolor='#abcdef', alpha=0.5)

    # Upper graph of prediction power.
    axis1.plot(times, ratios)
    axis1.set_title('Correct Predictions')
    axis1.set_xlabel('time (m)')
    axis1.set_ylabel('correctness')

    maxSamples = max(samples)
    minBlock = min(blocks)
    maxBlock = min(blocks)
    for block in sorted(blocks):
        numGames = samples[block]
        if numGames > 50 and numGames * 4 > maxSamples:
            maxBlock = block

    bestAccuracy = max(ratios[minBlock:maxBlock + 1])
    time = times[ratios.index(bestAccuracy)]
    accuracyText = '{:.3f} (@{:2.0f}m)'.format(bestAccuracy, time)
    axis3.text(
        time / max(times), 0.1,
        accuracyText, transform=axis1.transAxes, fontsize=14,
        bbox=props,
        verticalalignment='bottom', horizontalalignment='center')

    # Middle graph of log loss.
    maxLogLoss = max(1.4, min(10, 1.2 * max(logLosses)))

    axis2.plot(times, logLosses)
    axis2.set_title('Log Loss')
    axis2.set_xlabel('time (m)')
    axis2.set_ylabel('loss (log)')
    axis2.set_ylim([0, maxLogLoss])

    minLogLoss = min(logLosses[minBlock:maxBlock + 1])
    time = times[logLosses.index(minLogLoss)]
    logLossText = '{:.3f} (@{:2.0f}m)'.format(minLogLoss, time)
    axis2.text(
        time / max(times), 0.7,
        logLossText, transform=axis2.transAxes, fontsize=14,
        bbox=props,
        verticalalignment='bottom', horizontalalignment='center')

    # Lower graph of sample data.
    incorrects = [s - c for s, c in zip(samples, corrects)]

    axis3.plot(times, samples, 'b',
               times, corrects, 'g',
               times, incorrects, 'r')
    axis3.set_title('Number of samples')
    axis3.set_xlabel('time (m)')
    axis3.set_ylabel('samples')

    pyplot.show()


# Plot game predictions vs time.
def plotGame(times, results, winPredictions):
    fig, (axis1, axis2) = pyplot.subplots(2, 1)
    axis2_2 = axis2.twinx()

    fig.subplots_adjust(hspace=0.65)

    # Note: I didn't have luck with subplots(3, 1) and resizing so I used this.
    sliderAxis = pyplot.axes(
        [0.125, 0.48, 0.775, 0.03],
        axisbg='lightgoldenrodyellow')

    resultColors = {
        True: 'g',
        False: 'r'
    }

    sample = zip(results, winPredictions)
    if len(results) > 5000:
        sample = random.sample(list(sample), 5000)

    # For every game (if less than 5000 games) print prediction through out the game.
    for result, gamePredictions in sample:
        # TODO Set alpha dependant on number of games
        color = resultColors[result]
        axis1.plot(times[:len(gamePredictions)], [gP[1] for gP in gamePredictions], color, alpha=0.05)

    axis1.set_title('Predictions of win rate across the game')
    axis1.set_xlabel('time (m)')
    axis1.set_ylabel('prediction confidence')

    # At X minutes print confidences.
    # TODO set val_init to first block with interesting data.
    sliderTime = Slider(sliderAxis, 'Time', 0, 60, valinit=20, valfmt='%d')
    sliderTime.valtext.set_visible(False)
    # TODO show value, # of games, % of games at this point in a text box somewhere

    percentBuckets = 101
    percents = [p / (percentBuckets - 1) for p in range(percentBuckets)]

    def plotConfidentAtTime(requestedTime):
        ti = min([(abs(requestedTime - t), i) for i, t in enumerate(times)])[1]

        cdfTrue = [0] * len(percents)
        cdfFalse = [0] * len(percents)
        pdfTrue = [0] * len(percents)
        pdfFalse = [0] * len(percents)

        # TODO are end buckets being filled?
        for gameResult, predictions in zip(results, winPredictions):
            if len(predictions) <= ti:
                continue

            prediction = predictions[ti][1]
            for pi, percent in enumerate(percents):
                if percent >= prediction:
                    break
                if gameResult:
                    cdfTrue[pi] += 1
                else:
                    cdfFalse[pi] += 1

            bucket = int(round((percentBuckets - 1) * prediction))
            if gameResult:
                pdfTrue[bucket] += 1
            else:
                # ~ is a fun trick to get the negative index (0 => -1, 1 => -2, ...) of an item
                pdfFalse[~bucket] += 1

        axis2.cla()
        axis2_2.cla()

        axis2.plot(percents, cdfTrue, color=resultColors[True], alpha=0.9)
        axis2.plot(percents, cdfFalse, color=resultColors[False], alpha=0.9)

        axis2_2.bar(percents, pdfTrue, width=0.008, color=resultColors[True], alpha=0.5)
        axis2_2.bar(percents, pdfFalse, width=0.008, color=resultColors[False], alpha=0.5)

        axis2.set_xlabel('confidence')
        axis2.set_ylabel('count of games (cdf)')
        axis2_2.set_ylabel('count of games (pdf)')

        # 1.01 need to avoid cutting off largest bucket.
        axis2.set_xlim([0, 1.01])
        axis2_2.set_xlim([0, 1.01])

        # Draw the vertical line on the upper time graph
        for line in axis1.lines:
            if line.get_gid() == 'vline':
                line.remove()
        # TODO line kinda disappears if in first or last index :(
        axis1.axvline(x=times[ti], linewidth=4, color='b', gid='vline')

        fig.canvas.draw_idle()

    plotConfidentAtTime(20)
    sliderTime.on_changed(plotConfidentAtTime)

    pyplot.show()


def stats(blocks, samples, corrects, ratios, logLosses):
    startBlock = util.timeToBlock(10 * 60)
    endBlock = util.timeToBlock(30 * 60)

    startBlock = max(startBlock, min(blocks))
    endBlock = min(endBlock, max(blocks))

    # TODO: Should this be weighted by number of games?
    sumLosses = sum(logLosses[startBlock:endBlock + 1])
    totalSamples = sum(samples[startBlock:endBlock + 1])
    totalCorrect = sum(corrects[startBlock:endBlock + 1])
    totalIncorrect = totalSamples - totalCorrect
    mediumRatio = np.median(ratios[startBlock:endBlock + 1])

    print()
    print("Global Stats 10 to 30 minutes ({} Games)".format(max(samples)))
    print()
    print("Sum LogLoss: {:.3f}".format(sumLosses))
    print("Correct Predictions:", totalCorrect)
    print("Incorrect Predictions:", totalIncorrect)
    print("Global Ratio: {:2.1f}".format(100 * totalCorrect / totalSamples))
    print("Median Ratio: {:2.1f}".format(100 * mediumRatio))

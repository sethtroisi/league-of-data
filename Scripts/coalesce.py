import argparse
import re

from os import listdir
from os.path import isfile, join

import util


def getArgParse():
    parser = argparse.ArgumentParser(
        description='Coalesces downloaded json matches to training and test data')

    parser.add_argument(
        '-d', '--directory',
        type=str,
        default='../Data/',
        help='Directory of match json files')

    parser.add_argument(
        '--queue',
        type=str,
        default='ranked',
        help='Queue type "ranked" or "aram"')

    parser.add_argument(
        '--output-file',
        type=str,
        default='../Data/matchesAll.json',
        help='File to store concatenated matches')

    return parser


def main(args):
    queueDir = 'matches/' + args.queue + '/'
    baseDir = args.directory + queueDir

    timelines = {}
    matches = {}

    for f in listdir(baseDir):
        fileName = join(baseDir, f)

        if not isfile(fileName):
            continue

        isMatch = re.match('^getMatch-[0-9]{10}$', f)
        isTimeline = re.match('^getTimeline-[0-9]{10}$', f)
        matchId = f[-10:]
        assert isMatch or isTimeline, f

        if isMatch:
            matches[matchId] = queueDir + f
        else:
            timelines[matchId] = queueDir + f

    files = []
    for matchId in matches.keys():
        if matchId in timelines:
            files.append((matches[matchId], timelines[matchId]))

    print ('coalescing {} matches, {} timelines into {} pairs'.format(
        len(matches), len(timelines), len(files), args.output_file))

    util.writeJsonFile(args.output_file, files)


if __name__ == "__main__":
    parsedArgs = getArgParse().parse_args()
    main(parsedArgs)

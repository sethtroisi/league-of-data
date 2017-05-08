import argparse
import json
import re

from os import listdir
from os.path import isfile, join

from Util import *

def getArgParse():
  parser = argparse.ArgumentParser(
      description='Coalesces downloaded json matches to training and test data')

  parser.add_argument(
      '-d', '--directory',
      type=str,
      default='../Data/matches/',
      help='Diretory of match json files')

  parser.add_argument(
      '--output-file',
      type=str,
      default='../Data/matchesAll.json',
      help='File to store concatinated matches')

  return parser


def main(args):
  baseDir = args.directory
  
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
      matches[matchId] = f
    else:
      timelines[matchId] = f

    # TODO(sethtroisi): add filters based on dates and stuff here
    #match = loadJsonFile(fileName)


  files = []
  for matchId in matches.keys():
    if matchId in timelines:
      files.append((matches[matchId], timelines[matchId]))


  print ('coalescing {} matches, {} timelines into {} pairs'.format(
      len(matches), len(timelines), len(files), args.output_file))

  writeJsonFile(args.output_file, files)


# START CODE HERE
args = getArgParse().parse_args()
main(args)

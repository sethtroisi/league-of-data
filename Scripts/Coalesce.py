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
  matches = []
  files = 0

  baseDir = args.directory
  for f in listdir(baseDir):
    fileName = join(baseDir, f)

    if not isfile(fileName):
      continue

    assert re.match('^getMatch-[0-9]{10}$', f)

    files += 1

    match = loadJsonFile(fileName)
    matches.append(match)

  print ('coalescing {} matches'.format(files))

  writeJsonFile(args.output_file, matches)


# START CODE HERE
args = getArgParse().parse_args()
main(args)

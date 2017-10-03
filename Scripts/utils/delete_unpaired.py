#!/usr/bin/env python3

import os
import re

from collections import defaultdict

directory = "../Data/"

allFiles = set()

for root, dirs, files in os.walk(directory):
    print (root, len(dirs), len(files))
    allFiles.update( files )


items = defaultdict(list)

regex = re.compile('^([a-zA-Z]*)-([0-9]*)$')
for fileName in allFiles:
    match = regex.match(fileName)
    if not match:
        print ("no match:", fileName)
        continue

    fileType = match.group(1)
    mId = match.group(2)

    items[mId].append(fileName)


matchOnly = []
timelineOnly = []

for matchId, types in items.items():
    if len(types) != 2:
        if len(types) > 2:
            print (matchId, types)
            continue

        t = types[0]
        if 'Match' in t:
            matchOnly.append(t)
        else:
            timelineOnly.append(t)

print ("{} considered".format(len(items)))
print ("{} and {}".format(len(matchOnly), len(timelineOnly)))

if len(matchOnly) + len(timelineOnly) > 0:
    print ("deleteCommand part1:")
    print ("rm " + " ".join( (matchOnly + timelineOnly)[:40]))

import pandas as pd
from collections import deque
import os
import sys


#inPath = "/shared/3/projects/benlitterer/podcastData/diarization/mayJune/mcdn.podbean.com/0e/httpsmcdn.podbean.commfweb0e8donliveshow_202005311600.mp3.rttm"
inPath = sys.argv[1]
outStem = "/shared/3/projects/benlitterer/podcastData/diarizationMerged/"
cols = ["dummy1", "dummy2", "dummy3", "start", "duration", "dummy4", "dummy5", "speakerNum", "dummy6", "dummy7"]
diarizeDf = pd.read_csv(inPath, sep=" ", names=cols)

#get column for whether files exist 
def fileExists(fPath): 
    STEM_1 = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/floydMonth"
    #STEM_2 = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/5_4_5_10"
    #STEM_3 = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/6_9_6_15"
    STEM_2 = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/mayJuneCleanup"
    STEM_3 = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/mayJuneRemaining"
    stemList = [STEM_1, STEM_2, STEM_3]
    for stem in stemList: 
        potPath = stem + "/" + fPath
        print(potPath)
        if os.path.exists(potPath): 
            return potPath 
    return "" 

#get the path where the merged data should be 
fName = "/".join(inPath.split("/")[-3:]).replace(".rttm", "MERGED")

outPath = outStem + fName
print(outPath)

if os.path.exists(outPath): 
    print("Already Merged")
    sys.exit()

mergedPath = fileExists(fName)

if mergedPath == "": 
    print("didn't exist")
    sys.exit()

mergedDf = pd.read_csv(mergedPath)
#"/shared/3/projects/benlitterer/podcastData/prosodyMerged/floydMonth/mcdn.podbean.com/0e/httpsmcdn.podbean.commfweb0e8donliveshow_202005311600.mp3MERGED"



#whether the diarization chunk has passed and we need to no longer
#keep it in the queue
def hasPassed(tStart, tEnd, dStart, dEnd): 
    if tStart > dEnd: 
        return True
    return False 

def isOverlapping(tStart, tEnd, dStart, dEnd): 
    if tStart < dEnd and tEnd > dStart: 
        return True
    return False 


#get speaker segments as lists of tuples
diarizeDf["end"] = diarizeDf["start"] + diarizeDf["duration"]
dTuples = [(row["speakerNum"], row["start"], row["end"]) for i, row in diarizeDf.iterrows()]

#we want to merge these!
#NOTE: this works because the diarization is in order of start time
#once we hit a non-overlapping diarization segment, we know the rest of the segments 
#in the queue don't overlap
q = deque(dTuples) 
overlappingSegs = []
for i, row in mergedDf.iterrows(): 
    tStart = row["start"]
    tEnd = row["end"]

    #go through queue and keep removing while the speaker segment is no longer overlapping
    passed = True 

    #if there is more queue left and we have non-overlapping segments in it
    while len(q) > 0 and passed == True: 
        #get first item in queue, unpack tuple into variables 
        sNum, dStart, dEnd = q[0]
            
        passed = hasPassed(tStart, tEnd, dStart, dEnd)

        #if the current queue item no longer overlaps with this diarization segment, remove it 
        if passed == True: 
            q.popleft()

    #go through elements in queue that overlap with current word
    overlapping = True
    tupIter = 0 
    currentOverlap = []
    while tupIter < len(q) and overlapping == True: 
        #get speaker segment
        sNum, dStart, dEnd = q[tupIter]

        #check whether we are overlapping
        overlapping = isOverlapping(tStart, tEnd, dStart, dEnd)

        if overlapping: 
            currentOverlap.append(sNum)
        tupIter += 1
    
    #add the overlapping segments for the current word
    overlappingSegs.append(currentOverlap)

mergedDf["speakers"] = overlappingSegs

#print(mergedDf.head())

print(outPath)
print("processed new")
os.makedirs(os.path.dirname(outPath), exist_ok=True)
mergedDf.to_json(outPath, orient="records", lines=True)
#print(os.path.exists(outPath))
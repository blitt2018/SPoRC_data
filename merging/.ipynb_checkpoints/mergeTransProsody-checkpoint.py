#!/usr/bin/env python
# coding: utf-8

import warnings
import pandas as pd
import numpy as np
import srt
import sys

#where do we need to read our data from
#fileName = "httpsaudio1.redcircle.comepisodes2f8c0bc9a240433d934c8971d8da0a6cstream.mp3.srt"
#prosodyPath = "/shared/3/projects/benlitterer/podcastData/prosody/premadeTransProsody"
#diarizePath = "/shared/3/projects/benlitterer/podcastData/diarization/premadeTransDiarized"
#transcriptPath = "/shared/3/projects/benlitterer/podcastData/transcripts/premadeTransTranscripts"

#prosodyDf = pd.read_csv(f'{prosodyPath}/{fileName.replace(".srt", "LowLevel.csv")}')
print(f"reading: {sys.argv[2]}")
toKeep = ["start", "end", 'F0semitoneFrom27.5Hz_sma3nz', 'F1frequency_sma3nz', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3','mfcc4_sma3']
prosodyDf = pd.read_csv(sys.argv[2], usecols=toKeep)

def addMicroseconds(inStr): 
    if "." not in inStr: 
        return inStr + ".000000"
    else: 
        return inStr

prosodyDf["start"] = prosodyDf["start"].apply(addMicroseconds)
prosodyDf["end"] = prosodyDf["end"].apply(addMicroseconds)

#get columns to datetime format 
prosodyDf["start"] = prosodyDf["start"].apply(lambda x: x.split(" ")[2])
prosodyDf["start"] = pd.to_datetime(prosodyDf["start"], format="%H:%M:%S.%f")
prosodyDf["end"] = prosodyDf["end"].apply(lambda x: x.split(" ")[2])
prosodyDf["end"] = pd.to_datetime(prosodyDf["end"], format="%H:%M:%S.%f")

#get seconds that have passed in podcast 
prosodyDf["end"] = (prosodyDf["end"] - prosodyDf.loc[0, "start"]).dt.total_seconds()
prosodyDf["start"] = (prosodyDf["start"] - prosodyDf.loc[0, "start"]).dt.total_seconds()

#remove overlapping time chunks 
prosodyDf = prosodyDf[prosodyDf.index % 2 == 0].reset_index(drop=True)

transcriptFile = open(sys.argv[1]) 

transcript = transcriptFile.read() 
transTokens = srt.parse(transcript)

transcriptList = []

for tok in transTokens: 
    transcriptList.append([tok.start, tok.end, tok.content])

transcriptDf = pd.DataFrame(transcriptList, columns=["start", "end", "content"])

transcriptDf["start"] = transcriptDf["start"].dt.total_seconds()
transcriptDf["end"] = transcriptDf["end"].dt.total_seconds()

transcriptDf = transcriptDf.reset_index()

transcriptDf.head(4)

prosList = prosodyDf.values.tolist()

#probably just want averages but keep the actual prosody values just in case 
allProsAvgs = []
allProsVals = []

prosIndex = 0
prosStart = prosList[prosIndex][0]
prosEnd = prosList[prosIndex][1]
prosMid = (prosStart + prosEnd)/2.0
prosVals = prosList[prosIndex][2:]
    
for tokIndex, tokStart, tokEnd in transcriptDf[["index", "start", "end"]].values.tolist(): 

    
    #this is where the current prosody values go 
    currProsVals = [[] for i in range(len(prosVals))]
    
    #keep adding prosody values until the midpoint of the prosody interval 
    #is PAST the end of the token interval 
    while prosMid < tokEnd:
        
        #for each prosody value, append it to appropriate list 
        for i, prosVal in enumerate(prosVals): 
            currProsVals[i].append(prosVal)
            
        prosIndex += 1
        
        #if we hit the end of the loop exit 
        if prosIndex >= len(prosList): 
            break
            
        prosStart = prosList[prosIndex][0]
        prosEnd = prosList[prosIndex][1]
        prosMid = (prosStart + prosEnd)/2.0
        prosVals = prosList[prosIndex][2:]
    allProsVals.append(currProsVals)
    

prosodyGrouped = pd.DataFrame(allProsVals, columns=list(prosodyDf.columns)[2:])
#prosodyAvgd = pd.DataFrame(allProsAvgs, columns=list(prosodyDf.columns)[2:])

#get regression slope for a given list of prosody values 
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
def getSlope(inList): 
    if len(inList) < 2: 
        return 0
    
    x = list(np.arange(0, (len(inList)*.02)-.01, .02))
    x = np.array(x).reshape(-1, 1)
    y = np.array(inList).reshape(-1, 1)
    lr = LinearRegression(fit_intercept=True).fit(x, y)
    coef = lr.coef_.item()
    return coef



# block supresses the warnings from invalid values and taking mean/median/regression slope of empty list 
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    prosodyAvgd = prosodyGrouped.applymap(np.mean)
    prosodyMedianed = prosodyGrouped.applymap(np.median)
    prosodySlopes = prosodyGrouped.applymap(getSlope)

#add average vals to dataframe
transcriptDf[list(prosodyDf.columns)[2:]] = prosodyAvgd

#add slope columns to df
transcriptDf[[item + "Slope" for item in list(prosodyDf.columns)[2:]]] = prosodySlopes

#NOTE: the transcript df will occasionally give us tokens for which there is zero time passing
#this means we don't get any prosody values there 
#fill na values with the previous row (probably reasonable in this case)
transcriptDf = transcriptDf.fillna(method="ffill")

transcriptDf.to_csv(sys.argv[3])
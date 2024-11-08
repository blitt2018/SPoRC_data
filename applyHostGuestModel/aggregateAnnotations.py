# %%
import pandas as pd
import numpy as np
from collections import Counter
import sys

# %%
#inPath = "/shared/3/projects/benlitterer/podcastData/hostIdentification/hostGuestPredictions/10000LongPredictions.json"
df = pd.read_json(sys.argv[1], orient="records", lines=True)
df = df.dropna()

# %%
def getMode(inList): 
    if len(inList) == 1: 
        return inList[0]
    
    data = Counter(inList)
    modeVal, modeCount = data.most_common(1)[0]

    #we default to neither if we have a split decision
    
    if modeCount == 1: 
        return 2
    else: 
        return modeVal 
    
    return modeVal

#here we take the index of the maximum probability prediction 
#after mean pooling over columns 
def getConfidenceAggregation(inList): 
    inList = np.array(inList)
    return np.argmax(np.mean(inList, axis=0))

#we take in a 2d array of shape n x 3
#get the prediction for the row with the highest probability 
def getMostConfident(inList): 

    maxVal = 0 
    maxValIx = 2
    for row in inList: 
        for colNum, item in enumerate(row): 

            #if we have a new highest value, update 
            #note that maxValIx is just our prediction of 0, 1, or 2
            if item > maxVal: 
                maxVal = item 
                maxValIx = colNum
    return maxValIx

aggDf = df[["potentialOutPath", "left", "ent", "right", "pred", "prob"]].groupby(["potentialOutPath", "ent"]).agg(list)
aggDf = aggDf.reset_index()
aggDf["modalPred"] = aggDf["pred"].apply(getMode)
aggDf["confPred"] = aggDf["prob"].apply(getMostConfident)
aggDf["meanAggPred"] = aggDf["prob"].apply(getConfidenceAggregation)

# %%
#aggDf["numPreds"] = aggDf["pred"].apply(len)

# %%
#roughly 1/6th of entities have 2 predictions to use 
#aggDf["numPreds"].value_counts()

# %%
outPath = sys.argv[2] #"/shared/3/projects/benlitterer/podcastData/hostIdentification/hostGuestPredictions/10000AggPredictions.json"
aggDf.to_json(outPath, orient="records", lines=True)

# %%




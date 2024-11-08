# %%
from sklearn.metrics import roc_curve, RocCurveDisplay
from tqdm.auto import tqdm
import torch 
import transformers
from transformers import PreTrainedTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForTokenClassification
from datasets import Dataset
import pandas as pd
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast, RobertaTokenizerFast
import random
from torch import nn
import sys
#Build up to SBERT model 

# %%
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader,random_split,SubsetRandomSampler

#TODO: add an option to train on "easier" labels from batch1 before training on batch2

# %%
#set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

inArgs = sys.argv
# %%
OUT_STEM= inArgs[1] #"/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output"
PLOT_STEM= inArgs[2] #"/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output"

EPOCHS=int(inArgs[3]) #2
BATCH_SIZE=int(inArgs[4]) #4

#for command line: float("2e-5")
LR=float(inArgs[5]) #2e-4
F1_TYPE=inArgs[6] #"weighted"

DROPOUT=float(inArgs[7]) #0.0

TRAIN_SET=inArgs[8] #"full"
AGG_TYPE=inArgs[9] #"none"

#TODO: add this and lines below to script 
EVAL_SET=inArgs[10]

OUT_PATH=f"{OUT_STEM}/{AGG_TYPE}_{TRAIN_SET}_{EPOCHS}_{BATCH_SIZE}_{LR}_{F1_TYPE}_{DROPOUT}_{EVAL_SET}.log"
PLOT_PATH=f"{PLOT_STEM}/{AGG_TYPE}_{TRAIN_SET}_{EPOCHS}_{BATCH_SIZE}_{LR}_{F1_TYPE}_{EVAL_SET}_"


# %%
#create logfile 
logfile = open(OUT_PATH, "w")
logfile.write(f"aggregation type:{AGG_TYPE}\ntrain data:{TRAIN_SET}\neval set:{EVAL_SET}\nepochs:{EPOCHS}\nbatch size:{BATCH_SIZE}\nlearning rate:{LR}\nf1 type:{F1_TYPE}\ndropout:{DROPOUT}\n")
logfile.write("-------------------")

# %%
#df = pd.read_csv("/shared/3/projects/benlitterer/podcastData/hostIdentification/itunesGTsubset.tsv", sep="\t") 

# %%
#df = pd.read_json("/shared/3/projects/benlitterer/podcastData/annotation/label1000/MACE/1000annotTrain.jsonl", orient="records", lines=True)
df = pd.read_json("/shared/3/projects/benlitterer/podcastData/annotation/mergedBatches/trainMixedBatches.jsonl", orient="records", lines=True)

if TRAIN_SET == "batch1": 
    df = df[df["batch"] == 1]
if TRAIN_SET == "batch2": 
    df = df[df["batch"] == 2]


# %%
#try just training on the first part of the transcripts 
#removing mentions after a certain point 
#first get the number of words before the entity, we only use < 350 to train, so go with that 
def getEntPos(inRow): 
    return len(inRow["transcript"][:inRow["transStarts"]].split())

df["entPos"] = df.apply(getEntPos, axis=1)
df = df[df["entPos"] < 350]

# %%
df["entSnippets"] = df["left"] + df["ent"] + df["right"] 

df = df[["potentialOutPath", "left", "right", "ent",'transStarts', 'transEnds', 'groundTruth', 'entSnippets', 'batch']]
df = df.dropna()

# %%
#when we make the snippet, our spacing from the named entity extraction gets thrown off
#that's fine though! just re-extract the entity indices here 
df["snippetStart"] = df.apply(lambda x: x["entSnippets"].lower().find(x["ent"].lower()), axis=1)
df["snippetEnd"] = df["snippetStart"] + df["transEnds"] - df["transStarts"]


def extractEnt(inRow): 
    return inRow["entSnippets"][inRow["snippetStart"]:inRow["snippetEnd"]]

df["extractedEnt"] = df.apply(extractEnt, axis=1)

# %%
deviceNum = 6
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")

# %%
df = df.reset_index(drop=True)

# %%
print(device)

# %%

#put ground truth values into a list 
#trainDf = trainDf[["entSnippets", "groundTruth", "snippetStart", "snippetEnd"]] 
#trainDf = trainDf.reset_index(drop=True)

#valDf = valDf[["entSnippets", "groundTruth", "snippetStart", "snippetEnd"]] 
#valDf = valDf.reset_index(drop=True)

#get train, valid, test 
#trainDf, testDf = train_test_split(leanDf, test_size=0.3) 
#validDf, testDf = train_test_split(testDf, test_size=0.666) 


#validDataset = Dataset.from_pandas(valDf)
#testDataset = Datase|t.from_pandas(testDf)

# %%
# Preprocessing
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512, padding="max_length", truncation=True)

# %%
tokenized = []
for snip in df["entSnippets"]: 
    tokenized.append(tokenizer(snip, padding = "max_length", truncation=True, return_offsets_mapping=True))

# %%
df = pd.concat([df, pd.DataFrame.from_records(tokenized)], axis=1) 

# %%
#find the token indices which correspond to our entity 
def getTokenIndices(start, end, offsets):
    """
    print(start) 
    print(end) 
    print(offsets[:20]) 
    """

    currIndices = []
    for j, offset in enumerate(offsets): 
        offsetL, offsetR = offset
        if offsetL >= start and offsetR <= end: 
            currIndices.append(j)

    return currIndices

# %%

df["posTokens"] = df.apply(lambda row: getTokenIndices(row["snippetStart"], row["snippetEnd"], row["offset_mapping"]), axis=1)

labList = []
for i, row in df.iterrows(): 
    tokCount = sum(row["attention_mask"])
    paddingLen = len(row["attention_mask"]) - tokCount
    
    labels = ([0] * tokCount) + ([2] * paddingLen)
    
    for posIndex in row["posTokens"]: 
        labels[posIndex] = 1
    
    labList.append(labels) 

df["labels"] = labList

df["entsTokenized"] = df.apply(lambda row: [tokenizer.decode(row["input_ids"][i]) for i in row["posTokens"]], axis=1) 

# %%
#TODO: check what that <s> token is... 
#sanity check looks good!
#df.sample(10)

# %%
#validDataset.set_format(type='torch', columns=["entSnippets", "groundTruth", "input_ids", "snippetStart", "snippetEnd", "attention_mask", "offset_mapping"])

# %%
class Model(nn.Module):
    def __init__(self):
        #def __init__(self):
        super(Model,self).__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')

        #since we have three classes 
        self.l1 = nn.Linear(768, 3)

        #normalizes probabilities to sum to 1
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
        #self.ixList = ixList

        self.dropout = nn.Dropout(DROPOUT)
        
    def mean_pooling(self, token_embeddings, attention_mask): 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    #outIndices tells us the indices of the tokens corresponding to our word of interest
    #for each instance in our batch
    def forward(self, input_ids, attention_mask, outIndices): 
        
        #encode sentence and get mean pooled sentence representation 
        output = self.model(input_ids, attention_mask=attention_mask)
        
        #now we just have outIndices come to us in the forward pass 
        #outIndices = [self.ixList[ix] for ix in index]
        embeddingMeans = []
        batchIter = 0
        for batchIter in range(input_ids.shape[0]): 
            
            #get the last layer of the model 
            hiddenStates = output[0]
            
            #get the embeddings corresponding to the entity we're interested in 
            tokStates = [hiddenStates[batchIter][tokIndex,:] for tokIndex in outIndices[batchIter]]
            
            #take the mean over all embeddings for an entity 
            embeddingMean = torch.stack(tokStates).mean(dim=0)
            
            #append this so we get the mean embedding for each 
            #training example in this batch 
            embeddingMeans.append(embeddingMean) 
            #embeddingMeans.append(hiddenStates[batchIter][outIndices[batchIter][0],:])
        
        #we stack because this is for an entire batch 
        embeddingMeans = torch.stack(embeddingMeans)
        """
        working code just used this!
        embeddingMeans = self.mean_pooling(output[0], attention_mask)
        """
        probs = self.softmax(self.l1(self.dropout(embeddingMeans))).squeeze()
        
        return probs

# %%
from collections import Counter

# %%

#validation function 
def validate(model, validLoader, loss_func, tokenIxList, f1type="weighted", aggType="none"):
    model.eval()
    
    probsList = []
    validPreds = []
    validGts = []
    validLoss = []
    epKeys = []
    entList = []
    outList = [[], []]
    for batch in tqdm(validLoader): 
        
        optim.zero_grad()

        #just pass these through to list for later
        epKeys += batch["potentialOutPath"] #.cpu().tolist()
        entList += batch["ent"] #.cpu().tolist()

        input_ids = batch["input_ids"].to(device) 
        attention_mask = batch["attention_mask"].to(device) 
        index = batch["index"]
        outIndices = [tokenIxList[i] for i in index]
        
        gt = batch["groundTruth"].to(device) #.to(torch.float32)
        probs = model(input_ids, attention_mask, outIndices) #.to(torch.float32)
        
        
        #if we've hit the end of data, we may have a batch size of 1, which requires extra care
        if probs.size()[0] == 3 and len(probs.size()) == 1: 
            probs = probs.unsqueeze(0)
            print(probs.size())
            print(gt.size())

        loss = loss_func(probs, gt) 
        preds = torch.max(probs, 1).indices.to(int).cpu().tolist()
        gt = gt.to(int).detach().cpu().tolist()

        #update the lists of predictions, ground truths for train metrics
        probsList += probs.to("cpu").tolist()
        validPreds += preds
        validGts += gt
        validLoss.append(loss.cpu().detach().item())

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

    if aggType == "none": 
        f1 = f1_score(validPreds, validGts, average=f1type)

    if aggType == "mode": 
        metricDf = pd.DataFrame({"epKey":epKeys, "ent":entList, "preds":validPreds, "gts":validGts})
        metricDf = metricDf.groupby(["epKey", "ent"]).agg(list)

        #get mode 
        metricDf["preds"] = metricDf["preds"].apply(getMode)
        metricDf["gts"] = metricDf["gts"].apply(lambda x: x[0])

        f1 = f1_score(metricDf["preds"], metricDf["gts"], average=f1type)
    
    if aggType == "prob": 
        metricDf = pd.DataFrame({"epKey":epKeys, "ent":entList, "preds":validPreds, "gts":validGts, "probs":probsList})
        metricDf = metricDf.groupby(["epKey", "ent"]).agg(list)

        metricDf["preds"] = metricDf["probs"].apply(getMostConfident)
        metricDf["gts"] = metricDf["gts"].apply(lambda x: x[0])
        
        f1 = f1_score(metricDf["preds"], metricDf["gts"], average=f1type)

    validLossMean = np.mean(validLoss)
    
    model.train()
    return [probsList, entList, validPreds, validGts, f1, validLossMean]
    

# %%
def train(model, fold, epochs, optim, scheduler, loss_func, trainLoader, validLoader, tokenIxList, METRIC_FREQ=100, f1type="weighted"): 
    
    #how often should we be getting our train metrics
    print(len(trainLoader))
    validIndices = list(np.arange(0, len(trainLoader), METRIC_FREQ))
    
    #lists to put our f1 scores in 
    lossList = []
    trainMetrics = []
    validMetrics = []
    
    finalPreds = []
    
    for epoch in range(epochs):
        print("EPOCH: " + str(epoch))

        model.train()  # make sure model is in training mode

        #DEBUGGING
        i = 0 
        currLossList = []
        currPreds = []
        currGts = []
        
        for batch in tqdm(trainLoader):
            optim.zero_grad()

            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            index = batch["index"]
            gt = batch["groundTruth"].to(device) #.to(torch.)
            outIndices = [tokenIxList[i] for i in index]

            probs = model(input_ids, attention_mask, outIndices) #.to(torch.float32)

            #if we've hit the end of data, we may have a batch size of 1, which requires extra care
            if probs.size()[0] == 3 and len(probs.size()) == 1: 
                probs = probs.unsqueeze(0)
                print(probs.size())
                print(gt.size())

            loss = loss_func(probs, gt) 
            loss.backward()
            optim.step()
            scheduler.step()

            #preds = preds.detach().cpu().tolist()
            #gt = gt.detach().cpu().tolist() 

            #preds should be the index of the highest value 
            preds = torch.max(probs, 1).indices.to(int).cpu().tolist()
            gt = gt.to(int).detach().cpu().tolist()

            #update the lists of predictions, ground truths for train metrics
            currPreds += preds
            currGts += gt
            currLossList.append(loss.cpu().detach().item())

            #if we've hit the number of steps where we want to 
            #get training metrics 
            if i in validIndices: 
                trainF1 = f1_score(currPreds, currGts, average=f1type)
                avgLoss = np.mean(currLossList)
                
                #we don't want to get train metrics on the first step 
                if i != 0: 
                    trainMetrics.append([fold, i, i/len(trainLoader), epoch, trainF1, avgLoss]) 
                
                 
                probsList, entList, validPreds, validGts, validF1, validLossMean = validate(model, validLoader, loss_func, tokenIxList, f1type=f1type, aggType=AGG_TYPE)
                validMetrics.append([fold, i, i/len(trainLoader), epoch, validF1, validLossMean]) 
                                    
                """
                print(f"average loss: {np.mean(currLossList)}")
                print(f"F1: {f1_score(currPreds, currGts)}")
                print(currPreds[:20]) 
                print(currGts[:20]) 
                """
                
                #if this is our last run 
                #if i == validIndices[-1] and epoch == (epochs-1): 
                #    finalPreds.append([validPreds, validGts]) 
                
                currPreds = []
                currGts = []
                print(f"train f1: {trainF1}")
                print(f"valid f1: {validF1}") 
                print(f"learning rate {scheduler.get_last_lr()}")
                #print(model.l1.weight[0][:20]) 
                
            i += 1
    
    return [probsList, entList, validPreds, validGts, trainMetrics, validMetrics]
                                    
    """       
    print(f"average loss: {np.mean(currLossList)}")
    print(f"F1: {f1_score(currPreds, currGts)}")
    print(f"learning rate {scheduler.get_last_lr()}")  
    """

# %%
#design a way to get the f1 after averaging within episode 

# %%

uniqueEnts = df["ent"].unique()

#if TRAIN_SET=batch1, we just won't have any batch2ents, so should work automatically 
batch2Ents = df.loc[df["batch"] == 2, "ent"].unique()
if EVAL_SET == "mixed": 

    #split the unique entities into K_FOLDS segments 
    FOLDS=5
    splits=KFold(n_splits=FOLDS,shuffle=True,random_state=42)

    entSplits = [split for split in splits.split(uniqueEnts)]
    df = df.reset_index(drop=True).reset_index()

if EVAL_SET == "batch1": 
    #split the unique entities into K_FOLDS segments 
    FOLDS=5
    splits=KFold(n_splits=FOLDS,shuffle=True,random_state=42)

    batch1Ents = df.loc[df["batch"] == 1, "ent"].unique()
    entSplits = [split for split in splits.split(batch1Ents)]
    df = df.reset_index(drop=True).reset_index()
# %%

#trainDf = df[df["ent"].apply(lambda x: x in trainEnts)]
#trainDf.shape

# %%

#now do a version where we only validate on 1 entity 
#valDf = df[df["ent"].apply(lambda x: x in valEnts)]
#valDf = valDf.sort_values(["ent", "transStarts"]).drop_duplicates("ent")

#print(valDf.shape)

# %%
#how do we evaluate at the aggregated level? 
df.head() 

# %%
modelPath = "/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/"

allMetrics = []
bestF1 = 0

for fold, (train_idx,val_idx) in enumerate(entSplits):

    #get the entities associated with the train and validation indices 
    #if EVAL_SET is batch1, we add all of the ents for batch 2 into our training data 
    if EVAL_SET == "batch1": 
            #if TRAIN_SET is already batch1, then we should just get empty batch2 ents, since df doesn't have batch2 at all 
            trainEnts = list(batch1Ents[train_idx]) + list(batch2Ents) 
            valEnts = list(batch1Ents[val_idx]) 
        

    if EVAL_SET == "mixed": 
        trainEnts = uniqueEnts[train_idx]
        valEnts = uniqueEnts[val_idx]

    trainDf = df[df["ent"].apply(lambda x: x in trainEnts)]

    #now do a version where we only validate on 1 entity 
    valDf = df[df["ent"].apply(lambda x: x in valEnts)]
    valDf = valDf.sort_values(["ent", "transStarts"]).drop_duplicates("ent")
    
    print(valDf.shape)
    #get an "index" column that just indexes the row of the dataframe we have 
    #trainDf = trainDf.reset_index(drop=True).reset_index()
    #valDf = valDf.reset_index(drop=True).reset_index()

    #get a fresh index
    #the index is used to find the indices of the entity in the model we're training 
    tokenIxList = list(df["posTokens"])
    
    #here we just take the mixture we were already using to train and make batch1 go first 
    if TRAIN_SET=="batch1_2": 
        b1Df = trainDf[trainDf["batch"] == 1]
        b1Df = b1Df.sample(len(b1Df))

        b2Df = trainDf[trainDf["batch"] == 2]
        b2Df = b2Df.sample(len(b2Df))

        trainDf = pd.concat([b1Df, b2Df])
    
    trainDataset = Dataset.from_pandas(trainDf)
    trainDataset.set_format(type='torch', columns=["potentialOutPath", "ent", "index", "entSnippets", "groundTruth", "input_ids", "attention_mask"])
    
    valDataset = Dataset.from_pandas(valDf)
    valDataset.set_format(type='torch', columns=["potentialOutPath", "ent", "index", "entSnippets", "groundTruth", "input_ids", "attention_mask"])

    #initialize model 
    model = Model().to(device)
    
    print('Fold {}'.format(fold + 1))

    #if we are training on batch1 then batch 2, then we don't shuffle
    #as we've already done so within batches 
    toShuffle = TRAIN_SET != "batch1_2"

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=toShuffle)
    validLoader = torch.utils.data.DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    
    total_steps = int(len(trainLoader))*EPOCHS
    warmup_steps = int(0.10 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps - warmup_steps)
    loss_func = torch.nn.CrossEntropyLoss()
    
    metrics = train(model, fold, EPOCHS, optim, scheduler, loss_func, trainLoader, validLoader, tokenIxList, f1type=F1_TYPE)
    allMetrics.append(metrics)

    #find best f1 from validation
    currBestF1 = max([metList[4] for metList in metrics[5]])

    #save model if need be
    if currBestF1 > bestF1: 
        bestF1 = currBestF1
        torch.save(model.state_dict(), f'{modelPath}bestF1Params')

    del model 


# %%
#append all of the train matrices together 
trainOutput = []
validOutput = []
for i in range(len(allMetrics)): 
    trainOutput += allMetrics[i][4]
    validOutput += allMetrics[i][5]
    
trainDf = pd.DataFrame(trainOutput, columns=["fold", "step", "epochFrac", "epoch", "f1", "learningRate"]) 
trainDf["data"] =  "train"

validDf = pd.DataFrame(validOutput, columns=["fold", "step", "epochFrac", "epoch", "f1", "learningRate"]) 
validDf["data"] =  "valid"                     

#validDf["totalSteps"] = 
#validDf["overallEpochFrac"] = validDf["epoch"] + validDf["epochFrac"]
#trainDf["overallEpochFrac"] = trainDf["epoch"] + trainDf["epochFrac"]

#NOTE: this is approximate, since not every fold is the exact same length 
epochLen = len(trainLoader)
trainDf["epochFrac"] = (trainDf["step"] / epochLen) + trainDf["epoch"]
validDf["epochFrac"] = (validDf["step"] / epochLen) + validDf["epoch"]
metricDf = pd.concat([validDf, trainDf], axis=0) 

# %%
metricDf.head() 

# %%
#get best validation score achieved
print(f'highest validation f1 acheived: {max(metricDf.loc[metricDf["data"] == "valid", "f1"])}') 
logfile.write(f'highest validation f1 acheived: {max(metricDf.loc[metricDf["data"] == "valid", "f1"])}')

foldF1s = metricDf.loc[metricDf["data"] == "valid", ["fold", "f1"]].groupby("fold").agg(lambda x: list(x)[-1])
logfile.write(f"Fold 1-5 f1's:{list(foldF1s['f1'])}\n")
print(foldF1s)

#get mean, median F1 across folds 
logfile.write(f"Median f1:{np.median(list(foldF1s['f1']))}\n")
logfile.write(f"Mean f1:{np.mean(list(foldF1s['f1']))}\n")

# %%

#TODO: collect different types of F1 scores, also do for no duplicates in validation set 
fig, ax = plt.subplots(figsize=(4, 4))
sns.lineplot(data=metricDf, x="epochFrac", y="f1", hue="data", ax=ax)
ax.set_ylabel(f"Accuracy ({F1_TYPE} f1)")
ax.set_xlabel("Epoch (approximate)")
plt.title(f"Speaker Role Classification: {F1_TYPE} f1")
plt.tight_layout()
fig.savefig(f"{PLOT_PATH}trainValidPlot.jpg")

# %%
allPreds = []
allGts = []

for i in range(len(allMetrics)): 
    allPreds += allMetrics[i][2]
    allGts += allMetrics[i][3] 

# %%
from sklearn.metrics import confusion_matrix

# %%
logfile.write(str(confusion_matrix(allGts, allPreds)) + "\n")

#we want to get the ROC curve for the aggregated predictions 
entList = []
probList = []
gtList = []
for i in range(len(allMetrics)): 
    entList += allMetrics[i][1]
    probList += allMetrics[i][0]
    gtList += allMetrics[i][3]


rocDf = pd.DataFrame({"ent":entList, "prob":probList, "gt":gtList})
rocDf = rocDf.groupby("ent").agg(list)

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
    return [maxValIx, maxVal]

rocDf[["pred", "maxProb"]] = rocDf["prob"].apply(getMostConfident).tolist()

def getHostProb(inRow): 
    if inRow["pred"] == 0: 
        return inRow["maxProb"]
    
    #we assign it to zero probability if it wasn't predicted 
    else: 
        return 0


def getGuestProb(inRow): 
    if inRow["pred"] == 1: 
        return inRow["maxProb"]
    
    #we assign it to zero probability if it wasn't predicted 
    else: 
        return 0

#first think of the positive class as host and everything else as the negative class 

#get the maximum probability assigned to guessing host across the different predictions 
rocDf["gt"] = rocDf["gt"].apply(lambda x: x[0])
rocDf["hostBinaryProb"] = rocDf.apply(getHostProb, axis=1)
rocDf["guestBinaryProb"] = rocDf.apply(getGuestProb, axis=1)

#now we have probabilities for our different "positive" classes we're interested in for binary version of the task 
#get 0, 1 binary ground truth labels for host and guest as positive class 
rocDf["hostGt"] = rocDf["gt"] == 0
rocDf["hostGt"] = rocDf["hostGt"].astype(int)

rocDf["guestGt"] = rocDf["gt"] == 1
rocDf["guestGt"] = rocDf["guestGt"].astype(int)

from sklearn.metrics import precision_recall_fscore_support, f1_score

def cutPoints(y_true, y_prob, thresholds):

    outArr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        prec, rec, fBeta, support = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary")
        f1 = f1_score(y_true, y_pred)

        outArr.append([fpr, tpr, prec, rec, f1])

    return outArr

#we have a target we're intersted in, so we want the TPR and FPR for this target 
CUTOFFS=[.3, .4, .5, .6, .7, .8, .9, .95]

fpr, tpr, _ = roc_curve(rocDf["hostGt"], rocDf["hostBinaryProb"], pos_label=1, drop_intermediate=False)
hostOutput = pd.DataFrame({"fpr":fpr, "tpr":tpr, "cutoffs":_})
hostOutput["type"] = "host vs. rest"
hostCutPoints = cutPoints(rocDf["hostGt"], rocDf["hostBinaryProb"], CUTOFFS)

fpr, tpr, _ = roc_curve(rocDf["guestGt"], rocDf["guestBinaryProb"], pos_label=1)
guestOutput = pd.DataFrame({"fpr":fpr, "tpr":tpr, "cutoffs":_})
guestOutput["type"] = "guest vs. rest"
guestCutPoints = cutPoints(rocDf["guestGt"], rocDf["guestBinaryProb"], CUTOFFS)

rocOutput = pd.concat([hostOutput, guestOutput])
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter([item[0] for item in hostCutPoints],  [item[1] for item in hostCutPoints], color="red")
sns.lineplot(rocOutput[rocOutput["type"] == "host vs. rest"], x="fpr", y="tpr", ax=axs[0])
axs[0].set_title("Binary Host Prediction Task - ROC Curve")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")

axs[1].scatter([item[0] for item in guestCutPoints],  [item[1] for item in guestCutPoints], color="red")
sns.lineplot(rocOutput[rocOutput["type"] == "guest vs. rest"], x="fpr", y="tpr", ax=axs[1], color="orange")
axs[1].set_title("Binary Guest Prediction Task - ROC Curve")
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate")

plt.tight_layout()
plt.savefig(f"{PLOT_PATH}ROCplot.jpg")

logfile.write(f"cutpoints: {CUTOFFS}\n")
logfile.write(f"host fpr, tpr at cuts: {hostCutPoints}\n")
logfile.write(f"guest fpr, tpr at cuts: {guestCutPoints}\n")
logfile.close()
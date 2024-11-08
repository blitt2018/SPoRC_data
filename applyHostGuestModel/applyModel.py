# %%
import pandas as pd
import torch 
from torch import nn
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForTokenClassification
from datasets import Dataset
from tqdm import tqdm 
from collections import Counter
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerFast, RobertaTokenizerFast
import numpy as np
import sys

#best model didn't use dropout 
DROPOUT=0
# %%
#model class 
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
torch.cuda.is_available()

#arguments are: state_path, device_num, in_path, out_path
# %%
#load best model from training
STATE_PATH  = sys.argv[1] #"/shared/3/projects/benlitterer/podcastData/hostGuestModels/initialModel/bestF1Params"
deviceNum = int(sys.argv[2]) 
device = torch.device("cuda:" + str(deviceNum) if torch.cuda.is_available() else "cpu")
print(device)
model = Model().to(device)
model.load_state_dict(torch.load(STATE_PATH))

# %%
#load dataset
inPath = sys.argv[3]
df = pd.read_json(inPath, orient="records", lines=True)

print(df.shape)
# %%
toKeep = ["potentialOutPath", "transcript", "rssUrl", "epTitle", "epDescription", "transEnts", "transStarts", "transEnds", "transTypes"]
df = df[toKeep].explode(["transEnts", "transStarts", "transEnds", "transTypes"])

print(df.shape)
df = df[df["transTypes"] == "PERSON"]
print(df.shape)
# %%
#removing mentions after a certain point 
#first get the number of words before the entity, we only use < 350 to train, so go with that 
def getEntPos(inRow): 
    return len(inRow["transcript"][:inRow["transStarts"]].split())

print(df.head(5))

#only use entities in the first 350 tokens 
df["entPos"] = df.apply(getEntPos, axis=1)
df = df[df["entPos"] < 350]

#only use entities with two tokens 
# %%
df["transEntLen"] = df["transEnts"].apply(lambda x: len(x.split()))
df = df[df["transEntLen"] == 2]


# %%
df = df.sort_values(["potentialOutPath", "transEnts"]) 

# %%

BEFORE_BUFF = 50
AFTER_BUFF=50
#PUNCH IN HERE
def getSnippet(row): 
    #find where the entity starts quick 
   # row["snippetStart"] = trainDf.apply(lambda x: x["entSnippets"].lower().find(x["ent"].lower()), axis=1)
    
    snippet = row["transcript"]
    entStart = row["transStarts"]
    entEnd = row["transEnds"]

    
    beforeWords = snippet[0:entStart].split(" ")
    if len(beforeWords) >= BEFORE_BUFF: 
        buffStart = " ".join(beforeWords[-BEFORE_BUFF:]) 
    else: 
        buffStart = " ".join(beforeWords) 

    afterWords = snippet[entEnd:len(snippet)].split(" ")

    if len(afterWords) >= AFTER_BUFF: 
        buffEnd = " ".join(afterWords[:AFTER_BUFF]) 
    else: 
        buffEnd = " ".join(afterWords) 
    return [buffStart, snippet[entStart:entEnd], buffEnd]
            

df[["left", "ent", "right"]] = pd.DataFrame(df.apply(getSnippet, axis=1).tolist(), index=df.index)

# %%
#for the sake of memory 
df = df.drop(columns=["transcript"])

# %%
df["entSnippets"] = df["left"] + df["ent"] + df["right"] 

#df = df[["left", "right", "ent",'transStarts', 'transEnds', 'groundTruth', 'entSnippets']]
#df = df.dropna()

# %%
df["snippetStart"] = df.apply(lambda x: x["entSnippets"].lower().find(x["ent"].lower()), axis=1)
df["snippetEnd"] = df["snippetStart"] + df["transEnds"] - df["transStarts"]

def extractEnt(inRow): 
    return inRow["entSnippets"][inRow["snippetStart"]:inRow["snippetEnd"]]

df["extractedEnt"] = df.apply(extractEnt, axis=1)

# %%
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length=512, padding="max_length", truncation=True)

# %%
tokenized = []
for snip in df["entSnippets"]: 
    tokenized.append(tokenizer(snip, padding = "max_length", truncation=True, return_offsets_mapping=True))

df = pd.concat([df.reset_index(), pd.DataFrame.from_records(tokenized)], axis=1) 

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

#drop extra information about location of tokens that aren't those of interest
df = df.drop(columns=["offset_mapping"])

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
extractionErrorCount = len(df[df["extractedEnt"].apply(lambda x: x.lower()) != df["ent"].apply(lambda x: x.lower())]) 
print(f"Number of entities where we have error from extraction of entity: {extractionErrorCount}")

# %%
df = df.drop(columns=["index"])

# %%
df = df.dropna(subset=["attention_mask", "input_ids", "posTokens"])

#add an index to be used for getting the position of tokens later 
df = df.reset_index(drop=True).reset_index()

# %%
#tokenRef = dict(zip(df["index"], df["posTokens"]))
tokenIxList = list(df["posTokens"])

# %%
BATCH_SIZE=2
#make sure we are feeding the model something with each row 
df = df.dropna(subset=["input_ids", "attention_mask", "index"])
dataset = Dataset.from_pandas(df)
dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "index"])
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

import numpy as np
# %%
#input ids, mask, indices 
predList = [] #np.array([])
probList = [] #np.array([])

for batch in tqdm(loader):
    input_ids = batch["input_ids"].to(device) 
    attention_mask = batch["attention_mask"].to(device) 
    index = batch["index"]
    outIndices = [tokenIxList[i] for i in index]
    
    try: 
        if len(outIndices) > 0: 
            probs = model(input_ids, attention_mask, outIndices)  #.to(torch.float32)
            preds = torch.max(probs, 1).indices.to(int).cpu().tolist()
            predList += preds #np.append(predList, np.array(preds)) 
            probList += probs.to("cpu").tolist()  #np.append(probList, np.array(probs.to("cpu").tolist()))
    except:
        for i in range(len(index)): 
            print("error occured")
            predList.append(None) #np.append(predList, np.nan)
            probList.append([None]) #np.append(probList, np.nan)
            #predList += np.nan
    

# %%
df["pred"] = predList
df["prob"] = probList

# %%
outPath = sys.argv[4]
df[["potentialOutPath", "rssUrl", "left", "ent", "right", "pred", "prob"]].to_json(outPath, orient="records", lines=True)

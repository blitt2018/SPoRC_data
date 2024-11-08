import pandas as pd
from tqdm import tqdm 

#dataframe with the entities extracted from the introduction of each podcast 
entDf = pd.read_csv("/shared/3/projects/benlitterer/podcastData/NER/podIntroductions/floydMonth500wordNEs.csv", names=["potentialOutPath", "ent", "start", "end", "type"])
entDf = entDf[entDf["type"] == "PERSON"] 

#metadata (this has author field)  
metaDf = pd.read_csv("/shared/3/projects/benlitterer/podcastData/processed/floydMonth/floydMonthEn.csv")
metaDf["potentialOutPath"] = "/shared/3/projects/benlitterer/podcastData/prosodyMerged/floydMonth" + metaDf["potentialOutPath"]

#dataframe with the first bit (500 words) of each podcast 
introDf = pd.read_csv("/shared/3/projects/benlitterer/podcastData/NER/floydIntroductions.tsv", sep="\t", names=["potentialOutPath", "first500"])

#merges on only the transcripts we actually have 
#merge metadata, transcript beginnings, and entities
df = pd.merge(entDf, introDf, on="potentialOutPath", how="inner") 
df = pd.merge(df, metaDf[["itunesAuthor","rssUrl", "potentialOutPath"]], on="potentialOutPath", how="inner") \

df = df.dropna() 

#sort the dataframe so that within each episode (potentialOutPath)
#we have the starts of the entities in ascending order
df = df.sort_values(["potentialOutPath", "start"]) 

#we want to figure out where to create our snippets 
df = df.groupby("potentialOutPath").agg(list) 

BEFORE_BUFF = 10 
AFTER_BUFF = 10

#go through and create snippets of text 
entSnippets = []
for i, row in tqdm(df.iterrows()): 
    prevEntEnd = 0 
    currEntSnippets = []
    for j in range(len(row["start"])): 
        snippet = row["first500"][j]
        entStart = row["start"][j]
        entEnd = row["end"][j]
        
        #get position that is BUFFER words before and after entity
        beforeWords = snippet[0:entStart].split(" ")
        
        #only if we have enough words before to get entire buffer 
        if len(beforeWords) >= BEFORE_BUFF: 
            #get the snippet before 
            buffStart = entStart - len(" ".join(beforeWords[-BEFORE_BUFF:])) -1
        else: 
            buffStart = entStart - len(" ".join(beforeWords)) -1
        
        #get position that is BUFFER words before and after entity
        afterWords = snippet[entEnd:len(snippet)].split(" ")
        
        #only if we have enough words after to get entire buffer
        if len(afterWords) >= AFTER_BUFF: 
            #get the snippet after
            buffEnd = entEnd + len(" ".join(afterWords[:AFTER_BUFF])) + 1
        else:  
            buffEnd = entEnd + len(" ".join(afterWords)) + 1
        
        """
        testing
        print(row["ent"][j])
        print(snippet[entStart:buffEnd])
        print("---------------") 
        """
        
        """
        right now this is set up so that we don't include entities before 
        
        """
        snippetLeft = max(prevEntEnd, buffStart) 
        
        #if we have a next entity to look ahead to, see where it starts
        #if it starts before where our buffer will end, stop when we hit that next entity 
        if j + 1 < len(row["start"]):
            snippetRight = min(row["start"][j + 1], buffEnd)
        else: 
            snippetRight = buffEnd
            
        currEntSnippets.append(snippet[snippetLeft: snippetRight])
        prevEntEnd = entEnd 
    entSnippets.append(currEntSnippets) 
    
df["entSnippets"] = entSnippets

#now that we've gotten the snippets we can explode back out
df = df.explode(list(df.columns))

#now we want to determine whether an entity is a ground truth label or not 
#we can do so by checking if it is contained in the itunes author field 
def entIsAuthor(inRow):
    if inRow["ent"] == inRow["ent"]: 
        ent = inRow["ent"] 
    else: 
        return False 
    if inRow["itunesAuthor"] == inRow["itunesAuthor"]: 
        auth = inRow["itunesAuthor"]
    else: 
        return False 
        
    return ent.lower() in auth.lower()

df["entInAuthor"] = df.apply(lambda x: x["ent"].lower() in x["itunesAuthor"].lower(), axis=1)

df = df.drop(columns=["first500"])

posClass = df[df["entInAuthor"] == True] 
negClass = df[df["entInAuthor"] == False].sample(len(posClass) * 4) 

trainData = pd.concat([posClass, negClass], axis=0) 

trainData.to_csv("/shared/3/projects/benlitterer/podcastData/hostIdentification/itunesGTsubset.tsv", sep="\t") 
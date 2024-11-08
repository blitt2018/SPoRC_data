import torch
from pyannote.audio import Pipeline
import sys
from pyannote.audio import Audio

#we control which gpu is used 
#by setting it's visibility in bash script calling this script
gpuNum = sys.argv[1]
FILE_PATH=sys.argv[2]
OUT_PATH=sys.argv[3]
device = torch.device(f"cuda:{str(gpuNum)}") 
#print(device)
#device = torch.device("cuda:2")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1" ,use_auth_token="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").to(device) 

io = Audio(mono='downmix', sample_rate=16000)
waveform, sample_rate = io(FILE_PATH)

print("diarizing")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# apply the pipeline to an audio file
#diarization = pipeline(FILE_PATH)

print("writing")
# dump the diarization output to disk using RTTM format
with open(OUT_PATH, "w") as rttm:
    diarization.write_rttm(rttm)

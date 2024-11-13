#  SPoRC: the Structured Podcast Research Corpus!
Welcome to the github repository for the creation of SPoRC: the Structured Podcast Research Corpus!

You can access our data [here](https://huggingface.co/datasets/blitt/SPoRC), our data analysis pipeline [here](https://github.com/blitt2018/SPoRC_analysis), and our publication [here](FILL_IN).

SPoRC captures the inherently multi-modal nature of podcast data, with transcripts and metadata for over 1.1M episodes and speaker-turn and audio-feature data for over 370K episodes. As shown below, these features can be combined to provide rich insight into human communication:  

<p align="center">
  <img src="/figures/diarizationVisualization.png?raw=true">
  <em>Speaker-pitch information overlayed with speaker-turn information.</em>
</p>

<p align="center">
  <img src="/figures/transcriptHighlightingFigure.png?raw=true">
  <em>A podcast transcript colored by speaker turns. Multiple assigned speakers are depicted with grey text and no assigned speakers are assigned black text.</em>
</p>

<p align="center">
  <img src="/figures/pitchDemo.png?raw=true">
  <em>Speaker-pitch information displayed alongside token-level transcript information.</em>
</p>

To create our dataset, we begin with podcast-level metadata from [Podcast Index](https://podcastindex.org/) and collect episode-level metadata by scraping the RSS feeds associated with each English podcast from May-June 2020. We then feed mp3 url's from these RSS feeds into a three-phase pipeline that extracts transcript, audio, and speaker-turn information. Finally, all of these data types are merged together at the episode level and speaker-turn level and [released](https://huggingface.co/datasets/blitt/SPoRC) for future non-commercial use.

# Our 3-Phase Pipeline 
Here, we release code our three-phase pipeline such that transcripts, audio features, and speaker turns can be given a particular mp3 url as input. This pipeline runs using the [transcribeOne.sh](transcriptionProsody/transcribeOne.sh) script for transcription and audio-feauture extraction and the [diarizeOne.sh](diarization/diarizeOne.sh) script for diarization. These scripts are run seperately, with their outputs being merged after the fact by [mergeDiarization](merging/mergeDiarization.py), as the transcription+audio-feature extraction runs on cpu and the latter runs on gpu.

Below, we walk through [transcribeOne.sh](transcriptionProsody/transcribeOne.sh) and [diarizeOne.sh](diarization/diarizeOne.sh) to illustrate our pipeline.  

## Setup
To begin our pipeline, we declare an input mp3 url and set up our storage and script paths. We recommend using our [conda environment](SPoRCenvironment.yml) for easy setup.   

```
#the goal is to download a file, transcribe it, then delete it in one file 
#we then run this file in parallel!

source activate SPoRCenvironment 

#input url to mp3 of interest
inURL=path/to/url

#final output path to store merged data in 
FINAL_OUT_PATH=path/to/dir

#where to store intermediary mp3s, transcripts, and prosodic information 
STORAGE_DIR=my/storage/path
MP3_LOC=$STORAGE_DIR/mp3s/transcription
TRANSCRIPTS_PATH=$STORAGE_DIR/transcripts
PROSODY_PATH=$STORAGE_DIR/prosody

#paths to whisper and opensmile python script  
WHISPER_PATH=my/whisper/path/whisper.cpp
MODEL_NAME="ggml-base.en.bin"
OPENSMILE_PATH=transcriptionProsody/extractProsodicFeatures.py

#code to merge together the transcription and prosodic information at the word level  
MERGE_SCRIPT_PATH=merging/mergeTransProsody.py
MERGED_PATH=final/output/directory 

#for time logging 
start=`date +%s`

#don't continue processing if we fail on any given step 
set -e 

#download the file to MP3 location
curl --connect-timeout 10 --max-time 240 -L $inURL --output $MP3_LOC/$kURL

#echo "converting" 
#forcibly overwrite using the -y flag 
#turn that file into .wav format 

ffmpeg -y -i $MP3_LOC/$kURL -ar 16000 -ac 1 -c:a pcm_s16le $MP3_LOC/$kURL.wav

#delete the mp3, as we now have the .wav 
rm $MP3_LOC/$kURL
```

## Transcription and Audio-Feature extraction 
Extracting transcripts and audio-features is as simple as calling two scripts. We use the Whisper-base model and openSmile with the eGeMAPSv02feature set, though these settings are easily changeable in future use.  

```
#transcribe the file
$WHISPER_PATH/main -osrt -p 1 -t 1 -ml 1 -m $WHISPER_PATH/models/$MODEL_NAME -f $MP3_LOC/$kURL.wav -of $TRANSCRIPTS_PATH/$kURL

#run audio-feature extraction with opensmile  
python3 $OPENSMILE_PATH $MP3_LOC/$kURL.wav $PROSODY_PATH/$kURL
```

## Merging Transcripts and Audio-Features 
Even after deleting .mp3 and .wav files, storage can quickly become a limitation due to the large size of the audio-feature output. To minimize this factor, we merge our audio-feature information onto our word-level transcript information using [mergeTranscriptProsody.py](merging/mergeTranscriptProsody.py).  

```
python3 $MERGE_SCRIPT_PATH $TRANSCRIPTS_PATH/${kURL}.srt $PROSODY_PATH/${kURL}LowLevel.csv $MERGED_PATH/$FINAL_OUT_PATH/${kURL}MERGED

#remove all files other than merged version 
rm $TRANSCRIPTS_PATH/${kURL}.srt
rm $PROSODY_PATH/${kURL}LowLevel.csv
```

## Diarization 
The code below comes from [diarizeOne.sh](diarization/diarizeOne.sh) and demonstrates how to extract speaker-turn information given a single mp3 url. 

```
#specify gpu, input mp3 url, and final output directory 
GPU_NUM=my_gpu_number
inURL=my_input_url
DIARIZE_PATH=$STORAGE_DIR/diarization

STORAGE_DIR=my/storage/location
MP3_LOC=$STORAGE_DIR/mp3s/diarization

PYANNOTE_PATH=diarization/pyAnnoteGPU.py
URL_KEY_PATH=~/transcriptionProsody/cleanURL.py

#this makes the script fail and exit if any of the following lines fail 
set -e 

#download the file to MP3 location
curl -L $inURL --output $MP3_LOC/$kURL

#forcibly overwrite using the -y flag
#turn that file into .wav format
ffmpeg -hide_banner -loglevel error -y -i $MP3_LOC/$kURL -ar 16000 -ac 1 -c:a pcm_s16le $MP3_LOC/$kURL.wav

#delete the mp3
rm $MP3_LOC/$kURL

#echo "diarizing"
time python3 $PYANNOTE_PATH $GPU_NUM $MP3_LOC/$kURL.wav $DIARIZE_PATH/$kURL.rttm

#delete the wav
rm $MP3_LOC/$kURL.wav
```

## Final Merge
To create a single output file with transcription, audio-feature, and speaker-turn information, we perform a final merge using [mergeDiarization.py](merging/mergeDiarization.py). This file maintains our word-level transcript and audio-feature information but adds on speaker-information as a column. When multiple speakers are overlapping, the first speaker in the list corresponds to who was had been speaking first.  


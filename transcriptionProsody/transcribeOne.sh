#the goal is to download a file, transcribe it, then delete it in one file 
#we then run this file in parallel!

source activate transcribePodcasts

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

#transcribe the file
$WHISPER_PATH/main -osrt -p 1 -t 1 -ml 1 -m $WHISPER_PATH/models/$MODEL_NAME -f $MP3_LOC/$kURL.wav -of $TRANSCRIPTS_PATH/$kURL

#run audio-feature extraction with opensmile  
python3 $OPENSMILE_PATH $MP3_LOC/$kURL.wav $PROSODY_PATH/$kURL

#delete the wav
rm $MP3_LOC/$kURL.wav

python3 $MERGE_SCRIPT_PATH $TRANSCRIPTS_PATH/${kURL}.srt $PROSODY_PATH/${kURL}LowLevel.csv $MERGED_PATH/$FINAL_OUT_PATH/${kURL}MERGED 

#remove all files other than merged version 
rm $TRANSCRIPTS_PATH/${kURL}.srt
rm $PROSODY_PATH/${kURL}LowLevel.csv 

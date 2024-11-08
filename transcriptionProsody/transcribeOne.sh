#the goal is to download a file, transcribe it, then delete it in one file 
#we then run this file in parallel!

source activate transcribePodcasts

STORAGE_DIR=/shared/3/projects/benlitterer/podcastData
MP3_LOC=$STORAGE_DIR/mp3s/transcription
WHISPER_PATH=/shared/4/projects/podcasts/whisper.cpp
MODEL_NAME="ggml-base.en.bin"
TRANSCRIPTS_PATH=$STORAGE_DIR/transcripts/mayJune
PROSODY_PATH=$STORAGE_DIR/prosody/floydMonth/mayJune
OPENSMILE_PATH=~/projects/podcasts/mixedTranscription/extractProsodicFeatures.py
URL_KEY_PATH=~/projects/podcasts/mixedTranscription/cleanURL.py
#GET_URL_PATH=/home/blitt/projects/podcasts/mixedTranscription/getNextURL.py

GET_URL_PATH=/home/blitt/projects/podcasts/mixedTranscription/fileTracking/mayJuneRemaining/getNextURLSetProcessing.py
UPDATE_URL_PROCESSED=/home/blitt/projects/podcasts/mixedTranscription/fileTracking/mayJuneRemaining/updateUrlProcessed.py
#UPDATE_URL_PROCESSING=/home/blitt/projects/podcasts/mixedTranscription/fileTracking/floydMonths/updateUrlProcessing.py
MERGE_SCRIPT_PATH=/home/blitt/projects/podcasts/merging/mergeTransProsody.py
MERGED_PATH=/shared/3/projects/benlitterer/podcastData/prosodyMerged/mayJuneRemaining 
#LOG_PATH=$STORAGE_DIR/logs/floydMonth
GET_FINAL_DIR=/home/blitt/projects/podcasts/mixedTranscription/getHostPath.py


#for time logging 
start=`date +%s`

set -e 

#TODO: add while loop here so this runs forever 
#this searches the urls to find the first one 
#that hasn't been written to the finshed file
#this also updates to the database that this file is being processed 
inURL=`python3 $GET_URL_PATH`
kURL=`python3 $URL_KEY_PATH $inURL`

#python3 $UPDATE_URL_PROCESSING $inURL
echo $inURL 
echo "downloading" 
#download the file to MP3 location
curl --connect-timeout 10 --max-time 240 -L $inURL --output $MP3_LOC/$kURL

#echo "converting" 
#forcibly overwrite using the -y flag 
#turn that file into .wav format 
ffmpeg -y -i $MP3_LOC/$kURL -ar 16000 -ac 1 -c:a pcm_s16le $MP3_LOC/$kURL.wav

#echo "removing mp3" 
#delete the mp3
rm $MP3_LOC/$kURL

#echo "transcribing" 
#transcribe the file
$WHISPER_PATH/main -osrt -p 1 -t 1 -ml 1 -m $WHISPER_PATH/models/$MODEL_NAME -f $MP3_LOC/$kURL.wav -of $TRANSCRIPTS_PATH/$kURL
whisper=`date +%s`
#echo ffmpeg whisper time `expr $whisper - $start` seconds > $LOG_PATH/$kURL

#echo "prosody" 
python3 $OPENSMILE_PATH $MP3_LOC/$kURL.wav $PROSODY_PATH/$kURL
prosody=`date +%s`
#echo prosody time `expr $prosody - $whisper` seconds >> $LOG_PATH/$kURL

#echo "removing wav" 
#delete the wav
rm $MP3_LOC/$kURL.wav

#get the final directory for our merged file
#we do this so we dont have huge huge directories  
HIER_PATH=`python3 $GET_FINAL_DIR $inURL` 
mkdir -p $MERGED_PATH/$HIER_PATH

python3 $MERGE_SCRIPT_PATH $TRANSCRIPTS_PATH/${kURL}.srt $PROSODY_PATH/${kURL}LowLevel.csv $MERGED_PATH/$HIER_PATH/${kURL}MERGED 
#TODO: remove all files other than merged version 
rm $TRANSCRIPTS_PATH/${kURL}.srt
rm $PROSODY_PATH/${kURL}LowLevel.csv 
finished=`date +%s`
#echo merge time `expr $finished - $prosody` seconds >> $LOG_PATH/$kURL

#update that this url has been processed 
python3 $UPDATE_URL_PROCESSED $inURL 

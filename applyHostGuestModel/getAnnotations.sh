APPLY_MODEL=/home/blitt/projects/podcasts/analysis/applyHostGuestModel/applyModel.py 
AGG_ANNOTS=/home/blitt/projects/podcasts/analysis/applyHostGuestModel/aggregateAnnotations.py

#TODO: change this so that we first break up data_path, then loop over these chunks sequentially to avoid memory issues 
STATE_PATH="/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/5_4_2e-06_batch1_2_batch_1_bestParams"
DEVICE_NUM=0
#DATA_PATH="/shared/3/projects/benlitterer/podcastData/processed/floydMonth/floydMonthDataClean.jsonl"
DATA_PATH="/shared/3/projects/benlitterer/podcastData/processed/mayJune/mayJuneData.jsonl" 
OUT_PATH="/shared/3/projects/benlitterer/podcastData/hostIdentification/hostGuestPredictions/MJ_LongPredictions.jsonl"
python3 $APPLY_MODEL $STATE_PATH $DEVICE_NUM $DATA_PATH $OUT_PATH

AGG_OUT_PATH="/shared/3/projects/benlitterer/podcastData/hostIdentification/hostGuestPredictions/MJ_AggPredictions.jsonl"
python3 $AGG_ANNOTS $OUT_PATH $AGG_OUT_PATH

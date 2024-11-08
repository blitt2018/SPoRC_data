
OUT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
PLOT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
SCRIPT_PATH=/home/blitt/projects/podcasts/analysis/hostIdentification/robertaClassification/TrainRobMACEAnnots.py

:'
#TODO: we should probably actually only test on batch1? Since batch2 is not representative 
#out stem, plot stem, epochs, batch size, learning rate, f1 type, dropout, data subset, aggregation strategy
#learning rate
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-7 micro 0.0 full none 

#batch size
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 8 2e-5 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 16 2e-5 micro 0.0 full none 

#dropout
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.25 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.5 full none 

#dropout with slower learning rate 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.25 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-7 micro 0.5 full none 

#training with only batch1 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1 none 

#trying out the batch1 then batch2 approach 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1_2 none 

#aggregating with the modal annotation 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full mode 

#more epochs
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 full none 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 4 4 2e-5 micro 0.0 full none 
'

#just retrain a model that appeared to work well quickly to get one to use 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full none 
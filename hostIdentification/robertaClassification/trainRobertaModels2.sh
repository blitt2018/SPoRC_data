
OUT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
PLOT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
SCRIPT_PATH=/home/blitt/projects/podcasts/analysis/hostIdentification/robertaClassification/TrainRobMACEAnnots.py


#out stem, plot stem, epochs, batch size, learning rate, f1 type, dropout, data subset, aggregation strategy
#learning rate
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full prob
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full prob
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 full prob

#more epochs 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 full prob
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-6 micro 0.0 full prob

#training with only batch1 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 1 4 2e-5 micro 0.0 batch1 prob 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1 prob 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1 prob 

#trying out the batch1 then batch2 approach 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1_2 prob
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1_2 prob

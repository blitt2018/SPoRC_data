
OUT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
PLOT_STEM=/shared/3/projects/benlitterer/podcastData/hostGuestModels/fullData/output
SCRIPT_PATH=/home/blitt/projects/podcasts/analysis/hostIdentification/robertaClassification/TrainRobMACEAnnots.py

:'
#out stem, plot stem, epochs, batch size, learning rate, f1 type, dropout, data subset, aggregation strategy, evaluation set

#learning rate, train on batch1 then 2, eval on batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-4 micro 0.0 batch1_2 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1_2 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 batch1_2 none batch1

#learning rate just training on batch1, eval on batch
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-4 micro 0.0 batch1 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 batch1 none batch1

#full data for training, but randomized, eval on batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-4 micro 0.0 full none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 full none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-6 micro 0.0 full none batch1

#more epochs, eval on batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-4 micro 0.0 batch1_2 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1_2 none batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-6 micro 0.0 batch1_2 none batch1


#different aggregation methods, 2 and 3 epochs 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1_2 prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1_2 mode batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1_2 prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 2 4 2e-5 micro 0.0 batch1_2 mode batch1

#some different learning rates, small adjustments though 
#and one with many epochs just in case 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 1e-5 micro 0.0 batch1_2 prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 8e-5 micro 0.0 batch1_2 prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 5 4 2e-6 micro 0.0 batch1_2 prob batch1


#what Dallas requested 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 full prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 full mode batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1 prob batch1
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 3 4 2e-5 micro 0.0 batch1 mode batch1 
'

#just to get model weights 
python3 $SCRIPT_PATH $OUT_STEM $PLOT_STEM 5 4 2e-6 micro 0.0 batch1_2 prob batch1
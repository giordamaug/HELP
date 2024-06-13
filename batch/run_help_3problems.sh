#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/HELPpy/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/data
TARGET=$HOME/$ROOTDIR/HELP/data
SCOREDIR=$HOME/$ROOTDIR/HELP/data4rev/scoresveryfinal
LOGDIR=$HOME/$ROOTDIR/HELP/data4rev/logsveryfinal
TISSUE=Kidney
PROB1=EvsAE
PROB2=EvsSNE
PROB3=AEvsSNE
VOTERS1="-v 3"
VOTERS2="-v 10"
VOTERS3="-v 4"
ALIASES1="{'E':1, 'aNE': 0}"
ALIASES2="{'sNE': 0, 'E': 1}"
ALIASES3="{'aE':1, 'sNE': 0"
EXCLABELS1="-X sNE"
EXCLABELS2="-X aE"
EXCLABELS3="-X E"
LABELFILE=${TISSUE}_${METHOD}.csv

echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -c 1 5 1 -n std -ba $VOTERS1 $ESTIMATORS $LR $EXCLABELS1 $ALIASES1 -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB1}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB1}_bioccn2v.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -c 1 5 1 -n std -ba $VOTERS2 $ESTIMATORS $LR  $EXCLABELS2 $ALIASES2 -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB2}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB2}_bioccn2v.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -c 1 5 1 -n std -ba $VOTERS3 $ESTIMATORS $LR  $EXCLABELS3 $ALIASES3 -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB3}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB3}_bioccn2v.csv" | bash

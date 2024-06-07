#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/HELPpy/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/data
TARGET=$HOME/$ROOTDIR/HELP/data
SCOREDIR=$HOME/$ROOTDIR/HELP/data4rev/scoresfinal
LOGDIR=$HOME/$ROOTDIR/HELP/data4rev/logsfinal
TISSUE=Human
PROB=EvsNE
#ALIASES="{'aE':0, 'sNE': 0, 'E': 1}"
ALIASES="{'NE':0, 'E': 1}"
#LR="-lr 0.1" # 0.0945 from optuna
VOTERS="-v 13"
ESTIMATORS="-e 200"
#LABELFILE=${TISSUE}_${METHOD}.csv
LABELFILE=PanTissue_group_HELP.csv

echo "running CC"
echo "$CMD $PRG -i $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_cc.csv" | bash
echo "running BIO"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR  -j -1 -B -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bio.csv" | bash
echo "running N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_n2v.csv" | bash
#echo "running BIO+N2V"
#echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv" | bash
echo "running BIO+N2V+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
#echo "running BIO+N2V+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biocc.csv" | bash

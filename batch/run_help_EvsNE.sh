#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/HELPpy/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/data
TARGET=$HOME/$ROOTDIR/HELP/data
SCOREDIR=$HOME/$ROOTDIR/HELP/data4rev/scores
LOGDIR=$HOME/$ROOTDIR/HELP/data4rev/logs
TISSUE=Lung
PROB=EvsNE
ALIASES="{'aE':'NE', 'sNE': 'NE'}"
EXCLABELS=""
LR="-lr 0.35"
VOTERS="-v 11"
ESTIMATORS="-e 190"
LABELFILE=${TISSUE}_${METHOD}.csv

#echo "running BIO+CC"
#echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -ba $VOTERS $ESTIMATORS $LR $EXCLABELS -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biocc.csv" | bash
echo "running BIO"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR $EXCLABELS  -j -1 -B -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bio.csv" | bash
echo "running N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR $EXCLABELS -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_n2v.csv" | bash
echo "running BIO+N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR $EXCLABELS -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv" | bash
echo "running BIO+N2V+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR $EXCLABELS -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccn2v.csv" | bash

#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology/HELP_branches/v2.0
ROOTDIR2=PLOS_CompBiology/HELP_branches/main
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR2/HELP/data
TARGET=$HOME/$ROOTDIR2/HELP/data
SCOREDIR=$HOME/$ROOTDIR/HELP/scoresgm
LOGDIR=$HOME/$ROOTDIR/HELP/logsgm
TISSUE=Brain
PROB=EvsNE
ALIASES="{'aE':0, 'sNE': 0, 'E': 1}"
#ALIASES="{'NE':0, 'E': 1}"
LR="-lr 0.09" # 0.0945 from optuna
VOTERS="-v 12"
ESTIMATORS="-e 140"
LABELFILE=${TISSUE}_${METHOD}.csv
#LABELFILE=PanTissue_group_HELP.csv

echo "running CC"
echo "$CMD $PRG -i $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -pl 1 -c $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_cc.csv" | bash
#echo "running CCred"
#echo "$CMD $PRG -i $DATA/${TISSUE}_CCcfs_reduce_min5imp.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_cc_reduce.csv" | bash
echo "running BIO"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -pl 1 -c $VOTERS $ESTIMATORS $LR  -j -1 -B -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bio.csv" | bash
echo "running N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -pl 1 -c $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_n2v.csv" | bash
#echo "running BIO+N2V"
#echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv" | bash
#echo "running CCBeder"
#echo "$CMD $PRG -i $DATA/${TISSUE}_CCBeder.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_ccbeder.csv" | bash
echo "running BIO+N2V+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -pl 1 -c $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
echo "running BIO+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std -pl 1 -c $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biocc.csv" | bash
#echo "running BIO+N2V+CCcfs*"
#echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs_reduce_min5imp.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -n std $VOTERS $ESTIMATORS $LR -j -1 -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccredn2v.csv" | bash
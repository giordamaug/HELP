#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology
CMD=$HOME/miniconda3/bin/python
PRG=$HOME/$ROOTDIR/HELP/help/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/help/data
TARGET=$HOME/$ROOTDIR/HELP/help/data
SCOREDIR=$HOME/$ROOTDIR/HELP/help/scores
LOGDIR=$HOME/$ROOTDIR/HELP/help/logs
TISSUE=Kidney
PROB=EvsNE
ALIASES="{'aE':'NE', 'sNE': 'NE'}"
EXCLABELS=null
LABELFILE=${METHOD}_${TISSUE}_3class.csv

echo "running BIO"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS  -j -1 -P -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bio.csv
echo "running CC"
$CMD $PRG -i $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_cc.csv
echo "running CCBeder"
$CMD $PRG -i $DATA/${TISSUE}_CCBeder.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_ccbeder.csv
echo "running BPBeder"
$CMD $PRG -i $DATA/${TISSUE}_BPBeder.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j 1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bpbeder.csv
echo "running N2V"
$CMD $PRG -i $DATA/${TISSUE}_N2V.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_n2v.csv
echo "running BIO+CCcfs"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biocc.csv
echo "running BIO+CCBeder"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCBeder.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccbeder.csv
echo "running BIO+BPBeder"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_BPBeder.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j 1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biobpbeder.csv
echo "running BIO+N2V"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv
echo "running BIO+N2V+CCcfs"
$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A $ALIASES -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv
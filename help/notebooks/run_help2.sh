#!/bin/bash
METHOD=HELP
ROOTDIR=PLOS_CompBiology
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/help/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/help/data
TARGET=$HOME/$ROOTDIR/HELP/help/data
SCOREDIR=$HOME/$ROOTDIR/HELP/help/scoresfinal
LOGDIR=$HOME/$ROOTDIR/HELP/help/logsfinal
TISSUE=Lung
PROB=AEvsSNE
#ALIASES="{'aE':'NE', 'sNE': 'NE'}"
ALIASES="{}"
EXCLABELS="-X E"
LABELFILE=${TISSUE}_${METHOD}.csv

echo "running BIO"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS  -j -1 -P -B -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bio.csv" | bash
#echo "running CC"
#echo "$CMD $PRG -i $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_cc.csv" | bash
#echo "running CCBeder"
#echo "$CMD $PRG -i $DATA/${TISSUE}_CCBeder.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_ccbeder.csv" | bash
#echo "running BPBeder"
#echo "$CMD $PRG -i $DATA/${TISSUE}_BPBeder.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j 1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bpbeder.csv" | bash
echo "running N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_n2v.csv" | bash
echo "running BIO+CC"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biocc.csv" | bash
echo "running BIO+CCBeder"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCBeder.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccbeder.csv" | bash
echo "running BIO+N2V"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bion2v.csv" | bash
echo "running BIO+N2V+CCcfs"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
echo "running BIO+BPBeder"
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_BPBeder.csv -l $TARGET/${LABELFILE} -A \"$ALIASES\" -ba -sf 4 $EXCLABELS -j 1 -P -B  -o $LOGDIR/log_batch_${METHOD}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_${METHOD}_${TISSUE}_${PROB}_biobpbeder.csv" | bash

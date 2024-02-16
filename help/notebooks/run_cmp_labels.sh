#!/bin/bash
METHOD1=HELP
METHOD2=AdAM
METHOD3=FiPer
METHOD4=OGEE
ROOTDIR=PLOS_CompBiology
ENV=envs/mytorch
CMD=$HOME/miniconda3/$ENV/bin/python
PRG=$HOME/$ROOTDIR/HELP/help/notebooks/EG_prediction.py
DATA=$HOME/$ROOTDIR/HELP/help/data
TARGET=$HOME/$ROOTDIR/HELP/help/data
SCOREDIR=$HOME/$ROOTDIR/HELP/help/scores
LOGDIR=$HOME/$ROOTDIR/HELP/help/logs
TISSUE=Kidney
PROB=EvsNE
ALIASES="{'aE':'NE', 'sNE': 'NE'}"
EXCLABELS=null
LABELFILE1=${TISSUE}_${METHOD1}_shared.csv
LABELFILE2=${TISSUE}_${METHOD2}_shared.csv
LABELFILE3=${TISSUE}_${METHOD3}_shared.csv
LABELFILE4=${TISSUE}_${METHOD4}_shared.csv

echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE1} -A \"$ALIASES\" -S -X $EXCLABELS  -j -1 -P -B -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD1}_${TISSUE}_${PROB}_bio.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE2} -A \"$ALIASES\" -S -X $EXCLABELS  -j -1 -P -B -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD2}_${TISSUE}_${PROB}_bio.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE3} -A \"$ALIASES\" -S -X $EXCLABELS  -j -1 -P -B -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD3}_${TISSUE}_${PROB}_bio.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv -l $TARGET/${LABELFILE4} -A \"$ALIASES\" -S -X $EXCLABELS  -j -1 -P -B -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD4}_${TISSUE}_${PROB}_bio.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE1} -A \"$ALIASES\" -S -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD1}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE2} -A \"$ALIASES\" -S -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD2}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE3} -A \"$ALIASES\" -S -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD3}_${TISSUE}_${PROB}_bioccn2v.csv" | bash
echo "$CMD $PRG -i $DATA/${TISSUE}_BIO.csv $DATA/${TISSUE}_CCcfs.csv $DATA/${TISSUE}_EmbN2V_128.csv -l $TARGET/${LABELFILE4} -A \"$ALIASES\" -S -X $EXCLABELS -j -1 -P -B  -o $LOGDIR/log_batch_cmp_${METHOD1}_${METHOD2}_${METHOD3}_${METHOD4}_${TISSUE}_${PROB}.txt -s $SCOREDIR/score_cmp_${METHOD4}_${TISSUE}_${PROB}_bioccn2v.csv" | bash

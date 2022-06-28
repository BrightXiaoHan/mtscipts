LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh


# export wmt test data from 2017-2020
srcfile=$TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG}.${SRCLANG}
tgtfile=$TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG}.${TGTLANG}
touch $srcfile $tgtfile
for year in {17..20}; do
  sacrebleu -t wmt$year -l ${SRCLANG}-${TGTLANG} --echo src >> $srcfile
  sacrebleu -t wmt$year -l ${SRCLANG}-${TGTLANG} --echo ref >> $tgtfile
done

# tokenize
for file in $srcfile $tgtfile;do
  spm_encode --model=${DATA_DIR}/spm.model --output_format=piece < $file > ${file}.spm
done

fairseq-preprocess --source-lang ${SRCLANG}_RWSK --target-lang ${TGTLANG}_RWSK \
    --trainpref $TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG} \
    --srcdict $TRAIN_DIR/data/fairseq.vocab \
    --tgtdict $TRAIN_DIR/data/fairseq.vocab \
    --destdir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews \
    --workers 5

lang_pairs="${SRCLANG}_RWSK-${TGTLANG}_RWSK"
path_2_data=$TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews
lang_list=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
pretrained_model=$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}/average.pt

fairseq-train $path_2_data \
  --finetune-from-model $pretrained_model \
  --arch transformer_vaswani_wmt_en_de_big \
  --task translation_multi_simple_epoch \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr 3e-05 --max-epoch 10 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 12000 \
  --log-format simple --log-interval 1 \
  --save-dir "$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}-finetune" \
  > $TRAIN_DIR/finetune.${SRCLANG}-${TGTLANG}.log 2>&1

LANMT_TAINER_DIR=$(dirname $0)/../../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))/..
source $SOURCE_ROOT/env_check.sh

FINETUNE_EPOCH=${1:-10}
DOMAIN=${2:-RWSK}

srcfile=$TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG}.${SRCLANG}
# export wmt test data from 2017-2020
tgtfile=$TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG}.${TGTLANG}
touch $srcfile $tgtfile
for year in {17..20}; do
  sacrebleu -t wmt$year -l ${SRCLANG}-${TGTLANG} --echo src >> $srcfile
  sacrebleu -t wmt$year -l ${SRCLANG}-${TGTLANG} --echo ref >> $tgtfile
done

src_wmttest=$TRAIN_DIR/data/wmttest.${SRCLANG}-${TGTLANG}.${SRCLANG}
tgt_wmttest=$TRAIN_DIR/data/wmttest.${SRCLANG}-${TGTLANG}.${TGTLANG}
sacrebleu -t wmt21 -l ${SRCLANG}-${TGTLANG} --echo src > $src_wmttest
sacrebleu -t wmt21 -l ${SRCLANG}-${TGTLANG} --echo ref:A > $tgt_wmttest


# tokenize
for file in $srcfile $tgtfile $src_wmttest $tgt_wmttest;do
  spm_encode --model=${DATA_DIR}/spm.model --output_format=piece < $file > ${file}_${DOMAIN}
done

fairseq-preprocess --source-lang ${SRCLANG}_RWSK --target-lang ${TGTLANG}_RWSK \
    --trainpref $TRAIN_DIR/data/wmtdev.${SRCLANG}-${TGTLANG} \
    --validpref $TRAIN_DIR/data/wmttest.${SRCLANG}-${TGTLANG} \
    --srcdict $TRAIN_DIR/data/fairseq.vocab \
    --tgtdict $TRAIN_DIR/data/fairseq.vocab \
    --destdir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews \
    --workers 5

lang_pairs="${SRCLANG}_RWSK-${TGTLANG}_RWSK"
path_2_data=$TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews
lang_list=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
pretrained_model=$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}/average.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train $path_2_data \
  --finetune-from-model $pretrained_model \
  --arch transformer_vaswani_wmt_en_de_big \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --fp16 \
  --share-all-embeddings \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 3e-06 --max-epoch $FINETUNE_EPOCH \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 12000 \
  --log-format simple --log-interval 20 \
  --save-dir "$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}-finetune"

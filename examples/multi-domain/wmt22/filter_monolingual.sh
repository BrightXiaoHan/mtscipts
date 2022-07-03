LANMT_TAINER_DIR=$(dirname $0)/../../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))/..
source $SOURCE_ROOT/env_check.sh

# if [ -f $DATA_DIR/wmtnews/lid.176.bin ];then
#   echo "$DATA_DIR/wmtnews/lid.176.bin exists, skip download."
# else
#   wget -O $data_dir/wmtnews/lid.176.bin --no-check-certificate \
#     https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# fi

# cd $DATA_DER/wmtnews
# opusfilter $SOURCE_ROOT/opus_config_codeswitch.yml
# cd -

# if [ -f $DATA_DIR/monolingual.${TGTLANG}.${TGTLANG}_RWSK ];then
#   echo "$DATA_DIR/monolingual.${TGTLANG}.${TGTLANG}_RWSK exists, skip download."
# else
#   spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmtnews/${TGTLANG}.rules > monolingual.${TGTLANG}.${TGTLANG}_RWSK
# fi

mkdir -p $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/monolingual-${TGTLANG}
fairseq-preprocess --source-lang ${TGTLANG}_RWSK --target-lang ${SRCLANG}_RWSK \
    --testpref $DATA_DIR/wmtnews/monolingual.${TGTLANG} \
    --srcdict $TRAIN_DIR/data/fairseq.vocab \
    --tgtdict $TRAIN_DIR/data/fairseq.vocab \
    --destdir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/monolingual-${TGTLANG} \
    --workers 30 > $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/monolingual-${TGTLANG}/preprocess.log 2>&1

lang_pairs="${TGTLANG}_RWSK-${SRCLANG}_RWSK"
lang_list=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
pretrained_model=$TRAIN_DIR/checkpoints-${TGTLANG}-${SRCLANG}/average.pt

fairseq-generate $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/monolingual-${TGTLANG} \
  --path $pretrained_model --fp16 \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang ${TGTLANG}_RWSK \
  --target-lang ${SRCLANG}_RWSK \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --max-tokens 9192 \
  --beam 1 --sampling --sampling-topk 3 \
  --decoder-langtok \
  --skip-invalid-size-inputs-valid-test \
  > $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/monolingual-${TGTLANG}/btdata.out

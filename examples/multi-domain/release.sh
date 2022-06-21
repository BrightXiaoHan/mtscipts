LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

OUTPUT_DIR=$1
EVERAGE_LAST_N_EPOCHS=$2

# everage checkpoints
python $FAIRSEQ_PATH/scripts/average_checkpoints.py \
    --inputs $TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG} \
    --output $TRAIN_DIR/checkpoints-${srclang}-${tgtlang}/average.pt \
    --num-epoch-checkpoints $everage_last_n_epochs

if [ $SRC_LANG == "en" -a $TGTLANG == "zh" ]; then
  libtrans_release -m $TRAIN_DIR/checkpoints-${srclang}-${tgtlang}/average.pt \
      --data_dir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard0 \
      -o $OUTPUT_DIR -t fairseq -q float16  \
      --translate_lang_pairs ${SRCLANG}-${TGTLANG} \
      --term_mask_methods CodeSwitch \
      --term_mask_methods Url \
      --term_mask_wrapper "【】" \
      --postprocess_pipeline chinesepunc \
      --preprocess_pipeline normalize \
      --preprocess_pipeline uppercase \
      --preprocess_pipeline detruecase \
      --translate_prefix "lang_domain" \
      --translate_domain ""


elif [ $SRC_LANG == "zh" -a $TGTLANG == "en" ]; then
  libtrans_release -m $TRAIN_DIR/checkpoints-${srclang}-${tgtlang}/average.pt \
      --data_dir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard0 \
      -o $OUTPUT_DIR -t fairseq -q float16  \
      --translate_lang_pairs ${SRCLANG}-${TGTLANG} \
      --term_mask_methods CodeSwitch \
      --term_mask_methods Url \
      --term_mask_max_len 100

else
  echo "Language pair $SRCLANG-$TGTLANG not supported."
  exit 1
fi

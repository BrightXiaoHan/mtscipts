LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

OUTPUT_DIR=$1
EVERAGE_LAST_N_EPOCHS=$2
CHECKPOINT_DIR=$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}-finetune

# everage checkpoints
python $FAIRSEQ_PATH/scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --output $CHECKPOINT_DIR/average.pt \
    --num-epoch-checkpoints $EVERAGE_LAST_N_EPOCHS

iter_json_key()
{
  json_file=$1
  cat $json_file | python3 -c \
    "import sys, json; [print(key) for key in json.load(sys.stdin).keys()]"
}

translate_domain=""
for domain in $(iter_json_key $SOURCE_ROOT/domain_mapping.json); do
  translate_domain+="--translate_domain $domain "
done


if [[ "$SRCLANG" == "en" ]] && [[ "$TGTLANG" == "zh" ]]; then
  libtranslate release -m $CHECKPOINT_DIR/average.pt \
      --data_dir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews \
      -o $OUTPUT_DIR -t fairseq  \
      --max_sent_len 150 \
      --translate_beam_size 5 \
      --translate_lang_pairs ${SRCLANG}-${TGTLANG} \
      --term_mask_methods CodeSwitch \
      --term_mask_methods Url \
      --term_mask_wrapper "【】" \
      --postprocess_pipeline chinesepunc \
      --preprocess_pipeline normalize \
      --preprocess_pipeline uppercase \
      --preprocess_pipeline detruecase \
      --translate_target_prefix "lang_domain" \
      --translate_domain_prefix_mapping $SOURCE_ROOT/domain_mapping.json \
      --translate_language_prefix_mapping $SOURCE_ROOT/lang_mapping.json \
      --translate_model_device cuda:0 \
      --spm_model $DATA_DIR/spm.model \
      $translate_domain

elif [[ "$SRCLANG" == "zh" ]] && [[ "$TGTLANG" == "en" ]]; then
  libtranslate release -m $CHECKPOINT_DIR/checkpoint1.pt \
      --data_dir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/WMTNews \
      -o $OUTPUT_DIR -t fairseq \
      --max_sent_len 150 \
      --translate_beam_size 5 \
      --translate_lang_pairs ${SRCLANG}-${TGTLANG} \
      --term_mask_methods CodeSwitch \
      --term_mask_methods Url \
      --term_mask_wrapper "[]" \
      --term_mask_file $SOURCE_ROOT/countries.tsv \
      --term_mask_chinese_name_exclude_file $SOURCE_ROOT/names.tsv \
      --postprocess_pipeline latinpunc \
      --preprocess_pipeline normalize \
      --translate_target_prefix "lang_domain" \
      --translate_domain_prefix_mapping $SOURCE_ROOT/domain_mapping.json \
      --translate_language_prefix_mapping $SOURCE_ROOT/lang_mapping.json \
      --translate_model_device cuda:0 \
      --spm_model $DATA_DIR/spm.model \
      $translate_domain
else
  echo "Language pair $SRCLANG-$TGTLANG not supported."
  exit 1
fi

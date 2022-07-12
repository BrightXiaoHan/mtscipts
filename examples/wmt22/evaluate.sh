LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh


model=${TRAIN_DIR}/checkpoints/averaged.pt
result_dir=${TRAIN_DIR}/results
mkdir -p ${result_dir}

# # everage checkpoints
# python ${FAIRSEQ_PATH}/scripts/average_checkpoints.py \
#   --inputs ${TRAIN_DIR}/checkpoints \
#   --output $model \
#   --num-epoch-checkpoints 5

lang_pairs=""
for pair in $(cat $LANGPAIRS_FILE); do
  lang_pairs="$pair,${lang_pairs}"
done
lang_pairs=${lang_pairs::-1}
lang_list="WMT22-LANGS.txt"

for pair in $(cat $LANGPAIRS_FILE);do
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)

  fairseq-generate $TRAIN_DIR/data-bin/shard0 \
    --path ${TRAIN_DIR}/checkpoints/checkpoint_last.pt \
    --task translation_multi_simple_epoch \
    --gen-subset test \
    --source-lang $srclang \
    --target-lang $tgtlang \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 64 \
    --beam 3 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > $result_dir/generated.${srclang}_${tgtlang}.txt
  exit 1
done

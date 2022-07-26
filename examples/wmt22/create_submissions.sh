LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh


for pair in $(cat $LANGPAIRS_FILE); do
  echo "Processing $pair..."
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)

  sacrebleu -t wmt22 -l $pair --echo src \
    | sacremoses normalize \
    | spm_encode --model $DATA_DIR/spm.model --output_format=piece \
    >  $TRAIN_DIR/test.$pair.$srclang

  databin=$TRAIN_DIR/data-bin/shard0
  mkdir -p $databin
  cp $TRAIN_DIR/data-bin/dict.src.txt $databin/dict.$srclang.txt
  cp $TRAIN_DIR/data-bin/dict.tgt.txt $databin/dict.$tgtlang.txt
  fairseq-preprocess --source-lang $srclang --target-lang $tgtlang \
      --testpref $TRAIN_DIR/test.$pair --only-source \
      --srcdict $databin/dict.$srclang.txt \
      --tgtdict $databin/dict.$tgtlang.txt \
      --destdir $databin > /dev/null 2>&1
done

WMT_RESULT_DIR=${TRAIN_DIR}/wmt22_submissions
mkdir -p $WMT_RESULT_DIR

lang_pairs=""
for pair in $(cat $LANGPAIRS_FILE); do
  lang_pairs="$pair,${lang_pairs}"
done
lang_pairs=${lang_pairs::-1}
lang_list="${SOURCE_ROOT}/WMT22-LANGS.txt"

for pair in $(cat $LANGPAIRS_FILE);do
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)

  echo "Running generation for $srclang-$tgtlang."
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
    --beam 5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    | grep ^H | LC_ALL=C sort -V | cut -f3- \
    > $WMT_RESULT_DIR/${srclang}_${tgtlang}.hyp.txt
done

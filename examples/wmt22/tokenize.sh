LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

SPM_MODEL_PREFIX=$DATA_DIR/spm

function train_spm(){
  if [ -f $SPM_MODEL_PREFIX.model ]; then
    echo "SPM model exists, skip training"
    exit 0
  fi
  params=""
  for lang in $(cat $SOURCE_ROOT/WMT22-LANGS.txt);
  do
    echo "Merge and shuffle $lang files."
    allfiles=$(find $DATA_DIR -type f -name "${lang}.final")
    cat $allfiles > $TRAIN_DIR/${lang}.all
    $TERASHUF_PATH/terashuf < $TRAIN_DIR/${lang}.all > $TRAIN_DIR/${lang}.shuf 2>/dev/null
    rm $TRAIN_DIR/${lang}.all
    params="-c $TRAIN_DIR/${lang}.shuf -l $lang $params"
  done

  echo "Training sentencepiece model..."
  PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
  python ${LANMT_TAINER_DIR}/lanmttrainer/preprocessor/train_multilingual_spm_model.py \
    $params \
    --vocab-size 100000 \
    --input-sentence-size 20000000 \
    --sample-temprature 2 \
    -o $SPM_MODEL_PREFIX
  spm_export_vocab --model $SPM_MODEL_PREFIX.model --output $SPM_MODEL_PREFIX.vocab
}

function tokenize_all(){
  for file in $(find $DATA_DIR $BT_DATA_DIR -type f -name "*.final");
  do
    if [ -f $file.spm ];then
      echo "Skip tokenizing $file, $file.spm exists."
      continue
    fi
    spm_encode --model=$SPM_MODEL_PREFIX.model --output_format=piece < $file > $file.spm &
    pwait 20
  done

  for file in $(find $TEST_DATA_DIR -type f ! -name "*.spm");
  do
    spm_encode --model=$SPM_MODEL_PREFIX.model --output_format=piece < $file > $file.spm
  done
}

MODE=$1
if [ "$MODE" == "train" ]; then
  train_spm
elif [ "$MODE" == "tokenize" ]; then
  tokenize_all
else
  echo "Usage: $0 [train|tokenize]"
  exit 1
fi

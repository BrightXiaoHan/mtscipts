LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
MODE=${1:-train}

if [ "$MODE" == "train" ]; then
  echo "Training sentencepiece model..."
  PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
  python ${LANMT_TAINER_DIR}/lanmttrainer/preprocessor/train_multilingual_spm_model.py \
    -c $DATA_DIR/train.en -l en \
    -c $DATA_DIR/train.zh -l zh \
    --vocab-size 32000 \
    --input-sentence-size 20000000 \
    -o $DATA_DIR/spm

  echo "Done."
else
  echo "Tokenizing all data..."

  for file in $(find $DATA_DIR -name "train.*" ! -name "*.spm") $(find $DATA_DIR -name "test.*" ! -name "*.spm");
  do
    echo "Tokenizing $file..."
    if [ -f "$file.spm" ]; then
      echo "Skipping $file.spm already exists."
      continue
    fi

    if [[ $file =~ "train" ]]; then
      samplesize=5
    else
      samplesize=1
    fi

    if [[ $file =~ "en" ]]; then
      alpha=0.2
    else
      lang=0.1
    fi

    PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
    python ${LANMT_TAINER_DIR}/lanmttrainer/preprocessor/spm_encode_with_subword_regularization.py \
      --model $DATA_DIR/spm.model \
      --input-file $file \
      --output-suffix "spm" \
      --sample-size $samplesize \
      --alpha $alpha &
    pwait 10
  done
fi

if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

LANMT_TAINER_DIR=$(dirname $0)/../..

echo "Training sentencepiece model..."
PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
python ${LANMT_TAINER_DIR}/lanmttrainer/preprocessor/train_multilingual_spm_model.py \
  -c $DATA_DIR/train.en -l en \
  -c $DATA_DIR/train.zh -l zh \
  --vocab-size 32000 \
  --input-sentence-size 20000000 \
  -o $DATA_DIR/spm

echo "Done."


echo "Tokenizing all data..."
spm_encode

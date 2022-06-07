if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi
LANMT_TAINER_DIR=$(dirname $0)/../..

echo "Joining all the English files together..."
cat $(find $DATA_DIR -name 'train.en.final') > $DATA_DIR/merged.en
echo "Deduplicating..."
sort -u $DATA_DIR/merged.en > $DATA_DIR/merged.en.uniq
echo "Shuffling..."
shuf $DATA_DIR/merged.en.uniq > $DATA_DIR/merged.en.uniq.shuf

echo "Training sentencepiece model..."
PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
python ${LANMT_TAINER_DIR}/lanmttrainer/preprocessor/train_multilingual_spm_model.py \
  -c $DATA_DIR/wmt22-zhen/train.zh.final -l en \
  -c $DATA_DIR/wmt22-csen/train.cs.final -l cs \
  -c $DATA_DIR/wmt22-deen/train.de.final -l de \
  -c $DATA_DIR/wmt22-hren/train.hr.final -l hr \
  -c $DATA_DIR/wmt22-jaen/train.ja.final -l ja \
  -c $DATA_DIR/wmt22-ruen/train.ru.final -l ru \
  -c $DATA_DIR/wmt22-uken/train.uk.final -l uk \
  -c $DATA_DIR/wmt22-zhen/train.zh.final -l zh \
  --vocab-size 100000 \
  --input-sentence-size 20000000 \
  --sample-temprature 2 \
  -o $DATA_DIR/spm

echo "Done."

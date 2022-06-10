if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

echo "Merging all training data togather"
cat $(find $DATA_DIR/data -name "train.*.en") > $DATA_DIR/train.en
cat $(find $DATA_DIR/data -name "train.*" ! -name "train.*.en") > $DATA_DIR/train.ot

echo "Building vocabulary"
# Prapre for vocabularies
fairseq-preprocess --source-lang ot --target-lang en \
    --trainpref $DATA_DIR/train \
    --dict-only \
    --joined-dictionary \
    --thresholdtgt 10 --thresholdsrc 10 \
    --destdir $DATA_DIR/data-bin \
    --workers 30


echo "Shared Datasets"
LANMT_TAINER_DIR=$(dirname $0)/../..

PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
python ${LANMT_TAINER_DIR}/lanmttrainer/trainer/fairseq/shared_large_datasets.py \
  $DATA_DIR/data \
  --lang-pairs "cs-en,de-en,hr-en,ja-en,uk-en,zh-en,ru-en" \
  --epoch_sents 50000000 \
  --trainpref train


EPOCH_SIZE=50000000

# Shared large datasets into chunks
total_sentences=$(wc -l < $DATA_DIR/train.en)
total_epoch=$[$total_sentences / 50000000]
echo $total_epoch

for i in $(seq 0 $total_epoch); do
  echo "Sharding $i/$total_epoch..."
  for lang in cs de hr ja uk zh ru;do
    fairseq-preprocess --source-lang ${lang} --target-lang en \
        --trainpref $DATA_DIR/data/part${i}.train.${lang}-en \
        --validpref $DATA_DIR/data/dev.${lang}-en \
        --srcdict $DATA_DIR/data-bin/dict.ot.txt \
        --tgtdict $DATA_DIR/data-bin/dict.en.txt \
        --destdir $DATA_DIR/data-bin/shard${i} \
        --workers 30
  done
done
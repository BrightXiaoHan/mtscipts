LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

echo "Merging all training data togather"
source_file=$TRAIN_DIR/train.src
target_file=$TRAIN_DIR/train.tgt
rm -f $source_file $target_file
for pair in $(cat $LANGPAIRS_FILE); do
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)
  cat $BT_DATA_DIR/$pair/$srclang.final.spm >> $source_file
  cat $BT_DATA_DIR/$pair/$tgtlang.final.spm >> $target_file
  if [ ! -d "$DATA_DIR/$pair" ]; then
    pair=${tgtlang}-${srclang}
  fi
  cat $DATA_DIR/$pair/$srclang.final.spm >> $source_file
  cat $DATA_DIR/$pair/$tgtlang.final.spm >> $target_file
done


echo "Building vocabulary"
# Prapre for vocabularies
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref $TRAIN_DIR/train \
    --dict-only \
    --joined-dictionary \
    --thresholdtgt 10 --thresholdsrc 10 \
    --destdir $TRAIN_DIR/data-bin \
    --workers 40

LANGPAIRS=""
for pair in $(cat $LANGPAIRS_FILE); do
  echo "Merge and shuf $pair data."
  LANGPAIRS="$pair,$LANGPAIRS"
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)
  cp $TEST_DATA_DIR/$pair/$srclang.spm $TRAIN_DIR/valid.$pair.$srclang
  cp $TEST_DATA_DIR/$pair/$tgtlang.spm $TRAIN_DIR/valid.$pair.$tgtlang
  prefix=$TRAIN_DIR/train.$pair
  srcfile=$prefix.$srclang
  tgtfile=$prefix.$tgtlang
  cat $BT_DATA_DIR/$pair/$srclang.final.spm > $srcfile
  cat $BT_DATA_DIR/$pair/$tgtlang.final.spm > $tgtfile
  if [ ! -d "$DATA_DIR/$pair" ]; then
    pair=${tgtlang}-${srclang}
  fi
  cat $DATA_DIR/$pair/$srclang.final.spm >> $srcfile
  cat $DATA_DIR/$pair/$tgtlang.final.spm >> $tgtfile

  paste $srcfile $tgtfile > $prefix.merge
  rm $srcfile $tgtfile
  $TERASHUF_PATH/terashuf < $prefix.merge > $prefix.shuf 2>/dev/null
  rm $prefix.merge
  cut -f 1 $prefix.shuf > $srcfile
  cut -f 2 $prefix.shuf > $tgtfile
  rm $prefix.shuf
done
# remove final , from LANGPAIRS
LANGPAIRS=${LANGPAIRS::-1}

EPOCH_SIZE=50000000

PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
python ${LANMT_TAINER_DIR}/lanmttrainer/trainer/fairseq/shared_large_datasets.py \
  $TRAIN_DIR \
  --lang-pairs $LANGPAIRS \
  --epoch_sents $EPOCH_SIZE \
  --trainpref train \
  --suffix-pair

# Shared large datasets into chunks
num_pairs=$(cat $LANGPAIRS_FILE | wc -l)
num_parts=$(find $TRAIN_DIR -type f -name part*.train.* | wc -l)
total_epoch=$[$num_parts / $num_pairs / 2]
echo "Total shared: $total_epoch."

for i in $(seq 0 $total_epoch); do
  echo "Sharding $i/$total_epoch..."
  for pair in $(cat $LANGPAIRS_FILE); do
    srclang=$(echo $pair | cut -d '-' -f 1)
    tgtlang=$(echo $pair | cut -d '-' -f 2)
    databin=$TRAIN_DIR/data-bin/shard${i}
    mkdir -p $databin
    cp $TRAIN_DIR/data-bin/dict.src.txt $databin/dict.$srclang.txt
    cp $TRAIN_DIR/data-bin/dict.tgt.txt $databin/dict.$tgtlang.txt
    fairseq-preprocess --source-lang $srclang --target-lang $tgtlang \
        --trainpref $TRAIN_DIR/part${i}.train.$pair \
        --validpref $TRAIN_DIR/valid.$pair \
        --srcdict $databin/dict.$srclang.txt \
        --tgtdict $databin/dict.$tgtlang.txt \
        --destdir $TRAIN_DIR/data-bin/shard${i} \
        --workers 40 > $databin/preprocess.$pair.log 2>&1
  done
done

LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))

EPOCH_SIZE=$1

SRCLANG=en
TGTLANG=zh

DATA_BIN_DIR="$DATA_DIR/data-bin/$SRCLANG-$TGTLANG"

echo "Convert sentencepiece vocab to fairseq vocab"
spm_export_vocab --model $DATA_DIR/spm.model --output $DATA_DIR/spm.vocab

# general data
if [ ! -d "$TRAIN_DIR/data" ]; then
  mkdir -p $TRAIN_DIR/data
fi
cut -f1 $DATA_DIR/spm.vocab | tail -n +4 | sed "s/$/ 100/g" > $TRAIN_DIR/data/fairseq.vocab

echo "Copy training corpus to $TRAIN_DIR/data"
function merge_and_shuf(){
  folder=$1
  domain=$2
  cat $folder/train.$SRCLANG.spm $folder/train.$SRCLANG.codeswitch.spm $folder/train.$TGTLANG.codeswitch.spm > $TRAIN_DIR/data/train.${SRCLANG}_$domain.sort
  cat $folder/train.$TGTLANG.spm > $TRAIN_DIR/data/train.${TGTLANG}_$domain.sort
  awk 'NR%5==1 || NR%5==2' $folder/train.$TGTLANG.spm >> $TRAIN_DIR/data/train.${TGTLANG}_$domain.sort
  awk 'NR%5==3 || NR%5==4' $folder/train.$TGTLANG.spm >> $TRAIN_DIR/data/train.${TGTLANG}_$domain.sort

  paste $TRAIN_DIR/data/train.${SRCLANG}_$domain.sort $TRAIN_DIR/data/train.${TGTLANG}_$domain.sort | shuf \
    | awk -F "\t" -v out1="$TRAIN_DIR/data/train.${SRCLANG}_$domain" -v out2="$TRAIN_DIR/data/train.${TGTLANG}_$domain" '{ print $1 > out1 ; print $2 > out2 }'
  rm $TRAIN_DIR/data/train.${SRCLANG}_$domain.sort $TRAIN_DIR/data/train.${TGTLANG}_$domain.sort

  awk -v train="$TRAIN_DIR/data/valid.${SRCLANG}_$domain" -v test="$TRAIN_DIR/data/test.${SRCLANG}_$domain" \
    '{if(rand()<0.5) {print > train} else {print > test}}' $folder/test.$SRCLANG
  awk -v train="$TRAIN_DIR/data/valid.${TGTLANG}_$domain" -v test="$TRAIN_DIR/data/test.${TGTLANG}_$domain" \
    '{if(rand()<0.5) {print > train} else {print > test}}' $folder/test.$TGTLANG
}

for folder in $(find $DATA_DIR/online -maxdepth 2 -mindepth 2 -type d);do
  echo "Copy $folder"
  domainname=$(basename $folder)
  domain=$(cat $SOURCE_ROOT/DOMAIN_LIST.txt | grep "$domainname " | awk '{print $2}')
  merge_and_shuf $folder $domain
done

echo "Copy $DATA_DIR"
merge_and_shuf $DATA_DIR GEN

echo "Shared Datasets"
langpairs=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
  '{printf "%s_%s-%s_%s",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)
# join by ','
langpairs_str=$(echo $langpairs | paste -sd ',')

PYTHONPATH=${LANMT_TAINER_DIR}:${PYTHONPATH} \
python ${LANMT_TAINER_DIR}/lanmttrainer/trainer/fairseq/shared_large_datasets.py \
  $TRAIN_DIR/data \
  --lang-pairs $langpairs_str \
  --epoch_sents $EPOCH_SIZE \
  --trainpref train > /dev/null 2>&1

# Shared large datasets into chunks
total_sentences=$(find $TRAIN_DIR/data -type f -name "train.${SRCLANG}*" -exec cat {} + | wc -l)
total_epoch=$[$total_sentences / $EPOCH_SIZE]
echo "Toatal epochs: $total_epoch"

for i in $(seq 0 $total_epoch); do
  echo "Sharding $i/$total_epoch..."
  for pair in $langpairs;do
    # split by '-'
    srclang=$(echo $pair | cut -d'-' -f1)
    tgtlang=$(echo $pair | cut -d'-' -f2)
    fairseq-preprocess --source-lang $srclang --target-lang tgtlang \
        --trainpref $TRAIN_DIR/data/part${i}.train \
        --validpref $TRAIN_DIR/data/valid \
        --srcdict $TRAIN_DIR/data/fairseq.vocab \
        --tgtdict $TRAIN_DIR/data/fairseq.vocab \
        --destdir $TRAIN_DIR/data-bin/shard${i} \
        --workers 30 > $TRAIN_DIR/data-bin-$SRCLANG-$TGTLANG/shard${i}/preprocess.log 2>&1
  done
done

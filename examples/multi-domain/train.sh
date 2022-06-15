LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))

MODE=$1
NUM_SHARDS=$2
SRCLANG=en
TGTLANG=zh

if [ ! -z "$MODE" || ! -z "$NUM_SHARDS" ]; then
  echo "Usage: $0 <mode> <num_shards>"
fi

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

PATH_TO_DATA="$TRAIN_DIR/data-bin/shard0"
for i in `seq 1 $[${NUM_SHARDS}-1]`; do
    PATH_TO_DATA="${PATH_TO_DATA}:$TRAIN_DIR/data-bin/shard${i}"
done

echo "PATH_TO_DATA: $PATH_TO_DATA"

langlist=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
  '{printf "%s%s-%s%s",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)
# join by ','
langpairs=$(echo $langpairs | paste -sd ',')

langlist_file=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
echo $(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
  '{printf "%s_%s\n%s_%s",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt) > $langlist_file

fairseq-train "$PATH_TO_DATA" \
  --arch transformer_vaswani_wmt_en_de_big \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --fp16 \
  --share-all-embeddings \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 0.001 --warmup-updates 10000 --warmup-init-lr '1e-07' --max-update 1000000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 10000 --update-freq 16 \
  --save-interval 1 --keep-interval-updates 10 --keep-interval-updates-pattern 10 \
  --tensorboard-logdir "$TRAIN_DIR/tensorboard-${SRCLANG}-${TGTLANG}" \
  --seed 221 --log-format simple --log-interval 100 \
  --save-dir "$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}"

LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

# Shared large datasets into chunks
num_pairs=$(cat $LANGPAIRS_FILE | wc -l)
num_parts=$(find $TRAIN_DIR -type f -name part*.train.* | wc -l)
total_epoch=$[$num_parts / $num_pairs / 2]
echo "Total shared: $total_epoch."

PATH_TO_DATA="$TRAIN_DIR/data-bin/shard0"
for i in `seq 1 $[${total_epoch}-1]`; do
    PATH_TO_DATA="${PATH_TO_DATA}:$TRAIN_DIR/data-bin/shard${i}"
done

echo "PATH_TO_DATA: $PATH_TO_DATA"

lang_list="$SOURCE_ROOT/WMT22-LANGS.txt"
lang_pairs=""
for pair in $(cat $LANGPAIRS_FILE); do
  lang_pairs="$pair,${lang_pairs}"
done
lang_pairs=${lang_pairs::-1}
echo "lang_pairs: $lang_pairs"

fairseq-train "$PATH_TO_DATA" \
  --arch transformer_wmt_en_de_big_t2t --layernorm-embedding \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --encoder-langtok "src" \
  --decoder-langtok \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --fp16 \
  --share-all-embeddings \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 10000 --warmup-init-lr '1e-07' --max-update 1000000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 10000 --update-freq 16 \
  --save-interval 1 \
  --tensorboard-logdir "$TRAIN_DIR/tensorboard" \
  --seed 221 --log-format simple --log-interval 100 \
  --save-dir "$TRAIN_DIR/checkpoints"
  # > $TRAIN_DIR/train.log 2>&1

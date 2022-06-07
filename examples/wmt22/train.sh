NUM_SHARDS=$1

if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

PATH_TO_DATA="$DATA_DIR/data-bin/shard0"
for i in `seq 1 $[${NUM_SHARDS}-1]`; do
    PATH_TO_DATA="${PATH_TO_DATA}:$DATA_DIR/data-bin/shard${i}"
done

echo "PATH_TO_DATA: $PATH_TO_DATA"

lang_list="WMT22-LANGS.txt"
lang_pairs="cs-en,de-en,hr-en,ja-en,ru-en,uk-en,zh-en"

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
  --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 10000 --warmup-init-lr '1e-07' --max-update 1000000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0001 \
  --max-tokens 10000 --update-freq 8 \
  --save-interval 1 --keep-interval-updates 10 --keep-interval-updates-pattern 10 \
  --tensorboard-logdir "$DATA_DIR/tensorboard" \
  --seed 221 --log-format simple --log-interval 100 \
  --save-dir "$DATA_DIR/checkpoints"

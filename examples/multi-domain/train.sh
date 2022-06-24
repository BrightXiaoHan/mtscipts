LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

MODE=$1
NUM_SHARDS=$2

if [ -z "$MODE" ] || [ -z "$NUM_SHARDS" ]; then
  echo "Usage: $0 <mode> <num_shards>"
  exit 1
fi

if [ "$MODE" == "preprocess" ]; then

  echo "Preprocessing data..."
  langpairs=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
    '{printf "%s_%s-%s_%s ",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)

  for i in $(seq 0 $[$NUM_SHARDS-1]); do
    echo "Sharding $[$i+1]/$NUM_SHARDS..."
    rm -f $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard${i}/preprocess.log
    for pair in $langpairs;do
      # split by '-'
      srclang=$(echo $pair | cut -d'-' -f1)
      tgtlang=$(echo $pair | cut -d'-' -f2)
      mkdir -p $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard${i}
      fairseq-preprocess --source-lang $srclang --target-lang $tgtlang \
          --trainpref $TRAIN_DIR/data/part${i}.train \
          --validpref $TRAIN_DIR/data/valid \
          --testpref $TRAIN_DIR/data/test \
          --srcdict $TRAIN_DIR/data/fairseq.vocab \
          --tgtdict $TRAIN_DIR/data/fairseq.vocab \
          --destdir $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard${i} \
          --workers 30 >> $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard${i}/preprocess.log 2>&1
      if [ $? -ne 0 ]; then
        echo "Error: training failed. Please check the log file."
        exit 1
      fi
    done
  done

elif [ "$MODE" == "train" ]; then
  PATH_TO_DATA="$TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard0"
  for i in `seq 1 $[${NUM_SHARDS}-1]`; do
      PATH_TO_DATA="${PATH_TO_DATA}:$TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard${i}"
  done

  echo "PATH_TO_DATA: $PATH_TO_DATA"

  langpairs=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
    '{printf "%s_%s-%s_%s,",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)
  # join by ','
  langpairs=${langpairs::-1}
  langlist_file=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
  awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
    '{printf "%s_%s\n%s_%s\n",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt > $langlist_file

  echo "langlist_file: $langlist_file"
  echo "langpairs: $langpairs"
  echo "training..."
  echo "You can find logs in $TRAIN_DIR/train.log"

  fairseq-train "$PATH_TO_DATA" \
    --arch transformer_vaswani_wmt_en_de_big \
    --task translation_multi_simple_epoch \
    --sampling-method "temperature" \
    --sampling-temperature 1.5 \
    --decoder-langtok \
    --lang-dict "$langlist_file" \
    --lang-pairs "$langpairs" \
    --fp16 \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --max-update 100000 --patience 5 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 12000 --update-freq 16 \
    --tensorboard-logdir "$TRAIN_DIR/tensorboard-${SRCLANG}-${TGTLANG}" \
    --seed 221 --log-format simple --log-interval 10 \
    --save-dir "$TRAIN_DIR/checkpoints-${SRCLANG}-${TGTLANG}" \
    > $TRAIN_DIR/train.${SRCLANG}-${TGTLANG}.log 2>&1

  # if error, exit
  if [ $? -ne 0 ]; then
    echo "Error: training failed. Please check the log file."
    exit 1
  fi
  echo done
elif [ "$MODE" == "eval" ]; then
  model=${TRAIN_DIR}/checkpoints-$SRCLANG-$TGTLANG/checkpoint_best.pt
  space_seperated_langpairs=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
    '{printf "%s_%s-%s_%s ",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)
  langlist_file=$TRAIN_DIR/DOMAIN_LIST.${SRCLANG}-${TGTLANG}.txt
  comma_seperated_langpairs=$(awk -v srclang=$SRCLANG -v tgtlang=$TGTLANG \
    '{printf "%s_%s-%s_%s,",srclang,$2,tgtlang,$2}' $SOURCE_ROOT/DOMAIN_LIST.txt)
  comma_seperated_langpairs=${comma_seperated_langpairs::-1}
  
  for pair in $space_seperated_langpairs;do
    echo "Evaluating $pair..."
    srclang=$(echo $pair | cut -d'-' -f1)
    tgtlang=$(echo $pair | cut -d'-' -f2)
    fairseq-generate $TRAIN_DIR/data-bin-${SRCLANG}-${TGTLANG}/shard0 \
      --path $model \
      --task translation_multi_simple_epoch \
      --gen-subset test \
      --source-lang $srclang \
      --target-lang $tgtlang \
      --lang-dict "$langlist_file" \
      --lang-pairs "$comma_seperated_langpairs" \
      --sacrebleu --remove-bpe 'sentencepiece'\
      --batch-size 128 \
      --beam 5 \
      --decoder-langtok \
      > $TRAIN_DIR/generated.${srclang}-${tgtlang}.txt 2>&1

    if [ $? -ne 0 ]; then
      echo "Error: evaluation failed. Please check the log file."
      exit 1
    fi
  done
else
  echo "Usage: $0 <mode> <num_shards>"
fi

if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

if [ -z "${FAIRSEQ_SOURCE_ROOT}" ]; then
  echo "Please set FAIRSEQ_SOURCE_ROOT in the environment variables."
  exit 1
fi

# everage checkpoints
python ${FAIRSEQ_SOURCE_ROOT}/scripts/average_checkpoints.py \
  --inputs ${DATA_DIR}/checkpoints \
  --output ${DATA_DIR}/checkpoints/averaged.pt \
  --num-epoch-checkpoints 5

model=${DATA_DIR}/checkpoints/averaged.pt
lang_pairs="cs-en,de-en,hr-en,ja-en,ru-en,uk-en,zh-en"
lang_list="WMT22-LANGS.txt"


for source_lang in cs de hr ja uk zh ru;do
  target_lang=en
  fairseq-generate $DATA_DIR/data-bin/shard0 \
    --path ${DATA_DIR}/checkpoints/averaged.pt \
    --task translation_multi_simple_epoch \
    --gen-subset valid \
    --source-lang $source_lang \
    --target-lang $target_lang \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" \
    --sacrebleu --remove-bpe 'sentencepiece'\
    --batch-size 128 \
    --beam 5 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --lang-dict "$lang_list" \
    --lang-pairs "$lang_pairs" > $DATA_DIR/generated.${source_lang}_${target_lang}.txt
done

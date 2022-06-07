if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

if [ ! -d $DATA_DIR/data ]; then
    mkdir $DATA_DIR/data
fi

echo "Tokenizing training data..."
for lang in cs de hr ja ru uk zh; do
  echo "Tokenizing $lang corpus..."
  spm_encode --model=$DATA_DIR/spm.model --output_format=sample_piece < $DATA_DIR/wmt22-${lang}en/train.${lang}.final > $DATA_DIR/data/train.${lang}-en.${lang} &
  spm_encode --model=$DATA_DIR/spm.model --output_format=sample_piece < $DATA_DIR/wmt22-${lang}en/train.en.final > $DATA_DIR/data/train.${lang}-en.en &
done
wait
echo "Done."

echo "Tokenizing dev data..."
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-csen/dev.ces > $DATA_DIR/data/dev.cs-en.cs
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-csen/dev.eng > $DATA_DIR/data/dev.cs-en.en

spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-deen/dev.deu > $DATA_DIR/data/dev.de-en.de
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-deen/dev.eng > $DATA_DIR/data/dev.de-en.en

spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-jaen/dev.jpn > $DATA_DIR/data/dev.ja-en.ja
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-jaen/dev.eng > $DATA_DIR/data/dev.ja-en.en

spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-ruen/dev.rus > $DATA_DIR/data/dev.ru-en.ru
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-ruen/dev.eng > $DATA_DIR/data/dev.ru-en.en

spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-zhen/dev.zho > $DATA_DIR/data/dev.zh-en.zh
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/wmt22-zhen/dev.eng > $DATA_DIR/data/dev.zh-en.en

# floresdata
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/flores101/dev/hrv.dev > $DATA_DIR/data/dev.hr-en.hr
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/flores101/dev/eng.dev > $DATA_DIR/data/dev.hr-en.en

spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/flores101/dev/ukr.dev > $DATA_DIR/data/dev.uk-en.uk
spm_encode --model=$DATA_DIR/spm.model --output_format=piece < $DATA_DIR/flores101/dev/eng.dev > $DATA_DIR/data/dev.uk-en.en

echo "Done."


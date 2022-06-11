LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh

SOURCE_ROOT=$(dirname ${BASH_SOURCE[0]})
# 移除旧的数据
rm -f $DATA_DIR/raw.en $DATA_DIR/raw.zh
rm -f $DATA_DIR/test.en $DATA_DIR/test.zh

for folder in $(find $ROOT/datasets/online -maxdepth 2 -mindepth 2 -type d );
do
  cat ${folder}/train.en >> $DATA_DIR/raw.en
  cat ${folder}/train.zh >> $DATA_DIR/raw.zh
  cat ${folder}/test.en >> $DATA_DIR/test.en
  cat ${folder}/test.zh >> $DATA_DIR/test.zh
done


for folder in $(find $ROOT/datasets/opensource -maxdepth 1 -mindepth 1 -type d );
do
  cat ${folder}/en >> $DATA_DIR/raw.en
  cat ${folder}/zh >> $DATA_DIR/raw.zh
done


cd $DATA_DIR

opusfilter $SOURCE_ROOT/opus_config_opensource.yml

rm train.en train.zh
ln zh.rules train.zh
ln en.rules train.en

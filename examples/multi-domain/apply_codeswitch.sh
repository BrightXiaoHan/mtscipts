LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

function run() {
  folder=$1
  cd $folder
  opusfilter $SOURCE_ROOT/opus_config_codeswitch.yml --single 1 > /dev/null 2>&1

  $EFLOMAL_PATH/align.py \
    -s train.zh.tok -t train.en.tok \
    --priors $DATA_DIR/align.priors \
    --model 3 \
    -f zh-en.fwd \
    -r zh-en.rev

  opusfilter $SOURCE_ROOT/opus_config_codeswitch.yml --single 2 > /dev/null 2>&1

}

for folder in $(find $DATA_DIR/online -maxdepth 2 -mindepth 2 -type d ) $DATA_DIR;
do
  echo "Processing $folder"
  run $folder &
  pwait 10
done


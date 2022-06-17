LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

function run() {
  folder=$1
  cd $folder
  opusfilter $SOURCE_ROOT/opus_config_online.yml
  
  if [ $? -eq 0 ]; then
    echo "Successfully filtered $folder"
  else
    echo "Failed to filter $folder"
  fi
  
  paste zh.rules en.rules | sort -u > train.all
  paste test.zh test.en | sort -u > test.all

  comm -3 train.all test.all | shuf > train.uniq
  cut -f1 train.uniq > train.zh
  cut -f2 train.uniq > train.en

  rm $(find . -type f -name "*.*" ! -name "train.zh" ! -name "train.en" ! -name "test.zh" ! -name "test.en")
}

for folder in $(find $DATA_DIR/online -maxdepth 2 -mindepth 2 -type d );
do
  # if test.zh and test.en exist, continue
  if [ -f $folder/test.zh ] && [ -f $folder/test.en ]; then
    echo "$folder skip"
    continue
  fi
  run $folder &
  pwait 10  # max 10 jobs running at the same time
done
wait

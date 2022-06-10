if [ -z "${DATA_DIR}" ]; theR
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

SOURCE_ROOT=$(dirname "${BASH_SOURCE[0]}")

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

function run() {
  folder=$1
  cd $folder
  opusfilter $SOURCE_ROOt/opus_config_online.yml
  
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
  pwait 20  # max 10 jobs running at the same time
done

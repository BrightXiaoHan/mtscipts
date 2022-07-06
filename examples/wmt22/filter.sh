LANMT_TAINER_DIR=$(dirname $0)/../..
source $LANMT_TAINER_DIR/lanmttrainer/shell_utils.sh
SOURCE_ROOT=$(realpath $(dirname $0))
source $SOURCE_ROOT/env_check.sh

if [ ! -f "$SOURCE_ROOT/lid.176.bin" ]; then
  echo "Downloading fasttext language detection model..."
  wget -q -O $SOURCE_ROOT/lid.176.bin --no-check-certificate \
      https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
fi

for pair in $(cat $LANGPAIRS_FILE); do
  srclang=$(echo $pair | cut -d '-' -f 1)
  tgtlang=$(echo $pair | cut -d '-' -f 2)
  
  opus_config_file=$SOURCE_ROOT/assets/opus_${srclang}_${tgtlang}.yml
  if [ ! -f $opus_config_file ]; then
    opus_config_file=$SOURCE_ROOT/assets/opus_${tgtlang}_${srclang}.yml > /dev/null 2>&1
  fi

  echo "Clearing $pair..."
  cd $DATA_DIR/$pair
  if [ ! -f "lid.176.bin" ]; then
    ln $SOURCE_ROOT/lid.176.bin .
  fi
  opusfilter $opus_config_file
  cd -

  echo "Clearing $pair BT data..."
  cd $BT_DATA_DIR/$pair
  if [ ! -f "lid.176.bin" ]; then
    ln $SOURCE_ROOT/lid.176.bin .
  fi
  opusfilter $opus_config_file
  cd -

done

echo "Done."

SOURCE_ROOT=$(realpath $(dirname $0))

if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

if [ -z "${BT_DATA_DIR}" ]; then
  echo "Please set BT_DATA_DIR in the environment variables."
  exit 1
fi

if [ -z "${TEST_DATA_DIR}" ]; then
  echo "Please set TEST_DATA_DIR in the environment variables."
  exit 1
fi

if [ -z "${TRAIN_DIR}" ]; then
  echo "Please set TRAIN_DIR in the environment variables."
  exit 1
fi

# create TRAIN_DIR if it doesn't exist
if [ ! -d "${TRAIN_DIR}" ]; then
  mkdir -p ${TRAIN_DIR}
fi

if [ ! -z "${MODE}" ]; then
  echo "Please set MODE in the environment variables. Avaliable MODE is X2EN or EN2X"
fi

if [ "$MODE" == "X2EN" ]; then
  LANGPAIRS_FILE=$SOURCE_ROOT/WMT22-LANGPAIRS-X2EN.txt
elif [ "$MODE" == "EN2X" ]; then
  LANGPAIRS_FILE=$SOURCE_ROOT/WMT22-LANGPAIRS-EN2X.txt
else
  echo "Please set MODE in the environment variables. Avaliable MODE is X2EN or EN2X instead of $MODE."
  exit 1
fi

# check terashuf
if [ -z "$TERASHUF_PATH" -o -z "$MEMORY" ];then
  echo "Please set TERASHUF_PATH and MEMORY in the environment variables."
  exit 1
fi

# check fairseq
if [ -z "$FAIRSEQ_PATH" ]; then
  echo "Please set FAIRSEQ_PATH in the environment variables."
  exit 1
fi

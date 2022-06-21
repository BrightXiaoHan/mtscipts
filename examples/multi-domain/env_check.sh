# check if env DATA_DIR exists
if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

if [ -z "${TRAIN_DIR}" ]; then
  echo "Please set TRAIN_DIR in the environment variables."
fi

# create TRAIN_DIR if it doesn't exist
if [ ! -d "${TRAIN_DIR}" ]; then
  mkdir -p ${TRAIN_DIR}
fi

# check SRCLANG and TGTLANG
if [ -z "${SRCLANG}" ]; then
  echo "Please set SRCLANG in the environment variables."
  exit 1
fi

if [ -z "${TGTLANG}" ]; then
  echo "Please set TGTLANG in the environment variables."
  exit 1
fi

# check eflomal
if [ -z "$EFLOMAL_PATH" ]; then
  echo "Please set EFLOMAL_PATH in the environment variables."
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

# Bash scripts util functions
# Usage:
# source lanmttrainer/shell_utils.sh

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

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}


# Bash scripts util functions
# Usage:
# source lanmttrainer/shell_utils.sh

# check if env DATA_DIR exists
if [ -z "${DATA_DIR}" ]; then
  echo "Please set DATA_DIR in the environment variables."
  exit 1
fi

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}


# Bash scripts util functions
# Usage:
# source lanmttrainer/shell_utils.sh

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

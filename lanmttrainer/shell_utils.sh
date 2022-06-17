# Bash scripts util functions
# Usage:
# source lanmttrainer/shell_utils.sh

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


#!/usr/bin/env bash
# Stops the overnight_train.sh run started with `nohup ./overnight_train.sh &`.
#
# overnight_train.sh and everything it spawns (the rsl_rl train.py subprocess,
# any sub-shells) share the script's process group, since nohup without job
# control keeps the default PGID == the launching bash's PID. So instead of
# hunting down individual PIDs (train.py can outlive a plain `pkill
# overnight_train.sh` since that only matches the shell, not its python
# child), we signal the whole process group at once: SIGTERM first for a
# clean shutdown, then SIGKILL anything still standing after a grace period.

set -uo pipefail

PIDS=$(pgrep -f 'overnight_train\.sh' || true)

if [[ -z "$PIDS" ]]; then
    echo "No overnight_train.sh process found. Nothing to stop."
    exit 0
fi

# Collect the unique process groups these PIDs belong to.
PGIDS=$(ps -o pgid= -p $PIDS | tr -d ' ' | sort -u)

if [[ -z "$PGIDS" ]]; then
    echo "Found overnight_train.sh PIDs ($PIDS) but couldn't resolve their process group. Aborting."
    exit 1
fi

for PGID in $PGIDS; do
    echo "Sending SIGTERM to process group $PGID (overnight_train.sh and its children)..."
    kill -TERM -- "-$PGID" 2>/dev/null || true
done

echo "Waiting up to 15s for graceful shutdown..."
for i in $(seq 1 15); do
    sleep 1
    STILL_ALIVE=""
    for PGID in $PGIDS; do
        if pgrep -g "$PGID" >/dev/null 2>&1; then
            STILL_ALIVE="1"
        fi
    done
    [[ -z "$STILL_ALIVE" ]] && break
done

for PGID in $PGIDS; do
    if pgrep -g "$PGID" >/dev/null 2>&1; then
        echo "Process group $PGID still alive, sending SIGKILL..."
        kill -KILL -- "-$PGID" 2>/dev/null || true
    fi
done

sleep 1
REMAINING=$(pgrep -f 'overnight_train\.sh|scripts/rsl_rl/train\.py' || true)
if [[ -n "$REMAINING" ]]; then
    echo "Warning: some processes may still be running:"
    ps -o pid,ppid,pgid,cmd -p $REMAINING
    exit 1
else
    echo "overnight_train.sh and all its child processes have been stopped."
fi

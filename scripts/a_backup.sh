#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE_HOST="${BAND2TAB_SSH_HOST:-aiexp}"
DEST="${BAND2TAB_BACKUP_DIR:-$PWD/backups/azure}"
mkdir -p "$DEST/runs" "$DEST/reports" "$DEST/logs" "$DEST/data" "$DEST/models"
for folder in runs reports logs; do
  rsync -az --partial --delay-updates --exclude '*.tmp' --exclude '*.part' \
    -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
    "$REMOTE_HOST:music_ai_exp/$folder/" "$DEST/$folder/"
done
rsync -az --partial --delay-updates \
  --include '*/' --include '*.pt' --include '*.json' --exclude '*' \
  -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  "$REMOTE_HOST:music_ai_exp/models/" "$DEST/models/"
if [ "${BAND2TAB_BACKUP_DATA:-0}" = 1 ]; then
  rsync -az --partial --delay-updates --exclude '*.tmp' --exclude '*.part' \
    -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
    "$REMOTE_HOST:music_ai_exp/data/" "$DEST/data/"
  # The audit includes the pinned model source/base checkpoint. Keep this
  # specific public-model folder, never the unrelated private credential folder.
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$REMOTE_HOST" \
      'test -d music_ai_exp/vendor/yourmt3'; then
    mkdir -p "$DEST/vendor/yourmt3"
    rsync -az --partial --delay-updates --exclude '__pycache__/' --exclude '*.pyc' \
      --exclude '*.tmp' --exclude '*.part' \
      -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
      "$REMOTE_HOST:music_ai_exp/vendor/yourmt3/" "$DEST/vendor/yourmt3/"
  fi
else
  rsync -az --partial --delay-updates \
    --include '*/' --include '*.json' --exclude '*' \
  -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  "$REMOTE_HOST:music_ai_exp/data/" "$DEST/data/"
fi
printf 'Backup completed: %s\n' "$DEST"

#!/usr/bin/env bash
# scripts/quit_jupyter.sh
# Stop JupyterLab by removing the proxy container and the DemARK devcontainer.

set -euo pipefail

PROXY_NAME="demark-proxy"
WORKSPACE_DIR="$(pwd)"

echo "🛑  Stopping proxy container (if running) …"
docker rm -f "$PROXY_NAME" >/dev/null 2>&1 && echo "Removed $PROXY_NAME" || echo "No proxy container running"

echo "🛑  Stopping devcontainer(s) for this workspace …"
CID_LIST=$(docker ps -q --filter "label=devcontainer.local_folder=$WORKSPACE_DIR")
if [[ -z "$CID_LIST" ]]; then
  echo "No devcontainer currently running."
else
  docker stop $CID_LIST
  echo "Stopped container(s): $CID_LIST (use docker rm to delete permanently)"
fi

echo "✅  All stopped." 
#!/usr/bin/env bash
# scripts/start_jupyter.sh
# One-stop script to build / reuse the DemARK devcontainer,
# start JupyterLab inside it, and expose it on http://localhost:8888
# Works on macOS, Windows (WSL/PowerShell with bash), and Linux.

set -euo pipefail

PORT="8888"          # Host and container port for JupyterLab
ENV_NAME="DemARK"    # Conda environment inside the container
WORKSPACE_DIR="$(pwd)"
PROXY_NAME="demark-proxy"

# --- Helpers ---------------------------------------------------------
_exists() { command -v "$1" >/dev/null 2>&1; }
_die()    { echo "❌  $*" >&2; exit 1; }

# --- 1. Ensure devcontainer CLI is available ------------------------
if ! _exists devcontainer; then
  _die "devcontainer CLI not found. Install with: npm i -g @devcontainers/cli"
fi

# --- 2. Create or reuse the devcontainer ----------------------------
 echo "🔧  Creating (or reusing) devcontainer …"
 # If a container for this workspace already exists, devcontainer CLI will
 # simply start it; it only rebuilds when configuration/image has changed.
 devcontainer up --workspace-folder "$WORKSPACE_DIR"

# --- 3. Get the container ID ---------------------------------------
CID=$(docker ps \
        --filter "label=devcontainer.local_folder=$WORKSPACE_DIR" \
        -q | head -n1)
[[ -n "$CID" ]] || _die "Cannot locate devcontainer after startup."
 echo "✅  Using container $CID"

# --- 4. Start JupyterLab inside the container if not already running-
 echo "🚀  Ensuring JupyterLab is running on port $PORT …"
  if docker exec "$CID" bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME && pgrep -fl \"jupyter.*lab.*--port=$PORT\" >/dev/null"; then
    echo "JupyterLab already running."
  else
    docker exec "$CID" bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME && nohup jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true >/tmp/jlab.log 2>&1 &"
  fi

# --- 5. Determine the container's internal IP -----------------------
CIP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CID")
 echo "🌐  Container IP: $CIP"

# --- 6. Start/replace the lightweight socat proxy -------------------
 echo "🔄  Exposing http://localhost:$PORT → $CIP:$PORT"
 docker rm -f "$PROXY_NAME" >/dev/null 2>&1 || true
 docker run -d --rm --name "$PROXY_NAME" -p $PORT:$PORT \
   alpine/socat tcp-listen:$PORT,reuseaddr,fork tcp:$CIP:$PORT >/dev/null

# --- 7. Done ---------------------------------------------------------
 printf "\n🎉  JupyterLab is ready!  Open: http://localhost:%s\n\n" "$PORT"
# --- 8. Attempt to open default browser ---------------------------------
URL="http://localhost:$PORT"
if _exists open; then
  # macOS
  open "$URL" &>/dev/null &
elif _exists xdg-open; then
  # Linux desktop environments
  xdg-open "$URL" &>/dev/null &
elif _exists wslview; then
  # WSL with wslview installed
  wslview "$URL" &>/dev/null &
elif [[ "${OS-}" == "Windows_NT" ]]; then
  # Git Bash / MSYS / Cygwin on Windows
  (cmd.exe /c start "$URL" 2>/dev/null || powershell.exe -Command "Start-Process '$URL'" 2>/dev/null || explorer.exe "$URL" 2>/dev/null || true) &
else
  echo "(Could not auto-launch browser; please open the URL manually.)"
fi

 echo "To stop:   docker rm -f $PROXY_NAME $CID" 
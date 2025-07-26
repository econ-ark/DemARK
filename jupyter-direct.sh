#!/bin/bash

echo "🚀 DemARK Jupyter Lab - Direct Access"
echo "====================================="
echo ""

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Find an available port
PORT=8890
while ! check_port $PORT; do
    echo "Port $PORT is in use, trying next port..."
    PORT=$((PORT + 1))
done

echo "✅ Using port $PORT"
echo ""

# Stop any existing container with this name
echo "🔄 Stopping any existing Jupyter container..."
docker stop demark-jupyter-direct 2>/dev/null || true
docker rm demark-jupyter-direct 2>/dev/null || true

# Start a new container with proper port forwarding
echo "🌐 Starting Jupyter Lab with direct port forwarding..."
docker run -d \
  --name demark-jupyter-direct \
  --shm-size=8g \
  -p $PORT:$PORT \
  -v "$(pwd):/workspaces/DemARK" \
  -w /workspaces/DemARK \
  --entrypoint bash \
  vsc-demark-b628be7871d06175eec02b2c798294043c784d07e4276962ea59033dc2611b52-features \
  -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate DemARK
jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true
"

# Wait for container to start
sleep 5

# Check if container is running
if docker ps | grep -q "demark-jupyter-direct"; then
    echo ""
    echo "✅ Jupyter Lab is running with direct port forwarding!"
    echo ""
    echo "🌐 ACCESS INSTRUCTIONS:"
    echo "========================"
    echo ""
    echo "Jupyter Lab is now accessible at:"
    echo ""
    echo "  http://localhost:$PORT"
    echo ""
    echo "📁 Available Notebooks:"
    echo "========================"
    echo "- test_jupyter.ipynb (Math and HARK test notebook)"
    echo "- notebooks/ (All DemARK example notebooks)"
    echo ""
    echo "🎉 Ready to use! Open http://localhost:$PORT in your browser."
    echo ""
    echo "🛑 To stop: docker stop demark-jupyter-direct"
else
    echo "❌ Failed to start Jupyter Lab"
    echo "Please check Docker and try again"
fi 
#!/bin/bash

echo "🚀 DemARK Jupyter Lab - Simple Launcher"
echo "======================================="
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

# Check if container is running
if ! docker ps | grep -q "ee2179993603"; then
    echo "❌ DemARK container is not running. Starting it..."
    devcontainer up --workspace-folder . > /dev/null 2>&1
    sleep 5
fi

echo "✅ Container is running"
echo ""

# Kill any existing Jupyter processes
echo "🔄 Stopping any existing Jupyter processes..."
docker exec -it ee2179993603 pkill -f jupyter-lab 2>/dev/null || true
sleep 2

# Start Jupyter Lab in the container
echo "🌐 Starting Jupyter Lab on port $PORT..."
docker exec -it ee2179993603 bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate DemARK
jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true
" &

# Wait for Jupyter to start
sleep 5

# Check if Jupyter is running
if docker exec -it ee2179993603 pgrep -f jupyter-lab > /dev/null 2>&1; then
    echo ""
    echo "✅ Jupyter Lab is running in container on port $PORT"
    echo ""
    echo "🌐 ACCESS INSTRUCTIONS:"
    echo "========================"
    echo ""
    echo "Jupyter Lab is running in the container."
    echo ""
    echo "To access it from your browser, run this command in a NEW terminal:"
    echo ""
    echo "  socat TCP-LISTEN:$PORT,fork TCP:localhost:$PORT"
    echo ""
    echo "Then open your browser to: http://localhost:$PORT"
    echo ""
    echo "📁 Available Notebooks:"
    echo "========================"
    echo "- test_jupyter.ipynb (Math and HARK test notebook)"
    echo "- notebooks/ (All DemARK example notebooks)"
    echo ""
    echo "🎉 Ready to use!"
    echo ""
    echo "💡 Tip: If you don't have socat installed, run: brew install socat"
else
    echo "❌ Failed to start Jupyter Lab"
    echo "Please check the container logs for errors"
fi 
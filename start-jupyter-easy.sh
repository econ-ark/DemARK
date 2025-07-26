#!/bin/bash

echo "🚀 DemARK Jupyter Lab - Easy Access"
echo "==================================="
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
echo "🌐 Starting Jupyter Lab..."
docker exec -it ee2179993603 bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate DemARK
jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true
" &

# Wait for Jupyter to start
sleep 5

# Check if Jupyter is running
if docker exec -it ee2179993603 pgrep -f jupyter-lab > /dev/null 2>&1; then
    echo ""
    echo "✅ Jupyter Lab is running in container"
    echo ""
    echo "🌐 ACCESS INSTRUCTIONS:"
    echo "========================"
    echo ""
    echo "Jupyter Lab is running in the container on port 8890."
    echo ""
    echo "To access it from your browser, run this command in a NEW terminal:"
    echo ""
    echo "  socat TCP-LISTEN:8890,fork TCP:localhost:8890"
    echo ""
    echo "Then open your browser to: http://localhost:8890"
    echo ""
    echo "📁 Available Notebooks:"
    echo "========================"
    echo "- test_jupyter.ipynb (Math and HARK test notebook)"
    echo "- notebooks/ (All DemARK example notebooks)"
    echo ""
    echo "🎉 Ready to use!"
else
    echo "❌ Failed to start Jupyter Lab"
    echo "Please check the container logs for errors"
fi 
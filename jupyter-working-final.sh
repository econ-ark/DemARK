#!/bin/bash

echo "🚀 DemARK Jupyter Lab - Working Final Solution"
echo "=============================================="
echo ""

# Stop any existing containers
echo "🔄 Stopping any existing containers..."
docker stop demark-jupyter-working 2>/dev/null || true
docker rm demark-jupyter-working 2>/dev/null || true

# Start a new container with proper port forwarding
echo "🌐 Starting container with port forwarding..."
docker run -d \
  --name demark-jupyter-working \
  --shm-size=8g \
  -p 8890:8890 \
  -v "$(pwd):/workspaces/DemARK" \
  -w /workspaces/DemARK \
  --entrypoint bash \
  vsc-demark-b628be7871d06175eec02b2c798294043c784d07e4276962ea59033dc2611b52-features \
  -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate DemARK
jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true
"

# Wait for container to start
sleep 5

# Check if container is running
if docker ps | grep -q "demark-jupyter-working"; then
    echo ""
    echo "✅ Jupyter Lab is running with port forwarding!"
    echo ""
    echo "🌐 ACCESS INSTRUCTIONS:"
    echo "========================"
    echo ""
    echo "Jupyter Lab is now accessible at:"
    echo ""
    echo "  http://localhost:8890"
    echo ""
    echo "📁 Available Notebooks:"
    echo "========================"
    echo "- test_jupyter.ipynb (Math and HARK test notebook)"
    echo "- notebooks/ (All DemARK example notebooks)"
    echo ""
    echo "🎉 Ready to use! Open http://localhost:8890 in your browser."
    echo ""
    echo "🛑 To stop: docker stop demark-jupyter-working"
else
    echo "❌ Failed to start Jupyter Lab"
    echo "Please check Docker and try again"
fi 
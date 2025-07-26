#!/bin/bash

echo "🚀 DemARK Jupyter Lab - Final Test"
echo "=================================="
echo ""

# Step 1: Start devcontainer
echo "Step 1: Starting devcontainer..."
devcontainer up --workspace-folder . > /dev/null 2>&1
sleep 5

# Step 2: Get container ID
echo "Step 2: Getting container ID..."
CONTAINER_ID=$(docker ps | grep vsc-demark | awk '{print $1}')
if [ -z "$CONTAINER_ID" ]; then
    echo "❌ No container found"
    exit 1
fi
echo "Container ID: $CONTAINER_ID"

# Step 3: Start Jupyter Lab
echo "Step 3: Starting Jupyter Lab..."
docker exec -d $CONTAINER_ID bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate DemARK
jupyter lab --ip=0.0.0.0 --port=8890 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' --ServerApp.disable_check_xsrf=true
"

# Step 4: Wait and check
echo "Step 4: Waiting for Jupyter to start..."
sleep 5

# Step 5: Check if running
echo "Step 5: Checking if Jupyter is running..."
if docker exec $CONTAINER_ID pgrep -f jupyter-lab > /dev/null 2>&1; then
    echo "✅ Jupyter Lab is running!"
    echo ""
    echo "🌐 ACCESS INSTRUCTIONS:"
    echo "========================"
    echo ""
    echo "Jupyter Lab is running in container $CONTAINER_ID on port 8890"
    echo ""
    echo "To access it, run this command in a NEW terminal:"
    echo ""
    echo "  docker exec -it $CONTAINER_ID socat TCP-LISTEN:8890,fork TCP:localhost:8890"
    echo ""
    echo "Then open your browser to: http://localhost:8890"
    echo ""
    echo "🎉 Ready to use!"
else
    echo "❌ Jupyter Lab failed to start"
fi 
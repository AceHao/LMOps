#!/bin/bash
# Usage: 
#   On Head Pod:   ./setup_cluster.sh head
#   On Worker Pod: ./setup_cluster.sh worker <HEAD_POD_IP>

ROLE=$1
HEAD_IP=$2
MY_IP=$(hostname -i)
PORT=6379

# Clean up previous runs
ray stop --force 2>/dev/null

if [ "$ROLE" == "head" ]; then
    echo "------------------------------------------------"
    echo "🚀 STARTING RAY HEAD on $MY_IP"
    echo "------------------------------------------------"
    
    # Start Head (runs in background)
    # --num-gpus=8: Explicitly tell Ray this pod has 8 GPUs
    ray start --head \
        --node-ip-address=$MY_IP \
        --port=$PORT \
        --dashboard-host=0.0.0.0 \
        --num-gpus=8 \
        --disable-usage-stats \
        --block &
    
    echo ""
    echo "✅ Ray Head Started."
    echo "👉 Run this command on the WORKER Pod:"
    echo "./setup_cluster.sh worker $MY_IP"

elif [ "$ROLE" == "worker" ]; then
    if [ -z "$HEAD_IP" ]; then
        echo "❌ Error: Missing Head IP."
        echo "Usage: ./setup_cluster.sh worker <HEAD_IP>"
        exit 1
    fi

    echo "------------------------------------------------"
    echo "🔗 CONNECTING WORKER ($MY_IP) -> HEAD ($HEAD_IP)"
    echo "------------------------------------------------"
    
    # Start Worker Node
    ray start --address=$HEAD_IP:$PORT \
        --node-ip-address=$MY_IP \
        --num-gpus=8 \
        --disable-usage-stats \
        --block &

    echo "✅ Ray Worker Connected."

else
    echo "Usage: ./setup_cluster.sh [head | worker <HEAD_IP>]"
fi
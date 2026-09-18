#!/bin/bash
# deploy.sh — Deploy the ML API to AWS EC2 via SSH
# Usage: ./scripts/deploy.sh <EC2_HOST> <SSH_KEY_PATH>

set -euo pipefail

EC2_HOST="${1:-$EC2_HOST}"
SSH_KEY="${2:-$EC2_SSH_KEY_PATH}"
APP_DIR="/home/ubuntu/mlops-churn-prediction"

if [ -z "$EC2_HOST" ]; then
    echo "ERROR: EC2_HOST not set. Pass as arg or set environment variable."
    exit 1
fi

echo "Deploying to $EC2_HOST ..."

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "ubuntu@$EC2_HOST" << EOF
    set -e
    echo "=== Pulling latest code ==="
    cd $APP_DIR || (git clone \$REPO_URL $APP_DIR && cd $APP_DIR)
    git pull origin main

    echo "=== Rebuilding Docker image ==="
    docker compose down --remove-orphans
    docker compose up -d --build

    echo "=== Waiting for API to start ==="
    sleep 10
    curl -f http://localhost:8000/health || { echo 'Health check failed!'; exit 1; }
    echo "=== Deploy complete! ==="
EOF

echo "Deployment to $EC2_HOST finished successfully."

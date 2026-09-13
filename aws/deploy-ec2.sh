#!/bin/bash

set -euo pipefail

# ============================================================
# Sentiment MLOps - ECR -> EC2 Deployment
# Region: ap-south-1
# ============================================================

AWS_REGION="${AWS_REGION:-ap-south-1}"
ECR_REPOSITORY="${ECR_REPOSITORY:-sentiment-mlops}"
APP_DIR="/home/ec2-user/sentiment-mlops"
COMPOSE_FILE="$APP_DIR/aws/docker-compose.ec2.yml"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "============================================================"
echo " Sentiment MLOps - ECR -> EC2 Deployment"
echo "============================================================"

# ------------------------------------------------------------
# 1. Check required commands
# ------------------------------------------------------------
echo ""
echo "1. Checking required commands..."

for cmd in aws docker curl; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' is not installed."
        exit 1
    fi
done

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker Compose plugin is not available."
    echo "Install Docker Compose v2 and run this script again."
    exit 1
fi

# ------------------------------------------------------------
# 2. Get AWS account ID from the EC2 IAM role
# ------------------------------------------------------------
echo ""
echo "2. Checking AWS identity..."

AWS_ACCOUNT_ID="$(aws sts get-caller-identity \
    --query Account \
    --output text \
    --region "$AWS_REGION")"

if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "None" ]; then
    echo "ERROR: Cannot determine AWS account ID."
    echo "Make sure the EC2 instance has an IAM role with AWS credentials."
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "AWS Account : $AWS_ACCOUNT_ID"
echo "AWS Region  : $AWS_REGION"
echo "ECR Image   : $IMAGE"

# ------------------------------------------------------------
# 3. Create application directory
# ------------------------------------------------------------
echo ""
echo "3. Creating application directory..."

mkdir -p "$APP_DIR"
cd "$APP_DIR"

# ------------------------------------------------------------
# 4. Verify ECR repository exists
# ------------------------------------------------------------
echo ""
echo "4. Checking ECR repository..."

if ! aws ecr describe-repositories \
    --repository-names "$ECR_REPOSITORY" \
    --region "$AWS_REGION" >/dev/null 2>&1; then

    echo "ERROR: ECR repository '$ECR_REPOSITORY' was not found."
    echo "Create it first, or set ECR_REPOSITORY to the correct name."
    exit 1
fi

# ------------------------------------------------------------
# 5. Login to Amazon ECR
# ------------------------------------------------------------
echo ""
echo "5. Logging in to Amazon ECR..."

aws ecr get-login-password --region "$AWS_REGION" | \
docker login \
    --username AWS \
    --password-stdin "$ECR_REGISTRY"

# ------------------------------------------------------------
# 6. Pull latest image
# ------------------------------------------------------------
echo ""
echo "6. Pulling Docker image..."

docker pull "$IMAGE"

# ------------------------------------------------------------
# 7. Stop old container
# ------------------------------------------------------------
echo ""
echo "7. Stopping old container..."

if [ -f "$COMPOSE_FILE" ]; then
    ECR_REGISTRY="$ECR_REGISTRY" \
    ECR_REPOSITORY="$ECR_REPOSITORY" \
    IMAGE_TAG="$IMAGE_TAG" \
    docker compose -f "$COMPOSE_FILE" down || true
else
    echo "ERROR: $COMPOSE_FILE not found."
    echo "Copy docker-compose.ec2.yml into $APP_DIR/aws/ first."
    exit 1
fi

# ------------------------------------------------------------
# 8. Start new container
# ------------------------------------------------------------
echo ""
echo "8. Starting new container..."

ECR_REGISTRY="$ECR_REGISTRY" \
ECR_REPOSITORY="$ECR_REPOSITORY" \
IMAGE_TAG="$IMAGE_TAG" \
docker compose -f "$COMPOSE_FILE" up -d

# ------------------------------------------------------------
# 9. Wait for application
# ------------------------------------------------------------
echo ""
echo "9. Waiting for application..."

sleep 15

# ------------------------------------------------------------
# 10. Show containers
# ------------------------------------------------------------
echo ""
echo "10. Docker containers..."

docker ps

# ------------------------------------------------------------
# 11. Health check
# ------------------------------------------------------------
echo ""
echo "11. Health check..."

if curl --fail --silent --show-error \
    --max-time 10 \
    http://127.0.0.1:5000/health >/dev/null; then

    echo "Health check: PASSED"
else
    echo "Health check: FAILED"
    echo ""
    echo "Container logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=100
    exit 1
fi

# ------------------------------------------------------------
# 12. Get public IPv4
# ------------------------------------------------------------
echo ""
PUBLIC_IP="$(curl --fail --silent --max-time 5 \
    http://169.254.169.254/latest/meta-data/public-ipv4 || true)"

echo "============================================================"
echo " Deployment successful!"
echo "============================================================"

if [ -n "$PUBLIC_IP" ]; then
    echo "Application: http://${PUBLIC_IP}:5000"
else
    echo "Application is running on port 5000."
    echo "Public IP could not be detected automatically."
fi

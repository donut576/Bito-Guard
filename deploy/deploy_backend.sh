#!/bin/bash
# deploy/deploy_backend.sh
#
# 將後端 FastAPI 部署到 AWS App Runner（透過 ECR）
#
# 使用方式：
#   bash deploy/deploy_backend.sh \
#     --account 123456789012 \
#     --region ap-northeast-1 \
#     --service aml-api
#
# 前置條件：
#   - AWS CLI 已安裝並設定好 credentials
#   - Docker 已安裝並執行中
#   - 第一次執行會自動建立 ECR repo 和 App Runner service

set -e

# ── 參數解析 ──────────────────────────────────────────────────
ACCOUNT_ID=""
REGION="ap-northeast-1"
SERVICE_NAME="aml-api"
IMAGE_TAG="latest"

while [[ $# -gt 0 ]]; do
  case $1 in
    --account)  ACCOUNT_ID="$2";   shift 2 ;;
    --region)   REGION="$2";       shift 2 ;;
    --service)  SERVICE_NAME="$2"; shift 2 ;;
    --tag)      IMAGE_TAG="$2";    shift 2 ;;
    *) echo "未知參數: $1"; exit 1 ;;
  esac
done

# fallback 到環境變數
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"

if [[ -z "$ACCOUNT_ID" ]]; then
  echo "[ERROR] 無法取得 AWS Account ID，請確認 aws configure 已設定"
  exit 1
fi

ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$SERVICE_NAME"

echo "======================================"
echo "  AML Backend 部署"
echo "  Account : $ACCOUNT_ID"
echo "  Region  : $REGION"
echo "  ECR     : $ECR_REPO"
echo "  Service : $SERVICE_NAME"
echo "======================================"

# ── Step 1：確保 ECR repo 存在 ────────────────────────────────
echo "[INFO] 確認 ECR repo..."
aws ecr describe-repositories --repository-names "$SERVICE_NAME" --region "$REGION" \
  > /dev/null 2>&1 || \
aws ecr create-repository \
  --repository-name "$SERVICE_NAME" \
  --region "$REGION" \
  --image-scanning-configuration scanOnPush=true \
  > /dev/null
echo "[INFO] ✓ ECR repo: $SERVICE_NAME"

# ── Step 2：Docker build ──────────────────────────────────────
echo "[INFO] Docker build..."
cd "$(dirname "$0")/.."   # 切到專案根目錄

docker build \
  --platform linux/amd64 \
  -t "$SERVICE_NAME:$IMAGE_TAG" \
  -f Dockerfile \
  .

echo "[INFO] ✓ Build 完成"

# ── Step 3：推送到 ECR ────────────────────────────────────────
echo "[INFO] ECR login..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker tag "$SERVICE_NAME:$IMAGE_TAG" "$ECR_REPO:$IMAGE_TAG"

# 同時打上日期 tag 方便回滾
DATE_TAG=$(date +%Y%m%d_%H%M)
docker tag "$SERVICE_NAME:$IMAGE_TAG" "$ECR_REPO:$DATE_TAG"

echo "[INFO] 推送 image..."
docker push "$ECR_REPO:$IMAGE_TAG"
docker push "$ECR_REPO:$DATE_TAG"
echo "[INFO] ✓ 推送完成: $ECR_REPO:$IMAGE_TAG"

# ── Step 4：建立或更新 App Runner service ─────────────────────
APP_RUNNER_ROLE="arn:aws:iam::$ACCOUNT_ID:role/AppRunnerECRAccessRole"

# 確認 App Runner ECR access role 存在（若不存在則建立）
aws iam get-role --role-name AppRunnerECRAccessRole > /dev/null 2>&1 || {
  echo "[INFO] 建立 AppRunnerECRAccessRole..."
  aws iam create-role \
    --role-name AppRunnerECRAccessRole \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Allow",
        "Principal":{"Service":"build.apprunner.amazonaws.com"},
        "Action":"sts:AssumeRole"
      }]
    }' > /dev/null
  aws iam attach-role-policy \
    --role-name AppRunnerECRAccessRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  echo "[INFO] ✓ IAM role 建立完成"
}

# 檢查 service 是否已存在
SERVICE_ARN=$(aws apprunner list-services --region "$REGION" \
  --query "ServiceSummaryList[?ServiceName=='$SERVICE_NAME'].ServiceArn" \
  --output text 2>/dev/null || echo "")

if [[ -z "$SERVICE_ARN" ]]; then
  # ── 第一次：建立 service ──────────────────────────────────
  echo "[INFO] 建立 App Runner service..."
  aws apprunner create-service \
    --region "$REGION" \
    --service-name "$SERVICE_NAME" \
    --source-configuration "{
      \"ImageRepository\": {
        \"ImageIdentifier\": \"$ECR_REPO:$IMAGE_TAG\",
        \"ImageRepositoryType\": \"ECR\",
        \"ImageConfiguration\": {
          \"Port\": \"8080\",
          \"RuntimeEnvironmentVariables\": {
            \"AWS_DEFAULT_REGION\": \"$REGION\",
            \"AML_S3_BUCKET\": \"${AML_S3_BUCKET:-your-aml-bucket}\"
          }
        }
      },
      \"AuthenticationConfiguration\": {
        \"AccessRoleArn\": \"$APP_RUNNER_ROLE\"
      },
      \"AutoDeploymentsEnabled\": false
    }" \
    --instance-configuration '{
      "Cpu": "1 vCPU",
      "Memory": "2 GB"
    }' \
    --health-check-configuration '{
      "Protocol": "HTTP",
      "Path": "/health",
      "Interval": 10,
      "Timeout": 5,
      "HealthyThreshold": 1,
      "UnhealthyThreshold": 5
    }' > /tmp/apprunner_create.json

  SERVICE_ARN=$(python3 -c "import json; d=json.load(open('/tmp/apprunner_create.json')); print(d['Service']['ServiceArn'])")
  SERVICE_URL=$(python3 -c "import json; d=json.load(open('/tmp/apprunner_create.json')); print(d['Service']['ServiceUrl'])")
  echo "[INFO] ✓ App Runner service 建立中..."
  echo "[INFO]   ARN: $SERVICE_ARN"
  echo "[INFO]   URL: https://$SERVICE_URL"

else
  # ── 已存在：更新 image ────────────────────────────────────
  echo "[INFO] 更新 App Runner service: $SERVICE_ARN"
  aws apprunner update-service \
    --region "$REGION" \
    --service-arn "$SERVICE_ARN" \
    --source-configuration "{
      \"ImageRepository\": {
        \"ImageIdentifier\": \"$ECR_REPO:$IMAGE_TAG\",
        \"ImageRepositoryType\": \"ECR\",
        \"ImageConfiguration\": {
          \"Port\": \"8080\"
        }
      },
      \"AuthenticationConfiguration\": {
        \"AccessRoleArn\": \"$APP_RUNNER_ROLE\"
      }
    }" > /dev/null
  echo "[INFO] ✓ 部署已觸發，通常 2-3 分鐘生效"

  SERVICE_URL=$(aws apprunner describe-service \
    --service-arn "$SERVICE_ARN" \
    --region "$REGION" \
    --query "Service.ServiceUrl" \
    --output text)
  echo "[INFO]   URL: https://$SERVICE_URL"
fi

echo ""
echo "======================================"
echo "  部署完成"
echo "  API URL: https://$SERVICE_URL"
echo "  健康檢查: https://$SERVICE_URL/health"
echo "  API 文件: https://$SERVICE_URL/docs"
echo ""
echo "  更新前端 API URL："
echo "  bash deploy/deploy_frontend.sh \\"
echo "    --bucket <frontend-bucket> \\"
echo "    --api-url https://$SERVICE_URL"
echo "======================================"

#!/bin/bash
# deploy/deploy_ec2.sh
#
# 一鍵將後端 (FastAPI) + 前端 (React) 部署到 EC2，並設定 CloudFront
#
# 使用方式（在本機執行）：
#   bash deploy/deploy_ec2.sh
#
# 預設值（可用環境變數覆蓋）：
#   EC2_HOST  = 54.191.133.100
#   EC2_KEY   = ~/Downloads/yxx.pem
#   EC2_USER  = ubuntu
#   AWS_REGION = ap-northeast-1

set -e

# ── 設定區 ────────────────────────────────────────────────────
EC2_HOST="${EC2_HOST:-54.191.133.100}"
EC2_KEY="${EC2_KEY:-$HOME/Downloads/yxx.pem}"
EC2_USER="${EC2_USER:-ubuntu}"
AWS_REGION="${AWS_REGION:-ap-northeast-1}"
APP_DIR="/home/ubuntu/aml"
BACKEND_PORT=8080

# CloudFront S3 bucket（前端靜態檔）
# 若不想用 S3+CloudFront，設 USE_CLOUDFRONT=false 改用 nginx 直接 serve
USE_CLOUDFRONT="${USE_CLOUDFRONT:-true}"
FRONTEND_BUCKET="${FRONTEND_BUCKET:-}"   # 留空會自動用 EC2 IP 產生名稱

# ─────────────────────────────────────────────────────────────

SSH="ssh -i $EC2_KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST"
SCP="scp -i $EC2_KEY -o StrictHostKeyChecking=no"

echo "======================================"
echo "  AML EC2 部署"
echo "  Host   : $EC2_HOST"
echo "  Key    : $EC2_KEY"
echo "  AppDir : $APP_DIR"
echo "======================================"

# ── 確認 key 權限 ─────────────────────────────────────────────
chmod 400 "$EC2_KEY"

# ── Step 1：EC2 環境安裝 ──────────────────────────────────────
echo ""
echo "[1/6] 安裝 EC2 環境（Python、Node.js、nginx）..."

$SSH << 'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive

# 更新套件
sudo apt-get update -qq

# Python 3.11
if ! python3.11 --version &>/dev/null; then
  sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip
fi

# Node.js 20
if ! node --version &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - > /dev/null 2>&1
  sudo apt-get install -y -qq nodejs
fi

# nginx
if ! nginx -v &>/dev/null 2>&1; then
  sudo apt-get install -y -qq nginx
fi

# AWS CLI v2
if ! aws --version &>/dev/null 2>&1; then
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  sudo apt-get install -y -qq unzip
  unzip -q /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install
  rm -rf /tmp/awscliv2.zip /tmp/aws
fi

echo "✓ 環境就緒"
REMOTE

# ── Step 2：上傳專案程式碼 ────────────────────────────────────
echo ""
echo "[2/6] 上傳專案程式碼..."

# 建立遠端目錄
$SSH "mkdir -p $APP_DIR/app $APP_DIR/aml-frontend"

# 上傳後端
$SCP -r app/ "$EC2_USER@$EC2_HOST:$APP_DIR/"
$SCP requirements.txt "$EC2_USER@$EC2_HOST:$APP_DIR/"

# 上傳前端原始碼
$SCP -r aml-frontend/src aml-frontend/index.html aml-frontend/package.json \
        aml-frontend/package-lock.json aml-frontend/vite.config.js \
        "$EC2_USER@$EC2_HOST:$APP_DIR/aml-frontend/"

echo "✓ 程式碼上傳完成"

# ── Step 3：安裝 Python 依賴 & 啟動後端 ──────────────────────
echo ""
echo "[3/6] 安裝 Python 依賴 & 啟動後端..."

$SSH << REMOTE
set -e
cd $APP_DIR

# 建立 venv
if [ ! -d venv ]; then
  python3.11 -m venv venv
fi
source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 建立 .env（若不存在）
if [ ! -f .env ]; then
  cat > .env << 'ENV'
model_s3_uri=s3://aml-models/model_registry/latest
database_url=postgresql://postgres:postgres@localhost:5432/aml
aws_region=$AWS_REGION
stream_broker_type=kafka
redis_url=redis://localhost:6379/0
ENV
fi

# 建立 systemd service
sudo tee /etc/systemd/system/aml-api.service > /dev/null << 'SERVICE'
[Unit]
Description=AML Fraud Detection API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment=PATH=$APP_DIR/venv/bin
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port $BACKEND_PORT --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable aml-api
sudo systemctl restart aml-api

sleep 3
if sudo systemctl is-active --quiet aml-api; then
  echo "✓ 後端服務啟動成功 (port $BACKEND_PORT)"
else
  echo "⚠ 後端服務啟動失敗，查看 log："
  sudo journalctl -u aml-api -n 30 --no-pager
fi
REMOTE

# ── Step 4：Build 前端 & 設定 nginx ──────────────────────────
echo ""
echo "[4/6] Build 前端 & 設定 nginx..."

$SSH << REMOTE
set -e
cd $APP_DIR/aml-frontend

# 設定 API URL（指向本機後端）
echo "VITE_API_BASE_URL=http://$EC2_HOST" > .env.production

npm install --silent
npm run build
echo "✓ 前端 build 完成"

# nginx 設定：前端 + 後端 proxy
sudo tee /etc/nginx/sites-available/aml << 'NGINX'
server {
    listen 80;
    server_name _;

    # 前端靜態檔
    root $APP_DIR/aml-frontend/dist;
    index index.html;

    # SPA fallback
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # 後端 API proxy
    location ~ ^/(metrics|features|thresholds|shap|fraud_explanation|infer|health|summary|docs|openapi.json) {
        proxy_pass http://127.0.0.1:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 120s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/aml /etc/nginx/sites-enabled/aml
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
echo "✓ nginx 設定完成"
REMOTE

# ── Step 5：CloudFront（若啟用）──────────────────────────────
if [[ "$USE_CLOUDFRONT" == "true" ]]; then
  echo ""
  echo "[5/6] 設定 CloudFront..."

  # 用 EC2 IP 建立 CloudFront distribution（指向 EC2 的 nginx）
  CF_ORIGIN="$EC2_HOST"

  # 檢查是否已有 distribution 指向此 EC2
  EXISTING_CF=$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Origins.Items[0].DomainName=='$CF_ORIGIN'].Id" \
    --output text 2>/dev/null || echo "")

  if [[ -n "$EXISTING_CF" && "$EXISTING_CF" != "None" ]]; then
    echo "[INFO] 已有 CloudFront distribution: $EXISTING_CF"
    CF_DOMAIN=$(aws cloudfront get-distribution \
      --id "$EXISTING_CF" \
      --query "Distribution.DomainName" \
      --output text)
  else
    echo "[INFO] 建立新的 CloudFront distribution..."
    CF_RESULT=$(aws cloudfront create-distribution \
      --region us-east-1 \
      --distribution-config "{
        \"CallerReference\": \"aml-ec2-$(date +%s)\",
        \"Comment\": \"AML Dashboard via EC2\",
        \"DefaultCacheBehavior\": {
          \"TargetOriginId\": \"EC2-$EC2_HOST\",
          \"ViewerProtocolPolicy\": \"redirect-to-https\",
          \"CachePolicyId\": \"4135ea2d-6df8-44a3-9df3-4b5a84be39ad\",
          \"AllowedMethods\": {
            \"Quantity\": 7,
            \"Items\": [\"GET\",\"HEAD\",\"OPTIONS\",\"PUT\",\"POST\",\"PATCH\",\"DELETE\"],
            \"CachedMethods\": {\"Quantity\": 2, \"Items\": [\"GET\",\"HEAD\"]}
          },
          \"Compress\": true
        },
        \"CacheBehaviors\": {
          \"Quantity\": 1,
          \"Items\": [{
            \"PathPattern\": \"/api/*\",
            \"TargetOriginId\": \"EC2-$EC2_HOST\",
            \"ViewerProtocolPolicy\": \"redirect-to-https\",
            \"CachePolicyId\": \"4135ea2d-6df8-44a3-9df3-4b5a84be39ad\",
            \"AllowedMethods\": {
              \"Quantity\": 7,
              \"Items\": [\"GET\",\"HEAD\",\"OPTIONS\",\"PUT\",\"POST\",\"PATCH\",\"DELETE\"]
            },
            \"Compress\": false
          }]
        },
        \"Origins\": {
          \"Quantity\": 1,
          \"Items\": [{
            \"Id\": \"EC2-$EC2_HOST\",
            \"DomainName\": \"$EC2_HOST\",
            \"CustomOriginConfig\": {
              \"HTTPPort\": 80,
              \"HTTPSPort\": 443,
              \"OriginProtocolPolicy\": \"http-only\"
            }
          }]
        },
        \"CustomErrorResponses\": {
          \"Quantity\": 1,
          \"Items\": [{
            \"ErrorCode\": 404,
            \"ResponseCode\": \"200\",
            \"ResponsePagePath\": \"/index.html\"
          }]
        },
        \"Enabled\": true,
        \"DefaultRootObject\": \"index.html\"
      }")

    CF_DIST_ID=$(echo "$CF_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['Id'])")
    CF_DOMAIN=$(echo "$CF_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['DomainName'])")
    echo "✓ CloudFront distribution 建立完成: $CF_DIST_ID"
  fi

  echo "✓ CloudFront URL: https://$CF_DOMAIN"
else
  echo ""
  echo "[5/6] 跳過 CloudFront（USE_CLOUDFRONT=false）"
  CF_DOMAIN="$EC2_HOST"
fi

# ── Step 6：健康檢查 ──────────────────────────────────────────
echo ""
echo "[6/6] 健康檢查..."

sleep 3
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://$EC2_HOST/health" || echo "000")

if [[ "$HTTP_STATUS" == "200" ]]; then
  echo "✓ 後端健康檢查通過 (HTTP $HTTP_STATUS)"
else
  echo "⚠ 健康檢查回傳 HTTP $HTTP_STATUS，請確認 EC2 Security Group 已開放 port 80"
fi

echo ""
echo "======================================"
echo "  部署完成"
echo ""
echo "  EC2 直連  : http://$EC2_HOST"
if [[ "$USE_CLOUDFRONT" == "true" && -n "$CF_DOMAIN" ]]; then
echo "  CloudFront: https://$CF_DOMAIN"
echo "  （CloudFront 首次部署約需 5-10 分鐘生效）"
fi
echo ""
echo "  後端 API 文件: http://$EC2_HOST/docs"
echo ""
echo "  管理指令（SSH 進去後）："
echo "    sudo systemctl status aml-api    # 查看後端狀態"
echo "    sudo journalctl -u aml-api -f    # 即時 log"
echo "    sudo systemctl restart aml-api   # 重啟後端"
echo "======================================"

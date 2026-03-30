#!/bin/bash
# deploy/deploy_frontend.sh
#
# 將 aml-frontend 靜態網站部署到 S3 + CloudFront
#
# 使用方式：
#   bash deploy/deploy_frontend.sh \
#     --bucket my-aml-frontend-bucket \
#     --region ap-northeast-1 \
#     --api-url https://your-api.ap-northeast-1.awsapprunner.com
#
# 前置條件：
#   - AWS CLI 已安裝並設定好 credentials（aws configure）
#   - Node.js >= 18 已安裝
#   - 第一次執行需先手動建立 S3 bucket 與 CloudFront distribution
#     （或執行 deploy/setup_infra.sh 自動建立）

set -e

# ── 參數解析 ──────────────────────────────────────────────────
BUCKET=""
REGION="ap-northeast-1"
API_URL=""
CF_DIST_ID=""   # CloudFront distribution ID（選填，有才做 invalidation）

while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket)     BUCKET="$2";     shift 2 ;;
    --region)     REGION="$2";     shift 2 ;;
    --api-url)    API_URL="$2";    shift 2 ;;
    --cf-dist-id) CF_DIST_ID="$2"; shift 2 ;;
    *) echo "未知參數: $1"; exit 1 ;;
  esac
done

if [[ -z "$BUCKET" ]]; then
  echo "[ERROR] 請指定 --bucket <bucket-name>"
  exit 1
fi

# ── 讀取環境變數 fallback ─────────────────────────────────────
BUCKET="${BUCKET:-$AML_FRONTEND_BUCKET}"
API_URL="${API_URL:-$AML_API_URL}"
CF_DIST_ID="${CF_DIST_ID:-$AML_CF_DIST_ID}"

echo "======================================"
echo "  AML Frontend 部署"
echo "  Bucket : $BUCKET"
echo "  Region : $REGION"
echo "  API URL: ${API_URL:-（未設定，使用 vite proxy）}"
echo "======================================"

# ── Step 1：設定 API URL 環境變數給 Vite build ────────────────
cd "$(dirname "$0")/../aml-frontend"

if [[ -n "$API_URL" ]]; then
  echo "VITE_API_BASE_URL=$API_URL" > .env.production
  echo "[INFO] 寫入 .env.production: VITE_API_BASE_URL=$API_URL"
fi

# ── Step 2：安裝依賴 & build ──────────────────────────────────
echo "[INFO] npm install..."
npm install --silent

echo "[INFO] npm run build..."
npm run build

echo "[INFO] ✓ build 完成，輸出在 dist/"

# ── Step 3：同步到 S3 ─────────────────────────────────────────
echo "[INFO] 上傳到 s3://$BUCKET/ ..."

# HTML 不快取（確保使用者拿到最新版）
aws s3 sync dist/ "s3://$BUCKET/" \
  --region "$REGION" \
  --delete \
  --exclude "*.js" --exclude "*.css" --exclude "*.png" --exclude "*.svg" --exclude "*.ico"

# JS/CSS/圖片長期快取（Vite 會在檔名加 hash）
aws s3 sync dist/ "s3://$BUCKET/" \
  --region "$REGION" \
  --cache-control "public, max-age=31536000, immutable" \
  --exclude "*.html"

# HTML 不快取
aws s3 cp dist/index.html "s3://$BUCKET/index.html" \
  --region "$REGION" \
  --cache-control "no-cache, no-store, must-revalidate" \
  --content-type "text/html"

echo "[INFO] ✓ S3 同步完成"

# ── Step 4：CloudFront Invalidation（若有設定 distribution ID）─
if [[ -n "$CF_DIST_ID" ]]; then
  echo "[INFO] CloudFront invalidation: $CF_DIST_ID ..."
  aws cloudfront create-invalidation \
    --distribution-id "$CF_DIST_ID" \
    --paths "/*" \
    --region us-east-1   # CloudFront API 固定在 us-east-1
  echo "[INFO] ✓ Invalidation 已送出（通常 1-2 分鐘生效）"
else
  echo "[WARN] 未設定 --cf-dist-id，跳過 CloudFront invalidation"
fi

echo ""
echo "======================================"
echo "  部署完成"
if [[ -n "$CF_DIST_ID" ]]; then
  echo "  請到 AWS Console 查詢 CloudFront domain"
else
  echo "  S3 靜態網站 URL："
  echo "  http://$BUCKET.s3-website-$REGION.amazonaws.com"
fi
echo "======================================"

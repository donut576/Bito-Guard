#!/bin/bash
# deploy/setup_infra.sh
#
# 一次性建立 AWS 基礎設施（只需執行一次）：
#   - S3 bucket（前端靜態網站）
#   - S3 bucket（模型 & 資料）
#   - CloudFront distribution（前端 CDN）
#
# 使用方式：
#   bash deploy/setup_infra.sh \
#     --frontend-bucket my-aml-frontend \
#     --data-bucket my-aml-models \
#     --region ap-northeast-1

set -e

FRONTEND_BUCKET=""
DATA_BUCKET=""
REGION="ap-northeast-1"

while [[ $# -gt 0 ]]; do
  case $1 in
    --frontend-bucket) FRONTEND_BUCKET="$2"; shift 2 ;;
    --data-bucket)     DATA_BUCKET="$2";     shift 2 ;;
    --region)          REGION="$2";          shift 2 ;;
    *) echo "未知參數: $1"; exit 1 ;;
  esac
done

if [[ -z "$FRONTEND_BUCKET" || -z "$DATA_BUCKET" ]]; then
  echo "[ERROR] 請指定 --frontend-bucket 和 --data-bucket"
  exit 1
fi

echo "======================================"
echo "  建立 AWS 基礎設施"
echo "  Frontend bucket : $FRONTEND_BUCKET"
echo "  Data bucket     : $DATA_BUCKET"
echo "  Region          : $REGION"
echo "======================================"

# ── 建立前端 S3 bucket ────────────────────────────────────────
echo "[INFO] 建立前端 bucket: $FRONTEND_BUCKET"
if [[ "$REGION" == "us-east-1" ]]; then
  aws s3api create-bucket --bucket "$FRONTEND_BUCKET" --region "$REGION"
else
  aws s3api create-bucket --bucket "$FRONTEND_BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
fi

# 關閉 Block Public Access（靜態網站需要公開讀取）
aws s3api put-public-access-block \
  --bucket "$FRONTEND_BUCKET" \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# 設定靜態網站 hosting
aws s3 website "s3://$FRONTEND_BUCKET/" \
  --index-document index.html \
  --error-document index.html   # SPA fallback

# 設定 bucket policy（公開讀取）
aws s3api put-bucket-policy --bucket "$FRONTEND_BUCKET" --policy "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"PublicReadGetObject\",
    \"Effect\": \"Allow\",
    \"Principal\": \"*\",
    \"Action\": \"s3:GetObject\",
    \"Resource\": \"arn:aws:s3:::$FRONTEND_BUCKET/*\"
  }]
}"
echo "[INFO] ✓ 前端 bucket 建立完成"

# ── 建立資料/模型 S3 bucket ───────────────────────────────────
echo "[INFO] 建立資料 bucket: $DATA_BUCKET"
if [[ "$REGION" == "us-east-1" ]]; then
  aws s3api create-bucket --bucket "$DATA_BUCKET" --region "$REGION"
else
  aws s3api create-bucket --bucket "$DATA_BUCKET" --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"
fi

# 建立目錄結構（空物件當佔位符）
for prefix in "data/" "output/" "model_registry/latest/"; do
  aws s3api put-object --bucket "$DATA_BUCKET" --key "$prefix"
done
echo "[INFO] ✓ 資料 bucket 建立完成"

# ── 建立 CloudFront distribution ─────────────────────────────
echo "[INFO] 建立 CloudFront distribution..."
CF_ORIGIN="$FRONTEND_BUCKET.s3-website-$REGION.amazonaws.com"

CF_DIST=$(aws cloudfront create-distribution --distribution-config "{
  \"CallerReference\": \"aml-frontend-$(date +%s)\",
  \"Comment\": \"AML Dashboard Frontend\",
  \"DefaultCacheBehavior\": {
    \"TargetOriginId\": \"S3-$FRONTEND_BUCKET\",
    \"ViewerProtocolPolicy\": \"redirect-to-https\",
    \"CachePolicyId\": \"658327ea-f89d-4fab-a63d-7e88639e58f6\",
    \"Compress\": true,
    \"AllowedMethods\": {
      \"Quantity\": 2,
      \"Items\": [\"GET\", \"HEAD\"]
    }
  },
  \"Origins\": {
    \"Quantity\": 1,
    \"Items\": [{
      \"Id\": \"S3-$FRONTEND_BUCKET\",
      \"DomainName\": \"$CF_ORIGIN\",
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
}" --region us-east-1)

CF_DIST_ID=$(echo "$CF_DIST" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['Id'])")
CF_DOMAIN=$(echo "$CF_DIST" | python3 -c "import sys,json; print(json.load(sys.stdin)['Distribution']['DomainName'])")

echo "[INFO] ✓ CloudFront distribution 建立完成"
echo ""
echo "======================================"
echo "  基礎設施建立完成，請記錄以下資訊："
echo ""
echo "  前端 bucket    : $FRONTEND_BUCKET"
echo "  資料 bucket    : $DATA_BUCKET"
echo "  CloudFront ID  : $CF_DIST_ID"
echo "  CloudFront URL : https://$CF_DOMAIN"
echo ""
echo "  後續部署前端："
echo "  bash deploy/deploy_frontend.sh \\"
echo "    --bucket $FRONTEND_BUCKET \\"
echo "    --cf-dist-id $CF_DIST_ID \\"
echo "    --api-url https://your-api-url"
echo "======================================"

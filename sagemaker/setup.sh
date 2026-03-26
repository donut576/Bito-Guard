#!/bin/bash
# SageMaker Notebook 環境安裝腳本
# 在 Notebook Instance 的 terminal 執行一次即可：
#   bash sagemaker/setup.sh

set -e

pip install --quiet \
    xgboost>=2.0.0 \
    lightgbm>=4.0.0 \
    optuna>=3.0.0 \
    shap>=0.43.0 \
    s3fs>=2023.6.0

echo "✓ 所有依賴安裝完成"

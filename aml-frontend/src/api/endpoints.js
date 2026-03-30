/** API endpoint definitions */

// 本地開發：空字串（走 vite proxy）
// Production build：從 VITE_API_BASE_URL 環境變數注入
const BASE = import.meta.env.VITE_API_BASE_URL ?? '';

export const ENDPOINTS = {
  metrics:    (mode, model = 'xgb') => `${BASE}/metrics?mode=${mode}&model=${model}`,
  compare:    (mode)                 => `${BASE}/metrics/compare?mode=${mode}`,
  features:   (mode, model = 'xgb') => `${BASE}/features?mode=${mode}&model=${model}`,
  thresholds: (mode, model = 'xgb') => `${BASE}/thresholds?mode=${mode}&model=${model}`,
  shap:       (mode, model = 'xgb') => `${BASE}/shap?mode=${mode}&model=${model}`,
  summary:    ()                     => `${BASE}/summary`,
  fraud_explanation: (limit = 10)    => `${BASE}/fraud_explanation?limit=${limit}`,
  infer:      ()                     => `${BASE}/infer`,
};

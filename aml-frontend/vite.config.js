import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    allowedHosts: ['dicotyledonous-nonsegmentary-asha.ngrok-free.dev'],
    proxy: {
      '/metrics': 'http://localhost:8000',
      '/features': 'http://localhost:8000',
      '/thresholds': 'http://localhost:8000',
      '/shap': 'http://localhost:8000',
      '/fraud_explanation': 'http://localhost:8000',
      '/infer': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/cache': 'http://localhost:8000',
    },
  },
  // production build 時從環境變數注入 API base URL
  define: {
    __API_BASE_URL__: JSON.stringify(process.env.VITE_API_BASE_URL ?? ''),
  },
});

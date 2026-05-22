import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_BASE = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8765';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/healthz': API_BASE,
      '/reports': API_BASE,
      '/sessions': API_BASE,
      '/eval': API_BASE,
    },
  },
  preview: {
    port: 4173,
    proxy: {
      '/healthz': API_BASE,
      '/reports': API_BASE,
      '/sessions': API_BASE,
      '/eval': API_BASE,
    },
  },
});

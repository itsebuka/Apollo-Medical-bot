import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls during development to the FastAPI backend
    // This avoids CORS issues and allows clean relative URLs in the frontend
    proxy: {
      '/chat': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})

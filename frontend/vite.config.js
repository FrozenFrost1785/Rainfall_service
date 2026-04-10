import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ['parmesan-tidbit-unisexual.ngrok-free.dev'],
    proxy: {
      '/api': 'http://localhost:8002',
      '/alerts': { target: 'ws://localhost:8002', ws: true },
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 前端开发服务器
// 后端默认端口 8000；如需修改，同步修改这里与后端启动命令
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/v1': { target: 'http://localhost:8000', changeOrigin: true },
      '/rag': { target: 'http://localhost:8000', changeOrigin: true },
      '/admin': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const apiPort = process.env.ARENA_API_PORT ?? '8317'
const webPort = Number(process.env.ARENA_WEB_PORT ?? '5317')

const proxy = {
  '/api': {
    target: `http://127.0.0.1:${apiPort}`,
    ws: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  plugins: [react()],
  server: { host: '127.0.0.1', port: webPort, strictPort: true, proxy },
  // `make party` serves the built table to the whole LAN; any Host header is fine there (IP, name.local).
  preview: { port: webPort, strictPort: true, proxy, allowedHosts: true },
})

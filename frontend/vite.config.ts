import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

import { cloudflare } from "@cloudflare/vite-plugin";

// https://vite.dev/config/
export default defineConfig({
 plugins: [react(), cloudflare()],
 server: {
 host: '0.0.0.0',
 port: 21030,
 proxy: {
 '/ws': {
 target: 'ws://localhost:8765',
 ws: true,
 changeOrigin: true,
 },
 },
 },
})

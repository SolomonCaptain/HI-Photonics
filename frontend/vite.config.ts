import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  server: {
    allowedHosts: [
      '11477ja4cp718.vicp.fun',
    ],
  },
  plugins: [react()],
})

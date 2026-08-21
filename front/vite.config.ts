import { defineConfig, loadEnv } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  return {
    envDir: "..",
    plugins: [
      tailwindcss(),
      react(),
      babel({ presets: [reactCompilerPreset()] }),
    ],
    server: {
      proxy: {
        "/api": {
          target: env.VITE_DEV_API_TARGET || "http://127.0.0.1:8000",
          changeOrigin: true,
        },
      },
    },
  };
})

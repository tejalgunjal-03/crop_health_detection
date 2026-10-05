import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// The browser talks only to Vite (port 5173); /api and /files are forwarded to FastAPI (port 8000).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000", "/files": "http://127.0.0.1:8000" } },
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The console talks to the Python server (simheart.server) under /api.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});

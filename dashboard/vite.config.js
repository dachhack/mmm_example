import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build the static site into ../docs for GitHub Pages. emptyOutDir:false so the precomputed
// docs/data/* (produced by scripts/run_fits.sh) and the docs/*.md specs are preserved.
// base "./" makes asset URLs relative, so it works under a project-pages subpath.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../docs",
    emptyOutDir: false,
    assetsDir: "assets",
  },
});

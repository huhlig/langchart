import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, copyFileSync, mkdirSync } from "fs";

/** Vite plugin: copy WASM files from src/wasm/ to dist/assets/wasm/ if they exist. */
function copyWasmPlugin() {
  return {
    name: "copy-wasm",
    closeBundle() {
      const srcDir = resolve(__dirname, "src/wasm");
      const outDir = resolve(__dirname, "dist/assets/wasm");
      if (!existsSync(srcDir)) return;
      mkdirSync(outDir, { recursive: true });
      for (const name of ["langchart_wasm.js", "langchart_wasm_bg.wasm", "langchart_wasm_bg.wasm.d.ts"]) {
        const src = resolve(srcDir, name);
        if (existsSync(src)) {
          copyFileSync(src, resolve(outDir, name));
          console.info(`[copy-wasm] ${name} → dist/assets/wasm/`);
        }
      }
    },
  };
}

export default defineConfig(({ mode }) => {
  const isLib = mode === "lib";

  return {
    root: "src",
    publicDir: "../public",
    base: "./",
    build: isLib
      ? {
          // Library mode: emit ES module with type declarations
          lib: {
            entry: resolve(__dirname, "src/lib.ts"),
            formats: ["es"],
            fileName: "lib",
          },
          outDir: "../dist/lib",
          emptyOutDir: true,
          rollupOptions: {
            external: ["elkjs"],
            output: {
              globals: { elkjs: "ELK" },
            },
          },
        }
      : {
          // App mode: standalone HTML app
          outDir: "../dist",
          emptyOutDir: true,
          rollupOptions: {
            input: resolve(__dirname, "src/index.html"),
          },
          plugins: [copyWasmPlugin()],
        },
    server: {
      port: 5173,
      open: true,
    },
    // Allow top-level await for WASM initialisation.
    esbuild: {
      target: "es2022",
    },
    optimizeDeps: {
      exclude: ["langchart-wasm"],
    },
  };
});

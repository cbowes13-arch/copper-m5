// npm run langfuse:models: run once after adding the Langfuse keys (and again
// if the prices in lib/ai/pricing.ts change). Saves what each approved Gemini
// model costs in your Langfuse project, so every traced call shows its cost.
// It talks only to Langfuse; it makes no AI calls.
import { registerHooks } from "node:module";

// The app's TypeScript files import each other as "@/lib/...": point Node at
// them, and tell it they're ES modules so it doesn't print a warning.
const root = new URL("../", import.meta.url);
registerHooks({
  resolve: (specifier, context, next) =>
    next(
      specifier.startsWith("@/")
        ? new URL(`${specifier.slice(2)}.ts`, root).href
        : specifier,
      context,
    ),
  load: (url, context, next) =>
    next(
      url,
      url.endsWith(".ts")
        ? { ...context, format: "module-typescript" }
        : context,
    ),
});

try {
  process.loadEnvFile(new URL(".env.local", root));
} catch {
  // No .env.local: the keys may already be in the environment.
}

const { isTracingConfigured } = await import("../lib/ai/tracing.ts");
const { syncModels } = await import("../lib/ai/langfuse-models.ts");

if (!isTracingConfigured()) {
  console.error(
    "langfuse:models: add LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL to .env.local first.",
  );
  process.exit(1);
}

try {
  const results = await syncModels({
    baseUrl: process.env.LANGFUSE_BASE_URL?.trim() || undefined,
    publicKey: process.env.LANGFUSE_PUBLIC_KEY.trim(),
    secretKey: process.env.LANGFUSE_SECRET_KEY.trim(),
  });
  for (const { modelName, action } of results)
    console.log(`${action} ${modelName}`);
  console.log("Done: Langfuse now prices every Gemini call the app makes.");
} catch (error) {
  console.error(`langfuse:models: ${error.message}`);
  process.exit(1);
}

// Module 5: sends a trace of every AI call to Langfuse, so you can see which
// agent answered, the tools it used, and each call's tokens, cost and time.
// It's on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and
// LANGFUSE_BASE_URL for your region); without them the app runs exactly as
// before and nothing is sent anywhere.
//
// instrumentation.ts calls startTracing() once when the server starts. Scripts
// that make AI calls outside Next.js (like the evals) call it themselves, and
// call flushTracing() before they exit.
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { LangfuseVercelAiSdkIntegration } from "@langfuse/vercel-ai-sdk";
import { resourceFromAttributes } from "@opentelemetry/resources";
import type { SpanExporter } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerTelemetry } from "ai";

/** How traces name the app. */
export const SERVICE_NAME = "copper-state-labs";

export function isTracingConfigured(): boolean {
  return Boolean(
    process.env.LANGFUSE_PUBLIC_KEY?.trim() &&
    process.env.LANGFUSE_SECRET_KEY?.trim(),
  );
}

// Kept on globalThis so hot reloads in `next dev` don't start it twice.
const state = globalThis as typeof globalThis & {
  __copperTracing?: { processor: LangfuseSpanProcessor };
};

/**
 * Starts sending traces to Langfuse. Returns false when it isn't set up.
 * Tests pass their own `exporter` to catch the spans instead of sending them.
 */
export function startTracing({
  exporter,
}: { exporter?: SpanExporter } = {}): boolean {
  if (state.__copperTracing) return true;
  if (!isTracingConfigured()) return false;
  // "immediate" sends each span as it finishes: on Vercel the function can
  // be frozen soon after the reply, before a batch would be sent.
  const processor = new LangfuseSpanProcessor({
    exporter,
    exportMode: "immediate",
    // Lets Langfuse filter real chats from eval runs ("eval", "ci").
    environment:
      process.env.LANGFUSE_TRACING_ENVIRONMENT ??
      process.env.VERCEL_ENV ??
      process.env.NODE_ENV,
  });
  new NodeTracerProvider({
    // Names the app in each trace, instead of "unknown_service".
    resource: resourceFromAttributes({ "service.name": SERVICE_NAME }),
    spanProcessors: [processor],
  }).register();
  registerTelemetry(new LangfuseVercelAiSdkIntegration());
  state.__copperTracing = { processor };
  return true;
}

/** Waits until every finished span has reached Langfuse. */
export async function flushTracing(): Promise<void> {
  try {
    await state.__copperTracing?.processor.forceFlush();
  } catch (error) {
    // A Langfuse outage must never break the chat.
    console.warn("[tracing] could not send traces to Langfuse:", error);
  }
}

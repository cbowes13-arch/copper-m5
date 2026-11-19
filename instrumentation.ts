// Next.js runs register() once when the server starts (see
// node_modules/next/dist/docs/01-app/02-guides/instrumentation.md).
// Module 5: start sending AI traces to Langfuse, if its keys are set.
export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const { startTracing } = await import("@/lib/ai/tracing");
  if (startTracing()) console.info("[tracing] sending AI traces to Langfuse");
}

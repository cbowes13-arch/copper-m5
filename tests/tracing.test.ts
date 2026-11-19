// @vitest-environment node
// Module 5: every Ask Copper turn is traced in Langfuse: which agent answered,
// its tool calls, and each model call's tokens and time. These tests run the
// real Langfuse span processor with a mock model, and catch the spans in
// memory instead of sending them, so they need no Langfuse or Google key.
import type { LanguageModelV4StreamPart as ModelStreamPart } from "@ai-sdk/provider";
import { LangfuseOtelSpanAttributes as A } from "@langfuse/tracing";
import {
  InMemorySpanExporter,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import { convertArrayToReadableStream, MockLanguageModelV4 } from "ai/test";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { finalValue } from "@/lib/ai/live";
import {
  flushTracing,
  isTracingConfigured,
  SERVICE_NAME,
  startTracing,
} from "@/lib/ai/tracing";
import { askCopper } from "@/lib/copper/assistant/ask";

let model: MockLanguageModelV4;
vi.mock("@ai-sdk/google", () => ({
  createGoogleGenerativeAI: () => () => model,
}));

type StreamPart = Parameters<
  typeof convertArrayToReadableStream<ModelStreamPart>
>[0][number];

const finish = (reason: "stop" | "tool-calls"): StreamPart => ({
  type: "finish",
  finishReason: { unified: reason, raw: undefined },
  usage: {
    inputTokens: {
      total: 500,
      noCache: 500,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: { total: 50, text: 50, reasoning: undefined },
  },
});

const says = (text: string): StreamPart[] => [
  { type: "text-start", id: "t1" },
  { type: "text-delta", id: "t1", delta: text },
  { type: "text-end", id: "t1" },
  finish("stop"),
];

const calls = (toolName: string, input: unknown): StreamPart[] => [
  {
    type: "tool-call",
    toolCallId: "call-1",
    toolName,
    input: JSON.stringify(input),
  },
  finish("tool-calls"),
];

function modelStreaming(...steps: StreamPart[][]) {
  let step = 0;
  return new MockLanguageModelV4({
    doStream: async () => ({
      stream: convertArrayToReadableStream(steps[step++] ?? says("")),
    }),
  });
}

const exporter = new InMemorySpanExporter();

const keys = {
  LANGFUSE_PUBLIC_KEY: "pk-lf-test",
  LANGFUSE_SECRET_KEY: "sk-lf-test",
  LANGFUSE_BASE_URL: "http://localhost:1",
};

/** Runs one Ask Copper turn and returns the spans Langfuse would receive. */
async function traceOf(request: Parameters<typeof askCopper>[0]) {
  exporter.reset();
  const progress = await finalValue((await askCopper(request)).progress);
  await flushTracing();
  return { progress, spans: exporter.getFinishedSpans() };
}

const named = (spans: ReadableSpan[], name: string) =>
  spans.filter((s) => s.name === name);
const attr = (span: ReadableSpan | undefined, key: string) =>
  span?.attributes[key];
const parentOf = (span: ReadableSpan) => span.parentSpanContext?.spanId;

beforeAll(() => {
  for (const [name, value] of Object.entries(keys)) vi.stubEnv(name, value);
  expect(startTracing({ exporter })).toBe(true);
});

afterEach(() => {
  vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "");
  vi.spyOn(console, "info").mockImplementation(() => {});
});

describe("tracing setup (lib/ai/tracing.ts)", () => {
  it("is on only when both Langfuse keys are set", () => {
    expect(isTracingConfigured()).toBe(true);
    vi.stubEnv("LANGFUSE_SECRET_KEY", "");
    expect(isTracingConfigured()).toBe(false);
    vi.stubEnv("LANGFUSE_SECRET_KEY", keys.LANGFUSE_SECRET_KEY);
  });
});

describe("an Ask Copper turn in Langfuse", () => {
  it("is one trace: Copper hands off to the Refund agent, each with its tools and model calls", async () => {
    vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
    vi.spyOn(console, "info").mockImplementation(() => {});
    model = modelStreaming(
      calls("transfer_to_refund_agent", { reason: "refund order 1001" }),
      calls("request_refund", { orderId: "1001" }),
      says("Done: order #1001 is refunded."),
    );

    const { progress, spans } = await traceOf({
      messages: [{ role: "user", content: "Refund order 1001 please" }],
      useTools: true,
      desk: { limitUsd: 50, refunds: [] },
      sessionId: "chat-123",
    });
    expect(progress.reply).toBe("Done: order #1001 is refunded.");

    // One trace for the whole turn, and every span names the app.
    expect(new Set(spans.map((s) => s.spanContext().traceId)).size).toBe(1);
    for (const span of spans)
      expect(span.resource.attributes["service.name"]).toBe(SERVICE_NAME);

    // The turn: named, grouped into the chat's session, with what was asked,
    // the reply, and who answered.
    const [turn] = named(spans, "Ask Copper");
    expect(attr(turn, A.TRACE_NAME)).toBe("ask-copper");
    expect(attr(turn, A.TRACE_SESSION_ID)).toBe("chat-123");
    expect(attr(turn, A.OBSERVATION_INPUT)).toContain("Refund order 1001");
    expect(attr(turn, A.OBSERVATION_OUTPUT)).toContain("refunded");
    expect(
      JSON.stringify(
        Object.entries(turn.attributes).filter(([k]) =>
          k.startsWith(A.OBSERVATION_METADATA),
        ),
      ),
    ).toContain("Refund agent");

    // Each agent is its own span inside the turn.
    const [copper] = named(spans, "Copper");
    const [refund] = named(spans, "Refund agent");
    for (const agent of [copper, refund]) {
      expect(attr(agent, A.OBSERVATION_TYPE)).toBe("agent");
      expect(parentOf(agent)).toBe(turn.spanContext().spanId);
    }

    // Everything an agent did sits under its span: its model calls (named
    // for the feature, with tokens and model, which Langfuse prices) and its
    // tool calls (with what went in and came out). Langfuse reads these
    // standard gen_ai fields; a span's duration is the time it took.
    const under = (agent: ReadableSpan) => {
      const ids = new Set([agent.spanContext().spanId]);
      for (let grew = true; grew;) {
        grew = false;
        for (const s of spans)
          if (!ids.has(s.spanContext().spanId) && ids.has(parentOf(s)!)) {
            ids.add(s.spanContext().spanId);
            grew = true;
          }
      }
      return spans.filter((s) => ids.has(s.spanContext().spanId));
    };
    const toolsOf = (agent: ReadableSpan) =>
      under(agent)
        .map((s) => attr(s, "gen_ai.tool.name"))
        .filter(Boolean);
    const agentNamesOf = (agent: ReadableSpan) =>
      new Set(
        under(agent)
          .map((s) => attr(s, "gen_ai.agent.name"))
          .filter(Boolean),
      );
    expect(toolsOf(copper)).toEqual(["transfer_to_refund_agent"]);
    expect(toolsOf(refund)).toEqual(["request_refund"]);
    expect(agentNamesOf(copper)).toEqual(new Set(["ask-copper"]));
    expect(agentNamesOf(refund)).toEqual(new Set(["refund-agent"]));

    const modelCalls = spans.filter(
      (s) => attr(s, "gen_ai.operation.name") === "chat",
    );
    expect(modelCalls).toHaveLength(3);
    for (const call of modelCalls) {
      expect(attr(call, "gen_ai.request.model")).toBe("mock-model-id");
      expect(attr(call, "gen_ai.usage.input_tokens")).toBe(500);
      expect(attr(call, "gen_ai.usage.output_tokens")).toBe(50);
    }
  });

  it("marks a failed turn as an error", async () => {
    vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
    vi.spyOn(console, "info").mockImplementation(() => {});
    model = new MockLanguageModelV4({
      doStream: async () => {
        throw new Error("Gemini is down");
      },
    });

    const { progress, spans } = await traceOf({
      messages: [{ role: "user", content: "Hi" }],
      useTools: true,
    });
    expect(progress.status).toBe("error");
    const [turn] = named(spans, "Ask Copper");
    expect(attr(turn, "langfuse.observation.level")).toBe("ERROR");
  });
});

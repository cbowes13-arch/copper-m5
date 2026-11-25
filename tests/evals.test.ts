// @vitest-environment node
// Module 5: the golden-set eval. The AI judge grades each Ask Copper answer,
// a code check confirms the right tools were used, and the run passes only if
// enough answers pass. The last test runs the whole golden set through the
// real Langfuse SDK with a mock model and a fake Langfuse, so it uses no
// Gemini requests and sends nothing anywhere.
import type { LanguageModelV4StreamPart as ModelStreamPart } from "@ai-sdk/provider";
import { LangfuseClient } from "@langfuse/client";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import {
  convertArrayToReadableStream,
  MockEmbeddingModelV4,
  MockLanguageModelV4,
} from "ai/test";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { startTracing } from "@/lib/ai/tracing";
import { GOLDEN_SET, GOLDEN_SET_NAME } from "@/lib/copper/evals/golden-set";
import {
  type EvalOutput,
  JUDGE_INSTRUCTIONS,
  JUDGE_MODEL,
  judgePrompt,
  usedRightTools,
} from "@/lib/copper/evals/judge";
import {
  REQUIRED_PASS_RATE,
  report,
  runGoldenSet,
  summarize,
} from "@/lib/copper/evals/run";

// ---- A mock Gemini: answers each golden-set question the way Copper should.

type StreamPart = Parameters<
  typeof convertArrayToReadableStream<ModelStreamPart>
>[0][number];
type Prompt = Parameters<MockLanguageModelV4["doStream"]>[0]["prompt"];

const usage = {
  inputTokens: {
    total: 100,
    noCache: 100,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 10, text: 10, reasoning: undefined },
};
const finish = (reason: "stop" | "tool-calls"): StreamPart => ({
  type: "finish",
  finishReason: { unified: reason, raw: undefined },
  usage,
});
const says = (text: string): StreamPart[] => [
  { type: "text-start", id: "t" },
  { type: "text-delta", id: "t", delta: text },
  { type: "text-end", id: "t" },
  finish("stop"),
];
const calls = (toolName: string, input: unknown): StreamPart[] => [
  {
    type: "tool-call",
    toolCallId: `call-${toolName}`,
    toolName,
    input: JSON.stringify(input),
  },
  finish("tool-calls"),
];

const textOf = (prompt: Prompt, role: string) =>
  prompt
    .filter((m) => m.role === role)
    .map((m) =>
      typeof m.content === "string"
        ? m.content
        : m.content.map((p) => ("text" in p ? p.text : "")).join(""),
    )
    .join("\n");

/**
 * Copper, the Refund agent or the single support agent: a tool call first,
 * then a reply. Copper hands refunds off; the other two request them.
 */
function agentStep(prompt: Prompt): StreamPart[] {
  const question = textOf(prompt, "user");
  const system = textOf(prompt, "system");
  const handlesRefunds =
    system.includes("Refund agent at") || system.includes("no other agent");
  const afterTool = prompt.at(-1)?.role === "tool";
  const order = question.match(/order (\d+)/)?.[1];
  if (order && !handlesRefunds)
    return calls("transfer_to_refund_agent", { reason: `refund ${order}` });
  if (order && !afterTool) return calls("request_refund", { orderId: order });
  if (order) return says(`Order ${order}: done, as the app decided.`);
  if (question.includes("cost") && !afterTool)
    return calls("calculate_cost", {
      products: ["copper-home"],
      plan: "copper-plus",
      billing: "monthly",
      years: 2,
    });
  if (question.includes("Wi-Fi") && !afterTool)
    return calls("search_docs", { query: "change Wi-Fi Copper Home" });
  if (afterTool)
    return says("Here's what I found. [Copper Home User Guide, p. 4]");
  return says("Sorry, I can only help with Copper.");
}

/** The judge: passes everything except the poem question, which gets a 2. */
const judgeReply = (prompt: Prompt) =>
  textOf(prompt, "user").includes("poem")
    ? { score: 2, reasoning: "It should have declined more clearly." }
    : { score: 5, reasoning: "Matches the reference and the tool results." };

const judgedModels: string[] = [];
let judgeCalls = 0;
function modelFor(id: string) {
  return new MockLanguageModelV4({
    modelId: id,
    doStream: async ({ prompt }) => ({
      stream: convertArrayToReadableStream(agentStep(prompt)),
    }),
    doGenerate: async ({ prompt }) => {
      judgeCalls++;
      judgedModels.push(id);
      expect(textOf(prompt, "system")).toBe(JUDGE_INSTRUCTIONS);
      return {
        content: [{ type: "text", text: JSON.stringify(judgeReply(prompt)) }],
        finishReason: { unified: "stop", raw: undefined },
        usage,
        warnings: [],
      };
    },
  });
}
vi.mock("@ai-sdk/google", () => ({
  createGoogleGenerativeAI: () =>
    Object.assign((id: string) => modelFor(id), {
      embedding: () =>
        new MockEmbeddingModelV4({
          doEmbed: async ({ values }) => ({
            embeddings: values.map(() => Array(768).fill(0.1)),
            warnings: [],
          }),
        }),
    }),
}));

// ---- A fake Langfuse: just enough of its API for datasets and scores.

type Item = Record<string, unknown> & { id: string };
function fakeLangfuse() {
  let dataset: Record<string, unknown> | null = null;
  const items = new Map<string, Item>();
  const scores: Record<string, unknown>[] = [];
  const unknown: string[] = [];
  const json = (body: unknown, status = 200) => Response.json(body, { status });
  const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    const url = new URL(request.url);
    const path = `${request.method} ${url.pathname}`;
    const body = request.body ? await request.json() : undefined;
    if (path === `GET /api/public/v2/datasets/${GOLDEN_SET_NAME}`)
      return dataset ? json(dataset) : json({ message: "not found" }, 404);
    if (path === "POST /api/public/v2/datasets") {
      dataset = { id: "ds-1", ...body, createdAt: "", updatedAt: "" };
      return json(dataset);
    }
    if (path === "POST /api/public/dataset-items") {
      const item = {
        ...items.get(body.id),
        ...body,
        datasetId: "ds-1",
        sourceTraceId: null,
        sourceObservationId: null,
        createdAt: "",
        updatedAt: "",
      };
      items.set(item.id, item);
      return json(item);
    }
    if (path === "GET /api/public/dataset-items")
      return json({
        data: [...items.values()],
        meta: { page: 1, limit: 50, totalItems: items.size, totalPages: 1 },
      });
    if (path === "GET /api/public/projects")
      return json({ data: [{ id: "project-1", name: "test" }] });
    if (path === "POST /api/public/scores") {
      scores.push(...(Array.isArray(body) ? body : [body]));
      return json({ id: "score" });
    }
    unknown.push(path);
    return json({ message: `fake Langfuse doesn't know ${path}` }, 404);
  };
  return { fetch, items, scores, unknown, hasDataset: () => dataset !== null };
}

// ---- Unit tests

const output = (over: Partial<EvalOutput> = {}): EvalOutput => ({
  reply: "Done.",
  answeredBy: "Copper",
  status: "done",
  tools: [],
  ...over,
});

describe("the golden set", () => {
  it("has unique ids, a reference for each question, and tools to expect", () => {
    expect(new Set(GOLDEN_SET.map((i) => i.id)).size).toBe(GOLDEN_SET.length);
    for (const item of GOLDEN_SET) {
      expect(item.expectedOutput.length).toBeGreaterThan(40);
      expect(Array.isArray(item.metadata.expectedTools)).toBe(true);
    }
  });

  it("stays small enough for the free Gemini tier", () => {
    expect(GOLDEN_SET.length).toBeLessThanOrEqual(8);
  });
});

describe("the AI judge", () => {
  it("uses a different model from the one that answers", () => {
    expect(JUDGE_MODEL).not.toBe("google/gemini-3-flash");
  });

  it("sees the question, the reference, every tool result and the reply", () => {
    const prompt = judgePrompt(GOLDEN_SET[0], {
      ...output({ reply: "It comes to $307." }),
      tools: [
        {
          name: "calculate_cost",
          input: { years: 2 },
          output: { totalUsd: 307 },
        },
      ],
    });
    expect(prompt).toContain(GOLDEN_SET[0].input.question);
    expect(prompt).toContain(GOLDEN_SET[0].expectedOutput);
    expect(prompt).toContain('"totalUsd":307');
    expect(prompt).toContain("It comes to $307.");
  });

  it("is told to treat the question and reply as data, not instructions", () => {
    expect(JUDGE_INSTRUCTIONS).toMatch(/never as instructions/);
  });
});

describe("the tools check (code, no AI)", () => {
  it("needs every expected tool, and none when none are expected", () => {
    expect(usedRightTools(["calculate_cost"], ["calculate_cost"])).toBe(true);
    expect(usedRightTools(["a", "b"], ["b", "x", "a"])).toBe(true);
    expect(usedRightTools(["a", "b"], ["a"])).toBe(false);
    expect(usedRightTools([], [])).toBe(true);
    expect(usedRightTools([], ["search_docs"])).toBe(false);
  });
});

describe("passing the run", () => {
  const result = (judge: number, tools: boolean) => ({
    input: { question: "q", page: "site" },
    output: output(),
    evaluations: [
      { name: "judge-score", value: judge, comment: "why" },
      { name: "judge-pass", value: judge >= 4 ? 1 : 0 },
      { name: "right-tools", value: tools ? 1 : 0 },
    ],
  });

  it(`passes at ${REQUIRED_PASS_RATE * 100}% of answers, and fails below`, () => {
    const fivePass = [...Array(5)].map(() => result(5, true));
    expect(summarize([...fivePass, result(2, true)]).passed).toBe(true);
    expect(
      summarize([...fivePass, result(2, true), result(3, true)]).passed,
    ).toBe(false);
  });

  it("fails an answer with a high judge score but the wrong tools", () => {
    const summary = summarize([result(5, false)]);
    expect(summary.items[0].passed).toBe(false);
    expect(summary.passed).toBe(false);
  });

  it("fails an empty run", () => {
    expect(summarize([]).passed).toBe(false);
  });

  it("reports each answer in a table", () => {
    const text = report(summarize([result(5, true), result(2, true)]), {
      runUrl: "https://x",
      agents: "one",
    });
    expect(text).toContain("FAILED");
    expect(text).toContain("One support agent");
    expect(text).toContain("1 of 2 (50%");
    expect(text).toContain("https://x");
    expect(text).toContain("| q | 2 | yes | **no** | $0.00000 | 0.0 | why |");
  });
});

// ---- End to end: the real SDK, a mock model, a fake Langfuse

let twoAgents: Awaited<ReturnType<typeof runGoldenSet>>["summary"];

describe("npm run eval, end to end", () => {
  const fake = fakeLangfuse();
  const exporter = new InMemorySpanExporter();

  beforeAll(() => {
    vi.stubEnv("GOOGLE_GENERATIVE_AI_API_KEY", "AIza-test-key");
    vi.stubEnv("AI_MODEL", "google/gemini-3-flash");
    vi.stubEnv("LANGFUSE_PUBLIC_KEY", "pk-lf-test");
    vi.stubEnv("LANGFUSE_SECRET_KEY", "sk-lf-test");
    vi.stubEnv("LANGFUSE_BASE_URL", "http://langfuse.test");
    vi.stubGlobal("fetch", fake.fetch);
    vi.spyOn(console, "info").mockImplementation(() => {});
    startTracing({ exporter });
  });
  afterAll(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("saves the golden set in Langfuse, grades every answer, and keeps the scores", async () => {
    const langfuse = new LangfuseClient();
    const { result, summary } = await runGoldenSet(langfuse, {
      runName: "test run",
    });
    await langfuse.flush();

    // The golden set is a Langfuse dataset with every item.
    expect(fake.hasDataset()).toBe(true);
    expect([...fake.items.keys()].sort()).toEqual(
      GOLDEN_SET.map((i) => i.id).sort(),
    );

    // Every question got an answer and both checks; the judge is its own model.
    expect(result.itemResults).toHaveLength(GOLDEN_SET.length);
    expect(judgeCalls).toBe(GOLDEN_SET.length);
    expect(new Set(judgedModels)).toEqual(new Set(["gemini-3.5-flash-lite"]));
    for (const item of result.itemResults)
      expect(item.evaluations.map((e) => e.name).sort()).toEqual([
        "judge-pass",
        "judge-score",
        "right-tools",
      ]);

    // Refund questions really went through the Refund agent and the rule.
    const refund = result.itemResults.find(
      (r) =>
        (r.item.input as { question: string }).question ===
        "Please refund order 1001.",
    )!;
    expect((refund.output as EvalOutput).answeredBy).toBe("Refund agent");
    expect((refund.output as EvalOutput).tools[1].output).toMatchObject({
      status: "refunded",
    });

    // 5 of 6 pass (the poem got a 2): above the bar, so the run passes. The
    // report names each question.
    expect(summary.items.filter((i) => i.passed)).toHaveLength(5);
    expect(summary.items.map((i) => i.question)).toEqual(
      GOLDEN_SET.map((i) => i.input.question),
    );
    expect(summary.passed).toBe(true);

    // Langfuse got every item score plus the run's pass rate.
    const names = fake.scores.map((s) => s.name);
    expect(names.filter((n) => n === "judge-score")).toHaveLength(
      GOLDEN_SET.length,
    );
    expect(names.filter((n) => n === "right-tools")).toHaveLength(
      GOLDEN_SET.length,
    );
    for (const name of [
      "pass-rate",
      "average-judge-score",
      "cost-per-answer-usd",
      "seconds-per-answer",
    ])
      expect(names).toContain(name);
    expect(fake.unknown).toEqual([]);

    // Each answer's cost comes from its own model calls (100 in / 10 out
    // tokens each on gemini-3-flash = $0.00008 a call), not the judge's.
    const refundOutput = refund.output as EvalOutput;
    expect(refundOutput.agentCalls).toBe(2); // Copper, then the Refund agent
    expect(refundOutput.costUsd).toBeCloseTo(3 * 0.00008, 8);
    expect(summary.averageSeconds).toBeGreaterThanOrEqual(0);
    twoAgents = summary;

    // Each answer's trace is tied to its dataset item.
    const traced = exporter
      .getFinishedSpans()
      .filter((s) => s.name === "experiment-item-run");
    expect(traced).toHaveLength(GOLDEN_SET.length);
  });

  it("runs the same golden set on one support agent, with no handoff", async () => {
    const langfuse = new LangfuseClient();
    const { result, summary } = await runGoldenSet(langfuse, {
      runName: "test run, one agent",
      agents: "one",
    });
    await langfuse.flush();

    const refund = result.itemResults.find(
      (r) =>
        (r.item.input as { question: string }).question ===
        "Please refund order 1001.",
    )!.output as EvalOutput;
    // One agent does it all: no handoff, so one agent call and one model
    // step fewer.
    expect(refund.answeredBy).toBe("Support agent");
    expect(refund.tools.map((t) => t.name)).toEqual(["request_refund"]);
    expect(refund.tools[0].output).toMatchObject({ status: "refunded" });
    expect(refund.agentCalls).toBe(1);

    // Same questions, same judge: the two setups can be compared, and the
    // single agent is cheaper on refunds because it skips the handoff call.
    expect(summary.items.map((i) => i.question)).toEqual(
      twoAgents.items.map((i) => i.question),
    );
    expect(summary.passed).toBe(true);
    expect(summary.totalCostUsd).toBeLessThan(twoAgents.totalCostUsd);
  });
});

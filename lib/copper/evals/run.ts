// Module 5: runs the golden set as a Langfuse experiment. Each question goes
// to Ask Copper exactly as the chat panel sends it; the judge and the tools
// check grade each answer; Langfuse keeps the run, its traces and its scores,
// so runs can be compared in Datasets > ask-copper-golden-set > Runs.
// Part 4: the same golden set runs on the two agents (Copper and the Refund
// agent, what the app ships) or on one support agent, and each run records
// pass rate, cost and speed, so the two can be compared side by side.
// `npm run eval` / `npm run eval:single` (scripts/eval.mjs) run it.
import type {
  Evaluation,
  LangfuseClient,
  RunEvaluator,
} from "@langfuse/client";
import { type AiCallRow, watchAiCalls } from "@/lib/ai/call";
import { finalValue } from "@/lib/ai/live";
import { type AgentSetup, askCopper } from "@/lib/copper/assistant/ask";
import { AGENT_NAMES, toolCallsOf } from "@/lib/copper/assistant/progress";
import {
  GOLDEN_SET,
  GOLDEN_SET_NAME,
  type GoldenItem,
} from "@/lib/copper/evals/golden-set";
import {
  type EvalOutput,
  JUDGE_MODEL,
  judgeEvaluator,
  PASS_SCORE,
  toolsEvaluator,
} from "@/lib/copper/evals/judge";
import { DEFAULT_LIMIT_USD } from "@/lib/copper/orders";

/** The run fails (and so does CI) when fewer answers than this pass. */
export const REQUIRED_PASS_RATE = 0.8;

export const SETUP_NAMES: Record<AgentSetup, string> = {
  two: "Two agents (Copper + Refund agent)",
  one: "One support agent",
};

/**
 * Asks Ask Copper one golden-set question, as a fresh chat, and measures what
 * the answer cost (every model call it made, from the ai_calls log) and how
 * long it took. The judge's own call comes after, so it isn't counted.
 */
export async function askForEval(
  input: GoldenItem["input"],
  agents: AgentSetup = "two",
): Promise<EvalOutput> {
  const calls: AiCallRow[] = [];
  const stop = watchAiCalls((row) => calls.push(row));
  const started = performance.now();
  try {
    const { progress } = await askCopper({
      messages: [{ role: "user", content: input.question }],
      useTools: true,
      agents,
      // The Support page sends a fresh Refund Console: the starting limit and
      // no refunds yet, so every run starts from the same place.
      ...(input.page === "support" && {
        desk: { limitUsd: DEFAULT_LIMIT_USD, refunds: [] },
      }),
    });
    const final = await finalValue(progress);
    return {
      reply: final.reply,
      answeredBy: AGENT_NAMES[final.agent ?? "copper"],
      status: final.status,
      tools: toolCallsOf(final),
      costUsd: calls.reduce((sum, c) => sum + (c.cost_usd ?? 0), 0),
      seconds: (performance.now() - started) / 1000,
      agentCalls: calls.filter((c) => c.input_tokens !== null).length,
    };
  } finally {
    stop();
  }
}

/**
 * Saves the golden set as a Langfuse dataset (creating it the first time),
 * archives items no longer in the set, and returns the current items.
 */
export async function syncGoldenSet(
  langfuse: LangfuseClient,
): Promise<DatasetItem[]> {
  try {
    await langfuse.api.datasets.get(GOLDEN_SET_NAME);
  } catch {
    await langfuse.api.datasets.create({
      name: GOLDEN_SET_NAME,
      description:
        "Questions Ask Copper must get right, each with a reference answer for the AI judge (lib/copper/evals/golden-set.ts).",
    });
  }
  for (const item of GOLDEN_SET)
    await langfuse.dataset.createItem({
      datasetName: GOLDEN_SET_NAME,
      id: item.id,
      input: item.input,
      expectedOutput: item.expectedOutput,
      metadata: item.metadata,
      status: "ACTIVE",
    });
  const dataset = await langfuse.dataset.get(GOLDEN_SET_NAME);
  const ids = new Set(GOLDEN_SET.map((item) => item.id));
  for (const stale of dataset.items.filter((item) => !ids.has(item.id)))
    await langfuse.dataset.createItem({
      datasetName: GOLDEN_SET_NAME,
      id: stale.id,
      status: "ARCHIVED",
    });
  return dataset.items.filter((item) => ids.has(item.id));
}

type DatasetItem = Awaited<
  ReturnType<LangfuseClient["dataset"]["get"]>
>["items"][number];

const scoreOf = (evaluations: Evaluation[], name: string) => {
  const value = evaluations.find((e) => e.name === name)?.value;
  return typeof value === "number" ? value : null;
};

/** One answer passes when the judge passes it and it used the right tools. */
export function itemPassed(evaluations: Evaluation[]): boolean {
  return (
    scoreOf(evaluations, "judge-pass") === 1 &&
    scoreOf(evaluations, "right-tools") === 1
  );
}

export type ItemSummary = {
  question: string;
  judgeScore: number | null;
  judgeReason: string;
  rightTools: boolean;
  passed: boolean;
  reply: string;
  costUsd: number;
  seconds: number;
};

export type RunSummary = {
  items: ItemSummary[];
  passRate: number;
  averageJudgeScore: number | null;
  /** What all the answers cost together (not counting the judge). */
  totalCostUsd: number;
  averageCostUsd: number;
  averageSeconds: number;
  passed: boolean;
};

type ItemResultLike = {
  /** Set for plain data; dataset items keep their input on `item`. */
  input?: unknown;
  item?: { input?: unknown };
  output: unknown;
  evaluations: Evaluation[];
};

const average = (values: number[]) =>
  values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;

/** Pass rate, judge score, cost, speed, and whether the run clears the bar. */
export function summarize(itemResults: ItemResultLike[]): RunSummary {
  const items = itemResults.map(({ input, item, output, evaluations }) => {
    const answer = output as Partial<EvalOutput> | undefined;
    return {
      question:
        ((input ?? item?.input) as GoldenItem["input"] | undefined)?.question ??
        "?",
      judgeScore: scoreOf(evaluations, "judge-score"),
      judgeReason: String(
        evaluations.find((e) => e.name === "judge-score")?.comment ??
          "(the judge gave no score)",
      ),
      rightTools: scoreOf(evaluations, "right-tools") === 1,
      passed: itemPassed(evaluations),
      reply: answer?.reply ?? "(no reply)",
      costUsd: answer?.costUsd ?? 0,
      seconds: answer?.seconds ?? 0,
    };
  });
  const scores = items.flatMap((i) =>
    i.judgeScore === null ? [] : [i.judgeScore],
  );
  const passRate = items.length
    ? items.filter((i) => i.passed).length / items.length
    : 0;
  const totalCostUsd = items.reduce((sum, i) => sum + i.costUsd, 0);
  return {
    items,
    passRate,
    averageJudgeScore: scores.length ? average(scores) : null,
    totalCostUsd,
    averageCostUsd: items.length ? totalCostUsd / items.length : 0,
    averageSeconds: average(items.map((i) => i.seconds)),
    passed: items.length > 0 && passRate >= REQUIRED_PASS_RATE,
  };
}

/** Run-level scores in Langfuse: pass rate, judge score, cost and speed. */
const runScores: RunEvaluator = async ({ itemResults }) => {
  const summary = summarize(itemResults);
  return [
    {
      name: "pass-rate",
      value: summary.passRate,
      comment: `${summary.items.filter((i) => i.passed).length} of ${summary.items.length} passed (needs ${Math.round(REQUIRED_PASS_RATE * 100)}%)`,
      dataType: "NUMERIC",
    },
    ...(summary.averageJudgeScore === null
      ? []
      : [
          {
            name: "average-judge-score",
            value: summary.averageJudgeScore,
            dataType: "NUMERIC" as const,
          },
        ]),
    {
      name: "cost-per-answer-usd",
      value: summary.averageCostUsd,
      comment: `$${summary.totalCostUsd.toFixed(5)} for all ${summary.items.length} answers, not counting the judge`,
      dataType: "NUMERIC",
    },
    {
      name: "seconds-per-answer",
      value: summary.averageSeconds,
      dataType: "NUMERIC",
    },
  ];
};

/** Runs the golden set on Ask Copper and returns Langfuse's result and the summary. */
export async function runGoldenSet(
  langfuse: LangfuseClient,
  {
    runName,
    agents = "two",
    metadata = {},
  }: {
    runName: string;
    agents?: AgentSetup;
    metadata?: Record<string, unknown>;
  },
) {
  const data = await syncGoldenSet(langfuse);
  const result = await langfuse.experiment.run({
    name: "Ask Copper golden set",
    runName,
    description: `${SETUP_NAMES[agents]}, graded by an AI judge (${JUDGE_MODEL}); an answer passes at ${PASS_SCORE}/5 with the right tools.`,
    metadata: { agents: SETUP_NAMES[agents], ...metadata },
    data,
    task: async ({ input }) => askForEval(input as GoldenItem["input"], agents),
    evaluators: [judgeEvaluator, toolsEvaluator],
    runEvaluators: [runScores],
    // One question at a time, to stay under the free tier's rate limit (and
    // so each answer's cost and time are its own).
    maxConcurrency: 1,
  });
  return { result, summary: summarize(result.itemResults) };
}

const cell = (text: string) =>
  text.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();

/** A Markdown report: for the terminal, the job summary and the PR comment. */
export function report(
  summary: RunSummary,
  { runUrl, agents = "two" }: { runUrl?: string; agents?: AgentSetup } = {},
): string {
  const percent = Math.round(summary.passRate * 100);
  const passed = summary.items.filter((i) => i.passed).length;
  const lines = [
    `## Ask Copper golden set: ${summary.passed ? "passed" : "FAILED"}`,
    "",
    `**${SETUP_NAMES[agents]}**, graded by an AI judge (${JUDGE_MODEL}).`,
    "",
    "| Pass rate | Average judge score | Cost per answer | Seconds per answer |",
    "| --- | --- | --- | --- |",
    `| ${passed} of ${summary.items.length} (${percent}%; needs ${Math.round(REQUIRED_PASS_RATE * 100)}%) | ${summary.averageJudgeScore?.toFixed(1) ?? "n/a"} / 5 | $${summary.averageCostUsd.toFixed(5)} ($${summary.totalCostUsd.toFixed(4)} in all) | ${summary.averageSeconds.toFixed(1)} s |`,
    ...(runUrl ? ["", `Scores and traces in Langfuse: ${runUrl}`] : []),
    "",
    "| Question | Judge (1-5) | Right tools | Passed | Cost | Seconds | Why |",
    "| --- | --- | --- | --- | --- | --- | --- |",
    ...summary.items.map(
      (i) =>
        `| ${cell(i.question)} | ${i.judgeScore ?? "-"} | ${i.rightTools ? "yes" : "no"} | ${i.passed ? "yes" : "**no**"} | $${i.costUsd.toFixed(5)} | ${i.seconds.toFixed(1)} | ${cell(i.judgeReason)} |`,
    ),
  ];
  return lines.join("\n");
}

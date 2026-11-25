// npm run eval: asks Ask Copper every question in the golden set
// (lib/copper/evals/golden-set.ts), has an AI judge grade each answer, saves
// the run and its scores in Langfuse, and prints a report with the pass rate,
// cost and speed. Exits with an error when too few answers pass, which fails
// the GitHub Actions check.
// npm run eval:single (--agents=one) runs the same golden set on the single
// support agent instead of the two agents, for the Part 4 comparison.
// Uses real Gemini requests (about 15-20 per run): it needs
// GOOGLE_GENERATIVE_AI_API_KEY and the three LANGFUSE_ variables.
import { appendFileSync, writeFileSync } from "node:fs";
import { registerHooks } from "node:module";

// The app's TypeScript files import each other as "@/lib/...": point Node at
// them, and tell it they're ES modules so it doesn't print a warning. Next.js
// files like "next/server" need their ".js" outside Next's own bundler.
const root = new URL("../", import.meta.url);
registerHooks({
  resolve: (specifier, context, next) =>
    next(
      specifier.startsWith("@/")
        ? new URL(
            /\.\w+$/.test(specifier)
              ? specifier.slice(2)
              : `${specifier.slice(2)}.ts`,
            root,
          ).href
        : /^next\/[a-z-]+$/.test(specifier)
          ? `${specifier}.js`
          : specifier,
      context,
    ),
  // The app's data (catalog, orders, passages) is JSON imported without
  // `with { type: "json" }`, which Next.js allows and plain Node needs.
  load: (url, context, next) =>
    next(
      url,
      url.endsWith(".ts")
        ? { ...context, format: "module-typescript" }
        : url.endsWith(".json")
          ? {
              ...context,
              importAttributes: { ...context.importAttributes, type: "json" },
            }
          : context,
    ),
});

try {
  process.loadEnvFile(new URL(".env.local", root));
} catch {
  // No .env.local (as in GitHub Actions): the keys come from the environment.
}

const { isAiConfigured } = await import("../lib/ai/call.ts");
const { defaultModel } = await import("../lib/ai/models.ts");
const { flushTracing, isTracingConfigured, startTracing } =
  await import("../lib/ai/tracing.ts");
const { JUDGE_MODEL } = await import("../lib/copper/evals/judge.ts");
const { report, runGoldenSet } = await import("../lib/copper/evals/run.ts");
const { LangfuseClient } = await import("@langfuse/client");

// --agents=one runs the single support agent; two agents is the default.
const agentsArg = process.argv
  .find((arg) => arg.startsWith("--agents="))
  ?.slice("--agents=".length);
const agents = (agentsArg ?? process.env.EVAL_AGENTS ?? "").trim() || "two";
if (agents !== "one" && agents !== "two") {
  console.error(`eval: --agents must be "one" or "two", not "${agents}".`);
  process.exit(1);
}

/** Writes the report where CI shows it: the job summary and the PR comment. */
function publish(text) {
  if (process.env.GITHUB_STEP_SUMMARY)
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${text}\n`);
  if (process.env.EVAL_REPORT_FILE)
    writeFileSync(process.env.EVAL_REPORT_FILE, `${text}\n`);
}

const missing = [
  !isAiConfigured() && "GOOGLE_GENERATIVE_AI_API_KEY",
  !isTracingConfigured() && "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY",
].filter(Boolean);
if (missing.length) {
  const message = `eval: set ${missing.join(" and ")} (in .env.local, or as GitHub Actions secrets in CI).`;
  console.error(message);
  publish(`## Ask Copper golden set: FAILED\n\n${message}`);
  process.exit(1);
}

// Eval traces get their own environment in Langfuse, apart from real chats.
process.env.LANGFUSE_TRACING_ENVIRONMENT ??= "eval";
startTracing();
const langfuse = new LangfuseClient();

// In CI the run is named for the pull request and commit; locally, for now.
const commit = process.env.GITHUB_SHA?.slice(0, 7);
const pr = process.env.EVAL_PR_NUMBER;
const setup = agents === "one" ? "one agent" : "two agents";
const when = new Date().toISOString().slice(0, 16).replace("T", " ");
const runName =
  process.env.EVAL_RUN_NAME ??
  (pr
    ? `${setup}: PR #${pr} @ ${commit}`
    : `${setup}: ${commit ?? "local"} ${when}`);

let exitCode = 1;
try {
  console.log(`eval: running the golden set as "${runName}"...`);
  const { result, summary } = await runGoldenSet(langfuse, {
    runName,
    agents,
    metadata: {
      model: defaultModel(),
      judgeModel: JUDGE_MODEL,
      ...(commit && { commit }),
      ...(pr && { pullRequest: pr }),
    },
  });
  const text = report(summary, {
    runUrl: result.experimentUrl ?? result.datasetRunUrl,
    agents,
  });
  console.log(`\n${text}\n`);
  publish(text);
  exitCode = summary.passed ? 0 : 1;
} catch (error) {
  console.error(`eval: the run failed: ${error?.stack ?? error}`);
  publish(
    `## Ask Copper golden set: FAILED\n\nThe run stopped with an error: ${error?.message ?? error}`,
  );
} finally {
  // Send every trace and score to Langfuse before the process ends.
  await flushTracing();
  await langfuse.flush();
}
process.exit(exitCode);

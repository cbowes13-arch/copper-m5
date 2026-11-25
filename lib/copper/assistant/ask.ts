"use server";

// The server action the Ask Copper chat calls. It streams the conversation to
// the model through callStream (lib/ai/call.ts), with or without the tools,
// and sends back each step as it happens, so the panel can show Copper
// thinking and using its tools.
// Module 4: on the Support page the chat also sends the Refund Console's
// "desk" (the auto-refund limit and the refunds so far) and which agent is
// talking. Copper can hand the conversation to the Refund agent, a second
// agent with its own instructions and tools, and the Refund agent can hand it
// back. The agent that ends the turn answers the next message too.
// Module 5: each turn is one trace in Langfuse (lib/ai/tracing.ts): an "Ask
// Copper" span holding one span per agent that answered, and inside each its
// model calls and tool calls, with tokens, cost and time. Turns from the same
// chat share a session.
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";
import { hasToolCall, stepCountIs } from "ai";
import { after } from "next/server";
import { callStream, THOUGHT_SUMMARIES } from "@/lib/ai/call";
import { createLive, type Live } from "@/lib/ai/live";
import { flushTracing, isTracingConfigured } from "@/lib/ai/tracing";
import {
  COPPER_HANDOFF,
  COPPER_INSTRUCTIONS,
  REFUND_AGENT_INSTRUCTIONS,
  SUPPORT_AGENT_INSTRUCTIONS,
} from "@/lib/copper/assistant/instructions";
import {
  type Agent,
  AGENT_NAMES,
  type Progress,
  type ToolStep,
} from "@/lib/copper/assistant/progress";
import {
  copperTools,
  refundAgentTools,
  supportAgentTools,
  transferToRefundAgent,
} from "@/lib/copper/assistant/tools";
import type { Desk } from "@/lib/copper/orders";

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type AskCopperRequest = {
  messages: ChatMessage[];
  useTools: boolean;
  /** The Refund Console's limit and refunds (Support page only). */
  desk?: Desk;
  /** Who answers first: whoever ended the last turn. Copper by default. */
  agent?: Agent;
  /** Groups a chat's turns into one Langfuse session (Module 5). */
  sessionId?: string;
  /**
   * Module 5, Part 4: "two" (the default) is Copper plus the Refund agent;
   * "one" is a single support agent with every tool and no handoff.
   */
  agents?: AgentSetup;
};

export type AgentSetup = "two" | "one";

export type AskCopperResult = { progress: Live<Progress> };

/** At most this many model steps (tool calls plus the reply) per agent. */
const MAX_STEPS = 5;

/** Copper → Refund agent → Copper at most, so a turn can't bounce forever. */
const MAX_HANDOFFS = 2;

const HANDOFF_TOOLS: Record<string, Agent> = {
  transfer_to_refund_agent: "refund",
  transfer_to_copper: "copper",
};

// A handoff ends the agent's turn: the other agent answers.
const STOP = [
  stepCountIs(MAX_STEPS),
  hasToolCall("transfer_to_refund_agent", "transfer_to_copper"),
];

const SORRY = "Sorry, something went wrong on my end. Please try again.";

/** One agent's call: its instructions and only its own tools. */
function agentCall(agent: Agent, messages: ChatMessage[], desk?: Desk) {
  if (agent === "support" && desk)
    return callStream(
      { module: 5, feature: "support-agent" },
      {
        messages,
        providerOptions: THOUGHT_SUMMARIES,
        instructions: SUPPORT_AGENT_INSTRUCTIONS,
        tools: supportAgentTools(desk),
        stopWhen: STOP,
      },
    );
  if (agent === "refund" && desk)
    return callStream(
      { module: 4, feature: "refund-agent" },
      {
        messages,
        providerOptions: THOUGHT_SUMMARIES,
        instructions: REFUND_AGENT_INSTRUCTIONS,
        tools: refundAgentTools(desk),
        stopWhen: STOP,
      },
    );
  return callStream(
    { module: 2, feature: "ask-copper" },
    {
      messages,
      providerOptions: THOUGHT_SUMMARIES,
      instructions: desk
        ? `${COPPER_INSTRUCTIONS}\n\n${COPPER_HANDOFF}`
        : COPPER_INSTRUCTIONS,
      tools: desk
        ? { ...copperTools, transfer_to_refund_agent: transferToRefundAgent }
        : copperTools,
      stopWhen: STOP,
    },
  );
}

export async function askCopper({
  messages,
  useTools,
  desk,
  agent = "copper",
  sessionId,
  agents = "two",
}: AskCopperRequest): Promise<AskCopperResult> {
  const { live, update, done } = createLive<Progress>();
  // One agent answers everything on the Support page; two agents start
  // with whoever ended the last turn.
  const first: Agent =
    desk && useTools
      ? agents === "one"
        ? "support"
        : agent === "support"
          ? "copper"
          : agent
      : "copper";
  const progress: Progress = {
    steps: [],
    reply: "",
    status: "working",
    agent: first,
  };
  // A fresh copy each time, so earlier updates never change underneath the
  // panel.
  const snapshot = (): Progress => ({
    ...progress,
    steps: progress.steps.map((step) => ({ ...step })),
  });
  // Tool call ids are per agent call, so each call's are kept apart.
  let call = 0;
  const toolStep = (id: string) =>
    progress.steps.find(
      (step): step is ToolStep =>
        step.type === "tool" && step.id === `${call}:${id}`,
    );

  /** Reads one agent's stream into progress. Returns who to hand off to. */
  async function follow(
    current: Agent,
    result: ReturnType<typeof callStream>,
  ): Promise<Agent | undefined> {
    let handoff: Agent | undefined;
    for await (const part of result.fullStream) {
      switch (part.type) {
        case "reasoning-delta": {
          const last = progress.steps.at(-1);
          if (last?.type === "thinking" && last.agent === current)
            last.text += part.text;
          else
            progress.steps.push({
              type: "thinking",
              text: part.text,
              agent: current,
            });
          break;
        }
        case "tool-input-start":
          progress.steps.push({
            type: "tool",
            id: `${call}:${part.id}`,
            agent: current,
            name: part.toolName,
            state: "calling",
          });
          break;
        case "tool-call": {
          const step = toolStep(part.toolCallId);
          if (step) step.input = part.input;
          else
            progress.steps.push({
              type: "tool",
              id: `${call}:${part.toolCallId}`,
              agent: current,
              name: part.toolName,
              state: "calling",
              input: part.input,
            });
          break;
        }
        case "tool-result": {
          const step = toolStep(part.toolCallId);
          if (step) Object.assign(step, { output: part.output, state: "done" });
          if (part.toolName in HANDOFF_TOOLS)
            handoff = HANDOFF_TOOLS[part.toolName];
          break;
        }
        case "tool-error": {
          const step = toolStep(part.toolCallId);
          const message =
            part.error instanceof Error
              ? part.error.message
              : String(part.error);
          if (step) Object.assign(step, { output: message, state: "error" });
          break;
        }
        case "text-delta":
          progress.reply += part.text;
          break;
        case "error":
          throw part.error;
        default:
          continue;
      }
      update(snapshot());
    }
    return handoff;
  }

  /** One agent's part of the turn, as its own span in the Langfuse trace. */
  const traced = (current: Agent, run: () => ReturnType<typeof callStream>) =>
    startActiveObservation(
      AGENT_NAMES[current],
      async (span) => {
        const next = await follow(current, run());
        span.update({
          output: progress.reply,
          metadata: next ? { handedTo: AGENT_NAMES[next] } : undefined,
        });
        return next;
      },
      { asType: "agent" },
    );

  async function turn() {
    if (!useTools) {
      // Without tools, send no instructions either: "always use your
      // tools" with no tools makes the model answer with nothing.
      await traced("copper", () =>
        callStream(
          { module: 2, feature: "ask-copper" },
          { messages, providerOptions: THOUGHT_SUMMARIES },
        ),
      );
      return;
    }
    let current = first;
    for (let handoffs = 0; ; handoffs++) {
      const next = await traced(current, () =>
        agentCall(current, messages, desk),
      );
      if (!next || next === current || handoffs >= MAX_HANDOFFS) break;
      progress.steps.push({ type: "handoff", from: current, to: next });
      progress.agent = current = next;
      progress.reply = "";
      call++;
      update(snapshot());
    }
  }

  // Read the stream in the background, so the panel gets the first update now.
  void propagateAttributes(
    {
      traceName: "ask-copper",
      sessionId,
      tags: [
        desk ? "support-page" : "site",
        useTools ? "tools" : "no-tools",
        agents === "one" ? "one-agent" : "two-agents",
      ],
    },
    () =>
      startActiveObservation("Ask Copper", async (span) => {
        span.update({ input: messages.at(-1)?.content });
        try {
          await turn();
          progress.status = "done";
          if (!progress.reply.trim()) progress.reply = SORRY;
        } catch (error) {
          progress.status = "error";
          if (!progress.reply.trim()) progress.reply = SORRY;
          span.update({
            level: "ERROR",
            statusMessage:
              error instanceof Error ? error.message : String(error),
          });
        }
        const handoffs = progress.steps.filter((s) => s.type === "handoff");
        span.update({
          output: progress.reply,
          metadata: {
            answeredBy: AGENT_NAMES[progress.agent ?? "copper"],
            handoffs: handoffs.length,
            toolCalls: progress.steps.filter((s) => s.type === "tool").length,
            status: progress.status,
          },
        });
        done(snapshot());
      }),
  );

  // Send the trace once the reply has finished streaming to the browser.
  if (isTracingConfigured()) {
    try {
      after(flushTracing);
    } catch {
      // Not inside a request (a script or a test): the caller flushes.
    }
  }

  return { progress: live };
}

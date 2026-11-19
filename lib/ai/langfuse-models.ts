// Module 5: tells Langfuse what each approved Gemini model costs, so every
// traced model call shows its cost. Langfuse works cost out from a call's
// tokens and its model name; these are the same prices as lib/ai/pricing.ts,
// so Langfuse and the [ai_calls] log agree.
// Run once per Langfuse project: npm run langfuse:models
import { MODELS } from "@/lib/ai/models";
import { PRICING } from "@/lib/ai/pricing";

export type ModelDefinition = {
  modelName: string;
  matchPattern: string;
  unit: "TOKENS";
  pricingTiers: {
    name: string;
    isDefault: true;
    priority: 0;
    conditions: [];
    prices: { input: number; output: number };
  }[];
};

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** One Langfuse model definition per approved model, priced per token. */
export function modelDefinitions(): ModelDefinition[] {
  return MODELS.map(({ id, googleId }) => ({
    // Traces carry Google's own name for the model, like
    // "gemini-3.5-flash-lite", sometimes with "models/" or a version suffix.
    modelName: googleId,
    matchPattern: `(?i)^(models/)?${escape(googleId)}(-\\d+)?$`,
    unit: "TOKENS",
    pricingTiers: [
      {
        name: "Standard",
        isDefault: true,
        priority: 0,
        conditions: [],
        prices: {
          input: PRICING[id].input / 1_000_000,
          output: PRICING[id].output / 1_000_000,
        },
      },
    ],
  }));
}

type ExistingModel = {
  id: string;
  modelName: string;
  isLangfuseManaged: boolean;
};

export type SyncResult = { modelName: string; action: "created" | "replaced" };

/**
 * Saves the definitions in the Langfuse project the keys belong to. A model
 * this app defined before is replaced, so changed prices take effect.
 */
export async function syncModels({
  baseUrl = "https://cloud.langfuse.com",
  publicKey,
  secretKey,
  fetch = globalThis.fetch,
}: {
  baseUrl?: string;
  publicKey: string;
  secretKey: string;
  fetch?: typeof globalThis.fetch;
}): Promise<SyncResult[]> {
  const root = `${baseUrl.replace(/\/+$/, "")}/api/public/models`;
  const auth = `Basic ${Buffer.from(`${publicKey}:${secretKey}`).toString("base64")}`;
  const request = async (url: string, init: RequestInit = {}) => {
    const response = await fetch(url, {
      ...init,
      headers: {
        Authorization: auth,
        "Content-Type": "application/json",
        ...init.headers,
      },
    });
    if (!response.ok)
      throw new Error(
        `Langfuse said ${response.status} to ${init.method ?? "GET"} ${url.replace(root, "/api/public/models")}: ${await response.text()}`,
      );
    return response.status === 204 ? null : response.json();
  };

  const existing: ExistingModel[] = [];
  for (let page = 1; ; page++) {
    const { data, meta } = await request(`${root}?page=${page}&limit=100`);
    existing.push(...data);
    if (page >= (meta?.totalPages ?? 1)) break;
  }

  const results: SyncResult[] = [];
  for (const definition of modelDefinitions()) {
    const ours = existing.filter(
      (m) => !m.isLangfuseManaged && m.modelName === definition.modelName,
    );
    for (const model of ours)
      await request(`${root}/${encodeURIComponent(model.id)}`, {
        method: "DELETE",
      });
    await request(root, { method: "POST", body: JSON.stringify(definition) });
    results.push({
      modelName: definition.modelName,
      action: ours.length ? "replaced" : "created",
    });
  }
  return results;
}

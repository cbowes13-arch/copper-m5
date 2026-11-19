// @vitest-environment node
// Module 5: npm run langfuse:models tells Langfuse each Gemini model's price,
// so traced calls show their cost. A fake Langfuse stands in, so no keys.
import { describe, expect, it } from "vitest";
import { modelDefinitions, syncModels } from "@/lib/ai/langfuse-models";
import { MODELS } from "@/lib/ai/models";
import { PRICING } from "@/lib/ai/pricing";

/** Langfuse writes "(?i)" for case-insensitive; JavaScript uses the i flag. */
const matches = (pattern: string, model: string) =>
  new RegExp(pattern.replace(/^\(\?i\)/, ""), "i").test(model);

describe("modelDefinitions", () => {
  it("prices every approved model per token, from lib/ai/pricing.ts", () => {
    const definitions = modelDefinitions();
    expect(definitions).toHaveLength(MODELS.length);
    for (const { id, googleId } of MODELS) {
      const definition = definitions.find((d) => d.modelName === googleId)!;
      const [tier] = definition.pricingTiers;
      expect(tier.prices.input * 1_000_000).toBeCloseTo(PRICING[id].input);
      expect(tier.prices.output * 1_000_000).toBeCloseTo(PRICING[id].output);
      expect(tier).toMatchObject({ isDefault: true, priority: 0 });
    }
  });

  it("matches the model names Google puts in traces, and only those", () => {
    const pattern = (name: string) =>
      modelDefinitions().find((d) => d.modelName === name)!.matchPattern;
    const lite = pattern("gemini-3.5-flash-lite");
    expect(matches(lite, "gemini-3.5-flash-lite")).toBe(true);
    expect(matches(lite, "models/gemini-3.5-flash-lite")).toBe(true);
    expect(matches(lite, "Gemini-3.5-Flash-Lite-001")).toBe(true);
    expect(matches(lite, "gemini-3.5-flash-lite-preview")).toBe(false);

    const flash = pattern("gemini-2.5-flash");
    expect(matches(flash, "gemini-2.5-flash")).toBe(true);
    expect(matches(flash, "gemini-2.5-flash-lite")).toBe(false);
    expect(matches(flash, "gemini-2.5-flashy")).toBe(false);
  });
});

describe("syncModels", () => {
  it("replaces the app's earlier definitions, leaves Langfuse's own alone, and signs in with the keys", async () => {
    const requests: { method: string; url: string; auth: string | null }[] = [];
    const created: { modelName: string }[] = [];
    const fakeFetch: typeof fetch = async (input, init = {}) => {
      const url = String(input);
      const method = init.method ?? "GET";
      requests.push({
        method,
        url,
        auth: new Headers(init.headers).get("Authorization"),
      });
      if (method === "GET")
        return Response.json({
          data: [
            // Defined by this app last time: replaced.
            {
              id: "old-lite",
              modelName: "gemini-3.5-flash-lite",
              isLangfuseManaged: false,
            },
            // Langfuse's built-in price: never touched.
            {
              id: "builtin-flash",
              modelName: "gemini-2.5-flash",
              isLangfuseManaged: true,
            },
          ],
          meta: { page: 1, totalPages: 1 },
        });
      if (method === "POST") created.push(JSON.parse(String(init.body)));
      return method === "DELETE"
        ? new Response(null, { status: 204 })
        : Response.json({});
    };

    const results = await syncModels({
      baseUrl: "https://us.cloud.langfuse.com/",
      publicKey: "pk-lf-1",
      secretKey: "sk-lf-2",
      fetch: fakeFetch,
    });

    expect(results).toContainEqual({
      modelName: "gemini-3.5-flash-lite",
      action: "replaced",
    });
    expect(results).toContainEqual({
      modelName: "gemini-2.5-flash",
      action: "created",
    });
    expect(created.map((c) => c.modelName).sort()).toEqual(
      MODELS.map((m) => m.googleId).sort(),
    );
    const deletes = requests.filter((r) => r.method === "DELETE");
    expect(deletes.map((r) => r.url)).toEqual([
      "https://us.cloud.langfuse.com/api/public/models/old-lite",
    ]);
    const basic = `Basic ${Buffer.from("pk-lf-1:sk-lf-2").toString("base64")}`;
    for (const r of requests) expect(r.auth).toBe(basic);
  });

  it("says what Langfuse answered when a key is wrong", async () => {
    const fakeFetch: typeof fetch = async () =>
      new Response("Invalid credentials", { status: 401 });
    await expect(
      syncModels({ publicKey: "pk", secretKey: "sk", fetch: fakeFetch }),
    ).rejects.toThrow(/401.*Invalid credentials/);
  });
});

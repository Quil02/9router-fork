// Tests for JSON (non-streaming) zero-content failover in handleComboChat.
//
// Covers:
//   - isZeroContentResponse: the helper itself
//   - handleComboChat integration: JSON responses wired through real combo.js
//
// Requirement matrix:
//   R1a: JSON zero content with finish_reason "stop" on non-final seat → fallthrough
//   R1b: JSON reasoning-only response → zero content → fallthrough
//   R2:  JSON zero content on LAST candidate → error response (502), not empty 200
//   R3:  JSON with real content → returned as-is, no fallthrough
//   R4:  JSON content_filter finish with empty content → returned as-is (no-retry class)
import { describe, it, expect, vi } from "vitest";

// ── Keep I/O mocks off disk ──────────────────────────────────────────────────
vi.mock("@/lib/usageDb.js", () => ({
  saveRequestDetail: vi.fn(async () => {}),
  saveRequestUsage: vi.fn(async () => {}),
  appendRequestLog: vi.fn(async () => {}),
  trackPendingRequest: vi.fn(),
}));

import { isZeroContentResponse, handleComboChat } from "../../open-sse/services/combo.js";

// ─────────────────────────────────────────────────────────────────────────────
// isZeroContentResponse unit tests
// ─────────────────────────────────────────────────────────────────────────────
describe("isZeroContentResponse", () => {
  // Helpers to build a minimal chat-completion JSON body
  const chatBody = (content, finish_reason, extra = {}) => ({
    choices: [{ message: { content, ...extra }, finish_reason }],
  });

  describe("OpenAI chat format — real content is NOT zero", () => {
    it("non-empty string content + stop → NOT zero", () => {
      expect(isZeroContentResponse(chatBody("Hello!", "stop"))).toBe(false);
    });
    it("non-empty string content + length → NOT zero", () => {
      expect(isZeroContentResponse(chatBody("Hi", "length"))).toBe(false);
    });
    it("tool_calls + stop → NOT zero", () => {
      const data = {
        choices: [{
          message: { content: null, tool_calls: [{ id: "c1", type: "function", function: { name: "f" } }] },
          finish_reason: "stop",
        }],
      };
      expect(isZeroContentResponse(data)).toBe(false);
    });
    it("function_call (legacy) → NOT zero", () => {
      const data = {
        choices: [{
          message: { content: "", function_call: { name: "f", arguments: "{}" } },
          finish_reason: "stop",
        }],
      };
      expect(isZeroContentResponse(data)).toBe(false);
    });
    it("array content with parts → NOT zero", () => {
      const data = {
        choices: [{
          message: { content: [{ type: "text", text: "Hello" }] },
          finish_reason: "stop",
        }],
      };
      expect(isZeroContentResponse(data)).toBe(false);
    });
  });

  describe("OpenAI chat format — empty/no-content IS zero", () => {
    it("empty string content + stop → IS zero (the core bug fix)", () => {
      expect(isZeroContentResponse(chatBody("", "stop"))).toBe(true);
    });
    it("empty string content + length → IS zero", () => {
      expect(isZeroContentResponse(chatBody("", "length"))).toBe(true);
    });
    it("null content + stop → IS zero", () => {
      expect(isZeroContentResponse(chatBody(null, "stop"))).toBe(true);
    });
    it("whitespace-only content + stop → IS zero", () => {
      expect(isZeroContentResponse(chatBody("   ", "stop"))).toBe(true);
    });
    it("no message object at all → IS zero", () => {
      expect(isZeroContentResponse({ choices: [{ finish_reason: "stop" }] })).toBe(true);
    });
    it("empty choices array → IS zero", () => {
      expect(isZeroContentResponse({ choices: [] })).toBe(true);
    });
  });

  describe("OpenAI chat format — reasoning_content alone is NOT usable content", () => {
    it("reasoning_content non-empty, content empty + stop → IS zero", () => {
      const data = chatBody("", "stop", { reasoning_content: "I thought about it..." });
      expect(isZeroContentResponse(data)).toBe(true);
    });
    it("reasoning_content non-empty, content null + length → IS zero", () => {
      const data = chatBody(null, "length", { reasoning_content: "deep thoughts" });
      expect(isZeroContentResponse(data)).toBe(true);
    });
  });

  describe("OpenAI chat format — content_filter no-retry class", () => {
    it("empty content + content_filter → NOT zero (no-retry class, return as-is)", () => {
      expect(isZeroContentResponse(chatBody("", "content_filter"))).toBe(false);
    });
    it("null content + content_filter → NOT zero", () => {
      expect(isZeroContentResponse(chatBody(null, "content_filter"))).toBe(false);
    });
  });

  describe("Anthropic messages format", () => {
    it("text block with content → NOT zero", () => {
      const data = { content: [{ type: "text", text: "Hello!" }], stop_reason: "end_turn" };
      expect(isZeroContentResponse(data)).toBe(false);
    });
    it("tool_use block → NOT zero", () => {
      const data = { content: [{ type: "tool_use", id: "t1", name: "f" }], stop_reason: "tool_use" };
      expect(isZeroContentResponse(data)).toBe(false);
    });
    it("empty content array + end_turn → IS zero", () => {
      const data = { content: [], stop_reason: "end_turn" };
      expect(isZeroContentResponse(data)).toBe(true);
    });
    it("thinking-only block (not text/tool_use) + end_turn → IS zero", () => {
      const data = { content: [{ type: "thinking", thinking: "hmm" }], stop_reason: "end_turn" };
      expect(isZeroContentResponse(data)).toBe(true);
    });
    it("empty text block → IS zero", () => {
      const data = { content: [{ type: "text", text: "" }], stop_reason: "end_turn" };
      expect(isZeroContentResponse(data)).toBe(true);
    });
    it("refusal stop_reason (content_filter class) → NOT zero", () => {
      const data = { content: [], stop_reason: "refusal" };
      expect(isZeroContentResponse(data)).toBe(false);
    });
  });

  describe("edge cases", () => {
    it("null → false", () => expect(isZeroContentResponse(null)).toBe(false));
    it("string → false", () => expect(isZeroContentResponse("text")).toBe(false));
    it("empty object → false", () => expect(isZeroContentResponse({})).toBe(false));
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// handleComboChat integration: JSON (non-streaming) responses
// ─────────────────────────────────────────────────────────────────────────────
describe("handleComboChat JSON zero-content integration", () => {
  const LOG = {
    info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(),
    line: vi.fn(), errorLine: vi.fn(),
  };

  // Build a JSON Response with the given payload.
  function jsonResp(payload, status = 200) {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Build a minimal OpenAI chat completion body.
  function chatCompletion(content, finish_reason, extra = {}) {
    return {
      id: "chatcmpl-test",
      object: "chat.completion",
      choices: [{
        index: 0,
        message: { role: "assistant", content, ...extra },
        finish_reason,
      }],
      usage: { prompt_tokens: 10, completion_tokens: 0, total_tokens: 10 },
    };
  }

  // Run combo with the given per-model responses. Returns { res, tried }.
  async function comboWith(modelResponses) {
    const tried = [];
    const models = modelResponses.map((_, i) => `openai/m${i}`);
    const res = await handleComboChat({
      body: { model: "combo", stream: false, messages: [{ role: "user", content: "hi" }] },
      models,
      handleSingleModel: async (_body, modelStr) => {
        tried.push(modelStr);
        const idx = Number(modelStr.slice(-1));
        return modelResponses[idx];
      },
      log: LOG,
      comboName: `test-combo-${Math.random()}`,
      comboStrategy: "fallback",
      autoSwitch: false,
    });
    return { res, tried };
  }

  // R1a: JSON zero content (finish_reason "stop") on a non-final seat → falls through
  it("R1a: empty content + stop on non-final seat → falls through, returns next candidate", async () => {
    const emptyStop = jsonResp(chatCompletion("", "stop"));
    const realAnswer = jsonResp(chatCompletion("Hello from m1!", "stop"));

    const { res, tried } = await comboWith([emptyStop, realAnswer]);
    expect(tried).toEqual(["openai/m0", "openai/m1"]);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe("Hello from m1!");
  });

  // R1b: reasoning-only response → treated as zero content → falls through
  it("R1b: reasoning_content only (content empty) → zero content → falls through", async () => {
    const reasoningOnly = jsonResp(chatCompletion("", "stop", { reasoning_content: "Let me think..." }));
    const realAnswer = jsonResp(chatCompletion("Done reasoning!", "stop"));

    const { res, tried } = await comboWith([reasoningOnly, realAnswer]);
    expect(tried).toEqual(["openai/m0", "openai/m1"]);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe("Done reasoning!");
  });

  // R2: JSON zero content on the LAST candidate → error response, NOT empty 200
  it("R2: all candidates return zero content → returns error (not empty 200)", async () => {
    const empty0 = jsonResp(chatCompletion("", "stop"));
    const empty1 = jsonResp(chatCompletion("", "stop"));

    const { res, tried } = await comboWith([empty0, empty1]);
    expect(tried).toEqual(["openai/m0", "openai/m1"]);
    // Must NOT be a 200 (which would be an empty success delivery)
    expect(res.status).not.toBe(200);
    // Must be a 5xx error
    expect(res.status).toBeGreaterThanOrEqual(500);
    const body = await res.json();
    expect(body).toHaveProperty("error");
    expect(body.error.message).toMatch(/zero usable content|empty|unavailable/i);
  });

  // R3: JSON with real content → returned as-is, no fallthrough
  it("R3: real content → returned as-is, no fallthrough to m1", async () => {
    const realAnswer = jsonResp(chatCompletion("I have content.", "stop"));
    const neverUsed = jsonResp(chatCompletion("Should never see this", "stop"));

    const { res, tried } = await comboWith([realAnswer, neverUsed]);
    expect(tried).toEqual(["openai/m0"]);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe("I have content.");
  });

  // R4: content_filter finish with empty content → returned as-is (no-retry class)
  it("R4: content_filter + empty content → no-retry class, returned as-is (no fallthrough)", async () => {
    const filtered = jsonResp(chatCompletion("", "content_filter"));
    const neverUsed = jsonResp(chatCompletion("Should never see this", "stop"));

    const { res, tried } = await comboWith([filtered, neverUsed]);
    expect(tried).toEqual(["openai/m0"]);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.choices[0].finish_reason).toBe("content_filter");
  });

  // Additional: single-candidate combo with zero content → error
  it("single candidate zero content → error, not empty 200", async () => {
    const empty = jsonResp(chatCompletion("", "stop"));
    const { res, tried } = await comboWith([empty]);
    expect(tried).toEqual(["openai/m0"]);
    expect(res.status).not.toBe(200);
    expect(res.status).toBeGreaterThanOrEqual(500);
    const body = await res.json();
    expect(body).toHaveProperty("error");
  });

  // Additional: tool_calls response → NOT zero content, returned as-is
  it("tool_calls with empty content string → NOT zero content (tool call is usable)", async () => {
    const withTools = jsonResp({
      id: "chatcmpl-t",
      object: "chat.completion",
      choices: [{
        index: 0,
        message: {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "c1", type: "function", function: { name: "f", arguments: "{}" } }],
        },
        finish_reason: "tool_calls",
      }],
      usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    });
    const neverUsed = jsonResp(chatCompletion("fallback", "stop"));

    const { res, tried } = await comboWith([withTools, neverUsed]);
    expect(tried).toEqual(["openai/m0"]);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.choices[0].message.tool_calls).toHaveLength(1);
  });

  // Additional: null content (no key) + length → zero content, falls through
  it("null content + length finish → zero content → falls through", async () => {
    const nullLen = jsonResp(chatCompletion(null, "length"));
    const realAnswer = jsonResp(chatCompletion("Got it", "stop"));

    const { res, tried } = await comboWith([nullLen, realAnswer]);
    expect(tried).toEqual(["openai/m0", "openai/m1"]);
    const body = await res.json();
    expect(body.choices[0].message.content).toBe("Got it");
  });
});

/**
 * Behavioural regression tests for the streaming render path in frontend/app.js.
 *
 * These exist because of a real race: scheduleRender() queues a requestAnimationFrame
 * callback, but finalization (incomplete-stream warning, citations, telemetry, feedback)
 * runs synchronously when the read loop exits and *appends* to the bubble. A frame queued
 * by the last token fires afterwards and overwrites bubble.innerHTML, erasing all of it —
 * including the "this answer is incomplete" warning on a failed stream.
 *
 * rAF here is MANUALLY PUMPED (never auto-flushed), so we can force the exact hostile
 * interleaving rather than hoping the scheduler reproduces it.
 *
 * Run: node tests/frontend/stream_render.test.mjs
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import assert from "node:assert/strict";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_JS = join(HERE, "..", "..", "frontend", "app.js");

// ── Minimal DOM ──────────────────────────────────────────────────────────────

class El {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.className = "";
    this.dataset = {};
    this.style = {};
    this._html = "";
    this._text = "";
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this.selectedOptions = [];
    this.classList = {
      _s: new Set(),
      add: (c) => this.classList._s.add(c),
      remove: (c) => this.classList._s.delete(c),
      toggle: (c, f) => (f ? this.classList._s.add(c) : this.classList._s.delete(c)),
      contains: (c) => this.classList._s.has(c),
    };
  }
  // Setting innerHTML replaces content AND drops appended children — this is the
  // clobber we are testing for.
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get innerHTML() {
    return this._html + this.children.map((c) => c.outerHTML).join("");
  }
  get outerHTML() {
    const cls = this.className ? ` class="${this.className}"` : "";
    return `<${this.tagName.toLowerCase()}${cls}>${this.innerHTML}</${this.tagName.toLowerCase()}>`;
  }
  set textContent(v) { this._text = String(v); this._html = String(v); }
  get textContent() {
    return this._text + this.children.map((c) => c.textContent).join("");
  }
  appendChild(c) { this.children.push(c); return c; }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); }
  remove() {}
  addEventListener() {}
  removeEventListener() {}
  focus() {}
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  querySelectorAll(sel) {
    const want = sel.replace(/^\./, "");
    const out = [];
    for (const c of this.children) {
      if (c.className && c.className.split(/\s+/).includes(want)) out.push(c);
      out.push(...c.querySelectorAll(sel));
    }
    return out;
  }
}

function makeSandbox() {
  const frames = [];   // manually pumped rAF queue
  let nextId = 1;
  const cancelled = new Set();

  // app.js binds its DOM refs with top-level `const` (textarea, chatArea, sendBtn...),
  // which a vm sandbox property cannot shadow. So hand out STABLE elements per id and
  // drive the UI through those instead.
  const byId = new Map();
  const getEl = (id) => {
    if (!byId.has(id)) byId.set(id, new El());
    return byId.get(id);
  };

  const doc = {
    readyState: "complete",
    createElement: (t) => new El(t),
    getElementById: getEl,
    querySelector: () => new El(),
    querySelectorAll: () => [],
    addEventListener: () => {},
  };

  const store = () => ({ getItem: () => null, setItem: () => {}, removeItem: () => {} });

  const sandbox = {
    document: doc,
    window: { innerWidth: 1200, location: { search: "" } },
    localStorage: store(),
    sessionStorage: store(),
    performance: { now: () => 0 },
    console,
    URLSearchParams,
    JSON,
    TextDecoder,
    setTimeout,
    fetch: async () => ({ ok: false, statusText: "stub", json: async () => ({}) }),
    requestAnimationFrame: (cb) => { const id = nextId++; frames.push([id, cb]); return id; },
    cancelAnimationFrame: (id) => { cancelled.add(id); },
    // marked/DOMPurify absent -> renderMarkdownFallback is used (pure string work).
    marked: undefined,
    DOMPurify: undefined,
  };
  sandbox.globalThis = sandbox;
  sandbox.window.document = doc;

  // Pump every queued frame that wasn't cancelled. This is what a real browser would do
  // on the next repaint — including AFTER finalization has already run.
  // Returns the number of callbacks that actually EXECUTED (cancelled ones don't count).
  const pumpFrames = () => {
    const queued = frames.splice(0, frames.length);
    let ran = 0;
    for (const [id, cb] of queued) {
      if (cancelled.has(id)) continue;
      cb(0);
      ran++;
    }
    return ran;
  };

  return { sandbox, pumpFrames, frames, cancelled, getEl };
}

// Build an SSE body from a list of raw chunk strings.
function sseStream(chunks) {
  const enc = new TextEncoder();
  let i = 0;
  return {
    getReader: () => ({
      read: async () => (i < chunks.length
        ? { done: false, value: enc.encode(chunks[i++]) }
        : { done: true, value: undefined }),
    }),
  };
}

function loadApp(sandbox) {
  const src = readFileSync(APP_JS, "utf8");
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox, { filename: "app.js" });
  return sandbox;
}

// Drive sendQuestion() against a scripted SSE response, pumping frames afterwards.
async function runStream({ chunks, pumpDuringStream = false }) {
  const { sandbox, pumpFrames, getEl } = makeSandbox();
  const bubble = new El();

  loadApp(sandbox);

  // Top-level `function` declarations DO land on the vm global, so these override cleanly.
  // Pin the bubble the code streams into and neutralise unrelated UI plumbing.
  sandbox.createAssistantBubble = () => ({ msgDiv: new El(), bubble });
  sandbox.addMessage = () => {};
  sandbox.scrollIfNeeded = () => {};

  // The composer is a top-level `const`, so feed it via the element registry.
  getEl("user-input").value = "why does my arm go numb?";

  sandbox.fetch = async () => ({ ok: true, body: sseStream(chunks) });

  const p = vm.runInContext("sendQuestion()", sandbox);
  if (pumpDuringStream) {
    // Let the microtask queue advance mid-stream, then paint — simulates a browser
    // that actually repainted between tokens.
    await new Promise((r) => setTimeout(r, 0));
    pumpFrames();
  }
  await p;

  // The hostile move: a frame queued by the final token now fires, AFTER finalization.
  const pumpedAfterFinalize = pumpFrames();

  return { bubble, sandbox, pumpedAfterFinalize };
}

const tok = (t) => `data: ${JSON.stringify({ token: t })}\n\n`;
const done = (meta) => `event: done\ndata: ${JSON.stringify(meta)}\n\n`;

const OK_META = {
  complete: true,
  citations: ["mskneurology.com/tos — Mechanism"],
  retrieval_confidence: 0.42,
  retrieval_time: 0.3,
  generation_time: 1.1,
  answer_mode: "llm:openai",
  generation_model: "gpt-5.4-mini",
  category: "B",
  prompt_tokens: 100,
  output_tokens: 50,
};

const FAIL_META = {
  complete: false,
  error: "Generation failed mid-stream.",
  request_id: "abc123def456",
};

// ── Tests ────────────────────────────────────────────────────────────────────

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test("normal completion renders the answer and keeps citations + telemetry", async () => {
  const { bubble, pumpedAfterFinalize } = await runStream({
    chunks: [tok("Scapular "), tok("depression "), tok("narrows the space."), done(OK_META)],
    pumpDuringStream: true,
  });
  const html = bubble.innerHTML;
  assert.match(html, /Scapular depression narrows the space\./, "answer text must be present");
  assert.ok(bubble.querySelector(".citations"), "citations must survive finalization");
  assert.ok(bubble.querySelector(".telemetry-panel"), "telemetry must survive finalization");
  assert.ok(bubble.querySelector(".feedback-row"), "feedback controls must survive finalization");
  assert.equal(pumpedAfterFinalize, 0, "no frame may remain queued after finalization");
});

test("fast tokens immediately followed by done do not clobber appended metadata", async () => {
  // The original bug: every token arrives in ONE chunk with the done event, so the frame
  // queued by the last token is still pending when finalization runs.
  const { bubble } = await runStream({
    chunks: [tok("a") + tok("b") + tok("c") + done(OK_META)],
    pumpDuringStream: false,   // never painted mid-stream; frame is pending at finalize
  });
  assert.match(bubble.innerHTML, /abc/, "answer text must be present");
  assert.ok(bubble.querySelector(".citations"), "citations must not be erased by a late frame");
  assert.ok(bubble.querySelector(".telemetry-panel"), "telemetry must not be erased by a late frame");
  assert.ok(bubble.querySelector(".feedback-row"), "feedback must not be erased by a late frame");
});

test("partial/error completion after tokens rendered keeps BOTH the text and the warning", async () => {
  const { bubble } = await runStream({
    chunks: [tok("Partial clinical ans"), done(FAIL_META)],
    pumpDuringStream: true,   // tokens really painted, then the stream fails
  });
  const html = bubble.innerHTML;
  assert.match(html, /Partial clinical ans/, "the partial answer must be retained");
  assert.ok(bubble.querySelector(".stream-error"), "the incomplete-stream warning must be present");
  assert.match(html, /incomplete/i, "must tell the user the answer is incomplete");
  assert.match(html, /abc123def456/, "request_id must be surfaced");
});

test("incomplete response retains its warning even when a frame was queued by the last token", async () => {
  // Hardest case: last token queues a frame, THEN the stream fails. If the frame is not
  // cancelled it repaints plain markdown over the warning and the user sees a truncated
  // clinical answer presented as if it were complete.
  const { bubble, pumpedAfterFinalize } = await runStream({
    chunks: [tok("Do this exercise ") + tok("three times daily") + done(FAIL_META)],
    pumpDuringStream: false,
  });
  assert.ok(bubble.querySelector(".stream-error"),
    "SAFETY: the incomplete-stream warning must survive the queued frame");
  assert.match(bubble.innerHTML, /incomplete/i,
    "SAFETY: a truncated clinical answer must never be shown as complete");
  assert.equal(pumpedAfterFinalize, 0, "the queued frame must have been cancelled");
});

test("a failed stream is not committed to conversation history", async () => {
  const { sandbox } = await runStream({
    chunks: [tok("Partial"), done(FAIL_META)],
    pumpDuringStream: true,
  });
  const hist = vm.runInContext("history", sandbox);
  assert.equal(hist.length, 0, "an incomplete answer must not enter multi-turn history");
});

// ── Runner ───────────────────────────────────────────────────────────────────

let failed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failed++;
    console.error(`  FAIL ${name}\n       ${err.message}`);
  }
}
console.log(`\n${tests.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

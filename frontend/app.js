// ── MSK Triage Chatbot — Frontend Logic ──────────────────────────────────────

// ⚠️ UPDATE THIS to your Render backend URL after deploying
const API_URL = "https://msk-rag-chatbot.onrender.com";

// ── State ────────────────────────────────────────────────────────────────────
let history = [];
let isLoading = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatArea = document.getElementById("chat-area");
const textarea = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    addMessage("assistant", "Ask about symptoms, biomechanics, or exercise progressions.\n\nAll answers are grounded strictly in the MSK Neurology dataset.");
    textarea.focus();
    checkHealth();
});

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await fetch(`${API_URL}/health`);
        if (r.ok) {
            const data = await r.json();
            statusText.textContent = `Connected · ${data.chunk_count} chunks`;
        } else {
            statusText.textContent = "Backend unreachable";
        }
    } catch {
        statusText.textContent = "Backend offline";
    }
}

// ── Send question (streaming) ────────────────────────────────────────────────
async function sendQuestion() {
    const question = textarea.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    textarea.value = "";
    textarea.style.height = "auto";

    addMessage("user", question);

    const { msgDiv, bubble } = createAssistantBubble();
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    let fullText = "";
    let streamMeta = null;
    const streamStart = performance.now();

    try {
        const res = await fetch(`${API_URL}/ask/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: history.slice(-10),
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            bubble.innerHTML = `<p>⚠️ ${escapeHtml(err.detail || "Something went wrong.")}</p>`;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let gotDone = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("event: done")) {
                    gotDone = true;
                } else if (line.startsWith("data: ")) {
                    try {
                        const obj = JSON.parse(line.slice(6));
                        if (gotDone) {
                            streamMeta = obj;
                        } else if (obj.token) {
                            fullText += obj.token;
                            bubble.innerHTML = renderMarkdown(fullText);
                            chatArea.scrollTop = chatArea.scrollHeight;
                        }
                    } catch { /* skip */ }
                }
            }
        }

        // Remaining buffer
        if (buffer.startsWith("data: ") && !streamMeta) {
            try { streamMeta = JSON.parse(buffer.slice(6)); } catch { }
        }

        const endToEnd = ((performance.now() - streamStart) / 1000).toFixed(2);

        // Clear typing indicator if still showing
        const typingEl = bubble.querySelector(".typing-indicator");
        if (typingEl) {
            if (fullText) {
                bubble.innerHTML = renderMarkdown(fullText);
            } else if (streamMeta && streamMeta.error) {
                bubble.innerHTML = `<p>⚠️ ${escapeHtml(streamMeta.error)}</p>`;
            } else {
                bubble.innerHTML = `<p>⚠️ No response received.</p>`;
            }
        }

        // Citations
        if (streamMeta && streamMeta.citations && streamMeta.citations.length > 0) {
            appendCitations(bubble, streamMeta.citations);
        }

        // Telemetry
        if (streamMeta) {
            appendTelemetry(bubble, streamMeta, endToEnd);
        }

        history.push({ role: "user", content: question });
        history.push({ role: "assistant", content: fullText });

    } catch (err) {
        bubble.innerHTML = `<p>⚠️ Network error: ${escapeHtml(err.message)}</p>`;
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        textarea.focus();
    }
}

// ── Create empty assistant bubble ────────────────────────────────────────────
function createAssistantBubble() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "M";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    return { msgDiv, bubble };
}

// ── Add static message ───────────────────────────────────────────────────────
function addMessage(role, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "U" : "M";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "assistant") {
        bubble.innerHTML = renderMarkdown(text);
    } else {
        bubble.textContent = text;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Append citations ─────────────────────────────────────────────────────────
function appendCitations(bubble, citations) {
    const existing = bubble.querySelector(".citations");
    if (existing) existing.remove();

    const citDiv = document.createElement("div");
    citDiv.className = "citations";
    citations.forEach(c => {
        const tag = document.createElement("span");
        tag.className = "citation-tag";
        tag.textContent = c;
        citDiv.appendChild(tag);
    });
    bubble.appendChild(citDiv);
}

// ── Append telemetry panel ───────────────────────────────────────────────────
function appendTelemetry(bubble, meta, endToEnd) {
    const rt = (meta.retrieval_time || 0).toFixed(2);
    const gt = (meta.generation_time || 0).toFixed(2);
    const conf = (meta.retrieval_confidence || 0);
    const confClass = conf >= 0.5 ? "confidence-high" : conf >= 0.3 ? "confidence-mid" : "confidence-low";
    const pTok = meta.prompt_tokens || 0;
    const oTok = meta.output_tokens || 0;
    const cTok = meta.context_tokens || 0;
    const qTok = meta.question_tokens || 0;

    // Toggle button
    const toggleBtn = document.createElement("div");
    toggleBtn.className = "telemetry-toggle";
    toggleBtn.innerHTML = "📊 Stats";
    bubble.appendChild(toggleBtn);

    // Panel
    const panel = document.createElement("div");
    panel.className = "telemetry-panel";

    let statsHtml = `<div class="telemetry-stats">
    <span class="stat-badge">Retrieval: ${rt}s</span>
    <span class="stat-badge">LLM: ${gt}s</span>
    <span class="stat-badge">End-to-end: ${endToEnd}s</span>
    <span class="stat-badge">Prompt: ${pTok}</span>
    <span class="stat-badge">Output: ${oTok}</span>
    <span class="stat-badge">Context: ${cTok}</span>
    <span class="stat-badge">Question: ${qTok}</span>
    <span class="stat-badge ${confClass}">Confidence: ${conf.toFixed(2)}</span>
  </div>`;

    // Category + refined query
    let detailHtml = "";
    if (meta.category || meta.refined_query) {
        detailHtml = '<div class="telemetry-detail">';
        if (meta.category) {
            const label = meta.category_label || meta.category;
            detailHtml += `<strong>Category:</strong> ${escapeHtml(meta.category)} — ${escapeHtml(label)}<br>`;
        }
        if (meta.refined_query) {
            detailHtml += `<strong>Refined query:</strong> ${escapeHtml(meta.refined_query)}`;
        }
        detailHtml += "</div>";
    }

    panel.innerHTML = statsHtml + detailHtml;
    bubble.appendChild(panel);

    toggleBtn.addEventListener("click", () => {
        panel.classList.toggle("open");
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

// ── Markdown renderer ────────────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    html = html.replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/gs, (match) => `<ol>${match}</ol>`);

    html = html.replace(/^[-•] (.+)$/gm, "<li>$1</li>");

    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");

    return `<p>${html}</p>`;
}

function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return str.replace(/[&<>"']/g, c => map[c]);
}

// ── Event bindings ───────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendQuestion);

textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
});

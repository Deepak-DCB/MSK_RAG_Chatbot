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

    // User message
    addMessage("user", question);

    // Create assistant bubble + typing indicator
    const { msgDiv, bubble } = createAssistantBubble();
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    let fullText = "";

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

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const payload = line.slice(6);
                    try {
                        const obj = JSON.parse(payload);
                        if (obj.token) {
                            fullText += obj.token;
                            bubble.innerHTML = renderMarkdown(fullText);
                            chatArea.scrollTop = chatArea.scrollHeight;
                        }
                    } catch { /* skip malformed */ }
                } else if (line.startsWith("event: done")) {
                    // Next data line has metadata — handled in next iteration
                }
            }
        }

        // Process any remaining buffer
        if (buffer.startsWith("data: ")) {
            try {
                const meta = JSON.parse(buffer.slice(6));
                if (meta.citations && meta.citations.length > 0) {
                    appendCitations(bubble, meta.citations);
                }
            } catch { /* ignore */ }
        }

        // Update history
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

// ── Create an empty assistant message bubble ─────────────────────────────────
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

// ── Message rendering (for user messages and initial assistant welcome) ──────
function addMessage(role, text, citations) {
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

    if (citations && citations.length > 0) {
        appendCitations(bubble, citations);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Append citations to a bubble ─────────────────────────────────────────────
function appendCitations(bubble, citations) {
    // Remove existing citations if any
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

// ── Simple Markdown renderer ─────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");

    // Bold & italic
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Numbered lists
    html = html.replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/gs, (match) => `<ol>${match}</ol>`);

    // Bullet lists
    html = html.replace(/^[-•] (.+)$/gm, "<li>$1</li>");

    // Paragraphs
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

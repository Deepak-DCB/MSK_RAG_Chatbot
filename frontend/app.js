// ── MSK Triage Chatbot — Frontend Logic ──────────────────────────────────────

// ⚠️ UPDATE THIS to your Render backend URL after deploying
const API_URL = "https://your-render-app.onrender.com";

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

// ── Send question ────────────────────────────────────────────────────────────
async function sendQuestion() {
    const question = textarea.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    textarea.value = "";
    textarea.style.height = "auto";

    // User message
    addMessage("user", question);

    // Typing indicator
    const typingEl = showTyping();

    try {
        const res = await fetch(`${API_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                history: history.slice(-10),
            }),
        });

        removeTyping(typingEl);

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            addMessage("assistant", `⚠️ ${err.detail || "Something went wrong."}`);
            return;
        }

        const data = await res.json();

        // Build answer with citations
        let content = data.answer || "No answer returned.";
        addMessage("assistant", content, data.citations);

        // Update history
        history.push({ role: "user", content: question });
        history.push({ role: "assistant", content: data.answer });

    } catch (err) {
        removeTyping(typingEl);
        addMessage("assistant", `⚠️ Network error: ${err.message}`);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        textarea.focus();
    }
}

// ── Message rendering ────────────────────────────────────────────────────────
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

    // Citations
    if (citations && citations.length > 0) {
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

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

// ── Typing indicator ─────────────────────────────────────────────────────────
function showTyping() {
    const msgDiv = document.createElement("div");
    msgDiv.className = "message assistant";
    msgDiv.id = "typing-msg";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "M";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatArea.appendChild(msgDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
    return msgDiv;
}

function removeTyping(el) {
    if (el && el.parentNode) el.parentNode.removeChild(el);
}

// ── Simple Markdown renderer ─────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Headers (### → h3, ## → h2, # → h1)
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

    // Paragraphs (double newline)
    html = html.replace(/\n\n/g, "</p><p>");

    // Single newlines → <br>
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

// Auto-resize textarea
textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
});

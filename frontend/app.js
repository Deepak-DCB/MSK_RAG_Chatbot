// ── MSK Triage Chatbot — Frontend Logic ──────────────────────────────────────

const API_URL = "https://msk-rag-chatbot.onrender.com";

// ── Supabase ─────────────────────────────────────────────────────────────────
const SUPABASE_URL = "https://lmanobmmvrgpotioblih.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxtYW5vYm1tdnJncG90aW9ibGloIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI2MDE5OTgsImV4cCI6MjA4ODE3Nzk5OH0.gY-_zvAFsQd7Sfhq288Qvdn4uNa3tMQdr6BV4-IP5UI";

let sb = null;
try {
    // UMD build may export createClient at different levels
    const mod = window.supabase;
    if (mod && mod.createClient) {
        sb = mod.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    } else if (mod && mod.supabase && mod.supabase.createClient) {
        sb = mod.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    } else if (mod && mod.default && mod.default.createClient) {
        sb = mod.default.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    } else {
        console.warn("Supabase: createClient not found on window.supabase", mod);
    }
} catch (e) {
    console.warn("Supabase init failed:", e);
}
const sbClient = sb;

// ── State ────────────────────────────────────────────────────────────────────
let history = [];
let isLoading = false;
let currentUser = null;
let accessToken = null;
let userScrolledUp = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatArea = document.getElementById("chat-area");
const textarea = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const welcomeScreen = document.getElementById("welcome-screen");

// Auth modal elements
const authModal = document.getElementById("auth-modal");
const authForm = document.getElementById("auth-form");
const authEmail = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const authSubmit = document.getElementById("auth-submit");
const authTitle = document.getElementById("auth-title");
const authError = document.getElementById("auth-error");
const authSwitchText = document.getElementById("auth-switch-text");
const authSwitchLink = document.getElementById("auth-switch-link");
const googleBtn = document.getElementById("google-btn");
const skipBtn = document.getElementById("skip-btn");

// Sidebar elements
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const newChatBtn = document.getElementById("new-chat-btn");
const sidebarChats = document.getElementById("sidebar-chats");
const sidebarEmpty = document.getElementById("sidebar-empty");
const sidebarUser = document.getElementById("sidebar-user");
const sidebarAvatar = document.getElementById("sidebar-avatar");
const sidebarEmail = document.getElementById("sidebar-email");
const logoutBtn = document.getElementById("logout-btn");
const sidebarSigninBtn = document.getElementById("sidebar-signin-btn");

let isSignUp = false;

// ── Smart auto-scroll ────────────────────────────────────────────────────────
function shouldAutoScroll() {
    const threshold = 150;
    return chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < threshold;
}
chatArea.addEventListener("scroll", () => {
    if (isLoading) {
        userScrolledUp = !shouldAutoScroll();
    }
});
function scrollIfNeeded() {
    if (!userScrolledUp) {
        chatArea.scrollTop = chatArea.scrollHeight;
    }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
async function init() {
    // Auto-collapse sidebar on mobile
    if (window.innerWidth <= 768) {
        sidebar.classList.add("collapsed");
    }

    checkHealth();
    bindChips();
    bindAuth();

    if (sbClient) {
        try {
            // Check existing session
            const { data: { session } } = await sbClient.auth.getSession();
            if (session) {
                setUser(session.user, session.access_token);
                hideAuthModal();
            }

            // Listen for auth changes (Google redirect, etc.)
            sbClient.auth.onAuthStateChange((event, session) => {
                if (session) {
                    setUser(session.user, session.access_token);
                    hideAuthModal();
                } else {
                    clearUser();
                }
            });
        } catch (e) {
            console.warn("Supabase session check failed:", e);
        }
    }

    textarea.focus();
}

// Run init immediately if DOM is ready, otherwise wait
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}

// ── Auth UI ──────────────────────────────────────────────────────────────────
function bindAuth() {
    // Toggle sign-in / sign-up
    authSwitchLink.addEventListener("click", (e) => {
        e.preventDefault();
        isSignUp = !isSignUp;
        authTitle.textContent = isSignUp ? "Create account" : "Sign in";
        authSubmit.textContent = isSignUp ? "Sign up" : "Sign in";
        authSwitchText.textContent = isSignUp ? "Already have an account?" : "Don't have an account?";
        authSwitchLink.textContent = isSignUp ? "Sign in" : "Sign up";
        authError.textContent = "";
    });

    // Email/password submit
    authForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        authError.textContent = "";
        authSubmit.disabled = true;
        authSubmit.textContent = "Loading…";

        const email = authEmail.value.trim();
        const password = authPassword.value;

        try {
            if (!sbClient) {
                authError.textContent = "Auth service unavailable. Try again later.";
                return;
            }
            let result;
            if (isSignUp) {
                result = await sbClient.auth.signUp({ email, password });
            } else {
                result = await sbClient.auth.signInWithPassword({ email, password });
            }

            if (result.error) {
                authError.textContent = result.error.message;
            } else if (isSignUp && result.data?.user && !result.data.session) {
                authError.textContent = "";
                authTitle.textContent = "Check your email";
                authForm.style.display = "none";
                googleBtn.style.display = "none";
                document.querySelector(".divider").style.display = "none";
                document.querySelector(".auth-switch").textContent =
                    "We sent a confirmation link to " + email;
            }
        } catch (err) {
            authError.textContent = err.message;
        } finally {
            authSubmit.disabled = false;
            authSubmit.textContent = isSignUp ? "Sign up" : "Sign in";
        }
    });

    // Google OAuth
    googleBtn.addEventListener("click", async () => {
        if (!sbClient) { authError.textContent = "Auth service unavailable."; return; }
        const { error } = await sbClient.auth.signInWithOAuth({
            provider: "google",
            options: {
                redirectTo: window.location.origin,
            },
        });
        if (error) authError.textContent = error.message;
    });

    // Skip (guest mode)
    skipBtn.addEventListener("click", () => {
        hideAuthModal();
        textarea.focus();
    });

    // Logout (sidebar)
    logoutBtn.addEventListener("click", async () => {
        if (sbClient) await sbClient.auth.signOut();
        clearUser();
    });

    // Sidebar sign-in button
    sidebarSigninBtn.addEventListener("click", () => showAuthModal());

    // Sidebar toggle
    sidebarToggle.addEventListener("click", () => {
        sidebar.classList.remove("collapsed");
    });

    // New chat button
    newChatBtn.addEventListener("click", () => {
        startNewChat();
    });
}

function startNewChat() {
    // Clear current conversation
    history = [];
    chatArea.querySelectorAll(".message").forEach(el => el.remove());
    welcomeScreen.style.display = "";
    textarea.value = "";
    textarea.focus();
}

function setUser(user, token) {
    currentUser = user;
    accessToken = token;
    const initial = (user.email || "U")[0].toUpperCase();
    sidebarAvatar.textContent = initial;
    sidebarEmail.textContent = user.email || "User";
    sidebarUser.style.display = "flex";
    sidebarSigninBtn.style.display = "none";
    loadSidebarHistory();
}

function clearUser() {
    currentUser = null;
    accessToken = null;
    sidebarUser.style.display = "none";
    sidebarSigninBtn.style.display = "";
    sidebarChats.innerHTML = '<p class="sidebar-empty">Sign in to see history</p>';
}

function showAuthModal() {
    authModal.classList.remove("hidden");
    authError.textContent = "";
    authForm.style.display = "flex";
    googleBtn.style.display = "flex";
    const divider = document.querySelector(".divider");
    if (divider) divider.style.display = "flex";
}

function hideAuthModal() {
    authModal.classList.add("hidden");
}

// ── History ──────────────────────────────────────────────────────────────────
async function loadSidebarHistory() {
    if (!accessToken) return;

    sidebarChats.innerHTML = '<p class="sidebar-empty">Loading…</p>';

    try {
        const res = await fetch(`${API_URL}/history?limit=30`, {
            headers: { "Authorization": `Bearer ${accessToken}` },
        });

        if (!res.ok) {
            sidebarChats.innerHTML = '<p class="sidebar-empty">Could not load history</p>';
            return;
        }

        const data = await res.json();
        const convos = data.conversations || [];

        if (convos.length === 0) {
            sidebarChats.innerHTML = '<p class="sidebar-empty">No conversations yet</p>';
            return;
        }

        sidebarChats.innerHTML = "";
        convos.forEach(c => {
            const btn = document.createElement("button");
            btn.className = "sidebar-chat-item";
            btn.textContent = c.question;
            btn.title = c.question;
            btn.addEventListener("click", () => {
                // Load this conversation into chat
                welcomeScreen.style.display = "none";
                chatArea.querySelectorAll(".message").forEach(el => el.remove());
                addMessage("user", c.question);
                addMessage("assistant", c.answer);
                if (c.citations && c.citations.length > 0) {
                    const bubbles = chatArea.querySelectorAll(".message.assistant .message-bubble");
                    const lastBubble = bubbles[bubbles.length - 1];
                    if (lastBubble) appendCitations(lastBubble, c.citations);
                }
                // Collapse sidebar on mobile
                if (window.innerWidth <= 768) {
                    sidebar.classList.add("collapsed");
                }
            });
            sidebarChats.appendChild(btn);
        });
    } catch {
        sidebarChats.innerHTML = '<p class="sidebar-empty">Network error</p>';
    }
}

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await fetch(`${API_URL}/health`);
        if (r.ok) {
            const data = await r.json();
            statusText.textContent = `${data.chunk_count} sources`;
        } else {
            statusText.textContent = "Offline";
        }
    } catch {
        statusText.textContent = "Offline";
    }
}

// ── Chip click handlers ──────────────────────────────────────────────────────
function bindChips() {
    document.querySelectorAll("[data-query]").forEach(btn => {
        btn.addEventListener("click", () => {
            const q = btn.getAttribute("data-query");
            if (q && !isLoading) {
                textarea.value = q;
                sendQuestion();
            }
        });
    });
}

// ── Send question (streaming) ────────────────────────────────────────────────
async function sendQuestion() {
    const question = textarea.value.trim();
    if (!question || isLoading) return;

    isLoading = true;
    userScrolledUp = false;
    sendBtn.disabled = true;
    textarea.value = "";
    textarea.style.height = "auto";

    if (welcomeScreen) welcomeScreen.classList.add("hidden");

    addMessage("user", question);

    const { bubble } = createAssistantBubble();
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    let fullText = "";
    let streamMeta = null;
    const streamStart = performance.now();

    // Build headers — include JWT if logged in
    const headers = { "Content-Type": "application/json" };
    if (accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
    }

    try {
        const res = await fetch(`${API_URL}/ask/stream`, {
            method: "POST",
            headers,
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
                            scrollIfNeeded();
                        }
                    } catch { /* skip */ }
                }
            }
        }

        if (buffer.startsWith("data: ") && !streamMeta) {
            try { streamMeta = JSON.parse(buffer.slice(6)); } catch { }
        }

        const endToEnd = ((performance.now() - streamStart) / 1000).toFixed(2);

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

        if (streamMeta && streamMeta.citations && streamMeta.citations.length > 0) {
            appendCitations(bubble, streamMeta.citations);
        }

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

// ── Create assistant bubble ──────────────────────────────────────────────────
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
    scrollIfNeeded();

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
    scrollIfNeeded();
}

// ── Citations ────────────────────────────────────────────────────────────────
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

// ── Telemetry ────────────────────────────────────────────────────────────────
function appendTelemetry(bubble, meta, endToEnd) {
    const rt = (meta.retrieval_time || 0).toFixed(2);
    const gt = (meta.generation_time || 0).toFixed(2);
    const conf = (meta.retrieval_confidence || 0);
    const confClass = conf >= 0.5 ? "confidence-high" : conf >= 0.3 ? "confidence-mid" : "confidence-low";

    const toggleBtn = document.createElement("div");
    toggleBtn.className = "telemetry-toggle";
    toggleBtn.textContent = "📊 Stats";
    bubble.appendChild(toggleBtn);

    const panel = document.createElement("div");
    panel.className = "telemetry-panel";

    let html = `<div class="telemetry-stats">
        <span class="stat-badge">Retrieval ${rt}s</span>
        <span class="stat-badge">LLM ${gt}s</span>
        <span class="stat-badge">Total ${endToEnd}s</span>
        <span class="stat-badge">Prompt ${meta.prompt_tokens || 0}</span>
        <span class="stat-badge">Output ${meta.output_tokens || 0}</span>
        <span class="stat-badge ${confClass}">Confidence ${conf.toFixed(2)}</span>
    </div>`;

    if (meta.category || meta.refined_query) {
        html += '<div class="telemetry-detail">';
        if (meta.category) html += `<strong>Category:</strong> ${escapeHtml(meta.category_label || meta.category)}<br>`;
        if (meta.refined_query) html += `<strong>Query:</strong> ${escapeHtml(meta.refined_query)}`;
        html += "</div>";
    }

    panel.innerHTML = html;
    bubble.appendChild(panel);

    toggleBtn.addEventListener("click", () => {
        panel.classList.toggle("open");
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}

// ── Markdown ─────────────────────────────────────────────────────────────────
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
    html = html.replace(/(<li>.*<\/li>\n?)+/gs, match => `<ol>${match}</ol>`);
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

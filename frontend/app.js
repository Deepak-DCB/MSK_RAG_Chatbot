// ── MSK Triage Chatbot — Frontend Logic ──────────────────────────────────────

const API_URL = "https://msk-rag-chatbot.onrender.com";

// Guest-only v1 keeps the core triage product local/session-scoped while safety,
// retrieval, and UX quality mature. Backend auth/history remains available later.
const AUTH_ENABLED = false;
const sbClient = null;

// ── State ────────────────────────────────────────────────────────────────────
let history = [];
// Rolling conversation summary maintained by the backend and carried here between
// turns (the server is stateless). Keeps the chat coherent beyond the raw-history window.
let conversationSummary = null;
let isLoading = false;
let currentUser = null;
let accessToken = null;
let userScrolledUp = false;

// Phase 1 parity config: explicit request metadata
const REQUEST_CONFIG = {
    use_reranker: false,
    reranker_top_n: 10,
};

// ── Bring-your-own OpenAI key (optional, gated) ──────────────────────────────
// Held in memory for the request; by default only for this browser tab
// (sessionStorage). "Remember on this device" additionally saves it to localStorage.
// The key IS sent to the backend to make OpenAI calls for the user's own requests.
const API_KEY_STORAGE = "msk_openai_key";
let userApiKey = null;

// ── Generation model selection (optional, driven by GET /models) ─────────────
// { provider, model }. When set, the backend pins generation to this choice.
const MODEL_STORAGE = "msk_model_choice";
let modelCatalog = null;
let modelChoice = { provider: null, model: null };

function buildRequestConfig() {
    const cfg = { ...REQUEST_CONFIG };
    if (userApiKey) cfg.api_key = userApiKey;
    if (modelChoice.provider && modelChoice.model) {
        cfg.provider = modelChoice.provider;
        cfg.model = modelChoice.model;
    }
    return cfg;
}

function setupApiKeyPanel() {
    const btn = document.getElementById("api-key-btn");
    const panel = document.getElementById("api-key-panel");
    const input = document.getElementById("api-key-input");
    const remember = document.getElementById("api-key-remember");
    const saveBtn = document.getElementById("api-key-save");
    const clearBtn = document.getElementById("api-key-clear");
    const status = document.getElementById("api-key-status");
    const btnLabel = document.getElementById("api-key-btn-label");
    if (!btn || !panel) return;

    function reflectKeyState() {
        const active = Boolean(userApiKey);
        btn.classList.toggle("key-active", active);
        if (btnLabel) btnLabel.textContent = active ? "Key set" : "API key";
        if (status) status.textContent = active ? "Your key is active in this browser." : "";
        // Adding/clearing a key enables or disables premium (OpenAI) models.
        if (modelCatalog) renderModelOptions();
    }

    // Restore a previously entered key (localStorage takes precedence — it means the
    // user chose to remember it).
    const persisted = localStorage.getItem(API_KEY_STORAGE);
    const stored = persisted || sessionStorage.getItem(API_KEY_STORAGE);
    if (stored) {
        userApiKey = stored;
        if (remember) remember.checked = Boolean(persisted);
    }
    reflectKeyState();

    btn.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        if (!panel.hidden && input) input.focus();
    });

    saveBtn.addEventListener("click", () => {
        const val = (input.value || "").trim();
        if (!val) { if (status) status.textContent = "Enter a key first."; return; }
        userApiKey = val;
        sessionStorage.setItem(API_KEY_STORAGE, val);
        if (remember && remember.checked) localStorage.setItem(API_KEY_STORAGE, val);
        else localStorage.removeItem(API_KEY_STORAGE);
        input.value = "";
        reflectKeyState();
        panel.hidden = true;
    });

    clearBtn.addEventListener("click", () => {
        userApiKey = null;
        if (input) input.value = "";
        sessionStorage.removeItem(API_KEY_STORAGE);
        localStorage.removeItem(API_KEY_STORAGE);
        reflectKeyState();
        if (status) status.textContent = "Key cleared.";
    });
}

function openApiKeyPanel() {
    const panel = document.getElementById("api-key-panel");
    const input = document.getElementById("api-key-input");
    if (panel) {
        panel.hidden = false;
        if (input) input.focus();
    }
}

// ── Model dropdown ───────────────────────────────────────────────────────────
function persistModelChoice() {
    try { localStorage.setItem(MODEL_STORAGE, JSON.stringify(modelChoice)); } catch (_) { /* ignore */ }
}

async function setupModelSelect() {
    const picker = document.getElementById("model-picker");
    const sel = document.getElementById("model-select");
    const custom = document.getElementById("model-custom");
    if (!picker || !sel) return;

    try {
        const r = await fetch(`${API_URL}/models`);
        modelCatalog = r.ok ? await r.json() : null;
    } catch (_) {
        modelCatalog = null;
    }
    if (!modelCatalog || !Array.isArray(modelCatalog.providers) || !modelCatalog.providers.length) {
        picker.hidden = true;   // no catalog → hide the control entirely
        return;
    }

    // Restore a previously saved choice (may become unavailable → re-defaulted below).
    try {
        const saved = JSON.parse(localStorage.getItem(MODEL_STORAGE) || "null");
        if (saved && saved.provider && saved.model) modelChoice = saved;
    } catch (_) { /* ignore */ }

    picker.hidden = false;
    renderModelOptions();

    sel.addEventListener("change", () => {
        if (sel.value === "__custom__") {
            custom.hidden = false;
            custom.focus();
            return;
        }
        if (custom) custom.hidden = true;
        const opt = sel.selectedOptions[0];
        if (opt && opt.dataset.provider) {
            modelChoice = { provider: opt.dataset.provider, model: opt.value };
            persistModelChoice();
        }
    });

    if (custom) {
        custom.addEventListener("change", () => {
            const val = (custom.value || "").trim();
            const provider = custom.dataset.provider || "openai";
            if (val) { modelChoice = { provider, model: val }; persistModelChoice(); }
        });
    }
}

// (Re)build the grouped option list. Free providers are selectable only when the
// server has their key; premium (OpenAI) is enabled once the user adds a BYO key.
// Called again whenever the BYO-key state changes.
function renderModelOptions() {
    const sel = document.getElementById("model-select");
    const custom = document.getElementById("model-custom");
    if (!sel || !modelCatalog || !Array.isArray(modelCatalog.providers)) return;

    const hasKey = Boolean(userApiKey);
    sel.innerHTML = "";
    let firstEnabled = null;
    let matchedSaved = false;

    modelCatalog.providers.forEach((p) => {
        const enabled = p.tier === "free" ? Boolean(p.server_key) : (hasKey || Boolean(p.server_key));
        const group = document.createElement("optgroup");
        group.label = p.label + (enabled ? "" : (p.requires_user_key ? " — add your key" : " — unavailable"));
        (p.models || []).forEach((m) => {
            const o = document.createElement("option");
            o.value = m;
            o.textContent = m;
            o.dataset.provider = p.name;
            o.disabled = !enabled;
            if (enabled && !firstEnabled) firstEnabled = { provider: p.name, model: m, el: o };
            if (enabled && modelChoice.provider === p.name && modelChoice.model === m) {
                o.selected = true;
                matchedSaved = true;
            }
            group.appendChild(o);
        });
        sel.appendChild(group);
    });

    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom…";
    sel.appendChild(customOpt);

    // If the saved choice isn't currently selectable, fall back to the first enabled one.
    if (!matchedSaved) {
        if (firstEnabled) {
            firstEnabled.el.selected = true;
            modelChoice = { provider: firstEnabled.provider, model: firstEnabled.model };
        } else {
            modelChoice = { provider: null, model: null };
        }
        persistModelChoice();
    }
    if (custom) custom.dataset.provider = (firstEnabled && firstEnabled.provider) || "openai";
}

// Render an "API key unavailable" error with a call-to-action to add a personal key.
function renderApiKeyError(bubble, message, requestIdNote) {
    const msg = (typeof message === "string" && message) ||
        "The service's shared API key is currently unavailable or out of quota.";
    bubble.innerHTML = `<p>⚠️ ${escapeHtml(msg)}</p>` +
        `<p><button type="button" class="key-action key-cta">Add your OpenAI key</button></p>` +
        (requestIdNote || "");
    const cta = bubble.querySelector(".key-cta");
    if (cta) cta.addEventListener("click", openApiKeyPanel);
}

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatArea = document.getElementById("chat-area");
const textarea = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const welcomeScreen = document.getElementById("welcome-screen");
const mechanicsStudyToggle = document.getElementById("mechanics-study-toggle");

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
    setupGuestMode();
    setupApiKeyPanel();
    setupModelSelect();
    bindAuth();

    // Landing page deep links: chat.html?q=... prefills the composer
    const prefill = new URLSearchParams(window.location.search).get("q");
    if (prefill) {
        textarea.value = prefill;
        textarea.style.height = "auto";
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    }

    if (AUTH_ENABLED && sbClient) {
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

function setupGuestMode() {
    hideAuthModal();
    if (sidebarSigninBtn) sidebarSigninBtn.style.display = "none";
    if (sidebarUser) sidebarUser.style.display = "none";
    if (sidebarChats) {
        sidebarChats.innerHTML = '<p class="sidebar-empty">Guest mode: chats stay in this browser session only.</p>';
    }
}

// Run init immediately if DOM is ready, otherwise wait
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
} else {
    init();
}

// ── Auth UI ──────────────────────────────────────────────────────────────────
function bindAuth() {
    // In guest-only mode, keep only navigation controls active.
    if (!AUTH_ENABLED) {
        sidebarToggle.addEventListener("click", () => {
            sidebar.classList.toggle("collapsed");
        });
        newChatBtn.addEventListener("click", () => {
            startNewChat();
        });
        return;
    }

    // Toggle sign-in / sign-up
    authSwitchLink.addEventListener("click", (e) => {
        e.preventDefault();
        isSignUp = !isSignUp;
        authTitle.textContent = "Guest mode";
        authSubmit.textContent = "Unavailable";
        authSwitchText.textContent = "Accounts are off for this guest-only prototype.";
        authSwitchLink.textContent = "Guest mode";
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
            authSubmit.textContent = "Unavailable";
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
    conversationSummary = null;
    chatArea.querySelectorAll(".message").forEach(el => el.remove());
    welcomeScreen.style.display = "";
    welcomeScreen.classList.remove("hidden");
    if (window.innerWidth <= 768) {
        sidebar.classList.add("collapsed");
    }
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
    sidebarChats.innerHTML = '<p class="sidebar-empty">Guest mode: history is not saved.</p>';
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
    if (!AUTH_ENABLED) {
        setupGuestMode();
        return;
    }
    if (!accessToken) return;

    sidebarChats.innerHTML = '<p class="sidebar-empty">Loading…</p>';

    try {
        const res = await fetch(`${API_URL}/history?limit=30`, {
            headers: { "Authorization": `Bearer ${accessToken}` },
        });

        if (!res.ok) {
            let detail = "";
            try {
                const err = await res.json();
                detail = (err && err.detail) ? String(err.detail) : "";
            } catch {
                detail = "";
            }

            if (res.status === 401) {
                sidebarChats.innerHTML = '<p class="sidebar-empty">Session unavailable. Continue in guest mode.</p>';
            } else if (res.status === 503) {
                sidebarChats.innerHTML = '<p class="sidebar-empty">History storage is not configured.</p>';
            } else if (detail) {
                sidebarChats.innerHTML = `<p class="sidebar-empty">${escapeHtml(detail)}</p>`;
            } else {
                sidebarChats.innerHTML = '<p class="sidebar-empty">Could not load history</p>';
            }
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

    if (mechanicsStudyToggle && mechanicsStudyToggle.checked) {
        await sendMechanicsStudyQuestion(question, bubble);
        return;
    }

    let fullText = "";
    let streamMeta = null;
    const streamStart = performance.now();

    // Re-parsing the whole answer through marked + DOMPurify and rebuilding the DOM on
    // every token is O(n²) over the answer length and janks the tab on long replies.
    // Coalesce to at most one render per animation frame; the final render after the
    // stream ends guarantees the last tokens are always painted.
    //
    // A queued frame MUST NOT survive finalization. Finalization runs synchronously when
    // the read loop exits and it *appends* to the bubble (incomplete-stream warning,
    // citations, evidence spans, mechanism graph, telemetry, feedback). A frame queued by
    // the last token would fire afterwards and clobber the bubble's innerHTML, silently
    // erasing all of it — including the "this answer is incomplete" warning. So
    // finalizeRender() both cancels the queued frame and latches a flag, and any callback
    // that still runs becomes a no-op.
    let renderPending = false;
    let renderHandle = null;
    let streamFinalized = false;

    function scheduleRender() {
        if (streamFinalized || renderPending) return;
        renderPending = true;
        renderHandle = requestAnimationFrame(() => {
            renderPending = false;
            renderHandle = null;
            if (streamFinalized) return;
            bubble.innerHTML = renderMarkdown(fullText);
            scrollIfNeeded();
        });
    }

    function finalizeRender() {
        streamFinalized = true;
        if (renderHandle !== null) {
            cancelAnimationFrame(renderHandle);
            renderHandle = null;
        }
        renderPending = false;
    }

    // Build headers — include JWT if logged in
    const headers = { "Content-Type": "application/json" };
    if (AUTH_ENABLED && accessToken) {
        headers["Authorization"] = `Bearer ${accessToken}`;
    }

    try {
        const res = await fetch(`${API_URL}/ask/stream`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                question,
                history: history.slice(-10),
                conversation_summary: conversationSummary,
                config: buildRequestConfig(),
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            const detail = err.detail;
            const isObj = detail && typeof detail === "object";
            const code = isObj ? detail.error_code : null;
            const message = isObj ? detail.message : detail;
            if (code === "api_key_unavailable") {
                renderApiKeyError(bubble, message, "");
            } else {
                bubble.innerHTML = `<p>⚠️ ${escapeHtml((typeof message === "string" && message) || "Something went wrong.")}</p>`;
            }
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
                            scheduleRender();
                        }
                    } catch { /* skip */ }
                }
            }
        }

        if (buffer.startsWith("data: ") && !streamMeta) {
            try { streamMeta = JSON.parse(buffer.slice(6)); } catch { }
        }

        // Past this point the bubble is owned by finalization, which appends warnings,
        // citations, telemetry and feedback. Kill any frame the last token queued.
        finalizeRender();

        const endToEnd = ((performance.now() - streamStart) / 1000).toFixed(2);
        const assistantText = fullText.trim();
        const hasStreamError = Boolean(streamMeta && streamMeta.error);
        const isComplete = Boolean(streamMeta && streamMeta.complete !== false);
        const requestIdNote = streamMeta && streamMeta.request_id
            ? `<br><small>request_id: ${escapeHtml(streamMeta.request_id)}</small>`
            : "";

        const errorCode = streamMeta && streamMeta.error_code;

        // NOTE: do NOT gate this on the typing indicator still being present. The first
        // streamed token replaces the bubble's innerHTML and destroys it, so a mid-stream
        // failure used to skip every branch below — the user was left looking at a
        // truncated clinical answer with no indication it had been cut off.
        if (errorCode === "api_key_unavailable" && !assistantText) {
            renderApiKeyError(bubble, streamMeta && streamMeta.error, requestIdNote);
        } else if (!isComplete || hasStreamError) {
            const msg = (streamMeta && streamMeta.error) || "Response interrupted before completion.";
            if (assistantText) {
                // Partial answer already on screen: keep it, but mark it clearly as incomplete.
                bubble.innerHTML = renderMarkdown(fullText);
                const warn = document.createElement("p");
                warn.className = "stream-error";
                warn.innerHTML = `⚠️ ${escapeHtml(msg)}${requestIdNote}<br><small>This answer is incomplete — do not rely on it. Please resend.</small>`;
                bubble.appendChild(warn);
                if (errorCode === "api_key_unavailable") {
                    const cta = document.createElement("button");
                    cta.type = "button";
                    cta.className = "key-action key-cta";
                    cta.textContent = "Add your OpenAI key";
                    cta.addEventListener("click", openApiKeyPanel);
                    bubble.appendChild(cta);
                }
            } else {
                bubble.innerHTML = `<p>⚠️ ${escapeHtml(msg)}${requestIdNote}</p>`;
            }
        } else if (assistantText) {
            bubble.innerHTML = renderMarkdown(fullText);
        } else {
            bubble.innerHTML = `<p>⚠️ No response received.</p>`;
        }

        if (isComplete && !hasStreamError && streamMeta && streamMeta.citations && streamMeta.citations.length > 0) {
            appendCitations(bubble, streamMeta.citations);
        }

        if (isComplete && !hasStreamError && streamMeta && Array.isArray(streamMeta.evidence_spans) && streamMeta.evidence_spans.length > 0) {
            appendEvidenceUsed(bubble, streamMeta.evidence_spans);
        }

        if (isComplete && !hasStreamError && streamMeta && streamMeta.graph_available) {
            appendMechanismGraph(bubble, streamMeta);
        }

        if (isComplete && !hasStreamError && streamMeta) {
            appendTelemetry(bubble, streamMeta, endToEnd);
            appendFeedback(bubble);
        }

        if (isComplete && !hasStreamError && assistantText) {
            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: fullText });
            if (streamMeta && typeof streamMeta.conversation_summary === "string" && streamMeta.conversation_summary) {
                conversationSummary = streamMeta.conversation_summary;
            }
        }

    } catch (err) {
        // A frame queued mid-stream would otherwise repaint over this error.
        finalizeRender();
        bubble.innerHTML = `<p>⚠️ Network error: ${escapeHtml(err.message)}</p>`;
    } finally {
        finalizeRender();  // idempotent; also covers the early `!res.ok` return
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

    const citDiv = document.createElement("details");
    citDiv.className = "citations source-details";

    const summary = document.createElement("summary");
    summary.textContent = `Sources (${citations.length})`;
    citDiv.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-list";
    citations.forEach(c => {
        const item = document.createElement("div");
        item.className = "source-item";
        const parts = String(c).split(" — ");
        const source = document.createElement("span");
        source.className = "source-path";
        source.textContent = parts[0] || c;
        item.appendChild(source);
        if (parts[1]) {
            const section = document.createElement("span");
            section.className = "source-section";
            section.textContent = parts.slice(1).join(" — ");
            item.appendChild(section);
        }
        list.appendChild(item);
    });
    citDiv.appendChild(list);
    bubble.appendChild(citDiv);
}

async function sendMechanicsStudyQuestion(question, bubble) {
    const studyStart = performance.now();
    try {
        const res = await fetch(`${API_URL}/study/mechanics`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, mechanics_max_items: 8 }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            const detail = err.detail;
            const isObj = detail && typeof detail === "object";
            const code = isObj ? detail.error_code : null;
            const message = isObj ? detail.message : detail;
            if (code === "api_key_unavailable") {
                renderApiKeyError(bubble, message, "");
            } else {
                bubble.innerHTML = `<p>⚠️ ${escapeHtml((typeof message === "string" && message) || "Something went wrong.")}</p>`;
            }
            return;
        }

        const data = await res.json();
        bubble.innerHTML = renderMarkdown(data.answer || "No mechanics study answer received.");
        appendMechanicsMap(bubble, data);
        appendFeedback(bubble);
        history.push({ role: "user", content: question });
        history.push({ role: "assistant", content: data.answer || "" });

        const endToEnd = ((performance.now() - studyStart) / 1000).toFixed(2);
        const panel = document.createElement("div");
        panel.className = "mechanics-runtime-note";
        panel.textContent = `Mechanics study mode · Total ${endToEnd}s`;
        bubble.appendChild(panel);
    } catch (err) {
        bubble.innerHTML = `<p>⚠️ Network error: ${escapeHtml(err.message)}</p>`;
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        textarea.focus();
    }
}

function appendEvidenceUsed(bubble, spans) {
    const existing = bubble.querySelector(".evidence-used");
    if (existing) existing.remove();

    const details = document.createElement("details");
    details.className = "citations source-details evidence-used";

    const summary = document.createElement("summary");
    summary.textContent = `Evidence used (${spans.length})`;
    details.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-list";
    spans.slice(0, 8).forEach(span => {
        const item = document.createElement("div");
        item.className = "source-item";

        const source = document.createElement("span");
        source.className = "source-path";
        source.textContent = span.title || span.source_relpath || "Evidence span";
        item.appendChild(source);

        const section = document.createElement("span");
        section.className = "source-section";
        section.textContent = span.section_name || span.source_relpath || "";
        item.appendChild(section);

        const text = document.createElement("span");
        text.className = "source-section";
        const rawText = String(span.text || "").trim();
        text.textContent = rawText.length > 260 ? `${rawText.slice(0, 260)}...` : rawText;
        item.appendChild(text);

        list.appendChild(item);
    });
    details.appendChild(list);
    bubble.appendChild(details);
}

function appendMechanismGraph(bubble, meta) {
    const existing = bubble.querySelector(".mechanism-graph");
    if (existing) existing.remove();

    const nodes = Array.isArray(meta.graph_nodes) ? meta.graph_nodes : [];
    const paths = Array.isArray(meta.graph_paths) ? meta.graph_paths : [];
    const edges = Array.isArray(meta.graph_edges) ? meta.graph_edges : [];
    const spanCount = Array.isArray(meta.graph_supporting_spans) ? meta.graph_supporting_spans.length : 0;
    if (!nodes.length && !paths.length && !edges.length) return;

    const details = document.createElement("details");
    details.className = "citations source-details mechanism-graph";

    const summary = document.createElement("summary");
    summary.textContent = `Mechanism graph (${paths.length || edges.length})`;
    details.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-list";

    const conceptItem = document.createElement("div");
    conceptItem.className = "source-item";
    const concepts = nodes.slice(0, 8).map(n => n.canonical_name || n.node_id).filter(Boolean).join(", ");
    conceptItem.innerHTML = `<span class="source-path">Matched concepts</span><span class="source-section">${escapeHtml(concepts || "None")}</span>`;
    list.appendChild(conceptItem);

    paths.slice(0, 5).forEach(path => {
        const item = document.createElement("div");
        item.className = "source-item";
        const weakest = path.weakest_support_level ? `Weakest support: ${path.weakest_support_level}` : "Support not labeled";
        item.innerHTML = `<span class="source-path">${escapeHtml(path.path_text || path.path_id || "Mechanism path")}</span><span class="source-section">${escapeHtml(weakest)}</span>`;
        list.appendChild(item);
    });

    if (!paths.length) {
        edges.slice(0, 5).forEach(edge => {
            const item = document.createElement("div");
            item.className = "source-item";
            const label = `${edge.source || edge.source_node_id} ${edge.relation_type || "related_to"} ${edge.target || edge.target_node_id}`;
            item.innerHTML = `<span class="source-path">${escapeHtml(label)}</span><span class="source-section">Support: ${escapeHtml(edge.support_level || "unknown")}</span>`;
            list.appendChild(item);
        });
    }

    const stats = document.createElement("div");
    stats.className = "source-item";
    stats.innerHTML = `<span class="source-path">Graph support</span><span class="source-section">Evidence spans: ${spanCount} · Graph tokens: ${meta.graph_context_token_estimate || 0}</span>`;
    list.appendChild(stats);

    details.appendChild(list);
    bubble.appendChild(details);
}

function appendMechanicsMap(bubble, meta) {
    const existing = bubble.querySelector(".mechanics-map");
    if (existing) existing.remove();

    const nerves = Array.isArray(meta.mechanics_nerves) ? meta.mechanics_nerves : [];
    const sites = Array.isArray(meta.mechanics_entrapment_sites) ? meta.mechanics_entrapment_sites : [];
    const pairs = Array.isArray(meta.mechanics_muscle_pairs) ? meta.mechanics_muscle_pairs : [];
    const chains = Array.isArray(meta.mechanics_mechanism_chains) ? meta.mechanics_mechanism_chains : [];
    if (!nerves.length && !sites.length && !pairs.length && !chains.length) return;

    const details = document.createElement("details");
    details.className = "citations source-details mechanics-map";

    const summary = document.createElement("summary");
    summary.textContent = `Mechanics map (${nerves.length + sites.length + pairs.length + chains.length})`;
    details.appendChild(summary);

    const list = document.createElement("div");
    list.className = "source-list";

    function addRecord(group, label, support, detail) {
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `<span class="source-path">${escapeHtml(group)}: ${escapeHtml(label)}</span><span class="source-section">Support: ${escapeHtml(support || "unknown")} · ${escapeHtml(detail || "")}</span>`;
        list.appendChild(item);
    }

    nerves.slice(0, 5).forEach(record => addRecord("Nerve", record.name || record.nerve_id, record.support_level, record.course_summary));
    sites.slice(0, 5).forEach(record => addRecord("Site", record.site_name || record.site_id, record.support_level, record.mechanical_trigger));
    pairs.slice(0, 5).forEach(record => addRecord("Muscle pair", (record.muscles || []).join(", ") || record.pair_id, record.support_level, record.mechanical_role));
    chains.slice(0, 5).forEach(record => addRecord("Chain", record.chain_id, record.support_level, record.weakest_step));

    details.appendChild(list);
    bubble.appendChild(details);
}

// ── Telemetry ────────────────────────────────────────────────────────────────
function appendTelemetry(bubble, meta, endToEnd) {
    const rt = (meta.retrieval_time || 0).toFixed(2);
    const gt = (meta.generation_time || 0).toFixed(2);
    const conf = (meta.retrieval_confidence || 0);
    const rrMode = meta.reranker_mode || (meta.use_reranker ? "per_source" : "off");
    const confClass = conf >= 0.5 ? "confidence-high" : conf >= 0.3 ? "confidence-mid" : "confidence-low";

    // Answer/retrieval mode badges — highlighted amber when running on a fallback path.
    const answerMode = meta.answer_mode;
    const retrievalMode = meta.retrieval_mode;
    const answerDegraded = answerMode && answerMode !== "llm:openai" && answerMode !== "disabled";
    const retrievalDegraded = retrievalMode && retrievalMode !== "hybrid";
    const answerBadge = answerMode
        ? `<span class="stat-badge ${answerDegraded ? "mode-degraded" : ""}">Answer ${escapeHtml(answerMode)}</span>`
        : "";
    const retrievalBadge = retrievalDegraded
        ? `<span class="stat-badge mode-degraded">Retrieval ${escapeHtml(retrievalMode)}</span>`
        : "";
    const genModel = meta.generation_model;
    const modelBadge = genModel
        ? `<span class="stat-badge">Model ${escapeHtml(genModel)}</span>`
        : "";

    const toggleBtn = document.createElement("div");
    toggleBtn.className = "telemetry-toggle";
    toggleBtn.textContent = "Why this answer?";
    bubble.appendChild(toggleBtn);

    const panel = document.createElement("div");
    panel.className = "telemetry-panel";

    let html = `<div class="telemetry-stats">
        <span class="stat-badge">Retrieval ${rt}s</span>
        <span class="stat-badge">LLM ${gt}s</span>
        <span class="stat-badge">Total ${endToEnd}s</span>
        <span class="stat-badge">Reranker ${escapeHtml(rrMode)}</span>
        <span class="stat-badge">Context ${escapeHtml(meta.context_strategy || "chunk_pack")}</span>
        <span class="stat-badge">Graph ${escapeHtml(meta.graph_context_strategy || "off")}</span>
        <span class="stat-badge">Prompt ${meta.prompt_tokens || 0}</span>
        <span class="stat-badge">Output ${meta.output_tokens || 0}</span>
        <span class="stat-badge ${confClass}">Confidence ${conf.toFixed(2)}</span>
        ${modelBadge}
        ${answerBadge}
        ${retrievalBadge}
    </div>`;

    if (meta.category || meta.refined_query || meta.triage_level || meta.safety_gate_triggered) {
        html += '<div class="telemetry-detail">';
        if (meta.category) html += `<strong>Category:</strong> ${escapeHtml(meta.category_label || meta.category)}<br>`;
        if (meta.triage_level) html += `<strong>Triage:</strong> ${escapeHtml(meta.triage_level)}<br>`;
        if (meta.safety_gate_triggered) {
            const reasons = Array.isArray(meta.safety_gate_reasons) ? meta.safety_gate_reasons.join(", ") : "red_flag";
            html += `<strong>Safety gate:</strong> ${escapeHtml(reasons)}<br>`;
        }
        if (meta.scope_issue) html += `<strong>Scope boundary:</strong> ${escapeHtml(meta.scope_issue)}<br>`;
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

function appendFeedback(bubble) {
    const existing = bubble.querySelector(".feedback-row");
    if (existing) existing.remove();

    const wrap = document.createElement("div");
    wrap.className = "feedback-row";

    const label = document.createElement("span");
    label.className = "feedback-label";
    label.textContent = "Feedback:";
    wrap.appendChild(label);

    ["Helpful", "Unclear", "Felt unsafe", "Wrong source", "Not grounded"].forEach(text => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "feedback-btn";
        btn.textContent = text;
        btn.addEventListener("click", () => {
            wrap.querySelectorAll(".feedback-btn").forEach(el => el.classList.remove("selected"));
            btn.classList.add("selected");
            console.info("Local feedback selected:", text);
        });
        wrap.appendChild(btn);
    });

    bubble.appendChild(wrap);
}

// ── Markdown ─────────────────────────────────────────────────────────────────
function renderMarkdown(text) {
    if (!text) return "";
    // Full CommonMark/GFM rendering (tables, setext headings, * bullets, blockquotes)
    // via vendored marked + DOMPurify. Different generation providers emit different
    // markdown dialects; the old regex renderer only handled the GPT-4.1 subset.
    if (window.marked && window.DOMPurify) {
        const html = marked.parse(text, { gfm: true, breaks: true });
        return DOMPurify.sanitize(html);
    }
    return renderMarkdownFallback(text);
}

// Legacy regex renderer, kept only as a fallback if the vendored libs fail to load.
function renderMarkdownFallback(text) {
    let html = escapeHtml(text);
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/^(\d+)\. (.+)$/gm, "<li>$2</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/gs, match => `<ol>${match}</ol>`);
    html = html.replace(/^[-•*] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    return `<p>${html}</p>`;
}

function escapeHtml(str) {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    // Coerce: callers pass err.message (can be undefined) and server-supplied fields.
    return String(str ?? "").replace(/[&<>"']/g, c => map[c]);
}

// ── Event bindings ───────────────────────────────────────────────────────────
const sidebarScrim = document.getElementById("sidebar-scrim");
const sidebarCollapse = document.getElementById("sidebar-collapse");
if (sidebarScrim) {
    sidebarScrim.addEventListener("click", () => sidebar.classList.add("collapsed"));
}
if (sidebarCollapse) {
    sidebarCollapse.addEventListener("click", () => sidebar.classList.add("collapsed"));
}

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

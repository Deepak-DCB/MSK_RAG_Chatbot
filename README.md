<div style="font-family: Inter, system-ui, sans-serif; color: #e8ecf2; background: #0e1117; padding: 1rem; border-radius: 1rem; line-height: 1.4;">

  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 740" width="100%" style="background:#0e1117;">

    <!-- Title -->
    <text x="50%" y="40" text-anchor="middle" fill="#ffffff" font-size="22" font-weight="600">
      MSK Triage Chatbot — End-to-End RAG Pipeline
    </text>

    <!-- Data Layer -->
    <rect x="140" y="70" width="570" height="80" rx="12" fill="#1a3d7c" stroke="#4a78e0" stroke-width="2"/>
    <text x="425" y="110" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">Data Ingestion</text>
    <text x="425" y="130" text-anchor="middle" fill="#b9c5d8" font-size="13">textExtract.py → chunks.parquet + all_articles.jsonl</text>

    <!-- Arrow -->
    <polygon points="425,150 435,170 415,170" fill="#4a78e0"/>

    <!-- Embedding Layer -->
    <rect x="140" y="180" width="570" height="80" rx="12" fill="#1f4a1f" stroke="#47b347" stroke-width="2"/>
    <text x="425" y="220" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">Embedding Generation</text>
    <text x="425" y="240" text-anchor="middle" fill="#b9c5d8" font-size="13">embedding.py → mixedbread-ai/mxbai-large → embeddings.npy</text>

    <!-- Arrow -->
    <polygon points="425,250 435,270 415,270" fill="#47b347"/>

    <!-- Vector Store Layer -->
    <rect x="140" y="280" width="570" height="80" rx="12" fill="#1a383c" stroke="#41c0c8" stroke-width="2"/>
    <text x="425" y="320" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">Vector Database</text>
    <text x="425" y="340" text-anchor="middle" fill="#b9c5d8" font-size="13">ChromaDB.py → persistent Chroma store (msk_chunks)</text>

    <!-- Arrow -->
    <polygon points="425,350 435,370 415,370" fill="#41c0c8"/>

    <!-- Retrieval Layer -->
    <rect x="140" y="380" width="570" height="120" rx="12" fill="#433000" stroke="#f1b93e" stroke-width="2"/>
    <text x="425" y="415" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">Retrieval & QA Engine</text>
    <text x="425" y="435" text-anchor="middle" fill="#f2e2b0" font-size="13">qaEngine.py (v7.6)</text>
    <text x="425" y="455" text-anchor="middle" fill="#d6c27c" font-size="13">Biasing → Cross-Encoder Reranker → Token Budgeting → Ollama LLM</text>
    <text x="425" y="475" text-anchor="middle" fill="#d6c27c" font-size="13">Adaptive Conversation Memory (Decay, Penalty, Gating)</text>

    <!-- Arrow -->
    <polygon points="425,500 435,520 415,520" fill="#f1b93e"/>

    <!-- UI Layer -->
    <rect x="140" y="530" width="570" height="90" rx="12" fill="#3b2a4a" stroke="#a36cf0" stroke-width="2"/>
    <text x="425" y="565" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="600">User Interface</text>
    <text x="425" y="585" text-anchor="middle" fill="#d2c1f3" font-size="13">mskbot.py (Streamlit) → Chat UI + Metrics Strip + Citations Export</text>

    <!-- Arrow -->
    <polygon points="425,620 435,640 415,640" fill="#a36cf0"/>

    <!-- User Layer -->
    <rect x="140" y="650" width="570" height="60" rx="12" fill="#242424" stroke="#888" stroke-width="1.5"/>
    <text x="425" y="685" text-anchor="middle" fill="#f2f2f2" font-size="15" font-weight="600">User Queries ↔ Evidence-based Answers</text>

  </svg>

  <p style="color:#8fa0b8; font-size:13px; margin-top:0.5rem; text-align:center;">
    Blue = Data · Green = Embeddings · Cyan = Vector Store · Yellow = Retrieval · Purple = UI
  </p>
</div>
